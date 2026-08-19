"""Partitioned canonical Parquet storage with fail-closed atomic merges."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping

import pandas as pd

from src.data.contracts import CoverageKind, DatasetSpec


class CanonicalDataError(ValueError):
    pass


_NUMERIC_FIELDS = {
    "open", "high", "low", "close", "pre_close", "change", "pct_chg",
    "vol", "amount", "turnover_rate", "volume_ratio", "pe", "pe_ttm",
    "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm", "total_mv", "circ_mv",
    "adj_factor", "weight", "is_open",
}


def normalize_frame(spec: DatasetSpec, frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise CanonicalDataError("provider result must be a pandas DataFrame.")
    missing = [name for name in spec.required_fields if name not in frame.columns]
    if missing:
        raise CanonicalDataError(f"{spec.dataset_id} rows are missing required columns: {missing!r}.")
    result = frame.loc[:, list(spec.required_fields)].copy(deep=True)
    for column in ("trade_date", "cal_date", "list_date", "delist_date"):
        if column not in result.columns:
            continue
        values = result[column]
        nulls = values.isna() | values.astype(str).isin(("", "None", "NaT", "nan"))
        normalized = pd.to_datetime(values.mask(nulls), errors="coerce")
        invalid = normalized.isna() & ~nulls
        if invalid.any():
            raise CanonicalDataError(f"{column} contains invalid dates.")
        result[column] = normalized.dt.strftime("%Y-%m-%d").where(~nulls, None)
    for column in result.columns:
        if column in {"trade_date", "cal_date", "list_date", "delist_date"}:
            continue
        if column in _NUMERIC_FIELDS:
            converted = pd.to_numeric(result[column], errors="coerce")
            if (converted.isna() & result[column].notna()).any():
                raise CanonicalDataError(f"{column} contains non-numeric values.")
            result[column] = converted.astype("Int64" if column == "is_open" else "Float64")
        else:
            result[column] = result[column].astype("string")
    if "is_open" in result and not result["is_open"].dropna().isin((0, 1)).all():
        raise CanonicalDataError("is_open must contain only 0 or 1.")
    if result.loc[:, list(spec.primary_key)].isna().any().any():
        raise CanonicalDataError("primary-key fields must not be null.")
    if result.duplicated(list(spec.primary_key)).any():
        duplicates = result.loc[result.duplicated(list(spec.primary_key), False), list(spec.primary_key)]
        if duplicates.empty:
            raise CanonicalDataError("duplicate primary key.")
        grouped = result.groupby(list(spec.primary_key), dropna=False, sort=False)
        collapsed: list[pd.Series] = []
        for _, group in grouped:
            first = group.iloc[0]
            if not all(first.equals(group.iloc[index]) for index in range(1, len(group))):
                raise CanonicalDataError("conflicting duplicate primary-key payload.")
            collapsed.append(first)
        result = pd.DataFrame(collapsed, columns=result.columns)
    return result.sort_values(list(spec.primary_key), kind="mergesort", ignore_index=True)


def content_hash(spec: DatasetSpec, frame: pd.DataFrame) -> str:
    normalized = normalize_frame(spec, frame)
    payload = normalized.to_csv(index=False, lineterminator="\n", na_rep="<NA>", float_format="%.17g")
    return sha256(payload.encode("utf-8")).hexdigest()


def merge_canonical(spec: DatasetSpec, existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    left = normalize_frame(spec, existing)
    right = normalize_frame(spec, incoming)
    if left.empty:
        return right
    if right.empty:
        return left
    left_indexed = left.set_index(list(spec.primary_key), drop=False)
    right_indexed = right.set_index(list(spec.primary_key), drop=False)
    overlap = left_indexed.index.intersection(right_indexed.index)
    for key in overlap:
        old = left_indexed.loc[key]
        new = right_indexed.loc[key]
        if isinstance(old, pd.DataFrame) or isinstance(new, pd.DataFrame):
            raise CanonicalDataError("canonical store contains duplicate primary keys.")
        if not old.equals(new):
            if spec.dataset_id == "daily_basic" and pd.isna(old.get("dv_ttm")) and not pd.isna(new.get("dv_ttm")):
                shared = [name for name in spec.required_fields if name != "dv_ttm"]
                if not old.loc[shared].equals(new.loc[shared]):
                    raise CanonicalDataError("conflicting existing primary-key payload.")
                left_indexed.loc[key, "dv_ttm"] = new["dv_ttm"]
                continue
            raise CanonicalDataError("conflicting existing primary-key payload.")
    novel = right_indexed.loc[~right_indexed.index.isin(left_indexed.index)].reset_index(drop=True)
    updated_left = left_indexed.reset_index(drop=True)
    return normalize_frame(spec, pd.concat([updated_left, novel], ignore_index=True))


class PartitionedParquetStore:
    def __init__(self, root: str | Path, *, engine: str = "pyarrow", provider_id: str = "tushare_official") -> None:
        self.root = Path(root)
        self.engine = engine
        self.provider_id = provider_id

    @staticmethod
    def _entity(scope: tuple[tuple[str, str], ...]) -> str:
        values = dict(scope)
        return values.get("index_code") or values.get("exchange") or values.get("scope") or "default"

    def partition_path(self, spec: DatasetSpec, *, unit: str, scope: tuple[tuple[str, str], ...]) -> Path:
        parts = [self.root, Path(spec.dataset_id)]
        entity = self._entity(scope).replace("/", "_").replace("\\", "_")
        if "entity" in spec.storage_partition:
            parts.append(Path(f"entity={entity}"))
        if "year" in spec.storage_partition:
            parts.append(Path(f"year={unit[:4]}"))
        if "month" in spec.storage_partition:
            month = unit[5:7] if len(unit) >= 7 else "00"
            parts.append(Path(f"month={month}"))
        if "snapshot" in spec.storage_partition:
            parts.append(Path(f"snapshot={unit}"))
        return Path(*parts) / "data.parquet"

    def empty_marker_path(self, spec: DatasetSpec, *, unit: str, scope: tuple[tuple[str, str], ...]) -> Path:
        """Return the exact durable proof path for one valid empty unit."""
        scope_digest = sha256(
            json.dumps(dict(scope), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        safe_unit = unit.replace("/", "_").replace("\\", "_")
        return self.root / ".empty" / spec.dataset_id / scope_digest / f"{safe_unit}.json"

    def write_empty_marker(self, spec: DatasetSpec, *, unit: str, scope: tuple[tuple[str, str], ...]) -> Path:
        """Atomically persist a schema-bound proof for a provider-confirmed empty unit."""
        if not spec.allow_empty_complete:
            raise CanonicalDataError(f"{spec.dataset_id} does not permit empty completeness.")
        target = self.empty_marker_path(spec, unit=unit, scope=scope)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider_id": self.provider_id,
            "dataset_id": spec.dataset_id,
            "scope": dict(scope),
            "unit": unit,
            "schema_version": spec.schema_version,
            "required_fields": list(spec.required_fields),
        }
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent)
        os.close(descriptor)
        temp = Path(raw_path)
        try:
            temp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            with temp.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return target

    def has_empty_marker(self, spec: DatasetSpec, *, unit: str, scope: tuple[tuple[str, str], ...]) -> bool:
        path = self.empty_marker_path(spec, unit=unit, scope=scope)
        try:
            if path.is_symlink() or not path.is_file():
                return False
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return value == {
            "provider_id": self.provider_id,
            "dataset_id": spec.dataset_id,
            "scope": dict(scope),
            "unit": unit,
            "schema_version": spec.schema_version,
            "required_fields": list(spec.required_fields),
        }

    def load_partition(self, path: Path, spec: DatasetSpec) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=spec.required_fields)
        frame = pd.read_parquet(path, engine=self.engine)
        # daily_basic 1.1 added the provider-native dv_ttm column. Older local
        # partitions remain readable for targeted, missing-only repair, but the
        # ledger schema/hash check prevents them from being treated as 1.1
        # COMPLETE until every required unit is fetched and rewritten.
        if spec.dataset_id == "daily_basic" and "dv_ttm" not in frame.columns:
            frame = frame.copy()
            frame["dv_ttm"] = pd.NA
        return normalize_frame(spec, frame)

    def _write_temp(self, frame: pd.DataFrame, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp.parquet", dir=target.parent)
        os.close(descriptor)
        temp = Path(raw_path)
        try:
            frame.to_parquet(temp, index=False, engine=self.engine)
            with temp.open("r+b") as handle:
                os.fsync(handle.fileno())
            return temp
        except Exception:
            temp.unlink(missing_ok=True)
            raise

    def merge(
        self,
        spec: DatasetSpec,
        frame: pd.DataFrame,
        *,
        units: Iterable[str],
        scope: tuple[tuple[str, str], ...],
        provenance: Mapping[str, object] | None = None,
        transition: Callable[[str, str, Path, int, tuple[str, ...]], None] | None = None,
    ) -> tuple[Path, ...]:
        normalized = normalize_frame(spec, frame)
        paths: list[Path] = []
        by_path: dict[Path, list[str]] = {}
        for unit in units:
            by_path.setdefault(self.partition_path(spec, unit=unit, scope=scope), []).append(unit)
        for target, partition_units in by_path.items():
            if normalized.empty:
                continue
            date_col = "cal_date" if spec.coverage_kind is CoverageKind.CALENDAR_DATE else "trade_date"
            if spec.coverage_kind in {CoverageKind.GLOBAL_SNAPSHOT, CoverageKind.REFERENCE_EFFECTIVE_THROUGH}:
                incoming = normalized
            elif spec.coverage_kind is CoverageKind.ENTITY_MONTH:
                incoming = normalized.loc[normalized[date_col].str[:7].isin(partition_units)]
            else:
                incoming = normalized.loc[normalized[date_col].isin(partition_units)]
            if incoming.empty:
                continue
            merged = (
                incoming
                if spec.coverage_kind is CoverageKind.GLOBAL_SNAPSHOT
                else merge_canonical(spec, self.load_partition(target, spec), incoming)
            )
            temp = self._write_temp(merged, target)
            try:
                verified = normalize_frame(spec, pd.read_parquet(temp, engine=self.engine))
                if content_hash(spec, verified) != content_hash(spec, merged):
                    raise CanonicalDataError("temporary Parquet verification failed.")
                if transition is not None:
                    transition(
                        "CANONICAL_VALIDATED", "PARQUET_TEMP_VERIFIED", target,
                        len(verified), tuple(str(name) for name in verified.columns),
                    )
                os.replace(temp, target)
                if transition is not None:
                    transition(
                        "CANONICAL_COMMITTED", "PARQUET_ATOMIC_REPLACE", target,
                        len(verified), tuple(str(name) for name in verified.columns),
                    )
                manifest = target.with_suffix(target.suffix + ".manifest.json")
                descriptor, raw_manifest = tempfile.mkstemp(
                    prefix=f".{manifest.stem}.", suffix=".tmp", dir=manifest.parent
                )
                os.close(descriptor)
                manifest_temp = Path(raw_manifest)
                manifest_temp.write_text(json.dumps({
                    **dict(provenance or {}),
                    "provider_id": self.provider_id,
                    "dataset_id": spec.dataset_id,
                    "schema_version": spec.schema_version,
                    "row_count": len(verified),
                    "content_hash": content_hash(spec, verified),
                }, sort_keys=True, separators=(",", ":")), encoding="utf-8")
                try:
                    with manifest_temp.open("r+b") as handle:
                        os.fsync(handle.fileno())
                    os.replace(manifest_temp, manifest)
                    if transition is not None:
                        transition(
                            "CANONICAL_COMMITTED", "MANIFEST_ATOMIC_REPLACE", manifest,
                            len(verified), tuple(str(name) for name in verified.columns),
                        )
                finally:
                    manifest_temp.unlink(missing_ok=True)
            finally:
                temp.unlink(missing_ok=True)
            paths.append(target)
        return tuple(paths)

    def rows_for_unit(self, spec: DatasetSpec, *, unit: str, scope: tuple[tuple[str, str], ...]) -> pd.DataFrame:
        frame = self.load_partition(self.partition_path(spec, unit=unit, scope=scope), spec)
        if frame.empty or spec.coverage_kind in {CoverageKind.GLOBAL_SNAPSHOT, CoverageKind.REFERENCE_EFFECTIVE_THROUGH}:
            return frame
        column = "cal_date" if spec.coverage_kind is CoverageKind.CALENDAR_DATE else "trade_date"
        mask = frame[column].str[:7].eq(unit) if spec.coverage_kind is CoverageKind.ENTITY_MONTH else frame[column].eq(unit)
        return normalize_frame(spec, frame.loc[mask])


class RawParquetStore:
    def __init__(self, root: str | Path, *, engine: str = "pyarrow", provider_id: str = "tushare_official") -> None:
        self.root = Path(root)
        self.engine = engine
        self.provider_id = provider_id

    def save(self, dataset_id: str, fetch_id: str, frame: pd.DataFrame) -> Path:
        target = self.root / self.provider_id / dataset_id / f"{fetch_id}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError("raw fetch identity already exists.")
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{target.stem}.", suffix=".tmp.parquet", dir=target.parent
        )
        os.close(descriptor)
        temp = Path(raw_path)
        try:
            frame.to_parquet(temp, index=False, engine=self.engine)
            with temp.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return target
