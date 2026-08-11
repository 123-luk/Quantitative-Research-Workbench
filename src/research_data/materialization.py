"""Canonical ResearchInputBuilder with content identity and atomic publication."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Protocol

import pandas as pd

from src.factors.research_pipeline import FactorResearchRunner
from src.modeling_panel import ModelingPanelBuilder, ModelingPanelConfig
from src.ml import MLDatasetBuilder, MLDatasetConfig
from src.research_data.adjusted_prices import AdjustedPriceRequest, AdjustedPriceService, CanonicalMarketSlice
from src.research_data.calendar import ResearchCalendar
from src.research_data.planning import ResearchInputDataUnavailable, ResearchInputError, ResearchInputPlan
from src.universe import UniverseDataSource, UniverseService


class ResearchDatasetSource(Protocol):
    def load(self, dataset_id: str, dates: tuple[str, ...]) -> CanonicalMarketSlice: ...


_RETURN_COLUMNS = (
    "trade_date",
    "ts_code",
    "entry_trade_date",
    "exit_trade_date",
    "entry_price",
    "exit_price",
    "forward_return",
)


def _frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n", na_rep="<NA>", float_format="%.17g", date_format="%Y-%m-%d")
    return sha256(payload.encode("utf-8")).hexdigest()


def _json_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResearchInputMaterialization:
    materialization_id: str
    directory: Path
    reused: bool
    paths: Mapping[str, Path]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        root = self.directory.resolve()
        if not root.is_dir() or root.is_symlink():
            raise ResearchInputDataUnavailable("materialization directory must be a regular directory.")
        paths = {name: path.resolve() for name, path in dict(self.paths).items()}
        if any(not path.is_file() or path.is_symlink() or path.parent != root for path in paths.values()):
            raise ResearchInputDataUnavailable("materialization paths must be regular direct-child files.")
        object.__setattr__(self, "directory", root)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


class ResearchMaterializationStore:
    """Publish one validated content-addressed input set via directory rename."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _validated_existing(self, target: Path, materialization_id: str) -> ResearchInputMaterialization:
        manifest_path = target / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ResearchInputDataUnavailable("existing materialization manifest is invalid.") from exc
        if manifest.get("materialization_id") != materialization_id or set(manifest.get("files", {})) != set(manifest.get("targets", [])):
            raise ResearchInputDataUnavailable("existing materialization identity or target set is invalid.")
        paths: dict[str, Path] = {}
        for name, expected_hash in manifest["files"].items():
            if Path(name).name != name or name == "manifest.json":
                raise ResearchInputDataUnavailable("existing materialization contains an unsafe filename.")
            path = target / name
            if not path.is_file() or path.is_symlink() or self._hash_file(path) != expected_hash:
                raise ResearchInputDataUnavailable(f"existing materialization file failed validation: {name}.")
            paths[name] = path
        return ResearchInputMaterialization(materialization_id, target, True, paths, manifest.get("diagnostics", {}))

    def publish(self, materialization_id: str, frames: Mapping[str, pd.DataFrame], manifest_values: Mapping[str, object]) -> ResearchInputMaterialization:
        if not isinstance(materialization_id, str) or len(materialization_id) != 64 or any(character not in "0123456789abcdef" for character in materialization_id):
            raise ResearchInputError("materialization_id must be a SHA-256 hex identity.")
        target = self.root / materialization_id
        if target.exists():
            return self._validated_existing(target, materialization_id)
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{materialization_id}.", dir=self.root))
        try:
            files: dict[str, str] = {}
            paths: dict[str, Path] = {}
            for name in sorted(frames):
                if Path(name).name != name or not name.endswith(".parquet"):
                    raise ResearchInputError("materialization outputs must use safe Parquet filenames.")
                path = staging / name
                frames[name].to_parquet(path, index=False, engine="pyarrow")
                verified = pd.read_parquet(path, engine="pyarrow")
                if _frame_hash(verified) != _frame_hash(frames[name]):
                    raise ResearchInputDataUnavailable(f"staged materialization verification failed: {name}.")
                files[name] = self._hash_file(path)
                paths[name] = path
            manifest = dict(manifest_values)
            manifest.update({"materialization_id": materialization_id, "targets": sorted(frames), "files": files})
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
            with manifest_path.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(staging, target)
            return ResearchInputMaterialization(materialization_id, target, False, {name: target / name for name in paths}, manifest.get("diagnostics", {}))
        except Exception:
            if staging.exists() and staging.parent == self.root and staging.name.startswith(f".{materialization_id}."):
                shutil.rmtree(staging)
            raise


class ResearchInputBuilder:
    """Join P4C1/P4C2 services into existing factor/modeling input contracts."""

    def __init__(self, *, calendar: ResearchCalendar, universe_service: UniverseService, universe_data: UniverseDataSource, factor_registry: object, dataset_source: ResearchDatasetSource, adjusted_prices: AdjustedPriceService, factor_runner: FactorResearchRunner, store: ResearchMaterializationStore, modeling_builder: ModelingPanelBuilder | None = None) -> None:
        self.calendar = calendar
        self.universe_service = universe_service
        self.universe_data = universe_data
        self.factor_registry = factor_registry
        self.dataset_source = dataset_source
        self.adjusted_prices = adjusted_prices
        self.factor_runner = factor_runner
        self.store = store
        self.modeling_builder = modeling_builder

    @staticmethod
    def _dates(calendar: ResearchCalendar, start: str, end: str) -> tuple[str, ...]:
        return tuple(item for item in calendar.open_dates if start <= item <= end)

    @staticmethod
    def _normalized_slice(source: CanonicalMarketSlice, dates: tuple[str, ...], securities: tuple[str, ...], fields: tuple[str, ...]) -> pd.DataFrame:
        required = {"trade_date", "ts_code", *fields}
        if not required.issubset(source.frame.columns):
            raise ResearchInputDataUnavailable(f"Canonical {source.dataset_id} is missing factor input fields: {sorted(required - set(source.frame.columns))!r}.")
        frame = source.frame.loc[source.frame["trade_date"].isin(set(dates)) & source.frame["ts_code"].isin(set(securities)), ["trade_date", "ts_code", *fields]].copy()
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise ResearchInputDataUnavailable(f"Canonical {source.dataset_id} contains duplicate requested keys.")
        return frame

    def _factor_input(self, plan: ResearchInputPlan, securities: tuple[str, ...]) -> tuple[pd.DataFrame, tuple[str, ...]]:
        needed_fields = tuple(sorted({field for name in plan.factor_ids for field in self.factor_registry.get(name).metadata.source_fields}))
        field_datasets: dict[str, str] = {}
        dataset_starts: dict[str, str] = {}
        adjusted_start: str | None = None
        for name, spec in plan.factor_frequency_specs:
            factor_fields = tuple(self.factor_registry.get(name).metadata.source_fields)
            history_start = self.calendar.resolve_history(plan.formation_dates[0], spec.history_requirement).start_date
            for dataset, fields in spec.required_fields.items():
                for field in fields:
                    if field in {"trade_date", "ts_code"}:
                        continue
                    if field not in factor_fields:
                        continue
                    previous = field_datasets.setdefault(field, dataset)
                    if previous != dataset:
                        raise ResearchInputError(f"factor source field {field!r} has conflicting dataset owners.")
                    dataset_starts[dataset] = min(dataset_starts.get(dataset, history_start), history_start)
            if any(field.startswith("adj_") for field in factor_fields):
                adjusted_start = history_start if adjusted_start is None else min(adjusted_start, history_start)
        frames: list[pd.DataFrame] = []
        identities: list[str] = []
        for dataset in sorted(set(field_datasets.values())):
            fields = tuple(field for field in needed_fields if field_datasets.get(field) == dataset and not field.startswith("adj_"))
            if not fields:
                continue
            dataset_dates = self._dates(self.calendar, dataset_starts[dataset], plan.formation_dates[-1])
            source = self.dataset_source.load(dataset, dataset_dates)
            frames.append(self._normalized_slice(source, dataset_dates, securities, fields))
            identities.append(source.source_identity)
        adjusted_fields = tuple(field for field in needed_fields if field.startswith("adj_"))
        if adjusted_fields:
            if adjusted_start is None:
                raise ResearchInputDataUnavailable("adjusted factor fields have no history start.")
            adjusted_dates = self._dates(self.calendar, adjusted_start, plan.formation_dates[-1])
            raw_fields = tuple(field[4:] for field in adjusted_fields)
            adjusted = self.adjusted_prices.compute(AdjustedPriceRequest(securities, adjusted_dates, raw_fields))
            frames.append(adjusted.frame.loc[:, ["trade_date", "ts_code", *adjusted_fields]])
            identities.append(adjusted.source_identity)
        unresolved = tuple(field for field in needed_fields if field not in field_datasets and field not in adjusted_fields)
        if unresolved:
            raise ResearchInputDataUnavailable(f"factor source fields have no canonical dataset owner: {unresolved!r}.")
        if not frames:
            raise ResearchInputDataUnavailable("factor input materialization produced no source frames.")
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["trade_date", "ts_code"], how="outer", sort=False, validate="one_to_one")
        columns = ["trade_date", "ts_code", *needed_fields]
        merged["trade_date"] = pd.to_datetime(merged["trade_date"])
        result = merged.loc[:, columns].sort_values(["trade_date", "ts_code"], kind="mergesort", ignore_index=True)
        if result["trade_date"].max() > pd.Timestamp(plan.end_date):
            raise ResearchInputDataUnavailable("factor input contains a future observation beyond the research interval.")
        return result, tuple(identities)

    def _price_panel(self, plan: ResearchInputPlan, securities: tuple[str, ...]) -> tuple[pd.DataFrame, str]:
        forward = plan.forward_return_spec
        exit_date = self.calendar.shift_open_date(plan.formation_dates[-1], forward.entry_lag_periods + forward.horizon)
        dates = self._dates(self.calendar, plan.formation_dates[0], exit_date)
        adjusted = self.adjusted_prices.compute(AdjustedPriceRequest(securities, dates, ("close",)))
        panel = adjusted.frame.loc[:, ["trade_date", "ts_code", "adj_close"]].rename(columns={"adj_close": forward.price_column})
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        observed_dates = tuple(panel["trade_date"].drop_duplicates().sort_values().dt.strftime("%Y-%m-%d"))
        if observed_dates != dates:
            raise ResearchInputDataUnavailable("adjusted price panel cannot prove every open date required by the forward horizon.")
        return panel.sort_values(["trade_date", "ts_code"], kind="mergesort", ignore_index=True), adjusted.source_identity

    @staticmethod
    def _schedule(plan: ResearchInputPlan, snapshots: tuple[object, ...]) -> pd.DataFrame:
        rows = []
        for formation, snapshot in zip(plan.formation_dates, snapshots, strict=True):
            for code in snapshot.securities:
                rows.append({"trade_date": pd.Timestamp(formation), "ts_code": code})
        result = pd.DataFrame(rows, columns=("trade_date", "ts_code"))
        if result.empty:
            raise ResearchInputDataUnavailable("universe schedule contains no research members.")
        if result.duplicated(["trade_date", "ts_code"]).any():
            raise ResearchInputDataUnavailable("universe schedule contains duplicate keys.")
        return result.sort_values(["trade_date", "ts_code"], kind="mergesort", ignore_index=True)

    def _identity(self, plan: ResearchInputPlan, source_identities: tuple[str, ...], frames: Mapping[str, pd.DataFrame]) -> str:
        calculators = [{"factor_id": name, "calculator_id": spec.calculator_id, "version": self.factor_registry.get(name).metadata.version} for name, spec in plan.factor_frequency_specs]
        payload = {
            "plan_id": plan.plan_id,
            "source_identities": sorted(source_identities),
            "input_hashes": {name: _frame_hash(frame) for name, frame in sorted(frames.items())},
            "calculators": calculators,
            "runner": self.factor_runner.describe_config(),
            "schema": "research_input_1.0",
        }
        return _json_hash(payload)

    def build(self, plan: ResearchInputPlan) -> ResearchInputMaterialization:
        if not isinstance(plan, ResearchInputPlan):
            raise TypeError("plan must be a ResearchInputPlan.")
        if tuple(self.factor_runner.config.factor_names) != plan.factor_ids:
            raise ResearchInputError("factor runner factor_names differ from ResearchInputPlan.")
        if self.factor_runner.forward_return_config != plan.forward_return_spec.to_config():
            raise ResearchInputError("factor runner forward-return config differs from ResearchInputPlan.")
        if self.factor_runner.config.use_neutralization:
            raise ResearchInputError("P4C3 canonical exposure materialization is not implemented; neutralization must remain disabled.")
        snapshots = tuple(self.universe_service.resolve(plan.universe_spec, formation, self.universe_data) for formation in plan.formation_dates)
        score_panel = self._schedule(plan, snapshots)
        securities = tuple(sorted({code for snapshot in snapshots for code in snapshot.securities}))
        factor_input, factor_sources = self._factor_input(plan, securities)
        price_panel, price_source = self._price_panel(plan, securities)
        source_identities = tuple(snapshot.source_identity for snapshot in snapshots) + factor_sources + (price_source,)
        identity_inputs = {"factor_input.parquet": factor_input, "price_panel.parquet": price_panel, "score_panel.parquet": score_panel}
        materialization_id = self._identity(plan, source_identities, identity_inputs)
        target = self.store.root / materialization_id
        if target.exists():
            return self.store._validated_existing(target, materialization_id)
        try:
            result = self.factor_runner.run(factor_input, score_panel, price_panel)
        except Exception as exc:
            raise ResearchInputDataUnavailable(f"canonical factor/forward materialization failed: {exc}") from exc
        factor_panel = result.final_factor_panel.loc[:, ["trade_date", "ts_code", *plan.factor_ids]].sort_values(["trade_date", "ts_code"], kind="mergesort", ignore_index=True)
        return_columns = (*_RETURN_COLUMNS[:-1], plan.forward_return_spec.return_column)
        forward_returns = result.forward_returns.loc[:, list(return_columns)].sort_values(["trade_date", "ts_code"], kind="mergesort", ignore_index=True)
        labels = forward_returns.copy(deep=True)
        labels["available_at"] = labels["exit_trade_date"]
        score_keys = list(map(tuple, score_panel.loc[:, ["trade_date", "ts_code"]].itertuples(index=False, name=None)))
        for name, frame in (("modeling_factor_panel", factor_panel), ("modeling_forward_returns", forward_returns), ("labels_with_availability", labels)):
            keys = list(map(tuple, frame.loc[:, ["trade_date", "ts_code"]].itertuples(index=False, name=None)))
            if keys != score_keys:
                raise ResearchInputDataUnavailable(f"{name} keys differ from the point-in-time universe schedule.")
        modeling_builder = self.modeling_builder or ModelingPanelBuilder(ModelingPanelConfig(label_column=plan.forward_return_spec.return_column, include_features=plan.factor_ids, require_entry_after_signal=plan.forward_return_spec.entry_lag_periods > 0))
        try:
            modeling_builder.build(factor_panel, forward_returns)
            MLDatasetBuilder(MLDatasetConfig(label_col=plan.forward_return_spec.return_column)).build(factor_panel, forward_returns, plan.factor_ids)
        except Exception as exc:
            raise ResearchInputDataUnavailable(f"canonical modeling input validation failed: {exc}") from exc
        frames = {
            **identity_inputs,
            "modeling_factor_panel.parquet": factor_panel,
            "modeling_forward_returns.parquet": forward_returns,
            "labels_with_availability.parquet": labels,
        }
        diagnostics = {"formation_count": len(plan.formation_dates), "universe_rows": len(score_panel), "factor_input_rows": len(factor_input), "price_rows": len(price_panel), "score_panel_semantics": "formation/universe selection keys only; not ML predictions", "forward_formula": "exit_price / entry_price - 1", "recomputed": True}
        return self.store.publish(materialization_id, frames, {"schema_version": "research_input_1.0", "plan": plan.to_dict(), "plan_id": plan.plan_id, "source_identities": sorted(source_identities), "diagnostics": diagnostics})
