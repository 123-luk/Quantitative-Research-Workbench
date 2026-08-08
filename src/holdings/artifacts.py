"""Safe persistence and independent validation for V5 Holdings Artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from uuid import uuid4

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.holdings.builder import (
    WEIGHT_SUM_ABSOLUTE_TOLERANCE,
    HoldingsBuildResult,
)
from src.holdings.contracts import (
    HOLDINGS_KEY_COLUMNS,
    HOLDINGS_OUTPUT_COLUMNS,
    HOLDINGS_SCHEMA_VERSION,
    HoldingsContractError,
)


HOLDINGS_ARTIFACT_SCHEMA_VERSION = "1.0"
HOLDINGS_ARTIFACT_TYPE = "holdings"
HOLDINGS_PARQUET_FILENAME = "holdings.parquet"
HOLDINGS_CONFIG_FILENAME = "config.json"
HOLDINGS_AUDIT_FILENAME = "audit.json"
HOLDINGS_MANIFEST_FILENAME = "manifest.json"
HOLDINGS_ARTIFACT_FILENAMES = (
    HOLDINGS_PARQUET_FILENAME,
    HOLDINGS_CONFIG_FILENAME,
    HOLDINGS_AUDIT_FILENAME,
    HOLDINGS_MANIFEST_FILENAME,
)

_PAYLOAD_FILENAMES = HOLDINGS_ARTIFACT_FILENAMES[:-1]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISSUE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CHUNK_SIZE = 1024 * 1024
_CONFIG_FIELDS = {"top_n", "insufficient_universe_policy", "weighting"}
_PROVENANCE_FIELDS = {
    "signal_artifact_dir", "signal_path", "signal_schema_version",
    "signal_sha256",
}
_DATE_COUNT_FIELDS = {
    "trade_date", "available_count", "selected_count", "partial",
}
_AUDIT_FIELDS = {
    "holdings_schema_version", "input_rows", "output_rows", "trade_date_count",
    "min_trade_date", "max_trade_date", "requested_top_n",
    "insufficient_universe_policy", "weighting", "per_date_counts",
    "partial_dates", "weight_sum_min", "weight_sum_max",
    "weight_sum_tolerance", "duplicate_key_count",
    "source_signal_provenance", "warnings",
}
_MANIFEST_FIELDS = {
    "artifact_type", "artifact_schema_version", "holdings_schema_version",
    "created_at_utc", "row_count", "column_count", "columns",
    "pandas_dtypes", "top_n", "insufficient_universe_policy", "weighting",
    "source_signal_provenance", "files",
}


class HoldingsArtifactError(HoldingsContractError):
    """Base error for Holdings Artifact operations."""


class HoldingsArtifactExistsError(HoldingsArtifactError):
    """Raised when no-overwrite prevents publication."""


class HoldingsArtifactWriteError(HoldingsArtifactError):
    """Raised when a Holdings Artifact cannot be safely written."""


class HoldingsArtifactValidationError(HoldingsArtifactError):
    """Raised when strict Artifact metadata or API input is invalid."""


def _strict_keys(value: object, expected: set[str], context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise HoldingsArtifactValidationError(
            f"{context} must be a mapping with string keys."
        )
    if set(value) != expected:
        raise HoldingsArtifactValidationError(f"{context} fields are invalid.")
    return dict(value)


def _path(value: object) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise HoldingsArtifactValidationError(
            "artifact_dir must be str or os.PathLike."
        )
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise HoldingsArtifactValidationError("artifact_dir must be path-like.") from exc
    if not isinstance(raw, str) or not raw.strip() or raw != raw.strip() or "\x00" in raw:
        raise HoldingsArtifactValidationError(
            "artifact_dir must identify a non-empty trimmed directory."
        )
    result = Path(raw)
    if raw in {".", ".."} or result == Path(result.anchor):
        raise HoldingsArtifactValidationError(
            "artifact_dir must identify an explicit child directory."
        )
    return result


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _trimmed(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise HoldingsArtifactValidationError(
            f"{context} must be a non-empty trimmed string."
        )
    return value


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HoldingsArtifactValidationError(
            f"{context} must be a non-negative integer."
        )
    return value


@dataclass(frozen=True)
class SignalArtifactProvenance:
    """Minimal immutable identity of the direct Signal Artifact source."""

    signal_artifact_dir: Path
    signal_path: Path
    signal_schema_version: str
    signal_sha256: str

    def __post_init__(self) -> None:
        directory = _absolute(Path(self.signal_artifact_dir))
        signal_path = _absolute(Path(self.signal_path))
        if (
            signal_path.parent != directory
            or signal_path.name != "signals.parquet"
            or not directory.is_absolute()
        ):
            raise HoldingsArtifactValidationError(
                "source Signal provenance paths are invalid."
            )
        version = _trimmed(self.signal_schema_version, "signal_schema_version")
        if not isinstance(self.signal_sha256, str) or not _SHA256_RE.fullmatch(self.signal_sha256):
            raise HoldingsArtifactValidationError("signal_sha256 is invalid.")
        object.__setattr__(self, "signal_artifact_dir", directory)
        object.__setattr__(self, "signal_path", signal_path)
        object.__setattr__(self, "signal_schema_version", version)

    @classmethod
    def from_signal_write_result(cls, result: object) -> SignalArtifactProvenance:
        """Extract verified direct provenance from a public Signal write result."""
        from src.signals.artifacts import (
            SIGNAL_PARQUET_FILENAME,
            SignalArtifactWriteResult,
        )

        if not isinstance(result, SignalArtifactWriteResult) or not result.validation.is_valid:
            raise HoldingsArtifactValidationError(
                "result must be a valid SignalArtifactWriteResult."
            )
        record = next(
            (
                item for item in result.manifest.files
                if item.relative_path == SIGNAL_PARQUET_FILENAME
            ),
            None,
        )
        if record is None:
            raise HoldingsArtifactValidationError(
                "Signal write result has no Signal Parquet record."
            )
        return cls(
            signal_artifact_dir=result.artifact_dir,
            signal_path=result.signal_path,
            signal_schema_version=result.manifest.signal_schema_version,
            signal_sha256=record.sha256,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "signal_artifact_dir": str(self.signal_artifact_dir),
            "signal_path": str(self.signal_path),
            "signal_schema_version": self.signal_schema_version,
            "signal_sha256": self.signal_sha256,
        }


def _provenance(value: object) -> dict[str, str]:
    data = _strict_keys(value, _PROVENANCE_FIELDS, "source Signal provenance")
    normalized = {
        name: _trimmed(data[name], f"source Signal provenance {name}")
        for name in (
            "signal_artifact_dir", "signal_path", "signal_schema_version",
            "signal_sha256",
        )
    }
    if not _SHA256_RE.fullmatch(normalized["signal_sha256"]):
        raise HoldingsArtifactValidationError("source Signal SHA-256 is invalid.")
    directory = Path(normalized["signal_artifact_dir"])
    signal_path = Path(normalized["signal_path"])
    if (
        not directory.is_absolute()
        or not signal_path.is_absolute()
        or signal_path.parent != directory
        or signal_path.name != "signals.parquet"
    ):
        raise HoldingsArtifactValidationError(
            "source Signal provenance paths are invalid."
        )
    return normalized


@dataclass(frozen=True)
class HoldingsArtifactConfig:
    artifact_dir: Path
    parquet_compression: str = "zstd"
    verify_after_write: bool = True

    def __post_init__(self) -> None:
        path = _path(self.artifact_dir)
        compression = self.parquet_compression
        if not isinstance(compression, str):
            raise HoldingsArtifactValidationError(
                "parquet_compression must be a string."
            )
        compression = compression.strip().lower()
        if compression not in {"zstd", "snappy"}:
            raise HoldingsArtifactValidationError(
                "parquet_compression must be 'zstd' or 'snappy'."
            )
        if type(self.verify_after_write) is not bool:
            raise HoldingsArtifactValidationError(
                "verify_after_write must be a bool."
            )
        object.__setattr__(self, "artifact_dir", path)
        object.__setattr__(self, "parquet_compression", compression)


@dataclass(frozen=True)
class HoldingsArtifactFileRecord:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.relative_path not in _PAYLOAD_FILENAMES:
            raise HoldingsArtifactValidationError(
                "file record relative_path is unsafe or unsupported."
            )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise HoldingsArtifactValidationError(
                "file record size_bytes must be a positive integer."
            )
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise HoldingsArtifactValidationError(
                "file record sha256 must be lowercase hexadecimal."
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> HoldingsArtifactFileRecord:
        return cls(**_strict_keys(
            value, {"relative_path", "size_bytes", "sha256"}, "file record"
        ))  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _business_config(value: object) -> dict[str, object]:
    data = _strict_keys(value, _CONFIG_FIELDS, "Holdings config")
    top_n = data["top_n"]
    if type(top_n) is not int or top_n < 1:
        raise HoldingsArtifactValidationError("top_n must be a strict int >= 1.")
    policy = _trimmed(
        data["insufficient_universe_policy"], "insufficient_universe_policy"
    )
    if policy not in {"error", "allow_partial"}:
        raise HoldingsArtifactValidationError(
            "insufficient_universe_policy is invalid."
        )
    weighting = _trimmed(data["weighting"], "weighting")
    if weighting != "equal_weight":
        raise HoldingsArtifactValidationError("weighting is invalid.")
    return {
        "top_n": top_n,
        "insufficient_universe_policy": policy,
        "weighting": weighting,
    }


@dataclass(frozen=True)
class HoldingsArtifactManifest:
    artifact_type: str
    artifact_schema_version: str
    holdings_schema_version: str
    created_at_utc: str
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    pandas_dtypes: tuple[tuple[str, str], ...]
    top_n: int
    insufficient_universe_policy: str
    weighting: str
    source_signal_provenance: Mapping[str, str]
    files: tuple[HoldingsArtifactFileRecord, ...]

    def __post_init__(self) -> None:
        if self.artifact_type != HOLDINGS_ARTIFACT_TYPE:
            raise HoldingsArtifactValidationError("artifact_type is invalid.")
        if self.artifact_schema_version != HOLDINGS_ARTIFACT_SCHEMA_VERSION:
            raise HoldingsArtifactValidationError("artifact_schema_version is invalid.")
        if self.holdings_schema_version != HOLDINGS_SCHEMA_VERSION:
            raise HoldingsArtifactValidationError("holdings_schema_version is invalid.")
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc.endswith("Z"):
            raise HoldingsArtifactValidationError("created_at_utc must be UTC Z.")
        try:
            created = datetime.fromisoformat(self.created_at_utc[:-1] + "+00:00")
        except ValueError as exc:
            raise HoldingsArtifactValidationError(
                "created_at_utc is not valid ISO-8601."
            ) from exc
        if created.utcoffset() != timezone.utc.utcoffset(created):
            raise HoldingsArtifactValidationError("created_at_utc must be UTC.")
        rows = _nonnegative_int(self.row_count, "row_count")
        if rows == 0:
            raise HoldingsArtifactValidationError("row_count must be positive.")
        columns = tuple(self.columns)
        if columns != HOLDINGS_OUTPUT_COLUMNS or self.column_count != len(columns):
            raise HoldingsArtifactValidationError("manifest columns are invalid.")
        dtypes = tuple(tuple(item) for item in self.pandas_dtypes)
        if (
            len(dtypes) != len(columns)
            or any(len(item) != 2 for item in dtypes)
            or tuple(item[0] for item in dtypes) != columns
            or any(not isinstance(item[1], str) or not item[1] for item in dtypes)
        ):
            raise HoldingsArtifactValidationError("pandas_dtypes are invalid.")
        config = _business_config({
            "top_n": self.top_n,
            "insufficient_universe_policy": self.insufficient_universe_policy,
            "weighting": self.weighting,
        })
        provenance = _provenance(self.source_signal_provenance)
        files = tuple(self.files)
        if (
            len(files) != len(_PAYLOAD_FILENAMES)
            or any(not isinstance(item, HoldingsArtifactFileRecord) for item in files)
            or tuple(item.relative_path for item in files) != _PAYLOAD_FILENAMES
        ):
            raise HoldingsArtifactValidationError("manifest files are invalid.")
        object.__setattr__(self, "row_count", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "pandas_dtypes", dtypes)
        object.__setattr__(self, "top_n", config["top_n"])
        object.__setattr__(self, "insufficient_universe_policy", config["insufficient_universe_policy"])
        object.__setattr__(self, "weighting", config["weighting"])
        object.__setattr__(self, "source_signal_provenance", MappingProxyType(provenance))
        object.__setattr__(self, "files", files)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> HoldingsArtifactManifest:
        data = _strict_keys(value, _MANIFEST_FIELDS, "manifest")
        try:
            raw_files = data["files"]
            if (
                not isinstance(data["columns"], list)
                or not isinstance(data["pandas_dtypes"], list)
                or any(not isinstance(item, list) for item in data["pandas_dtypes"])
                or not isinstance(raw_files, list)
            ):
                raise TypeError("manifest sequences must be lists")
            return cls(
                artifact_type=data["artifact_type"],  # type: ignore[arg-type]
                artifact_schema_version=data["artifact_schema_version"],  # type: ignore[arg-type]
                holdings_schema_version=data["holdings_schema_version"],  # type: ignore[arg-type]
                created_at_utc=data["created_at_utc"],  # type: ignore[arg-type]
                row_count=data["row_count"],  # type: ignore[arg-type]
                column_count=data["column_count"],  # type: ignore[arg-type]
                columns=tuple(data["columns"]),  # type: ignore[arg-type]
                pandas_dtypes=tuple(tuple(item) for item in data["pandas_dtypes"]),  # type: ignore[arg-type]
                top_n=data["top_n"],  # type: ignore[arg-type]
                insufficient_universe_policy=data["insufficient_universe_policy"],  # type: ignore[arg-type]
                weighting=data["weighting"],  # type: ignore[arg-type]
                source_signal_provenance=data["source_signal_provenance"],  # type: ignore[arg-type]
                files=tuple(HoldingsArtifactFileRecord.from_dict(item) for item in raw_files),
            )
        except HoldingsArtifactValidationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise HoldingsArtifactValidationError("manifest cannot be parsed.") from exc

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_type": self.artifact_type,
            "artifact_schema_version": self.artifact_schema_version,
            "holdings_schema_version": self.holdings_schema_version,
            "created_at_utc": self.created_at_utc,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "pandas_dtypes": [list(item) for item in self.pandas_dtypes],
            "top_n": self.top_n,
            "insufficient_universe_policy": self.insufficient_universe_policy,
            "weighting": self.weighting,
            "source_signal_provenance": dict(self.source_signal_provenance),
            "files": [item.as_dict() for item in self.files],
        }


@dataclass(frozen=True)
class HoldingsArtifactValidationIssue:
    code: str
    message: str
    relative_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not _ISSUE_CODE_RE.fullmatch(self.code):
            raise HoldingsArtifactValidationError("issue code is invalid.")
        _trimmed(self.message, "issue message")
        if self.relative_path is not None and self.relative_path not in HOLDINGS_ARTIFACT_FILENAMES:
            raise HoldingsArtifactValidationError("issue relative_path is invalid.")

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class HoldingsArtifactValidationReport:
    artifact_dir: Path
    is_valid: bool
    issues: tuple[HoldingsArtifactValidationIssue, ...]
    manifest: HoldingsArtifactManifest | None

    def __post_init__(self) -> None:
        directory = _absolute(_path(self.artifact_dir))
        issues = tuple(self.issues)
        if any(not isinstance(item, HoldingsArtifactValidationIssue) for item in issues):
            raise HoldingsArtifactValidationError("issues contain invalid values.")
        if len(issues) != len(set(issues)):
            raise HoldingsArtifactValidationError("issues contain duplicates.")
        if type(self.is_valid) is not bool or self.is_valid != (not issues):
            raise HoldingsArtifactValidationError("is_valid must agree with issues.")
        if self.is_valid and not isinstance(self.manifest, HoldingsArtifactManifest):
            raise HoldingsArtifactValidationError("a valid report requires a manifest.")
        object.__setattr__(self, "artifact_dir", directory)
        object.__setattr__(self, "issues", issues)

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "is_valid": self.is_valid,
            "issues": [item.as_dict() for item in self.issues],
            "manifest": None if self.manifest is None else self.manifest.as_dict(),
        }


@dataclass(frozen=True)
class HoldingsArtifactWriteResult:
    artifact_dir: Path
    holdings_path: Path
    config_path: Path
    audit_path: Path
    manifest_path: Path
    rows: int
    schema_version: str
    manifest: HoldingsArtifactManifest
    validation: HoldingsArtifactValidationReport

    def __post_init__(self) -> None:
        directory = _absolute(Path(self.artifact_dir))
        paths = tuple(_absolute(Path(item)) for item in (
            self.holdings_path, self.config_path, self.audit_path, self.manifest_path
        ))
        for path, filename in zip(paths, HOLDINGS_ARTIFACT_FILENAMES, strict=True):
            if path.parent != directory or path.name != filename:
                raise HoldingsArtifactValidationError(
                    "write result paths must be fixed direct children."
                )
        if (
            self.schema_version != HOLDINGS_ARTIFACT_SCHEMA_VERSION
            or self.rows != self.manifest.row_count
            or not self.validation.is_valid
            or self.validation.manifest != self.manifest
        ):
            raise HoldingsArtifactValidationError("write result metadata is invalid.")
        object.__setattr__(self, "artifact_dir", directory)
        for name, value in zip(
            ("holdings_path", "config_path", "audit_path", "manifest_path"),
            paths,
            strict=True,
        ):
            object.__setattr__(self, name, value)


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number is forbidden")
    return parsed


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise HoldingsArtifactValidationError("strict JSON read failed.") from exc
    if not isinstance(value, dict):
        raise HoldingsArtifactValidationError("JSON top-level value must be an object.")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value, handle, ensure_ascii=False, allow_nan=False,
                indent=2, sort_keys=True,
            )
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise HoldingsArtifactWriteError("strict JSON write failed.") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HoldingsArtifactValidationError("checksum read failed.") from exc
    return digest.hexdigest()


def _record(path: Path) -> HoldingsArtifactFileRecord:
    try:
        return HoldingsArtifactFileRecord(path.name, path.stat().st_size, _sha256(path))
    except (OSError, HoldingsArtifactValidationError) as exc:
        raise HoldingsArtifactWriteError("payload metadata failed.") from exc


def _date_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") == value
    except ValueError:
        return False


def _audit(value: object) -> dict[str, object]:
    data = _strict_keys(value, _AUDIT_FIELDS, "audit")
    if data["holdings_schema_version"] != HOLDINGS_SCHEMA_VERSION:
        raise HoldingsArtifactValidationError("audit schema version is invalid.")
    for name in (
        "input_rows", "output_rows", "trade_date_count", "duplicate_key_count"
    ):
        _nonnegative_int(data[name], name)
    if (
        data["output_rows"] == 0
        or data["input_rows"] < data["output_rows"]
        or data["trade_date_count"] == 0
        or data["duplicate_key_count"] != 0
    ):
        raise HoldingsArtifactValidationError("audit counts are invalid.")
    if not _date_text(data["min_trade_date"]) or not _date_text(data["max_trade_date"]):
        raise HoldingsArtifactValidationError("audit dates are invalid.")
    config = _business_config({
        "top_n": data["requested_top_n"],
        "insufficient_universe_policy": data["insufficient_universe_policy"],
        "weighting": data["weighting"],
    })
    raw_counts = data["per_date_counts"]
    if not isinstance(raw_counts, list) or not raw_counts:
        raise HoldingsArtifactValidationError("per_date_counts are invalid.")
    counts: list[dict[str, object]] = []
    for value_item in raw_counts:
        item = _strict_keys(value_item, _DATE_COUNT_FIELDS, "per-date count")
        if not _date_text(item["trade_date"]):
            raise HoldingsArtifactValidationError("per-date trade_date is invalid.")
        available = _nonnegative_int(item["available_count"], "available_count")
        selected = _nonnegative_int(item["selected_count"], "selected_count")
        if type(item["partial"]) is not bool or available < 1 or selected < 1:
            raise HoldingsArtifactValidationError("per-date counts are invalid.")
        expected_selected = min(available, config["top_n"])
        expected_partial = available < config["top_n"]
        if selected != expected_selected or item["partial"] != expected_partial:
            raise HoldingsArtifactValidationError("per-date selection differs from config.")
        if config["insufficient_universe_policy"] == "error" and expected_partial:
            raise HoldingsArtifactValidationError("error policy contains a partial date.")
        counts.append(item)
    dates = [str(item["trade_date"]) for item in counts]
    if dates != sorted(set(dates)) or len(counts) != data["trade_date_count"]:
        raise HoldingsArtifactValidationError("per-date ordering is invalid.")
    if sum(int(item["available_count"]) for item in counts) != data["input_rows"]:
        raise HoldingsArtifactValidationError("per-date available counts differ.")
    if sum(int(item["selected_count"]) for item in counts) != data["output_rows"]:
        raise HoldingsArtifactValidationError("per-date selected counts differ.")
    partial_dates = [str(item["trade_date"]) for item in counts if item["partial"]]
    if data["partial_dates"] != partial_dates:
        raise HoldingsArtifactValidationError("partial_dates differ.")
    if data["min_trade_date"] != dates[0] or data["max_trade_date"] != dates[-1]:
        raise HoldingsArtifactValidationError("audit date range differs.")
    if data["weight_sum_tolerance"] != WEIGHT_SUM_ABSOLUTE_TOLERANCE:
        raise HoldingsArtifactValidationError("weight tolerance is invalid.")
    for name in ("weight_sum_min", "weight_sum_max"):
        value_number = data[name]
        if (
            isinstance(value_number, bool)
            or not isinstance(value_number, (int, float))
            or not math.isfinite(float(value_number))
            or not math.isclose(
                float(value_number), 1.0, rel_tol=0,
                abs_tol=WEIGHT_SUM_ABSOLUTE_TOLERANCE,
            )
        ):
            raise HoldingsArtifactValidationError("weight sum audit is invalid.")
    provenance = _provenance(data["source_signal_provenance"])
    warnings = data["warnings"]
    expected_warnings = [f"partial universe on {date}" for date in partial_dates]
    if warnings != expected_warnings:
        raise HoldingsArtifactValidationError("audit warnings are invalid.")
    return {
        **data,
        "requested_top_n": config["top_n"],
        "insufficient_universe_policy": config["insufficient_universe_policy"],
        "weighting": config["weighting"],
        "per_date_counts": counts,
        "partial_dates": partial_dates,
        "source_signal_provenance": provenance,
        "warnings": list(warnings),
    }


def _holdings_errors(
    frame: pd.DataFrame, config: Mapping[str, object]
) -> list[str]:
    if frame.empty:
        return ["Holdings payload is empty."]
    if not frame.columns.is_unique or tuple(frame.columns) != HOLDINGS_OUTPUT_COLUMNS:
        return ["Holdings columns or order are invalid."]
    errors: list[str] = []
    dates = frame["trade_date"]
    if (
        not pd.api.types.is_datetime64_ns_dtype(dates.dtype)
        or getattr(dates.dt, "tz", None) is not None
        or dates.isna().any()
        or not dates.eq(dates.dt.normalize()).all()
    ):
        errors.append("trade_date semantics are invalid.")
    codes = frame["ts_code"]
    if (
        codes.isna().any()
        or not codes.map(lambda item: isinstance(item, (str, np.str_))).all()
        or codes.astype("string").str.strip().eq("").any()
        or not codes.astype("string").eq(codes.astype("string").str.strip()).all()
    ):
        errors.append("ts_code semantics are invalid.")
    for name in ("target_weight", "score"):
        series = frame[name]
        if (
            pd.api.types.is_bool_dtype(series.dtype)
            or not pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype)
        ):
            errors.append(f"{name} dtype is invalid.")
            continue
        try:
            values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError):
            errors.append(f"{name} dtype is invalid.")
            continue
        if not np.isfinite(values).all():
            errors.append(f"{name} contains non-finite values.")
        if name == "target_weight" and bool((values <= 0).any()):
            errors.append("target_weight must be strictly positive.")
    ranks = frame["rank"]
    if not pd.api.types.is_integer_dtype(ranks.dtype) or bool((ranks <= 0).any()):
        errors.append("rank must be positive integer data.")
    if errors:
        return errors
    if frame.duplicated(list(HOLDINGS_KEY_COLUMNS)).any():
        errors.append("Holdings keys are duplicated.")
    expected_order = frame.sort_values(
        ["trade_date", "rank", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)
    try:
        pdt.assert_frame_equal(frame.reset_index(drop=True), expected_order)
    except AssertionError:
        errors.append("Holdings row order is not canonical.")
    top_n = int(config["top_n"])
    policy = str(config["insufficient_universe_policy"])
    for _, group in frame.groupby("trade_date", sort=False):
        count = len(group)
        if count > top_n or (policy == "error" and count != top_n):
            errors.append("Per-date Holdings count violates config.")
            break
        expected_ranks = np.arange(1, count + 1, dtype=np.int64)
        if not np.array_equal(group["rank"].to_numpy(), expected_ranks):
            errors.append("Selected ranks are not contiguous 1..K.")
            break
        weights = group["target_weight"].to_numpy(dtype=np.float64)
        if not np.allclose(weights, 1.0 / count, rtol=0, atol=1e-15):
            errors.append("Holdings are not equal weighted.")
            break
        if not math.isclose(
            float(weights.sum()), 1.0, rel_tol=0,
            abs_tol=WEIGHT_SUM_ABSOLUTE_TOLERANCE,
        ):
            errors.append("Per-date target weights do not sum to one.")
            break
    return errors


def _issue(
    code: str, message: str, path: str | None = None
) -> HoldingsArtifactValidationIssue:
    return HoldingsArtifactValidationIssue(code, message, path)


class HoldingsArtifactStore:
    """Write and independently validate one explicit Holdings Artifact."""

    def read_manifest(
        self, artifact_dir: str | os.PathLike[str]
    ) -> HoldingsArtifactManifest:
        directory = _absolute(_path(artifact_dir))
        path = directory / HOLDINGS_MANIFEST_FILENAME
        if (
            not directory.exists() or directory.is_symlink() or not directory.is_dir()
            or not path.exists() or path.is_symlink() or not path.is_file()
        ):
            raise HoldingsArtifactValidationError("manifest path is invalid.")
        return HoldingsArtifactManifest.from_dict(_read_json(path))

    def validate(
        self, artifact_dir: str | os.PathLike[str]
    ) -> HoldingsArtifactValidationReport:
        directory = _absolute(_path(artifact_dir))
        issues: list[HoldingsArtifactValidationIssue] = []
        manifest: HoldingsArtifactManifest | None = None
        if not directory.exists():
            return self._report(directory, [_issue("artifact_dir_missing", "Artifact directory is missing.")], None)
        if directory.is_symlink():
            return self._report(directory, [_issue("artifact_dir_symlink", "Artifact directory is a symlink.")], None)
        if not directory.is_dir():
            return self._report(directory, [_issue("artifact_dir_not_directory", "Artifact path is not a directory.")], None)
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            return self._report(directory, [_issue("artifact_dir_unreadable", "Artifact directory cannot be read.")], None)
        expected = set(HOLDINGS_ARTIFACT_FILENAMES)
        actual = {item.name for item in entries}
        for _ in sorted(actual - expected):
            issues.append(_issue("unexpected_entry", "Unexpected artifact entry."))
        for name in HOLDINGS_ARTIFACT_FILENAMES:
            if name not in actual:
                issues.append(_issue("missing_file", "Required artifact file is missing.", name))
        safe: dict[str, Path] = {}
        for name in HOLDINGS_ARTIFACT_FILENAMES:
            path = directory / name
            if name not in actual:
                continue
            if path.is_symlink():
                issues.append(_issue("artifact_file_symlink", "Artifact file is a symlink.", name))
            elif not path.is_file():
                issues.append(_issue("artifact_file_not_regular", "Artifact path is not a regular file.", name))
            else:
                safe[name] = path
        if (path := safe.get(HOLDINGS_MANIFEST_FILENAME)) is not None:
            try:
                manifest = HoldingsArtifactManifest.from_dict(_read_json(path))
            except HoldingsArtifactValidationError:
                issues.append(_issue("invalid_manifest_json", "Manifest JSON or schema is invalid.", HOLDINGS_MANIFEST_FILENAME))
        if manifest is not None:
            for record in manifest.files:
                path = safe.get(record.relative_path)
                if path is None:
                    continue
                try:
                    if path.stat().st_size != record.size_bytes:
                        issues.append(_issue("file_size_mismatch", "Payload size differs from manifest.", record.relative_path))
                    if _sha256(path) != record.sha256:
                        issues.append(_issue("checksum_mismatch", "Payload checksum differs from manifest.", record.relative_path))
                except (OSError, HoldingsArtifactValidationError):
                    issues.append(_issue("checksum_mismatch", "Payload metadata cannot be read.", record.relative_path))
        business: dict[str, object] | None = None
        if (path := safe.get(HOLDINGS_CONFIG_FILENAME)) is not None:
            try:
                business = _business_config(_read_json(path))
            except HoldingsArtifactValidationError:
                issues.append(_issue("invalid_config_json", "Config JSON is invalid.", HOLDINGS_CONFIG_FILENAME))
        audit: dict[str, object] | None = None
        if (path := safe.get(HOLDINGS_AUDIT_FILENAME)) is not None:
            try:
                audit = _audit(_read_json(path))
            except HoldingsArtifactValidationError:
                issues.append(_issue("invalid_audit_json", "Audit JSON is invalid.", HOLDINGS_AUDIT_FILENAME))
        if manifest is not None and business is not None:
            manifest_config = {
                "top_n": manifest.top_n,
                "insufficient_universe_policy": manifest.insufficient_universe_policy,
                "weighting": manifest.weighting,
            }
            if business != manifest_config:
                issues.append(_issue("config_manifest_mismatch", "Config differs from manifest.", HOLDINGS_CONFIG_FILENAME))
        if audit is not None and business is not None:
            audit_config = {
                "top_n": audit["requested_top_n"],
                "insufficient_universe_policy": audit["insufficient_universe_policy"],
                "weighting": audit["weighting"],
            }
            if audit_config != business:
                issues.append(_issue("audit_config_mismatch", "Audit differs from config.", HOLDINGS_AUDIT_FILENAME))
        if audit is not None and manifest is not None:
            if (
                audit["output_rows"] != manifest.row_count
                or audit["holdings_schema_version"] != manifest.holdings_schema_version
                or audit["source_signal_provenance"] != dict(manifest.source_signal_provenance)
            ):
                issues.append(_issue("audit_manifest_mismatch", "Audit differs from manifest.", HOLDINGS_AUDIT_FILENAME))
        frame: pd.DataFrame | None = None
        if (path := safe.get(HOLDINGS_PARQUET_FILENAME)) is not None:
            try:
                frame = pd.read_parquet(path, engine="pyarrow")
            except Exception:
                issues.append(_issue("parquet_read_error", "Holdings Parquet cannot be read.", HOLDINGS_PARQUET_FILENAME))
        if frame is not None and manifest is not None:
            if tuple(frame.columns) != manifest.columns or not frame.columns.is_unique:
                issues.append(_issue("parquet_column_mismatch", "Parquet columns differ from manifest.", HOLDINGS_PARQUET_FILENAME))
            if len(frame) != manifest.row_count:
                issues.append(_issue("parquet_row_count_mismatch", "Parquet row count differs from manifest.", HOLDINGS_PARQUET_FILENAME))
            dtypes = tuple((str(name), str(dtype)) for name, dtype in frame.dtypes.items())
            if dtypes != manifest.pandas_dtypes:
                issues.append(_issue("parquet_dtype_mismatch", "Parquet dtypes differ from manifest.", HOLDINGS_PARQUET_FILENAME))
            if tuple(frame.columns) == HOLDINGS_OUTPUT_COLUMNS:
                config_value = {
                    "top_n": manifest.top_n,
                    "insufficient_universe_policy": manifest.insufficient_universe_policy,
                    "weighting": manifest.weighting,
                }
                content_errors = _holdings_errors(frame, config_value)
                for message in content_errors:
                    issues.append(_issue("holdings_content_error", message, HOLDINGS_PARQUET_FILENAME))
                if audit is not None and not content_errors:
                    groups = frame.groupby("trade_date", sort=False)
                    sums = groups["target_weight"].sum()
                    expected = {
                        "output_rows": len(frame),
                        "trade_date_count": int(frame["trade_date"].nunique()),
                        "min_trade_date": pd.Timestamp(frame["trade_date"].min()).date().isoformat(),
                        "max_trade_date": pd.Timestamp(frame["trade_date"].max()).date().isoformat(),
                        "weight_sum_min": float(sums.min()),
                        "weight_sum_max": float(sums.max()),
                    }
                    if any(
                        not math.isclose(float(audit[name]), value, rel_tol=0, abs_tol=1e-15)
                        if name.startswith("weight_sum") else audit[name] != value
                        for name, value in expected.items()
                    ):
                        issues.append(_issue("audit_parquet_mismatch", "Audit differs from Holdings payload.", HOLDINGS_AUDIT_FILENAME))
                    audit_counts = audit["per_date_counts"]
                    selected_counts = {
                        pd.Timestamp(date).date().isoformat(): int(count)
                        for date, count in groups.size().items()
                    }
                    audited_selected = {
                        str(item["trade_date"]): int(item["selected_count"])
                        for item in audit_counts
                    } if isinstance(audit_counts, list) else {}
                    if audited_selected != selected_counts:
                        issues.append(_issue("audit_parquet_mismatch", "Per-date audit differs from Holdings payload.", HOLDINGS_AUDIT_FILENAME))
        return self._report(directory, issues, manifest)

    @staticmethod
    def _report(
        directory: Path,
        issues: list[HoldingsArtifactValidationIssue],
        manifest: HoldingsArtifactManifest | None,
    ) -> HoldingsArtifactValidationReport:
        unique: list[HoldingsArtifactValidationIssue] = []
        seen: set[HoldingsArtifactValidationIssue] = set()
        for issue in issues:
            if issue not in seen:
                seen.add(issue)
                unique.append(issue)
        return HoldingsArtifactValidationReport(
            directory, not unique, tuple(unique), manifest
        )

    def write(
        self,
        result: HoldingsBuildResult,
        provenance: SignalArtifactProvenance,
        config: HoldingsArtifactConfig,
    ) -> HoldingsArtifactWriteResult:
        if not isinstance(result, HoldingsBuildResult):
            raise HoldingsArtifactWriteError("result must be a HoldingsBuildResult.")
        if not isinstance(provenance, SignalArtifactProvenance):
            raise HoldingsArtifactWriteError(
                "provenance must be SignalArtifactProvenance."
            )
        if not isinstance(config, HoldingsArtifactConfig):
            raise HoldingsArtifactWriteError("config must be HoldingsArtifactConfig.")
        target = _absolute(config.artifact_dir)
        parent = target.parent
        if target.exists() or target.is_symlink():
            raise HoldingsArtifactExistsError("target artifact_dir already exists.")
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise HoldingsArtifactWriteError("artifact parent is invalid.")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HoldingsArtifactWriteError("artifact parent cannot be created.") from exc
        if parent.is_symlink():
            raise HoldingsArtifactWriteError("artifact parent must not be a symlink.")
        staging = parent / f".tmp-{target.name}-{uuid4().hex}"
        published = False
        try:
            staging.mkdir()
            holdings = result.holdings
            business = {
                "top_n": result.audit.requested_top_n,
                "insufficient_universe_policy": result.audit.insufficient_universe_policy,
                "weighting": result.audit.weighting,
            }
            errors = _holdings_errors(holdings, business)
            if errors:
                raise HoldingsArtifactWriteError(
                    "Holdings input is not canonical: " + " ".join(errors)
                )
            provenance_data = provenance.as_dict()
            weight_sums = holdings.groupby("trade_date", sort=False)["target_weight"].sum()
            audit = {
                "holdings_schema_version": HOLDINGS_SCHEMA_VERSION,
                **result.audit.as_dict(),
                "weight_sum_min": float(weight_sums.min()),
                "weight_sum_max": float(weight_sums.max()),
                "weight_sum_tolerance": WEIGHT_SUM_ABSOLUTE_TOLERANCE,
                "duplicate_key_count": 0,
                "source_signal_provenance": provenance_data,
            }
            _business_config(business)
            _audit(audit)
            holdings_path = staging / HOLDINGS_PARQUET_FILENAME
            holdings.to_parquet(
                holdings_path, engine="pyarrow",
                compression=config.parquet_compression, index=False,
            )
            _write_json(staging / HOLDINGS_CONFIG_FILENAME, business)
            _write_json(staging / HOLDINGS_AUDIT_FILENAME, audit)
            persisted = pd.read_parquet(holdings_path, engine="pyarrow")
            try:
                pdt.assert_frame_equal(
                    persisted, holdings, check_like=False, check_dtype=False,
                    check_exact=False, rtol=1e-12, atol=1e-12,
                )
            except AssertionError as exc:
                raise HoldingsArtifactWriteError(
                    "Parquet roundtrip changed Holdings semantics."
                ) from exc
            records = tuple(_record(staging / name) for name in _PAYLOAD_FILENAMES)
            manifest = HoldingsArtifactManifest(
                artifact_type=HOLDINGS_ARTIFACT_TYPE,
                artifact_schema_version=HOLDINGS_ARTIFACT_SCHEMA_VERSION,
                holdings_schema_version=HOLDINGS_SCHEMA_VERSION,
                created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                row_count=len(persisted),
                column_count=len(persisted.columns),
                columns=tuple(str(item) for item in persisted.columns),
                pandas_dtypes=tuple((str(name), str(dtype)) for name, dtype in persisted.dtypes.items()),
                top_n=result.audit.requested_top_n,
                insufficient_universe_policy=result.audit.insufficient_universe_policy,
                weighting=result.audit.weighting,
                source_signal_provenance=provenance_data,
                files=records,
            )
            _write_json(staging / HOLDINGS_MANIFEST_FILENAME, manifest.as_dict())
            pre = self.validate(staging)
            if not pre.is_valid:
                raise HoldingsArtifactWriteError("pre-publish validation failed.")
            if target.exists() or target.is_symlink():
                raise HoldingsArtifactExistsError("target appeared before publication.")
            os.replace(staging, target)
            published = True
            validation = self.validate(target) if config.verify_after_write else HoldingsArtifactValidationReport(target, True, (), manifest)
            if not validation.is_valid:
                try:
                    shutil.rmtree(target)
                except OSError as exc:
                    raise HoldingsArtifactWriteError(
                        "post-publish validation and cleanup failed."
                    ) from exc
                raise HoldingsArtifactWriteError("post-publish validation failed.")
            return HoldingsArtifactWriteResult(
                target,
                target / HOLDINGS_PARQUET_FILENAME,
                target / HOLDINGS_CONFIG_FILENAME,
                target / HOLDINGS_AUDIT_FILENAME,
                target / HOLDINGS_MANIFEST_FILENAME,
                len(persisted),
                HOLDINGS_ARTIFACT_SCHEMA_VERSION,
                manifest,
                validation,
            )
        except HoldingsArtifactError:
            if staging.exists() and not published:
                shutil.rmtree(staging, ignore_errors=True)
            raise
        except Exception as exc:
            if staging.exists() and not published:
                shutil.rmtree(staging, ignore_errors=True)
            raise HoldingsArtifactWriteError("artifact write failed.") from exc
