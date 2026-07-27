"""Safe, atomic persistence for validated Modeling Panel results."""

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
from uuid import uuid4

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.modeling_panel.contracts import (
    MODELING_PANEL_AUDIT_COLUMNS,
    MODELING_PANEL_KEY_COLUMNS,
    MODELING_PANEL_SCHEMA_VERSION,
    ModelingPanelConfig,
    ModelingPanelError,
    ModelingPanelResult,
)

MODELING_PANEL_ARTIFACT_SCHEMA_VERSION = "1.0"
MODELING_PANEL_ARTIFACT_TYPE = "modeling_panel"
MODELING_PANEL_PARQUET_FILENAME = "modeling_panel.parquet"
MODELING_PANEL_CONFIG_FILENAME = "config.json"
MODELING_PANEL_AUDIT_FILENAME = "audit.json"
MODELING_PANEL_MANIFEST_FILENAME = "manifest.json"
MODELING_PANEL_ARTIFACT_FILENAMES = (
    MODELING_PANEL_PARQUET_FILENAME,
    MODELING_PANEL_CONFIG_FILENAME,
    MODELING_PANEL_AUDIT_FILENAME,
    MODELING_PANEL_MANIFEST_FILENAME,
)

_PAYLOAD_FILENAMES = MODELING_PANEL_ARTIFACT_FILENAMES[:-1]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISSUE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CHUNK_SIZE = 1024 * 1024
_MANIFEST_FIELDS = {
    "artifact_type", "artifact_schema_version", "modeling_panel_schema_version",
    "created_at_utc", "label_column", "feature_names", "row_count",
    "column_count", "columns", "pandas_dtypes", "files",
}
_AUDIT_FIELDS = {
    "schema_version", "config", "label_column", "factor_input_rows",
    "return_input_rows", "matched_rows", "output_rows", "factor_only",
    "return_only", "date_count", "security_count", "first_trade_date",
    "last_trade_date", "first_entry_trade_date", "last_entry_trade_date",
    "first_exit_trade_date", "last_exit_trade_date", "feature_count",
    "feature_names", "feature_missing_counts", "feature_missing_rates",
    "feature_non_finite_counts", "all_missing_features", "constant_features",
    "suspicious_feature_names", "label_missing_count",
    "label_non_finite_count", "duplicate_factor_key_count",
    "duplicate_return_key_count", "entry_before_signal_count",
    "entry_equal_signal_count", "exit_not_after_entry_count",
    "label_formula_mismatch_count", "per_date_security_count_min",
    "per_date_security_count_median", "per_date_security_count_max",
    "per_security_observation_count_min",
    "per_security_observation_count_median",
    "per_security_observation_count_max", "warnings",
}


class ModelingPanelArtifactError(ModelingPanelError):
    """Base error for Modeling Panel artifact operations."""


class ModelingPanelArtifactConfigError(ModelingPanelArtifactError):
    """Raised for invalid artifact configuration."""


class ModelingPanelArtifactWriteError(ModelingPanelArtifactError):
    """Raised when an artifact cannot be safely written."""


class ModelingPanelArtifactValidationError(ModelingPanelArtifactError):
    """Raised when strict artifact metadata is invalid."""


def _strict_keys(
    value: object,
    expected: set[str],
    context: str,
    error: type[ModelingPanelArtifactError],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise error(f"{context} must be a mapping with string keys.")
    if set(value) != expected:
        raise error(f"{context} fields are invalid.")
    return dict(value)


def _path(value: object, error: type[ModelingPanelArtifactError]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise error("artifact_dir must be str or os.PathLike.")
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise error("artifact_dir must be str or os.PathLike.") from exc
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise error("artifact_dir must identify a non-empty directory.")
    result = Path(raw)
    if raw.strip() in {".", ".."} or result == Path(result.anchor):
        raise error("artifact_dir must identify an explicit child directory.")
    return result


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


@dataclass(frozen=True)
class ModelingPanelArtifactConfig:
    artifact_dir: Path
    parquet_compression: str = "zstd"
    verify_after_write: bool = True

    def __post_init__(self) -> None:
        path = _path(self.artifact_dir, ModelingPanelArtifactConfigError)
        compression = self.parquet_compression
        if not isinstance(compression, str):
            raise ModelingPanelArtifactConfigError(
                "parquet_compression must be a string."
            )
        compression = compression.strip().lower()
        if compression not in {"zstd", "snappy"}:
            raise ModelingPanelArtifactConfigError(
                "parquet_compression must be 'zstd' or 'snappy'."
            )
        if type(self.verify_after_write) is not bool:
            raise ModelingPanelArtifactConfigError(
                "verify_after_write must be a bool."
            )
        object.__setattr__(self, "artifact_dir", path)
        object.__setattr__(self, "parquet_compression", compression)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | ModelingPanelArtifactConfig
    ) -> ModelingPanelArtifactConfig:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
            raise ModelingPanelArtifactConfigError("artifact config must be a mapping.")
        allowed = {"artifact_dir", "parquet_compression", "verify_after_write"}
        if set(value) - allowed or "artifact_dir" not in value:
            raise ModelingPanelArtifactConfigError("artifact config fields are invalid.")
        return cls(**dict(value))  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "parquet_compression": self.parquet_compression,
            "verify_after_write": self.verify_after_write,
        }


@dataclass(frozen=True)
class ModelingPanelArtifactFileRecord:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.relative_path not in _PAYLOAD_FILENAMES:
            raise ModelingPanelArtifactValidationError(
                "file record relative_path is unsafe or unsupported."
            )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ModelingPanelArtifactValidationError(
                "file record size_bytes must be a positive integer."
            )
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ModelingPanelArtifactValidationError(
                "file record sha256 must be lowercase hexadecimal."
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ModelingPanelArtifactFileRecord:
        data = _strict_keys(
            value, {"relative_path", "size_bytes", "sha256"}, "file record",
            ModelingPanelArtifactValidationError,
        )
        return cls(**data)  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _name(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ModelingPanelArtifactValidationError(
            f"{context} must be a non-empty trimmed string."
        )
    return value


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelingPanelArtifactValidationError(
            f"{context} must be a positive integer."
        )
    return value


@dataclass(frozen=True)
class ModelingPanelArtifactManifest:
    artifact_type: str
    artifact_schema_version: str
    modeling_panel_schema_version: str
    created_at_utc: str
    label_column: str
    feature_names: tuple[str, ...]
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    pandas_dtypes: tuple[tuple[str, str], ...]
    files: tuple[ModelingPanelArtifactFileRecord, ...]

    def __post_init__(self) -> None:
        if self.artifact_type != MODELING_PANEL_ARTIFACT_TYPE:
            raise ModelingPanelArtifactValidationError("artifact_type is invalid.")
        if self.artifact_schema_version != MODELING_PANEL_ARTIFACT_SCHEMA_VERSION:
            raise ModelingPanelArtifactValidationError(
                "artifact_schema_version is invalid."
            )
        if self.modeling_panel_schema_version != MODELING_PANEL_SCHEMA_VERSION:
            raise ModelingPanelArtifactValidationError(
                "modeling_panel_schema_version is invalid."
            )
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc.endswith("Z"):
            raise ModelingPanelArtifactValidationError("created_at_utc must be UTC Z.")
        try:
            created = datetime.fromisoformat(self.created_at_utc[:-1] + "+00:00")
        except ValueError as exc:
            raise ModelingPanelArtifactValidationError(
                "created_at_utc is not valid ISO-8601."
            ) from exc
        if created.utcoffset() != timezone.utc.utcoffset(created):
            raise ModelingPanelArtifactValidationError("created_at_utc must be UTC.")
        label = _name(self.label_column, "label_column")
        try:
            features = tuple(self.feature_names)
            columns = tuple(self.columns)
            dtypes = tuple(tuple(item) for item in self.pandas_dtypes)
            files = tuple(self.files)
        except (TypeError, ValueError) as exc:
            raise ModelingPanelArtifactValidationError(
                "manifest sequence fields are invalid."
            ) from exc
        if (
            not features
            or any(_name(item, "feature_name") != item for item in features)
            or len(features) != len(set(features))
        ):
            raise ModelingPanelArtifactValidationError("feature_names are invalid.")
        row_count = _positive_int(self.row_count, "row_count")
        column_count = _positive_int(self.column_count, "column_count")
        expected_columns = (
            *MODELING_PANEL_KEY_COLUMNS,
            *features,
            *MODELING_PANEL_AUDIT_COLUMNS,
            label,
        )
        if columns != expected_columns or len(columns) != len(set(columns)):
            raise ModelingPanelArtifactValidationError("columns are invalid.")
        if column_count != len(columns):
            raise ModelingPanelArtifactValidationError(
                "column_count does not match columns."
            )
        if (
            len(dtypes) != len(columns)
            or any(len(item) != 2 for item in dtypes)
            or tuple(item[0] for item in dtypes) != columns
            or any(not isinstance(item[1], str) or not item[1] for item in dtypes)
        ):
            raise ModelingPanelArtifactValidationError("pandas_dtypes are invalid.")
        if (
            len(files) != 3
            or any(not isinstance(item, ModelingPanelArtifactFileRecord) for item in files)
            or tuple(item.relative_path for item in files) != _PAYLOAD_FILENAMES
        ):
            raise ModelingPanelArtifactValidationError("manifest files are invalid.")
        object.__setattr__(self, "label_column", label)
        object.__setattr__(self, "feature_names", features)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "column_count", column_count)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "pandas_dtypes", dtypes)
        object.__setattr__(self, "files", files)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ModelingPanelArtifactManifest:
        data = _strict_keys(
            value, _MANIFEST_FIELDS, "manifest",
            ModelingPanelArtifactValidationError,
        )
        try:
            if (
                not isinstance(data["feature_names"], list)
                or not isinstance(data["columns"], list)
                or not isinstance(data["pandas_dtypes"], list)
                or any(not isinstance(item, list) for item in data["pandas_dtypes"])
            ):
                raise TypeError("manifest sequences must be lists")
            raw_files = data["files"]
            if not isinstance(raw_files, list):
                raise TypeError("files must be a list")
            return cls(
                artifact_type=data["artifact_type"],  # type: ignore[arg-type]
                artifact_schema_version=data["artifact_schema_version"],  # type: ignore[arg-type]
                modeling_panel_schema_version=data["modeling_panel_schema_version"],  # type: ignore[arg-type]
                created_at_utc=data["created_at_utc"],  # type: ignore[arg-type]
                label_column=data["label_column"],  # type: ignore[arg-type]
                feature_names=tuple(data["feature_names"]),  # type: ignore[arg-type]
                row_count=data["row_count"],  # type: ignore[arg-type]
                column_count=data["column_count"],  # type: ignore[arg-type]
                columns=tuple(data["columns"]),  # type: ignore[arg-type]
                pandas_dtypes=tuple(
                    tuple(item) for item in data["pandas_dtypes"]  # type: ignore[union-attr]
                ),
                files=tuple(
                    ModelingPanelArtifactFileRecord.from_dict(item)
                    for item in raw_files
                ),
            )
        except ModelingPanelArtifactValidationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelingPanelArtifactValidationError(
                "manifest cannot be parsed."
            ) from exc

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_type": self.artifact_type,
            "artifact_schema_version": self.artifact_schema_version,
            "modeling_panel_schema_version": self.modeling_panel_schema_version,
            "created_at_utc": self.created_at_utc,
            "label_column": self.label_column,
            "feature_names": list(self.feature_names),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "pandas_dtypes": [list(item) for item in self.pandas_dtypes],
            "files": [item.as_dict() for item in self.files],
        }


@dataclass(frozen=True)
class ModelingPanelArtifactValidationIssue:
    code: str
    message: str
    relative_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not _ISSUE_CODE_RE.fullmatch(self.code):
            raise ModelingPanelArtifactValidationError("issue code is invalid.")
        _name(self.message, "issue message")
        if self.relative_path is not None and self.relative_path not in MODELING_PANEL_ARTIFACT_FILENAMES:
            raise ModelingPanelArtifactValidationError("issue relative_path is invalid.")

    def as_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "relative_path": self.relative_path}


@dataclass(frozen=True)
class ModelingPanelArtifactValidationReport:
    artifact_dir: Path
    is_valid: bool
    issues: tuple[ModelingPanelArtifactValidationIssue, ...]
    manifest: ModelingPanelArtifactManifest | None

    def __post_init__(self) -> None:
        path = _absolute(_path(self.artifact_dir, ModelingPanelArtifactValidationError))
        try:
            issues = tuple(self.issues)
        except TypeError as exc:
            raise ModelingPanelArtifactValidationError("issues must be iterable.") from exc
        if any(not isinstance(item, ModelingPanelArtifactValidationIssue) for item in issues):
            raise ModelingPanelArtifactValidationError("issues contain invalid values.")
        if len(issues) != len(set(issues)):
            raise ModelingPanelArtifactValidationError("issues contain duplicates.")
        if type(self.is_valid) is not bool or self.is_valid != (len(issues) == 0):
            raise ModelingPanelArtifactValidationError("is_valid must agree with issues.")
        if self.is_valid and not isinstance(self.manifest, ModelingPanelArtifactManifest):
            raise ModelingPanelArtifactValidationError("a valid report requires a manifest.")
        object.__setattr__(self, "artifact_dir", path)
        object.__setattr__(self, "issues", issues)

    def as_dict(self) -> dict[str, object]:
        return {"artifact_dir": str(self.artifact_dir), "is_valid": self.is_valid,
                "issues": [item.as_dict() for item in self.issues],
                "manifest": None if self.manifest is None else self.manifest.as_dict()}


@dataclass(frozen=True)
class ModelingPanelArtifactWriteResult:
    artifact_dir: Path
    panel_path: Path
    config_path: Path
    audit_path: Path
    manifest_path: Path
    manifest: ModelingPanelArtifactManifest
    validation: ModelingPanelArtifactValidationReport

    def __post_init__(self) -> None:
        directory = _absolute(Path(self.artifact_dir))
        normalized = tuple(_absolute(Path(item)) for item in (
            self.panel_path, self.config_path, self.audit_path, self.manifest_path
        ))
        for path, filename in zip(normalized, MODELING_PANEL_ARTIFACT_FILENAMES, strict=True):
            if path.parent != directory or path.name != filename:
                raise ModelingPanelArtifactValidationError("write result paths must be fixed direct children.")
        if (not isinstance(self.manifest, ModelingPanelArtifactManifest)
                or not isinstance(self.validation, ModelingPanelArtifactValidationReport)
                or not self.validation.is_valid or self.validation.manifest != self.manifest):
            raise ModelingPanelArtifactValidationError("write result manifest or validation is invalid.")
        object.__setattr__(self, "artifact_dir", directory)
        for field, value in zip(("panel_path", "config_path", "audit_path", "manifest_path"), normalized, strict=True):
            object.__setattr__(self, field, value)


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
        raise ModelingPanelArtifactValidationError("strict JSON read failed.") from exc
    if not isinstance(value, dict):
        raise ModelingPanelArtifactValidationError("JSON top-level value must be an object.")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise ModelingPanelArtifactWriteError("strict JSON write failed.") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelingPanelArtifactValidationError("checksum read failed.") from exc
    return digest.hexdigest()


def _record(path: Path) -> ModelingPanelArtifactFileRecord:
    try:
        size = path.stat().st_size
        digest = _sha256(path)
    except (OSError, ModelingPanelArtifactValidationError) as exc:
        raise ModelingPanelArtifactWriteError("payload metadata failed.") from exc
    return ModelingPanelArtifactFileRecord(path.name, size, digest)


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _date_text(value: object, optional: bool = False) -> bool:
    if optional and value is None:
        return True
    if not isinstance(value, str):
        return False
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") == value
    except ValueError:
        return False


def _validate_unmatched(value: object) -> bool:
    expected = {"row_count", "date_count", "first_trade_date", "last_trade_date", "sampled_keys"}
    if not isinstance(value, dict) or set(value) != expected:
        return False
    rows, dates, samples = value["row_count"], value["date_count"], value["sampled_keys"]
    if not _nonnegative_int(rows) or not _nonnegative_int(dates) or dates > rows or not isinstance(samples, list) or len(samples) > 20:
        return False
    normalized: list[tuple[str, str]] = []
    for item in samples:
        if (not isinstance(item, dict) or set(item) != {"trade_date", "ts_code"}
                or not _date_text(item["trade_date"]) or not isinstance(item["ts_code"], str)
                or not item["ts_code"].strip() or item["ts_code"] != item["ts_code"].strip()):
            return False
        normalized.append((item["trade_date"], item["ts_code"]))
    if normalized != sorted(set(normalized)) or len(normalized) > rows:
        return False
    first, last = value["first_trade_date"], value["last_trade_date"]
    if rows == 0:
        return dates == 0 and first is None and last is None and not normalized
    return dates > 0 and _date_text(first) and _date_text(last) and first <= last and all(first <= item[0] <= last for item in normalized)



def _audit_structure(
    audit: dict[str, object],
    panel_config: ModelingPanelConfig,
    manifest: ModelingPanelArtifactManifest | None,
) -> list[str]:
    errors: list[str] = []
    if set(audit) != _AUDIT_FIELDS:
        return ["audit fields are invalid"]
    features = manifest.feature_names if manifest else tuple(
        audit["feature_names"] if isinstance(audit["feature_names"], list) else ()
    )
    if audit["schema_version"] != MODELING_PANEL_SCHEMA_VERSION:
        errors.append("audit schema version differs")
    if audit["config"] != panel_config.as_dict():
        errors.append("audit config differs")
    if audit["label_column"] != panel_config.label_column:
        errors.append("audit label differs")
    count_fields = (
        "factor_input_rows", "return_input_rows", "matched_rows", "output_rows",
        "date_count", "security_count", "feature_count", "label_missing_count",
        "label_non_finite_count", "duplicate_factor_key_count",
        "duplicate_return_key_count", "entry_before_signal_count",
        "entry_equal_signal_count", "exit_not_after_entry_count",
        "label_formula_mismatch_count",
    )
    if any(not _nonnegative_int(audit[item]) for item in count_fields):
        return errors + ["audit count is invalid"]
    if (audit["matched_rows"] != audit["output_rows"]
            or audit["feature_count"] != len(features)
            or list(features) != audit["feature_names"]):
        errors.append("audit dimensions differ")
    if manifest and audit["output_rows"] != manifest.row_count:
        errors.append("audit rows differ from manifest")
    for side in ("factor_only", "return_only"):
        if not _validate_unmatched(audit[side]):
            errors.append(f"{side} is invalid")
    if isinstance(audit["factor_only"], dict) and (
        audit["factor_input_rows"] - audit["matched_rows"]
        != audit["factor_only"].get("row_count")
    ):
        errors.append("factor unmatched equation differs")
    if isinstance(audit["return_only"], dict) and (
        audit["return_input_rows"] - audit["matched_rows"]
        != audit["return_only"].get("row_count")
    ):
        errors.append("return unmatched equation differs")
    for name in ("feature_missing_counts", "feature_missing_rates", "feature_non_finite_counts"):
        value = audit[name]
        if not isinstance(value, dict) or tuple(value) != features:
            errors.append(f"{name} keys differ")
    for name in ("feature_missing_counts", "feature_non_finite_counts"):
        value = audit[name]
        if isinstance(value, dict) and any(not _nonnegative_int(v) for v in value.values()):
            errors.append(f"{name} values are invalid")
    rates = audit["feature_missing_rates"]
    if isinstance(rates, dict) and any(not _finite_number(v) or not 0 <= float(v) <= 1 for v in rates.values()):
        errors.append("feature missing rates are invalid")
    for name in ("all_missing_features", "constant_features", "suspicious_feature_names"):
        value = audit[name]
        if (not isinstance(value, list) or len(value) != len(set(value))
                or any(item not in features for item in value)):
            errors.append(f"{name} is invalid")
    warnings = audit["warnings"]
    if (not isinstance(warnings, list) or len(warnings) != len(set(warnings))
            or any(not isinstance(item, str) or not item.strip() for item in warnings)):
        errors.append("warnings are invalid")
    success_zero = (
        "duplicate_factor_key_count", "duplicate_return_key_count",
        "entry_before_signal_count", "exit_not_after_entry_count",
        "label_formula_mismatch_count", "label_non_finite_count",
    )
    if any(audit[name] != 0 for name in success_zero):
        errors.append("audit success counters are nonzero")
    if panel_config.require_entry_after_signal and audit["entry_equal_signal_count"] != 0:
        errors.append("strict entry counter is nonzero")
    if not panel_config.allow_missing_labels and audit["label_missing_count"] != 0:
        errors.append("missing label policy differs")
    if panel_config.unmatched_policy == "error":
        for side in ("factor_only", "return_only"):
            if isinstance(audit[side], dict) and audit[side].get("row_count") != 0:
                errors.append("unmatched policy differs")
    if audit["all_missing_features"]:
        errors.append("all-missing feature recorded")
    nonfinite = audit["feature_non_finite_counts"]
    if isinstance(nonfinite, dict) and any(nonfinite.values()):
        errors.append("feature nonfinite counter is nonzero")
    for name in (
        "first_trade_date", "last_trade_date", "first_entry_trade_date",
        "last_entry_trade_date", "first_exit_trade_date", "last_exit_trade_date",
    ):
        optional = name not in {"first_trade_date", "last_trade_date"}
        if not _date_text(audit[name], optional):
            errors.append(f"{name} is invalid")
    distribution = (
        "per_date_security_count_min", "per_date_security_count_median",
        "per_date_security_count_max", "per_security_observation_count_min",
        "per_security_observation_count_median",
        "per_security_observation_count_max",
    )
    if any(not _finite_number(audit[name]) or float(audit[name]) <= 0 for name in distribution):
        errors.append("distribution value is invalid")
    return errors


def _nonfinite_count(series: pd.Series) -> int:
    try:
        values = series.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        return -1
    return int(np.isinf(values).sum())


def _date_range(series: pd.Series) -> tuple[str | None, str | None]:
    present = series[series.notna()]
    if present.empty:
        return None, None
    return pd.Timestamp(present.min()).date().isoformat(), pd.Timestamp(present.max()).date().isoformat()


def _panel_audit_errors(
    panel: pd.DataFrame,
    audit: dict[str, object],
    config: ModelingPanelConfig,
    manifest: ModelingPanelArtifactManifest,
) -> list[str]:
    errors: list[str] = []
    features = manifest.feature_names
    expected: dict[str, object] = {
        "output_rows": len(panel),
        "date_count": int(panel["trade_date"].nunique()),
        "security_count": int(panel["ts_code"].nunique()),
        "first_trade_date": _date_range(panel["trade_date"])[0],
        "last_trade_date": _date_range(panel["trade_date"])[1],
        "first_entry_trade_date": _date_range(panel["entry_trade_date"])[0],
        "last_entry_trade_date": _date_range(panel["entry_trade_date"])[1],
        "first_exit_trade_date": _date_range(panel["exit_trade_date"])[0],
        "last_exit_trade_date": _date_range(panel["exit_trade_date"])[1],
        "label_missing_count": int(panel[manifest.label_column].isna().sum()),
        "label_non_finite_count": _nonfinite_count(panel[manifest.label_column]),
    }
    missing_counts = {name: int(panel[name].isna().sum()) for name in features}
    expected["feature_missing_counts"] = missing_counts
    expected["feature_missing_rates"] = {name: count / len(panel) for name, count in missing_counts.items()}
    expected["feature_non_finite_counts"] = {name: _nonfinite_count(panel[name]) for name in features}
    expected["constant_features"] = [name for name in features if int(panel[name].nunique(dropna=True)) <= 1]
    expected["suspicious_feature_names"] = [
        name for name in features if name.lower().startswith(("future_", "next_", "lead_", "target_", "label_"))
    ]
    per_date = panel.groupby("trade_date", sort=False).size()
    per_security = panel.groupby("ts_code", sort=False).size()
    expected.update({
        "per_date_security_count_min": int(per_date.min()),
        "per_date_security_count_median": float(per_date.median()),
        "per_date_security_count_max": int(per_date.max()),
        "per_security_observation_count_min": int(per_security.min()),
        "per_security_observation_count_median": float(per_security.median()),
        "per_security_observation_count_max": int(per_security.max()),
    })
    for name, value in expected.items():
        actual = audit.get(name)
        if name == "feature_missing_rates" and isinstance(actual, dict):
            if tuple(actual) != features or any(
                not math.isclose(float(actual[key]), float(value[key]), rel_tol=1e-12, abs_tol=1e-12)
                for key in features
            ):
                errors.append(f"{name} differs")
        elif actual != value:
            errors.append(f"{name} differs")
    duplicate = int(panel.duplicated(list(MODELING_PANEL_KEY_COLUMNS)).sum())
    entry_present = panel["entry_trade_date"].notna()
    exit_present = panel["exit_trade_date"].notna()
    checks = {
        "duplicate_output_key_count": duplicate,
        "entry_before_signal_count": int((entry_present & (panel["entry_trade_date"] < panel["trade_date"])).sum()),
        "entry_equal_signal_count": int((entry_present & (panel["entry_trade_date"] == panel["trade_date"])).sum()),
        "exit_not_after_entry_count": int((entry_present & exit_present & (panel["exit_trade_date"] <= panel["entry_trade_date"])).sum()),
    }
    if duplicate:
        errors.append("duplicate output key")
    for name in ("entry_before_signal_count", "entry_equal_signal_count", "exit_not_after_entry_count"):
        if audit.get(name) != checks[name]:
            errors.append(f"{name} differs")
    complete = panel[manifest.label_column].notna() & panel["entry_price"].notna() & panel["exit_price"].notna()
    mismatch = 0
    if bool(complete.any()):
        expected_label = panel.loc[complete, "exit_price"] / panel.loc[complete, "entry_price"] - 1.0
        actual_label = panel.loc[complete, manifest.label_column]
        mismatch = int((~np.isclose(actual_label.to_numpy(dtype=float), expected_label.to_numpy(dtype=float),
                                    rtol=1e-10, atol=1e-12, equal_nan=False)).sum())
    if audit.get("label_formula_mismatch_count") != mismatch:
        errors.append("label formula differs")
    return errors



def _issue(code: str, message: str, path: str | None = None) -> ModelingPanelArtifactValidationIssue:
    return ModelingPanelArtifactValidationIssue(code, message, path)


class ModelingPanelArtifactStore:
    """Write, inspect, and independently validate fixed-layout artifacts."""

    def read_manifest(self, artifact_dir: str | os.PathLike[str]) -> ModelingPanelArtifactManifest:
        directory = _absolute(_path(artifact_dir, ModelingPanelArtifactValidationError))
        if not directory.exists() or directory.is_symlink() or not directory.is_dir():
            raise ModelingPanelArtifactValidationError("artifact directory is invalid.")
        path = directory / MODELING_PANEL_MANIFEST_FILENAME
        if not path.exists() or path.is_symlink() or not path.is_file():
            raise ModelingPanelArtifactValidationError("manifest file is invalid.")
        return ModelingPanelArtifactManifest.from_dict(_read_json(path))

    def validate(self, artifact_dir: str | os.PathLike[str]) -> ModelingPanelArtifactValidationReport:
        directory = _absolute(_path(artifact_dir, ModelingPanelArtifactValidationError))
        issues: list[ModelingPanelArtifactValidationIssue] = []
        manifest: ModelingPanelArtifactManifest | None = None
        if not directory.exists():
            return self._report(directory, [_issue("artifact_dir_missing", "Artifact directory is missing.")], None)
        if directory.is_symlink():
            return self._report(directory, [_issue("artifact_dir_symlink", "Artifact directory is a symlink.")], None)
        if not directory.is_dir():
            return self._report(directory, [_issue("artifact_dir_not_directory", "Artifact path is not a directory.")], None)
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            return self._report(directory, [_issue("artifact_dir_not_directory", "Artifact directory cannot be read.")], None)
        expected = set(MODELING_PANEL_ARTIFACT_FILENAMES)
        actual = {item.name for item in entries}
        for name in sorted(actual - expected):
            issues.append(_issue("unexpected_entry", "Unexpected artifact entry.", None))
        for name in MODELING_PANEL_ARTIFACT_FILENAMES:
            if name not in actual:
                issues.append(_issue("missing_file", "Required artifact file is missing.", name))
        safe: dict[str, Path] = {}
        for name in MODELING_PANEL_ARTIFACT_FILENAMES:
            path = directory / name
            if name not in actual:
                continue
            if path.is_symlink():
                code = "payload_symlink" if name != MODELING_PANEL_MANIFEST_FILENAME else "manifest_symlink"
                issues.append(_issue(code, "Artifact file is a symlink.", name))
            elif not path.is_file():
                issues.append(_issue("payload_not_regular_file", "Artifact path is not a regular file.", name))
            else:
                safe[name] = path
        manifest_path = safe.get(MODELING_PANEL_MANIFEST_FILENAME)
        if manifest_path is not None:
            try:
                manifest = ModelingPanelArtifactManifest.from_dict(_read_json(manifest_path))
            except ModelingPanelArtifactValidationError:
                issues.append(_issue("invalid_manifest_json", "Manifest JSON or schema is invalid.", MODELING_PANEL_MANIFEST_FILENAME))
        if manifest is not None:
            for record in manifest.files:
                if record.relative_path not in _PAYLOAD_FILENAMES:
                    issues.append(_issue("unsafe_relative_path", "Manifest path is unsafe.", MODELING_PANEL_MANIFEST_FILENAME))
                    continue
                path = safe.get(record.relative_path)
                if path is None:
                    continue
                try:
                    if path.stat().st_size != record.size_bytes:
                        issues.append(_issue("file_size_mismatch", "Payload size differs from manifest.", record.relative_path))
                    if _sha256(path) != record.sha256:
                        issues.append(_issue("checksum_mismatch", "Payload checksum differs from manifest.", record.relative_path))
                except (OSError, ModelingPanelArtifactValidationError):
                    issues.append(_issue("checksum_mismatch", "Payload metadata cannot be read.", record.relative_path))
        panel_config: ModelingPanelConfig | None = None
        config_path = safe.get(MODELING_PANEL_CONFIG_FILENAME)
        if config_path is not None:
            try:
                panel_config = ModelingPanelConfig.from_dict(_read_json(config_path))
            except (ModelingPanelError, ModelingPanelArtifactValidationError):
                issues.append(_issue("invalid_config_json", "Config JSON is invalid.", MODELING_PANEL_CONFIG_FILENAME))
        audit: dict[str, object] | None = None
        audit_path = safe.get(MODELING_PANEL_AUDIT_FILENAME)
        if audit_path is not None:
            try:
                audit = _read_json(audit_path)
            except ModelingPanelArtifactValidationError:
                issues.append(_issue("invalid_audit_json", "Audit JSON is invalid.", MODELING_PANEL_AUDIT_FILENAME))
        if panel_config is not None and manifest is not None and panel_config.label_column != manifest.label_column:
            issues.append(_issue("config_manifest_mismatch", "Config label differs from manifest.", MODELING_PANEL_CONFIG_FILENAME))
        if audit is not None and panel_config is not None:
            audit_errors = _audit_structure(audit, panel_config, manifest)
            for message in audit_errors:
                code = "audit_config_mismatch" if "config" in message or "policy" in message else "audit_integrity_error"
                issues.append(_issue(code, message.capitalize() + ".", MODELING_PANEL_AUDIT_FILENAME))
        panel: pd.DataFrame | None = None
        panel_path = safe.get(MODELING_PANEL_PARQUET_FILENAME)
        if panel_path is not None:
            try:
                panel = pd.read_parquet(panel_path, engine="pyarrow")
            except Exception:
                issues.append(_issue("parquet_read_error", "Parquet payload cannot be read.", MODELING_PANEL_PARQUET_FILENAME))
        if panel is not None and manifest is not None:
            if not panel.columns.is_unique or tuple(panel.columns) != manifest.columns:
                issues.append(_issue("parquet_column_mismatch", "Parquet columns differ from manifest.", MODELING_PANEL_PARQUET_FILENAME))
            if len(panel) != manifest.row_count:
                issues.append(_issue("parquet_row_count_mismatch", "Parquet row count differs from manifest.", MODELING_PANEL_PARQUET_FILENAME))
            dtypes = tuple((str(name), str(dtype)) for name, dtype in panel.dtypes.items())
            if dtypes != manifest.pandas_dtypes:
                issues.append(_issue("parquet_dtype_mismatch", "Parquet dtypes differ from manifest.", MODELING_PANEL_PARQUET_FILENAME))
            if audit is not None and panel.columns.is_unique and tuple(panel.columns) == manifest.columns:
                for message in _panel_audit_errors(panel, audit, panel_config, manifest):  # type: ignore[arg-type]
                    issues.append(_issue("panel_content_mismatch", message.capitalize() + ".", MODELING_PANEL_PARQUET_FILENAME))
        return self._report(directory, issues, manifest)

    @staticmethod
    def _report(
        directory: Path,
        issues: list[ModelingPanelArtifactValidationIssue],
        manifest: ModelingPanelArtifactManifest | None,
    ) -> ModelingPanelArtifactValidationReport:
        unique: list[ModelingPanelArtifactValidationIssue] = []
        seen: set[ModelingPanelArtifactValidationIssue] = set()
        for item in issues:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return ModelingPanelArtifactValidationReport(directory, not unique, tuple(unique), manifest)

    def write(
        self,
        result: ModelingPanelResult,
        config: ModelingPanelArtifactConfig,
    ) -> ModelingPanelArtifactWriteResult:
        if not isinstance(result, ModelingPanelResult):
            raise ModelingPanelArtifactWriteError("result must be a ModelingPanelResult.")
        if not isinstance(config, ModelingPanelArtifactConfig):
            raise ModelingPanelArtifactConfigError("config must be ModelingPanelArtifactConfig.")
        target = _absolute(config.artifact_dir)
        parent = target.parent
        if target.exists() or target.is_symlink():
            raise ModelingPanelArtifactWriteError("target artifact_dir already exists.")
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ModelingPanelArtifactWriteError("artifact parent is invalid.")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelingPanelArtifactWriteError("artifact parent cannot be created.") from exc
        if parent.is_symlink():
            raise ModelingPanelArtifactWriteError("artifact parent must not be a symlink.")
        staging = parent / f".tmp-{target.name}-{uuid4().hex}"
        published = False
        try:
            staging.mkdir()
            panel = result.panel
            panel_path = staging / MODELING_PANEL_PARQUET_FILENAME
            panel.to_parquet(
                panel_path, engine="pyarrow", compression=config.parquet_compression, index=False
            )
            _write_json(staging / MODELING_PANEL_CONFIG_FILENAME, result.config.as_dict())
            _write_json(staging / MODELING_PANEL_AUDIT_FILENAME, result.audit.as_dict())
            persisted = pd.read_parquet(panel_path, engine="pyarrow")
            try:
                pdt.assert_frame_equal(
                    persisted, panel, check_like=False, check_dtype=False,
                    check_exact=False, rtol=1e-12, atol=1e-12,
                )
            except AssertionError as exc:
                raise ModelingPanelArtifactWriteError("Parquet roundtrip changed panel semantics.") from exc
            records = tuple(_record(staging / name) for name in _PAYLOAD_FILENAMES)
            created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            manifest = ModelingPanelArtifactManifest(
                artifact_type=MODELING_PANEL_ARTIFACT_TYPE,
                artifact_schema_version=MODELING_PANEL_ARTIFACT_SCHEMA_VERSION,
                modeling_panel_schema_version=result.schema_version,
                created_at_utc=created,
                label_column=result.label_column,
                feature_names=result.feature_names,
                row_count=len(persisted),
                column_count=len(persisted.columns),
                columns=tuple(str(item) for item in persisted.columns),
                pandas_dtypes=tuple((str(name), str(dtype)) for name, dtype in persisted.dtypes.items()),
                files=records,
            )
            _write_json(staging / MODELING_PANEL_MANIFEST_FILENAME, manifest.as_dict())
            pre = self.validate(staging)
            if not pre.is_valid:
                raise ModelingPanelArtifactWriteError("pre-publish validation failed.")
            if target.exists() or target.is_symlink():
                raise ModelingPanelArtifactWriteError("target appeared before publication.")
            os.replace(staging, target)
            published = True
            validation = self.validate(target) if config.verify_after_write else ModelingPanelArtifactValidationReport(
                target, True, (), manifest
            )
            if not validation.is_valid:
                try:
                    shutil.rmtree(target)
                except OSError as exc:
                    raise ModelingPanelArtifactWriteError("post-publish validation and cleanup failed.") from exc
                raise ModelingPanelArtifactWriteError("post-publish validation failed.")
            return ModelingPanelArtifactWriteResult(
                target,
                target / MODELING_PANEL_PARQUET_FILENAME,
                target / MODELING_PANEL_CONFIG_FILENAME,
                target / MODELING_PANEL_AUDIT_FILENAME,
                target / MODELING_PANEL_MANIFEST_FILENAME,
                manifest,
                validation,
            )
        except ModelingPanelArtifactError:
            if staging.exists() and not published:
                shutil.rmtree(staging, ignore_errors=True)
            raise
        except Exception as exc:
            if staging.exists() and not published:
                shutil.rmtree(staging, ignore_errors=True)
            raise ModelingPanelArtifactWriteError("artifact write failed.") from exc

