"""Safe, atomic persistence and independent validation for V5 Signal Artifacts."""

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

from src.signals.builder import SignalBuildResult
from src.signals.contracts import (
    SIGNAL_KEY_COLUMNS,
    SIGNAL_OUTPUT_COLUMNS,
    SIGNAL_SCHEMA_VERSION,
    SignalContractError,
)
from src.signals.sources import PredictionSourceProvenance


SIGNAL_ARTIFACT_SCHEMA_VERSION = "1.0"
SIGNAL_ARTIFACT_TYPE = "signal"
SIGNAL_PARQUET_FILENAME = "signals.parquet"
SIGNAL_CONFIG_FILENAME = "config.json"
SIGNAL_AUDIT_FILENAME = "audit.json"
SIGNAL_MANIFEST_FILENAME = "manifest.json"
SIGNAL_ARTIFACT_FILENAMES = (
    SIGNAL_PARQUET_FILENAME,
    SIGNAL_CONFIG_FILENAME,
    SIGNAL_AUDIT_FILENAME,
    SIGNAL_MANIFEST_FILENAME,
)

_PAYLOAD_FILENAMES = SIGNAL_ARTIFACT_FILENAMES[:-1]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISSUE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CHUNK_SIZE = 1024 * 1024
_CONFIG_FIELDS = {"prediction_column", "signal_direction"}
_PROVENANCE_FIELDS = {
    "artifact_dir", "prediction_path", "artifact_schema_version",
    "experiment_id", "model_name", "prediction_sha256",
}
_AUDIT_FIELDS = {
    "signal_schema_version", "input_rows", "output_rows", "trade_date_count",
    "min_trade_date", "max_trade_date", "prediction_column",
    "signal_direction", "score_finite", "duplicate_key_count",
    "rank_integrity", "source_provenance", "warnings",
}
_MANIFEST_FIELDS = {
    "artifact_type", "artifact_schema_version", "signal_schema_version",
    "created_at_utc", "row_count", "column_count", "columns",
    "pandas_dtypes", "prediction_column", "signal_direction",
    "source_provenance", "files",
}


class SignalArtifactError(SignalContractError):
    """Base error for Signal Artifact operations."""


class SignalArtifactExistsError(SignalArtifactError):
    """Raised when no-overwrite prevents publication."""


class SignalArtifactWriteError(SignalArtifactError):
    """Raised when a Signal Artifact cannot be safely written."""


class SignalArtifactValidationError(SignalArtifactError):
    """Raised when strict Artifact metadata or API input is invalid."""


def _strict_keys(
    value: object, expected: set[str], context: str
) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise SignalArtifactValidationError(
            f"{context} must be a mapping with string keys."
        )
    if set(value) != expected:
        raise SignalArtifactValidationError(f"{context} fields are invalid.")
    return dict(value)


def _path(value: object) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise SignalArtifactValidationError("artifact_dir must be str or os.PathLike.")
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise SignalArtifactValidationError("artifact_dir must be path-like.") from exc
    if not isinstance(raw, str) or not raw.strip() or raw != raw.strip() or "\x00" in raw:
        raise SignalArtifactValidationError(
            "artifact_dir must identify a non-empty trimmed directory."
        )
    result = Path(raw)
    if raw in {".", ".."} or result == Path(result.anchor):
        raise SignalArtifactValidationError(
            "artifact_dir must identify an explicit child directory."
        )
    return result


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _trimmed(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SignalArtifactValidationError(
            f"{context} must be a non-empty trimmed string."
        )
    return value


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SignalArtifactValidationError(
            f"{context} must be a non-negative integer."
        )
    return value


@dataclass(frozen=True)
class SignalArtifactConfig:
    """Filesystem-only write policy; business config comes from the build result."""

    artifact_dir: Path
    parquet_compression: str = "zstd"
    verify_after_write: bool = True

    def __post_init__(self) -> None:
        path = _path(self.artifact_dir)
        compression = self.parquet_compression
        if not isinstance(compression, str):
            raise SignalArtifactValidationError(
                "parquet_compression must be a string."
            )
        compression = compression.strip().lower()
        if compression not in {"zstd", "snappy"}:
            raise SignalArtifactValidationError(
                "parquet_compression must be 'zstd' or 'snappy'."
            )
        if type(self.verify_after_write) is not bool:
            raise SignalArtifactValidationError(
                "verify_after_write must be a bool."
            )
        object.__setattr__(self, "artifact_dir", path)
        object.__setattr__(self, "parquet_compression", compression)


@dataclass(frozen=True)
class SignalArtifactFileRecord:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.relative_path not in _PAYLOAD_FILENAMES:
            raise SignalArtifactValidationError(
                "file record relative_path is unsafe or unsupported."
            )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise SignalArtifactValidationError(
                "file record size_bytes must be a positive integer."
            )
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise SignalArtifactValidationError(
                "file record sha256 must be lowercase hexadecimal."
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SignalArtifactFileRecord:
        return cls(**_strict_keys(
            value, {"relative_path", "size_bytes", "sha256"}, "file record"
        ))  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _provenance(value: object) -> dict[str, str]:
    data = _strict_keys(value, _PROVENANCE_FIELDS, "source provenance")
    normalized: dict[str, str] = {}
    for name in _PROVENANCE_FIELDS:
        normalized[name] = _trimmed(data[name], f"source provenance {name}")
    if not _SHA256_RE.fullmatch(normalized["prediction_sha256"]):
        raise SignalArtifactValidationError(
            "source provenance prediction_sha256 is invalid."
        )
    artifact = Path(normalized["artifact_dir"])
    prediction = Path(normalized["prediction_path"])
    if (
        not artifact.is_absolute()
        or not prediction.is_absolute()
        or prediction.parent != artifact
        or prediction.name != "predictions.parquet"
    ):
        raise SignalArtifactValidationError("source provenance paths are invalid.")
    return {name: normalized[name] for name in (
        "artifact_dir", "prediction_path", "artifact_schema_version",
        "experiment_id", "model_name", "prediction_sha256",
    )}


@dataclass(frozen=True)
class SignalArtifactManifest:
    artifact_type: str
    artifact_schema_version: str
    signal_schema_version: str
    created_at_utc: str
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    pandas_dtypes: tuple[tuple[str, str], ...]
    prediction_column: str
    signal_direction: str
    source_provenance: Mapping[str, str]
    files: tuple[SignalArtifactFileRecord, ...]

    def __post_init__(self) -> None:
        if self.artifact_type != SIGNAL_ARTIFACT_TYPE:
            raise SignalArtifactValidationError("artifact_type is invalid.")
        if self.artifact_schema_version != SIGNAL_ARTIFACT_SCHEMA_VERSION:
            raise SignalArtifactValidationError("artifact_schema_version is invalid.")
        if self.signal_schema_version != SIGNAL_SCHEMA_VERSION:
            raise SignalArtifactValidationError("signal_schema_version is invalid.")
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc.endswith("Z"):
            raise SignalArtifactValidationError("created_at_utc must be UTC Z.")
        try:
            created = datetime.fromisoformat(self.created_at_utc[:-1] + "+00:00")
        except ValueError as exc:
            raise SignalArtifactValidationError(
                "created_at_utc is not valid ISO-8601."
            ) from exc
        if created.utcoffset() != timezone.utc.utcoffset(created):
            raise SignalArtifactValidationError("created_at_utc must be UTC.")
        rows = _nonnegative_int(self.row_count, "row_count")
        if rows == 0:
            raise SignalArtifactValidationError("row_count must be positive.")
        columns = tuple(self.columns)
        if columns != SIGNAL_OUTPUT_COLUMNS:
            raise SignalArtifactValidationError("manifest columns are invalid.")
        if self.column_count != len(columns):
            raise SignalArtifactValidationError("column_count does not match columns.")
        dtypes = tuple(tuple(item) for item in self.pandas_dtypes)
        if (
            len(dtypes) != len(columns)
            or any(len(item) != 2 for item in dtypes)
            or tuple(item[0] for item in dtypes) != columns
            or any(not isinstance(item[1], str) or not item[1] for item in dtypes)
        ):
            raise SignalArtifactValidationError("pandas_dtypes are invalid.")
        prediction_column = _trimmed(self.prediction_column, "prediction_column")
        direction = _trimmed(self.signal_direction, "signal_direction")
        if direction not in {"ascending", "descending"}:
            raise SignalArtifactValidationError("signal_direction is invalid.")
        provenance = _provenance(self.source_provenance)
        files = tuple(self.files)
        if (
            len(files) != len(_PAYLOAD_FILENAMES)
            or any(not isinstance(item, SignalArtifactFileRecord) for item in files)
            or tuple(item.relative_path for item in files) != _PAYLOAD_FILENAMES
        ):
            raise SignalArtifactValidationError("manifest files are invalid.")
        object.__setattr__(self, "row_count", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "pandas_dtypes", dtypes)
        object.__setattr__(self, "prediction_column", prediction_column)
        object.__setattr__(self, "signal_direction", direction)
        object.__setattr__(self, "source_provenance", MappingProxyType(provenance))
        object.__setattr__(self, "files", files)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SignalArtifactManifest:
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
                signal_schema_version=data["signal_schema_version"],  # type: ignore[arg-type]
                created_at_utc=data["created_at_utc"],  # type: ignore[arg-type]
                row_count=data["row_count"],  # type: ignore[arg-type]
                column_count=data["column_count"],  # type: ignore[arg-type]
                columns=tuple(data["columns"]),  # type: ignore[arg-type]
                pandas_dtypes=tuple(tuple(item) for item in data["pandas_dtypes"]),  # type: ignore[arg-type]
                prediction_column=data["prediction_column"],  # type: ignore[arg-type]
                signal_direction=data["signal_direction"],  # type: ignore[arg-type]
                source_provenance=data["source_provenance"],  # type: ignore[arg-type]
                files=tuple(SignalArtifactFileRecord.from_dict(item) for item in raw_files),
            )
        except SignalArtifactValidationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise SignalArtifactValidationError("manifest cannot be parsed.") from exc

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_type": self.artifact_type,
            "artifact_schema_version": self.artifact_schema_version,
            "signal_schema_version": self.signal_schema_version,
            "created_at_utc": self.created_at_utc,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "pandas_dtypes": [list(item) for item in self.pandas_dtypes],
            "prediction_column": self.prediction_column,
            "signal_direction": self.signal_direction,
            "source_provenance": dict(self.source_provenance),
            "files": [item.as_dict() for item in self.files],
        }


@dataclass(frozen=True)
class SignalArtifactValidationIssue:
    code: str
    message: str
    relative_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not _ISSUE_CODE_RE.fullmatch(self.code):
            raise SignalArtifactValidationError("issue code is invalid.")
        _trimmed(self.message, "issue message")
        if self.relative_path is not None and self.relative_path not in SIGNAL_ARTIFACT_FILENAMES:
            raise SignalArtifactValidationError("issue relative_path is invalid.")

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class SignalArtifactValidationReport:
    artifact_dir: Path
    is_valid: bool
    issues: tuple[SignalArtifactValidationIssue, ...]
    manifest: SignalArtifactManifest | None

    def __post_init__(self) -> None:
        directory = _absolute(_path(self.artifact_dir))
        issues = tuple(self.issues)
        if any(not isinstance(item, SignalArtifactValidationIssue) for item in issues):
            raise SignalArtifactValidationError("issues contain invalid values.")
        if len(issues) != len(set(issues)):
            raise SignalArtifactValidationError("issues contain duplicates.")
        if type(self.is_valid) is not bool or self.is_valid != (not issues):
            raise SignalArtifactValidationError("is_valid must agree with issues.")
        if self.is_valid and not isinstance(self.manifest, SignalArtifactManifest):
            raise SignalArtifactValidationError("a valid report requires a manifest.")
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
class SignalArtifactWriteResult:
    artifact_dir: Path
    signal_path: Path
    config_path: Path
    audit_path: Path
    manifest_path: Path
    rows: int
    schema_version: str
    manifest: SignalArtifactManifest
    validation: SignalArtifactValidationReport

    def __post_init__(self) -> None:
        directory = _absolute(Path(self.artifact_dir))
        paths = tuple(_absolute(Path(item)) for item in (
            self.signal_path, self.config_path, self.audit_path, self.manifest_path
        ))
        for path, filename in zip(paths, SIGNAL_ARTIFACT_FILENAMES, strict=True):
            if path.parent != directory or path.name != filename:
                raise SignalArtifactValidationError(
                    "write result paths must be fixed direct children."
                )
        if (
            self.schema_version != SIGNAL_ARTIFACT_SCHEMA_VERSION
            or self.rows != self.manifest.row_count
            or not self.validation.is_valid
            or self.validation.manifest != self.manifest
        ):
            raise SignalArtifactValidationError("write result metadata is invalid.")
        object.__setattr__(self, "artifact_dir", directory)
        for name, value in zip(
            ("signal_path", "config_path", "audit_path", "manifest_path"),
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
        raise SignalArtifactValidationError("strict JSON read failed.") from exc
    if not isinstance(value, dict):
        raise SignalArtifactValidationError("JSON top-level value must be an object.")
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
        raise SignalArtifactWriteError("strict JSON write failed.") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SignalArtifactValidationError("checksum read failed.") from exc
    return digest.hexdigest()


def _record(path: Path) -> SignalArtifactFileRecord:
    try:
        return SignalArtifactFileRecord(path.name, path.stat().st_size, _sha256(path))
    except (OSError, SignalArtifactValidationError) as exc:
        raise SignalArtifactWriteError("payload metadata failed.") from exc


def _issue(
    code: str, message: str, path: str | None = None
) -> SignalArtifactValidationIssue:
    return SignalArtifactValidationIssue(code, message, path)


def _config(value: object) -> dict[str, object]:
    data = _strict_keys(value, _CONFIG_FIELDS, "config")
    prediction = _trimmed(data["prediction_column"], "prediction_column")
    direction = _trimmed(data["signal_direction"], "signal_direction")
    if direction not in {"ascending", "descending"}:
        raise SignalArtifactValidationError("signal_direction is invalid.")
    return {"prediction_column": prediction, "signal_direction": direction}


def _date_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") == value
    except ValueError:
        return False


def _audit(value: object) -> dict[str, object]:
    data = _strict_keys(value, _AUDIT_FIELDS, "audit")
    if data["signal_schema_version"] != SIGNAL_SCHEMA_VERSION:
        raise SignalArtifactValidationError("audit signal schema version is invalid.")
    for name in ("input_rows", "output_rows", "trade_date_count", "duplicate_key_count"):
        _nonnegative_int(data[name], name)
    if (
        data["input_rows"] != data["output_rows"]
        or data["output_rows"] == 0
        or data["trade_date_count"] == 0
        or data["trade_date_count"] > data["output_rows"]
        or data["duplicate_key_count"] != 0
    ):
        raise SignalArtifactValidationError("audit row/key counts are invalid.")
    if not _date_text(data["min_trade_date"]) or not _date_text(data["max_trade_date"]):
        raise SignalArtifactValidationError("audit dates are invalid.")
    if data["min_trade_date"] > data["max_trade_date"]:
        raise SignalArtifactValidationError("audit date range is invalid.")
    config = _config({name: data[name] for name in _CONFIG_FIELDS})
    if data["score_finite"] is not True or data["rank_integrity"] is not True:
        raise SignalArtifactValidationError("audit integrity flags are invalid.")
    provenance = _provenance(data["source_provenance"])
    warnings = data["warnings"]
    if (
        not isinstance(warnings, list)
        or warnings != sorted(set(warnings))
        or any(not isinstance(item, str) or not item.strip() for item in warnings)
    ):
        raise SignalArtifactValidationError("audit warnings are invalid.")
    return {
        **data,
        **config,
        "source_provenance": provenance,
        "warnings": list(warnings),
    }


def _signal_errors(frame: pd.DataFrame, direction: str) -> list[str]:
    errors: list[str] = []
    if frame.empty:
        return ["Signal payload is empty."]
    if not frame.columns.is_unique or tuple(frame.columns) != SIGNAL_OUTPUT_COLUMNS:
        return ["Signal columns or order are invalid."]
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
    score = frame["score"]
    if (
        pd.api.types.is_bool_dtype(score.dtype)
        or not pd.api.types.is_numeric_dtype(score.dtype)
        or pd.api.types.is_complex_dtype(score.dtype)
    ):
        errors.append("score dtype is invalid.")
    else:
        try:
            values = score.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError):
            errors.append("score dtype is invalid.")
        else:
            if not np.isfinite(values).all():
                errors.append("score contains non-finite values.")
    ranks = frame["rank"]
    if not pd.api.types.is_integer_dtype(ranks.dtype) or bool((ranks <= 0).any()):
        errors.append("rank must be positive integer data.")
    if not errors:
        if frame.duplicated(list(SIGNAL_KEY_COLUMNS)).any():
            errors.append("Signal keys are duplicated.")
        expected_order = frame.sort_values(
            ["trade_date", "rank", "ts_code"], kind="mergesort"
        ).reset_index(drop=True)
        try:
            pdt.assert_frame_equal(frame.reset_index(drop=True), expected_order)
        except AssertionError:
            errors.append("Signal row order is not canonical.")
        for _, group in frame.groupby("trade_date", sort=False):
            expected_ranks = np.arange(1, len(group) + 1, dtype=np.int64)
            if not np.array_equal(group["rank"].to_numpy(), expected_ranks):
                errors.append("Ranks are not unique contiguous 1..N per date.")
                break
        expected_ranked = frame.sort_values(
            ["trade_date", "score", "ts_code"],
            ascending=[True, direction == "ascending", True],
            kind="mergesort",
        ).reset_index(drop=True)
        expected_ranked["rank"] = (
            expected_ranked.groupby("trade_date", sort=False).cumcount() + 1
        ).astype(np.int64)
        if not frame.reset_index(drop=True).equals(expected_ranked):
            errors.append("Ranks disagree with direction, score, or tie-break semantics.")
    return errors


class SignalArtifactStore:
    """Write and independently validate one explicit fixed-layout Signal Artifact."""

    def read_manifest(
        self, artifact_dir: str | os.PathLike[str]
    ) -> SignalArtifactManifest:
        directory = _absolute(_path(artifact_dir))
        path = directory / SIGNAL_MANIFEST_FILENAME
        if (
            not directory.exists() or directory.is_symlink() or not directory.is_dir()
            or not path.exists() or path.is_symlink() or not path.is_file()
        ):
            raise SignalArtifactValidationError("manifest path is invalid.")
        return SignalArtifactManifest.from_dict(_read_json(path))

    def validate(
        self, artifact_dir: str | os.PathLike[str]
    ) -> SignalArtifactValidationReport:
        directory = _absolute(_path(artifact_dir))
        issues: list[SignalArtifactValidationIssue] = []
        manifest: SignalArtifactManifest | None = None
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
        expected = set(SIGNAL_ARTIFACT_FILENAMES)
        actual = {item.name for item in entries}
        for _ in sorted(actual - expected):
            issues.append(_issue("unexpected_entry", "Unexpected artifact entry."))
        for name in SIGNAL_ARTIFACT_FILENAMES:
            if name not in actual:
                issues.append(_issue("missing_file", "Required artifact file is missing.", name))
        safe: dict[str, Path] = {}
        for name in SIGNAL_ARTIFACT_FILENAMES:
            path = directory / name
            if name not in actual:
                continue
            if path.is_symlink():
                issues.append(_issue("artifact_file_symlink", "Artifact file is a symlink.", name))
            elif not path.is_file():
                issues.append(_issue("artifact_file_not_regular", "Artifact path is not a regular file.", name))
            else:
                safe[name] = path
        manifest_path = safe.get(SIGNAL_MANIFEST_FILENAME)
        if manifest_path is not None:
            try:
                manifest = SignalArtifactManifest.from_dict(_read_json(manifest_path))
            except SignalArtifactValidationError:
                issues.append(_issue("invalid_manifest_json", "Manifest JSON or schema is invalid.", SIGNAL_MANIFEST_FILENAME))
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
                except (OSError, SignalArtifactValidationError):
                    issues.append(_issue("checksum_mismatch", "Payload metadata cannot be read.", record.relative_path))
        config: dict[str, object] | None = None
        if (path := safe.get(SIGNAL_CONFIG_FILENAME)) is not None:
            try:
                config = _config(_read_json(path))
            except SignalArtifactValidationError:
                issues.append(_issue("invalid_config_json", "Config JSON is invalid.", SIGNAL_CONFIG_FILENAME))
        audit: dict[str, object] | None = None
        if (path := safe.get(SIGNAL_AUDIT_FILENAME)) is not None:
            try:
                audit = _audit(_read_json(path))
            except SignalArtifactValidationError:
                issues.append(_issue("invalid_audit_json", "Audit JSON is invalid.", SIGNAL_AUDIT_FILENAME))
        if manifest is not None and config is not None:
            if (
                config["prediction_column"] != manifest.prediction_column
                or config["signal_direction"] != manifest.signal_direction
            ):
                issues.append(_issue("config_manifest_mismatch", "Config differs from manifest.", SIGNAL_CONFIG_FILENAME))
        if audit is not None and config is not None:
            if any(audit[name] != config[name] for name in _CONFIG_FIELDS):
                issues.append(_issue("audit_config_mismatch", "Audit differs from config.", SIGNAL_AUDIT_FILENAME))
        if audit is not None and manifest is not None:
            if (
                audit["output_rows"] != manifest.row_count
                or audit["signal_schema_version"] != manifest.signal_schema_version
                or audit["prediction_column"] != manifest.prediction_column
                or audit["signal_direction"] != manifest.signal_direction
                or audit["source_provenance"] != dict(manifest.source_provenance)
            ):
                issues.append(_issue("audit_manifest_mismatch", "Audit differs from manifest.", SIGNAL_AUDIT_FILENAME))
        frame: pd.DataFrame | None = None
        if (path := safe.get(SIGNAL_PARQUET_FILENAME)) is not None:
            try:
                frame = pd.read_parquet(path, engine="pyarrow")
            except Exception:
                issues.append(_issue("parquet_read_error", "Signal Parquet cannot be read.", SIGNAL_PARQUET_FILENAME))
        if frame is not None and manifest is not None:
            if tuple(frame.columns) != manifest.columns or not frame.columns.is_unique:
                issues.append(_issue("parquet_column_mismatch", "Parquet columns differ from manifest.", SIGNAL_PARQUET_FILENAME))
            if len(frame) != manifest.row_count:
                issues.append(_issue("parquet_row_count_mismatch", "Parquet row count differs from manifest.", SIGNAL_PARQUET_FILENAME))
            dtypes = tuple((str(name), str(dtype)) for name, dtype in frame.dtypes.items())
            if dtypes != manifest.pandas_dtypes:
                issues.append(_issue("parquet_dtype_mismatch", "Parquet dtypes differ from manifest.", SIGNAL_PARQUET_FILENAME))
            if tuple(frame.columns) == SIGNAL_OUTPUT_COLUMNS:
                for message in _signal_errors(frame, manifest.signal_direction):
                    issues.append(_issue("signal_content_error", message, SIGNAL_PARQUET_FILENAME))
                if audit is not None and not _signal_errors(frame, manifest.signal_direction):
                    dates = frame["trade_date"]
                    expected_audit = {
                        "output_rows": len(frame),
                        "trade_date_count": int(dates.nunique()),
                        "min_trade_date": pd.Timestamp(dates.min()).date().isoformat(),
                        "max_trade_date": pd.Timestamp(dates.max()).date().isoformat(),
                    }
                    if any(audit[name] != value for name, value in expected_audit.items()):
                        issues.append(_issue("audit_parquet_mismatch", "Audit differs from Signal payload.", SIGNAL_AUDIT_FILENAME))
        return self._report(directory, issues, manifest)

    @staticmethod
    def _report(
        directory: Path,
        issues: list[SignalArtifactValidationIssue],
        manifest: SignalArtifactManifest | None,
    ) -> SignalArtifactValidationReport:
        unique: list[SignalArtifactValidationIssue] = []
        seen: set[SignalArtifactValidationIssue] = set()
        for issue in issues:
            if issue not in seen:
                seen.add(issue)
                unique.append(issue)
        return SignalArtifactValidationReport(
            directory, not unique, tuple(unique), manifest
        )

    def write(
        self,
        result: SignalBuildResult,
        provenance: PredictionSourceProvenance,
        config: SignalArtifactConfig,
    ) -> SignalArtifactWriteResult:
        if not isinstance(result, SignalBuildResult):
            raise SignalArtifactWriteError("result must be a SignalBuildResult.")
        if not isinstance(provenance, PredictionSourceProvenance):
            raise SignalArtifactWriteError(
                "provenance must be PredictionSourceProvenance."
            )
        if not isinstance(config, SignalArtifactConfig):
            raise SignalArtifactWriteError("config must be SignalArtifactConfig.")
        target = _absolute(config.artifact_dir)
        parent = target.parent
        if target.exists() or target.is_symlink():
            raise SignalArtifactExistsError("target artifact_dir already exists.")
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise SignalArtifactWriteError("artifact parent is invalid.")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SignalArtifactWriteError("artifact parent cannot be created.") from exc
        if parent.is_symlink():
            raise SignalArtifactWriteError("artifact parent must not be a symlink.")
        staging = parent / f".tmp-{target.name}-{uuid4().hex}"
        published = False
        try:
            staging.mkdir()
            signals = result.signals
            business_config = {
                "prediction_column": result.audit.prediction_column,
                "signal_direction": result.audit.signal_direction,
            }
            errors = _signal_errors(signals, result.audit.signal_direction)
            if errors:
                raise SignalArtifactWriteError(
                    "Signal input is not canonical: " + " ".join(errors)
                )
            provenance_data = provenance.as_dict()
            audit = {
                "signal_schema_version": SIGNAL_SCHEMA_VERSION,
                "input_rows": result.audit.input_rows,
                "output_rows": result.audit.output_rows,
                "trade_date_count": result.audit.trade_date_count,
                "min_trade_date": result.audit.first_trade_date.strftime("%Y-%m-%d"),
                "max_trade_date": result.audit.last_trade_date.strftime("%Y-%m-%d"),
                **business_config,
                "score_finite": True,
                "duplicate_key_count": 0,
                "rank_integrity": True,
                "source_provenance": provenance_data,
                "warnings": [],
            }
            _config(business_config)
            _audit(audit)
            signal_path = staging / SIGNAL_PARQUET_FILENAME
            signals.to_parquet(
                signal_path, engine="pyarrow",
                compression=config.parquet_compression, index=False,
            )
            _write_json(staging / SIGNAL_CONFIG_FILENAME, business_config)
            _write_json(staging / SIGNAL_AUDIT_FILENAME, audit)
            persisted = pd.read_parquet(signal_path, engine="pyarrow")
            try:
                pdt.assert_frame_equal(
                    persisted, signals, check_like=False, check_dtype=False,
                    check_exact=False, rtol=1e-12, atol=1e-12,
                )
            except AssertionError as exc:
                raise SignalArtifactWriteError(
                    "Parquet roundtrip changed Signal semantics."
                ) from exc
            records = tuple(_record(staging / name) for name in _PAYLOAD_FILENAMES)
            manifest = SignalArtifactManifest(
                artifact_type=SIGNAL_ARTIFACT_TYPE,
                artifact_schema_version=SIGNAL_ARTIFACT_SCHEMA_VERSION,
                signal_schema_version=SIGNAL_SCHEMA_VERSION,
                created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                row_count=len(persisted),
                column_count=len(persisted.columns),
                columns=tuple(str(item) for item in persisted.columns),
                pandas_dtypes=tuple((str(name), str(dtype)) for name, dtype in persisted.dtypes.items()),
                prediction_column=result.audit.prediction_column,
                signal_direction=result.audit.signal_direction,
                source_provenance=provenance_data,
                files=records,
            )
            _write_json(staging / SIGNAL_MANIFEST_FILENAME, manifest.as_dict())
            pre = self.validate(staging)
            if not pre.is_valid:
                raise SignalArtifactWriteError("pre-publish validation failed.")
            if target.exists() or target.is_symlink():
                raise SignalArtifactExistsError("target appeared before publication.")
            os.replace(staging, target)
            published = True
            validation = self.validate(target) if config.verify_after_write else SignalArtifactValidationReport(target, True, (), manifest)
            if not validation.is_valid:
                try:
                    shutil.rmtree(target)
                except OSError as exc:
                    raise SignalArtifactWriteError(
                        "post-publish validation and cleanup failed."
                    ) from exc
                raise SignalArtifactWriteError("post-publish validation failed.")
            return SignalArtifactWriteResult(
                target,
                target / SIGNAL_PARQUET_FILENAME,
                target / SIGNAL_CONFIG_FILENAME,
                target / SIGNAL_AUDIT_FILENAME,
                target / SIGNAL_MANIFEST_FILENAME,
                len(persisted),
                SIGNAL_ARTIFACT_SCHEMA_VERSION,
                manifest,
                validation,
            )
        except SignalArtifactError:
            if staging.exists() and not published:
                shutil.rmtree(staging, ignore_errors=True)
            raise
        except Exception as exc:
            if staging.exists() and not published:
                shutil.rmtree(staging, ignore_errors=True)
            raise SignalArtifactWriteError("artifact write failed.") from exc
