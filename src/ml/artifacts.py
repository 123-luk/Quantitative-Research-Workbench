"""Safe, atomic, and verifiable persistence for V3 ML experiment results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyarrow
import scipy
import sklearn

from src.ml.orchestration import MLExperimentResult


ML_ARTIFACT_SCHEMA_VERSION = "1.0"

_JSON_MEDIA_TYPE = "application/json"
_PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
_EXPERIMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_SUFFIXES = {".pkl", ".pickle", ".joblib", ".bin", ".model"}
_BASE_PATHS = (
    "experiment_config.json",
    "experiment_audit.json",
    "environment.json",
    "dataset_audit.json",
    "walk_forward_plan.json",
    "training_audit.json",
    "predictions.parquet",
    "evaluation/evaluation_audit.json",
    "evaluation/regression_metrics.json",
    "evaluation/pearson_ic_summary.json",
    "evaluation/rank_ic_summary.json",
    "evaluation/date_metrics.parquet",
    "evaluation/fold_metrics.parquet",
)
_IMPORTANCE_PATHS = (
    "permutation_importance/importance_audit.json",
    "permutation_importance/feature_importance.parquet",
    "permutation_importance/fold_importance.parquet",
    "permutation_importance/repeat_importance.parquet",
)
_PREDICTION_COLUMNS = (
    "trade_date",
    "ts_code",
    "entry_trade_date",
    "exit_trade_date",
    "target",
    "prediction",
    "fold_id",
)


class MLArtifactError(Exception):
    """Base error for ML artifact persistence."""


class MLArtifactConfigError(MLArtifactError):
    """Raised when artifact configuration is invalid."""


class MLArtifactDataError(MLArtifactError):
    """Raised when an experiment result violates its public contract."""


class MLArtifactWriteError(MLArtifactError):
    """Raised when artifact files cannot be safely written."""


class MLArtifactExistsError(MLArtifactWriteError):
    """Raised when a target experiment path already exists."""


class MLArtifactValidationError(MLArtifactError):
    """Raised when an artifact directory or file is invalid."""


class MLArtifactIntegrityError(MLArtifactValidationError):
    """Raised when valid files disagree across artifact boundaries."""


def _strict_fields(
    values: Mapping[str, object], expected: set[str], context: str
) -> dict[str, object]:
    if not isinstance(values, Mapping):
        raise MLArtifactConfigError(f"{context} must be a Mapping")
    if any(not isinstance(key, str) for key in values):
        raise MLArtifactConfigError(
            f"{context} field names must be strings")
    unknown = set(values) - expected
    if unknown:
        raise MLArtifactConfigError(
            f"{context} contains unknown field(s): {sorted(unknown)!r}"
        )
    return dict(values)


@dataclass(frozen=True)
class MLArtifactConfig:
    """Filesystem destination and Parquet settings for one experiment."""

    artifact_root: Path
    experiment_id: str
    parquet_compression: str = "zstd"

    def __post_init__(self) -> None:
        root = self.artifact_root
        if isinstance(root, str):
            if not root.strip():
                raise MLArtifactConfigError("artifact_root must not be empty")
            root = Path(root)
        elif not isinstance(root, Path):
            raise MLArtifactConfigError("artifact_root must be str or Path")
        if not str(root).strip():
            raise MLArtifactConfigError("artifact_root must not be empty")
        object.__setattr__(self, "artifact_root", root)

        experiment_id = self.experiment_id
        if (
            not isinstance(experiment_id, str)
            or not experiment_id
            or experiment_id != experiment_id.strip()
            or experiment_id in {".", ".."}
            or not _EXPERIMENT_ID_RE.fullmatch(experiment_id)
        ):
            raise MLArtifactConfigError(
                "experiment_id must be 1-128 characters, start with an "
                "alphanumeric character, and contain only letters, digits, "
                "dot, hyphen, or underscore"
            )
        compression = self.parquet_compression
        if not isinstance(compression, str):
            raise MLArtifactConfigError("parquet_compression must be a string")
        compression = compression.strip().lower()
        if compression not in {"zstd", "snappy", "none"}:
            raise MLArtifactConfigError(
                "parquet_compression must be zstd, snappy, or none"
            )
        object.__setattr__(self, "parquet_compression", compression)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> MLArtifactConfig:
        data = _strict_fields(
            values,
            {"artifact_root", "experiment_id", "parquet_compression"},
            "artifact config",
        )
        if "artifact_root" not in data or "experiment_id" not in data:
            raise MLArtifactConfigError(
                "artifact config requires artifact_root and experiment_id"
            )
        return cls(
            artifact_root=data["artifact_root"],  # type: ignore[arg-type]
            experiment_id=data["experiment_id"],  # type: ignore[arg-type]
            parquet_compression=data.get("parquet_compression", "zstd"),  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_root": str(self.artifact_root),
            "experiment_id": self.experiment_id,
            "parquet_compression": self.parquet_compression,
        }


def _to_json_safe(value: object, _active: set[int] | None = None) -> object:
    """Convert only explicitly supported values to strict JSON-safe values."""
    active = set() if _active is None else _active
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MLArtifactDataError("JSON value contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise MLArtifactDataError("JSON value contains NaT")
        return value.isoformat()
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        raise MLArtifactDataError(
            f"JSON value cannot contain {type(value).__name__}"
        )
    if callable(value):
        raise MLArtifactDataError("JSON value cannot contain a callable")
    marker = id(value)
    if isinstance(value, Mapping):
        if marker in active:
            raise MLArtifactDataError("JSON value contains a circular reference")
        active.add(marker)
        try:
            result: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise MLArtifactDataError("JSON mapping keys must be strings")
                result[key] = _to_json_safe(item, active)
            return result
        finally:
            active.remove(marker)
    if isinstance(value, (list, tuple)):
        if marker in active:
            raise MLArtifactDataError("JSON value contains a circular reference")
        active.add(marker)
        try:
            return [_to_json_safe(item, active) for item in value]
        finally:
            active.remove(marker)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        if marker in active:
            raise MLArtifactDataError("JSON value contains a circular reference")
        active.add(marker)
        try:
            converted = as_dict()
            if not isinstance(converted, Mapping):
                raise MLArtifactDataError("as_dict() must return a Mapping")
            return _to_json_safe(converted, active)
        finally:
            active.remove(marker)
    raise MLArtifactDataError(
        f"unsupported JSON value type: {type(value).__name__}"
    )


def _raise_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _read_json(path: Path, relative_path: str) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text, parse_constant=_raise_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise MLArtifactValidationError(
            f"{relative_path}: strict JSON parsing failed"
        ) from exc
    if not isinstance(value, dict):
        raise MLArtifactValidationError(
            f"{relative_path}: JSON top level must be an object"
        )
    return value


def _write_json(path: Path, value: object, relative_path: str) -> None:
    try:
        safe = _to_json_safe(value)
        if not isinstance(safe, dict):
            raise MLArtifactDataError(
                f"{relative_path}: JSON top level must be an object"
            )
        text = json.dumps(
            safe,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
        if _read_json(path, relative_path) != safe:
            raise MLArtifactValidationError(
                f"{relative_path}: JSON round-trip mismatch"
            )
    except MLArtifactError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise MLArtifactWriteError(f"{relative_path}: JSON write failed") from exc


def _validate_relative_path(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or Path(value).is_absolute()
        or ".." in PurePosixPath(value).parts
        or value.startswith("/")
    ):
        raise MLArtifactValidationError(
            f"{value!r}: artifact path must be a safe POSIX relative path"
        )


@dataclass(frozen=True)
class ArtifactFileRecord:
    relative_path: str
    artifact_type: str
    media_type: str
    size_bytes: int
    sha256: str
    row_count: int | None = None
    columns: tuple[str, ...] = ()
    dtypes: tuple[tuple[str, str], ...] = ()
    index_stored: bool = False
    index_name: str | None = None

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path)
        if self.relative_path == "experiment_manifest.json":
            raise MLArtifactValidationError(
                "experiment_manifest.json: manifest cannot record itself"
            )
        if self.artifact_type not in {"json", "parquet"}:
            raise MLArtifactValidationError(
                f"{self.relative_path}: invalid artifact_type"
            )
        expected_media = (
            _JSON_MEDIA_TYPE
            if self.artifact_type == "json"
            else _PARQUET_MEDIA_TYPE
        )
        if self.media_type != expected_media:
            raise MLArtifactValidationError(
                f"{self.relative_path}: media_type is inconsistent"
            )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise MLArtifactValidationError(
                f"{self.relative_path}: size_bytes must be positive"
            )
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(
            self.sha256
        ):
            raise MLArtifactValidationError(
                f"{self.relative_path}: sha256 is invalid"
            )
        try:
            columns = tuple(self.columns)
            dtypes = tuple(tuple(item) for item in self.dtypes)
            if any(len(item) != 2 for item in dtypes):
                raise ValueError("dtype records require name and dtype")
        except (TypeError, ValueError) as exc:
            raise MLArtifactValidationError(
                f"{self.relative_path}: table metadata is invalid"
            ) from exc
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "dtypes", dtypes)
        if not isinstance(self.index_stored, bool):
            raise MLArtifactValidationError(
                f"{self.relative_path}: index_stored must be boolean")
        if self.artifact_type == "json":
            if (
                self.row_count is not None
                or columns
                or dtypes
                or self.index_stored
                or self.index_name is not None
            ):
                raise MLArtifactValidationError(
                    f"{self.relative_path}: JSON table metadata must be empty"
                )
        else:
            if (
                isinstance(self.row_count, bool)
                or not isinstance(self.row_count, int)
                or self.row_count < 0
                or any(not isinstance(column, str) for column in columns)
                or tuple(name for name, _ in dtypes) != columns
                or any(
                    not isinstance(dtype, str) for _, dtype in dtypes
                )
                or (not self.index_stored and self.index_name is not None)
            ):
                raise MLArtifactValidationError(
                    f"{self.relative_path}: Parquet table metadata is invalid"
                )

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "artifact_type": self.artifact_type,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "columns": list(self.columns),
            "dtypes": [list(item) for item in self.dtypes],
            "index_stored": self.index_stored,
            "index_name": self.index_name,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> ArtifactFileRecord:
        expected = {
            "relative_path",
            "artifact_type",
            "media_type",
            "size_bytes",
            "sha256",
            "row_count",
            "columns",
            "dtypes",
            "index_stored",
            "index_name",
        }
        if not isinstance(values, Mapping) or set(values) != expected:
            raise MLArtifactValidationError(
                "artifact file record fields are invalid"
            )
        try:
            return cls(
                relative_path=values["relative_path"],  # type: ignore[arg-type]
                artifact_type=values["artifact_type"],  # type: ignore[arg-type]
                media_type=values["media_type"],  # type: ignore[arg-type]
                size_bytes=values["size_bytes"],  # type: ignore[arg-type]
                sha256=values["sha256"],  # type: ignore[arg-type]
                row_count=values["row_count"],  # type: ignore[arg-type]
                columns=tuple(values["columns"]),  # type: ignore[arg-type]
                dtypes=tuple(tuple(item) for item in values["dtypes"]),  # type: ignore[arg-type]
                index_stored=values["index_stored"],  # type: ignore[arg-type]
                index_name=values["index_name"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MLArtifactValidationError(
                "artifact file record cannot be parsed"
            ) from exc


@dataclass(frozen=True)
class MLArtifactManifest:
    schema_version: str
    experiment_id: str
    model_name: str
    permutation_importance_enabled: bool
    artifact_count: int
    artifacts: tuple[ArtifactFileRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ML_ARTIFACT_SCHEMA_VERSION:
            raise MLArtifactValidationError(
                f"unsupported artifact schema version {self.schema_version!r}"
            )
        if (
            not isinstance(self.experiment_id, str)
            or not _EXPERIMENT_ID_RE.fullmatch(self.experiment_id)
            or not isinstance(self.model_name, str)
            or not self.model_name
            or not isinstance(self.permutation_importance_enabled, bool)
        ):
            raise MLArtifactValidationError("manifest identity fields are invalid")
        artifacts = tuple(self.artifacts)
        if (
            isinstance(self.artifact_count, bool)
            or not isinstance(self.artifact_count, int)
            or self.artifact_count != len(artifacts)
            or any(not isinstance(item, ArtifactFileRecord) for item in artifacts)
        ):
            raise MLArtifactValidationError(
                "manifest artifact_count or records are invalid"
            )
        paths = tuple(item.relative_path for item in artifacts)
        if len(paths) != len(set(paths)):
            raise MLArtifactValidationError(
                "manifest contains duplicate relative paths"
            )
        object.__setattr__(self, "artifacts", artifacts)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "permutation_importance_enabled": (
                self.permutation_importance_enabled
            ),
            "artifact_count": self.artifact_count,
            "artifacts": [record.as_dict() for record in self.artifacts],
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> MLArtifactManifest:
        expected = {
            "schema_version",
            "experiment_id",
            "model_name",
            "permutation_importance_enabled",
            "artifact_count",
            "artifacts",
        }
        if not isinstance(values, Mapping) or set(values) != expected:
            raise MLArtifactValidationError("manifest fields are invalid")
        try:
            raw_artifacts = values["artifacts"]
            if not isinstance(raw_artifacts, list):
                raise TypeError("artifacts must be a list")
            return cls(
                schema_version=values["schema_version"],  # type: ignore[arg-type]
                experiment_id=values["experiment_id"],  # type: ignore[arg-type]
                model_name=values["model_name"],  # type: ignore[arg-type]
                permutation_importance_enabled=values[
                    "permutation_importance_enabled"
                ],  # type: ignore[arg-type]
                artifact_count=values["artifact_count"],  # type: ignore[arg-type]
                artifacts=tuple(
                    ArtifactFileRecord.from_dict(item)
                    for item in raw_artifacts
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MLArtifactValidationError("manifest cannot be parsed") from exc


@dataclass(frozen=True)
class MLArtifactValidationReport:
    schema_version: str
    experiment_id: str
    artifact_count: int
    validated_artifacts: tuple[str, ...]
    checksums_verified: bool
    sizes_verified: bool
    json_verified: bool
    parquet_verified: bool
    cross_file_integrity_verified: bool

    def __post_init__(self) -> None:
        checks = (
            self.checksums_verified,
            self.sizes_verified,
            self.json_verified,
            self.parquet_verified,
            self.cross_file_integrity_verified,
        )
        paths = tuple(self.validated_artifacts)
        if (
            self.schema_version != ML_ARTIFACT_SCHEMA_VERSION
            or not isinstance(self.experiment_id, str)
            or not _EXPERIMENT_ID_RE.fullmatch(self.experiment_id)
            or isinstance(self.artifact_count, bool)
            or not isinstance(self.artifact_count, int)
            or self.artifact_count != len(paths)
            or len(paths) != len(set(paths))
        ):
            raise MLArtifactValidationError(
                "validation report identity or artifact fields are invalid"
            )
        for path in paths:
            _validate_relative_path(path)
        object.__setattr__(self, "validated_artifacts", paths)
        if not all(value is True for value in checks):
            raise MLArtifactValidationError(
                "successful validation report checks must all be true"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "artifact_count": self.artifact_count,
            "validated_artifacts": list(self.validated_artifacts),
            "checksums_verified": self.checksums_verified,
            "sizes_verified": self.sizes_verified,
            "json_verified": self.json_verified,
            "parquet_verified": self.parquet_verified,
            "cross_file_integrity_verified": (
                self.cross_file_integrity_verified
            ),
        }


@dataclass(frozen=True)
class MLArtifactWriteResult:
    experiment_dir: Path
    manifest: MLArtifactManifest
    validation_report: MLArtifactValidationReport

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_dir", self.experiment_dir.resolve())
        if not isinstance(self.manifest, MLArtifactManifest) or not isinstance(
            self.validation_report, MLArtifactValidationReport
        ):
            raise MLArtifactValidationError("write result contracts are invalid")
        if (
            self.manifest.experiment_id
            != self.validation_report.experiment_id
            or self.manifest.artifact_count
            != self.validation_report.artifact_count
            or self.experiment_dir.name != self.manifest.experiment_id
        ):
            raise MLArtifactValidationError(
                "write result directory, manifest, and report disagree"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "experiment_dir": str(self.experiment_dir),
            "manifest": self.manifest.as_dict(),
            "validation_report": self.validation_report.as_dict(),
        }


def _sha256(path: Path, relative_path: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise MLArtifactWriteError(
            f"{relative_path}: SHA-256 read failed"
        ) from exc
    return digest.hexdigest()


def _record_for_json(root: Path, relative_path: str) -> ArtifactFileRecord:
    path = root / PurePosixPath(relative_path)
    return ArtifactFileRecord(
        relative_path=relative_path,
        artifact_type="json",
        media_type=_JSON_MEDIA_TYPE,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path, relative_path),
    )


def _write_parquet(
    root: Path,
    relative_path: str,
    frame: pd.DataFrame,
    *,
    compression: str,
    index: bool,
) -> ArtifactFileRecord:
    if not isinstance(frame, pd.DataFrame):
        raise MLArtifactDataError(f"{relative_path}: expected a DataFrame")
    original = frame.copy(deep=True)
    if original.empty:
        raise MLArtifactDataError(f"{relative_path}: table must not be empty")
    if index:
        if original.index.name != "dataset_index":
            original.index = original.index.copy()
            original.index.name = "dataset_index"
    path = root / PurePosixPath(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        original.to_parquet(
            path,
            engine="pyarrow",
            compression=None if compression == "none" else compression,
            index=index,
        )
    except Exception as exc:
        raise MLArtifactWriteError(
            f"{relative_path}: Parquet write failed"
        ) from exc
    try:
        restored = pd.read_parquet(path, engine="pyarrow")
        pdt.assert_frame_equal(original, restored)
    except Exception as exc:
        raise MLArtifactValidationError(
            f"{relative_path}: Parquet read verification failed"
        ) from exc
    return ArtifactFileRecord(
        relative_path=relative_path,
        artifact_type="parquet",
        media_type=_PARQUET_MEDIA_TYPE,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path, relative_path),
        row_count=len(restored),
        columns=tuple(restored.columns),
        dtypes=tuple((name, str(dtype)) for name, dtype in restored.dtypes.items()),
        index_stored=index,
        index_name=restored.index.name if index else None,
    )


def _plan_as_dict(plan: object) -> dict[str, object]:
    required = (
        "config",
        "splits",
        "all_score_dates",
        "skipped_initial_prediction_dates",
        "first_prediction_date",
        "last_prediction_date",
    )
    if any(not hasattr(plan, name) for name in required):
        raise MLArtifactDataError("walk_forward_plan public fields are incomplete")
    splits: list[dict[str, object]] = []
    split_fields = (
        "retrain_id",
        "train_indices",
        "validation_indices",
        "prediction_indices",
        "train_dates",
        "validation_dates",
        "prediction_dates",
        "train_start_date",
        "train_end_date",
        "validation_start_date",
        "validation_end_date",
        "prediction_start_date",
        "prediction_end_date",
        "n_train_rows",
        "n_validation_rows",
        "n_prediction_rows",
        "max_train_exit_date",
        "max_validation_exit_date",
        "embargo_dates",
        "train_validation_purged_dates",
        "label_unavailable_dates",
    )
    for split in plan.splits:
        if any(not hasattr(split, name) for name in split_fields):
            raise MLArtifactDataError("walk_forward split fields are incomplete")
        splits.append(
            {
                name: _to_json_safe(getattr(split, name))
                for name in split_fields
            }
        )
    return {
        "config": _to_json_safe(plan.config.as_dict()),
        "splits": splits,
        "all_score_dates": _to_json_safe(plan.all_score_dates),
        "skipped_initial_prediction_dates": _to_json_safe(
            plan.skipped_initial_prediction_dates
        ),
        "first_prediction_date": _to_json_safe(plan.first_prediction_date),
        "last_prediction_date": _to_json_safe(plan.last_prediction_date),
    }


def _environment() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "pyarrow_version": pyarrow.__version__,
        "artifact_schema_version": ML_ARTIFACT_SCHEMA_VERSION,
    }


def _result_payloads(result: MLExperimentResult) -> tuple[
    dict[str, object], dict[str, pd.DataFrame]
]:
    audit = result.audit
    training = result.training_result
    evaluation = result.evaluation_result
    importance = result.permutation_importance_result
    json_values: dict[str, object] = {
        "experiment_config.json": audit.config.as_dict(),
        "experiment_audit.json": audit.as_dict(),
        "environment.json": _environment(),
        "dataset_audit.json": result.dataset_audit.as_dict(),
        "walk_forward_plan.json": _plan_as_dict(result.walk_forward_plan),
        "training_audit.json": training.audit.as_dict(),
        "evaluation/evaluation_audit.json": evaluation.audit.as_dict(),
        "evaluation/regression_metrics.json": (
            evaluation.regression_metrics.as_dict()
        ),
        "evaluation/pearson_ic_summary.json": (
            evaluation.pearson_ic_summary.as_dict()
        ),
        "evaluation/rank_ic_summary.json": evaluation.rank_ic_summary.as_dict(),
    }
    tables = {
        "predictions.parquet": training.predictions,
        "evaluation/date_metrics.parquet": evaluation.date_metrics,
        "evaluation/fold_metrics.parquet": evaluation.fold_metrics,
    }
    if importance is not None:
        json_values["permutation_importance/importance_audit.json"] = (
            importance.audit.as_dict()
        )
        tables.update(
            {
                "permutation_importance/feature_importance.parquet": (
                    importance.feature_importance
                ),
                "permutation_importance/fold_importance.parquet": (
                    importance.fold_importance
                ),
                "permutation_importance/repeat_importance.parquet": (
                    importance.repeat_importance
                ),
            }
        )
    return json_values, tables


def _validate_result(result: object) -> MLExperimentResult:
    if not isinstance(result, MLExperimentResult):
        raise MLArtifactDataError("result must be MLExperimentResult")
    audit = result.audit
    training = result.training_result
    evaluation = result.evaluation_result
    importance = result.permutation_importance_result
    predictions = training.predictions
    if predictions.empty or tuple(predictions.columns) != _PREDICTION_COLUMNS:
        raise MLArtifactDataError("predictions.parquet: prediction table is invalid")
    train_audit = training.audit
    eval_audit = evaluation.audit
    if (
        not audit.evaluation_completed
        or len(predictions) != train_audit.n_prediction_rows
        or eval_audit.n_rows != train_audit.n_prediction_rows
        or eval_audit.n_dates != train_audit.n_prediction_dates
        or eval_audit.row_coverage != 1.0
        or eval_audit.date_coverage != 1.0
        or audit.model_name != train_audit.model_name
        or audit.resolved_model_parameters
        != train_audit.resolved_model_parameters
        or audit.n_folds != train_audit.n_folds
        or audit.n_prediction_rows != train_audit.n_prediction_rows
        or audit.n_prediction_dates != train_audit.n_prediction_dates
    ):
        raise MLArtifactDataError("experiment result audits are inconsistent")
    for fold in train_audit.fold_audits:
        if tuple(fold.model_fit_audit.feature_names) != audit.feature_names:
            raise MLArtifactDataError("training feature names are inconsistent")
    if audit.permutation_importance_enabled != (importance is not None):
        raise MLArtifactDataError("permutation importance state is inconsistent")
    if importance is not None:
        imp = importance.audit
        if (
            not audit.permutation_importance_completed
            or imp.model_name != train_audit.model_name
            or imp.resolved_model_parameters
            != train_audit.resolved_model_parameters
            or imp.n_folds != audit.n_folds
            or imp.n_features != audit.n_features
            or imp.first_prediction_date != audit.first_prediction_date
            or imp.last_prediction_date != audit.last_prediction_date
        ):
            raise MLArtifactDataError("permutation importance audit is inconsistent")
    for value in (
        audit,
        result.dataset_audit,
        train_audit,
        eval_audit,
    ):
        _to_json_safe(value.as_dict())
    return result


def _same_json(left: object, right: object) -> bool:
    return _to_json_safe(left) == _to_json_safe(right)


class MLExperimentArtifactStore:
    """Write and validate fixed-schema experiment artifact directories."""

    def write(
        self, result: MLExperimentResult, config: MLArtifactConfig
    ) -> MLArtifactWriteResult:
        if not isinstance(config, MLArtifactConfig):
            raise MLArtifactConfigError("config must be MLArtifactConfig")
        valid_result = _validate_result(result)
        try:
            root = config.artifact_root.resolve()
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MLArtifactWriteError("artifact_root: cannot create directory") from exc
        target = root / config.experiment_id
        if target.exists() or target.is_symlink():
            raise MLArtifactExistsError(
                f"{config.experiment_id}: target already exists"
            )
        staging = root / f".{config.experiment_id}.tmp-{uuid4().hex}"
        created = False
        renamed = False
        try:
            staging.mkdir()
            created = True
            json_values, tables = _result_payloads(valid_result)
            records_by_path: dict[str, ArtifactFileRecord] = {}
            for relative_path in _BASE_PATHS + (
                _IMPORTANCE_PATHS
                if valid_result.permutation_importance_result is not None
                else ()
            ):
                if relative_path in json_values:
                    path = staging / PurePosixPath(relative_path)
                    _write_json(path, json_values[relative_path], relative_path)
                    records_by_path[relative_path] = _record_for_json(
                        staging, relative_path
                    )
                else:
                    records_by_path[relative_path] = _write_parquet(
                        staging,
                        relative_path,
                        tables[relative_path],
                        compression=config.parquet_compression,
                        index=relative_path == "predictions.parquet",
                    )
            ordered_paths = _BASE_PATHS + (
                _IMPORTANCE_PATHS
                if valid_result.permutation_importance_result is not None
                else ()
            )
            manifest = MLArtifactManifest(
                schema_version=ML_ARTIFACT_SCHEMA_VERSION,
                experiment_id=config.experiment_id,
                model_name=valid_result.audit.model_name,
                permutation_importance_enabled=(
                    valid_result.permutation_importance_result is not None
                ),
                artifact_count=len(ordered_paths),
                artifacts=tuple(records_by_path[path] for path in ordered_paths),
            )
            _write_json(
                staging / "experiment_manifest.json",
                manifest.as_dict(),
                "experiment_manifest.json",
            )
            self._validate(staging, expected_experiment_id=config.experiment_id)
            if target.exists() or target.is_symlink():
                raise MLArtifactExistsError(
                    f"{config.experiment_id}: target appeared before rename"
                )
            os.replace(staging, target)
            renamed = True
            report = self.validate(target)
            return MLArtifactWriteResult(target, manifest, report)
        except MLArtifactError as exc:
            if created and not renamed and staging.exists():
                try:
                    shutil.rmtree(staging)
                except OSError:
                    raise MLArtifactWriteError(
                        f"{exc}; staging cleanup failed"
                    ) from exc
            raise
        except OSError as exc:
            error = MLArtifactWriteError(
                f"{config.experiment_id}: artifact write or rename failed"
            )
            if created and not renamed and staging.exists():
                try:
                    shutil.rmtree(staging)
                except OSError:
                    raise MLArtifactWriteError(
                        f"{error}; staging cleanup failed"
                    ) from exc
            raise error from exc
        except Exception as exc:
            error = MLArtifactWriteError(
                f"{config.experiment_id}: artifact write failed"
            )
            if created and not renamed and staging.exists():
                try:
                    shutil.rmtree(staging)
                except OSError:
                    raise MLArtifactWriteError(
                        f"{error}; staging cleanup failed"
                    ) from exc
            raise error from exc

    def read_manifest(self, experiment_dir: str | Path) -> MLArtifactManifest:
        directory = Path(experiment_dir)
        if (
            not directory.exists()
            or not directory.is_dir()
            or directory.is_symlink()
        ):
            raise MLArtifactValidationError(
                "experiment_manifest.json: experiment directory is invalid"
            )
        manifest_path = directory / "experiment_manifest.json"
        if (
            not manifest_path.exists()
            or not manifest_path.is_file()
            or manifest_path.is_symlink()
        ):
            raise MLArtifactValidationError(
                "experiment_manifest.json: missing or invalid manifest"
            )
        return MLArtifactManifest.from_dict(
            _read_json(manifest_path, "experiment_manifest.json")
        )

    def validate(
        self, experiment_dir: str | Path
    ) -> MLArtifactValidationReport:
        directory = Path(experiment_dir)
        return self._validate(directory, expected_experiment_id=directory.name)

    def _validate(
        self, directory: Path, *, expected_experiment_id: str
    ) -> MLArtifactValidationReport:
        manifest = self.read_manifest(directory)
        if manifest.experiment_id != expected_experiment_id:
            raise MLArtifactValidationError(
                "experiment_manifest.json: experiment_id differs from directory"
            )
        expected = _BASE_PATHS + (
            _IMPORTANCE_PATHS
            if manifest.permutation_importance_enabled
            else ()
        )
        paths = tuple(record.relative_path for record in manifest.artifacts)
        if paths != expected or manifest.artifact_count != len(expected):
            raise MLArtifactValidationError(
                "experiment_manifest.json: artifact set or order is invalid"
            )
        allowed_files = set(expected) | {"experiment_manifest.json"}
        actual_files: set[str] = set()
        allowed_dirs = {"evaluation"}
        if manifest.permutation_importance_enabled:
            allowed_dirs.add("permutation_importance")
        for path in directory.rglob("*"):
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink():
                raise MLArtifactValidationError(
                    f"{relative}: symbolic links are forbidden"
                )
            if path.is_dir():
                if relative not in allowed_dirs:
                    raise MLArtifactValidationError(
                        f"{relative}: unexpected artifact subdirectory"
                    )
            else:
                actual_files.add(relative)
                if path.suffix.lower() in _FORBIDDEN_SUFFIXES:
                    raise MLArtifactValidationError(
                        f"{relative}: model persistence file is forbidden"
                    )
        if actual_files != allowed_files:
            raise MLArtifactValidationError(
                "experiment_manifest.json: missing or extra artifacts"
            )
        json_values: dict[str, dict[str, object]] = {}
        tables: dict[str, pd.DataFrame] = {}
        for record in manifest.artifacts:
            path = directory / PurePosixPath(record.relative_path)
            if not path.is_file() or path.is_symlink():
                raise MLArtifactValidationError(
                    f"{record.relative_path}: missing or invalid artifact"
                )
            if path.stat().st_size != record.size_bytes:
                raise MLArtifactValidationError(
                    f"{record.relative_path}: file size mismatch"
                )
            try:
                digest = _sha256(path, record.relative_path)
            except MLArtifactWriteError as exc:
                raise MLArtifactValidationError(
                    f"{record.relative_path}: checksum read failed"
                ) from exc
            if digest != record.sha256:
                raise MLArtifactValidationError(
                    f"{record.relative_path}: SHA-256 mismatch"
                )
            if record.artifact_type == "json":
                json_values[record.relative_path] = _read_json(
                    path, record.relative_path
                )
            else:
                try:
                    frame = pd.read_parquet(path, engine="pyarrow")
                except Exception as exc:
                    raise MLArtifactValidationError(
                        f"{record.relative_path}: Parquet read failed"
                    ) from exc
                actual_dtypes = tuple(
                    (name, str(dtype)) for name, dtype in frame.dtypes.items()
                )
                expected_index = (
                    record.relative_path == "predictions.parquet"
                )
                range_index_ok = (
                    isinstance(frame.index, pd.RangeIndex)
                    and frame.index.start == 0
                    and frame.index.step == 1
                )
                if (
                    len(frame) != record.row_count
                    or tuple(frame.columns) != record.columns
                    or actual_dtypes != record.dtypes
                    or record.index_stored is not expected_index
                    or frame.index.name != record.index_name
                    or (
                        not expected_index
                        and (record.index_name is not None or not range_index_ok)
                    )
                ):
                    raise MLArtifactValidationError(
                        f"{record.relative_path}: Parquet schema mismatch"
                    )
                tables[record.relative_path] = frame
        self._validate_cross_file(manifest, json_values, tables)
        return MLArtifactValidationReport(
            schema_version=manifest.schema_version,
            experiment_id=manifest.experiment_id,
            artifact_count=manifest.artifact_count,
            validated_artifacts=paths,
            checksums_verified=True,
            sizes_verified=True,
            json_verified=True,
            parquet_verified=True,
            cross_file_integrity_verified=True,
        )

    @staticmethod
    def _validate_cross_file(
        manifest: MLArtifactManifest,
        values: dict[str, dict[str, object]],
        tables: dict[str, pd.DataFrame],
    ) -> None:
        experiment = values["experiment_audit.json"]
        config = values["experiment_config.json"]
        training = values["training_audit.json"]
        evaluation = values["evaluation/evaluation_audit.json"]
        regression = values["evaluation/regression_metrics.json"]
        predictions = tables["predictions.parquet"]
        date_metrics = tables["evaluation/date_metrics.parquet"]
        fold_metrics = tables["evaluation/fold_metrics.parquet"]
        try:
            dates = pd.to_datetime(predictions["trade_date"])
            fold_ids = set(int(value) for value in predictions["fold_id"].unique())
            if (
                not _same_json(experiment["config"], config)
                or experiment["model_name"] != manifest.model_name
                or experiment["permutation_importance_enabled"]
                is not manifest.permutation_importance_enabled
                or tuple(predictions.columns) != _PREDICTION_COLUMNS
                or predictions.index.name != "dataset_index"
                or len(predictions) != training["n_prediction_rows"]
                or dates.nunique() != training["n_prediction_dates"]
                or dates.min().strftime("%Y-%m-%d")
                != training["first_prediction_date"]
                or dates.max().strftime("%Y-%m-%d")
                != training["last_prediction_date"]
                or len(fold_ids) != training["n_folds"]
                or not np.isfinite(predictions["target"]).all()
                or not np.isfinite(predictions["prediction"]).all()
                or evaluation["n_rows"] != len(predictions)
                or evaluation["n_dates"] != dates.nunique()
                or evaluation["n_folds"] != len(fold_ids)
                or len(date_metrics) != evaluation["n_dates"]
                or len(fold_metrics) != evaluation["n_folds"]
                or pd.to_datetime(date_metrics["trade_date"]).min()
                != dates.min()
                or pd.to_datetime(date_metrics["trade_date"]).max()
                != dates.max()
                or set(int(value) for value in fold_metrics["fold_id"])
                != fold_ids
                or regression["n_obs"] != len(predictions)
                or evaluation["row_coverage"] != 1.0
                or evaluation["date_coverage"] != 1.0
            ):
                raise MLArtifactIntegrityError(
                    "artifact files disagree on experiment/training/evaluation"
                )
            if manifest.permutation_importance_enabled:
                importance = values[
                    "permutation_importance/importance_audit.json"
                ]
                feature = tables[
                    "permutation_importance/feature_importance.parquet"
                ]
                fold = tables[
                    "permutation_importance/fold_importance.parquet"
                ]
                repeat = tables[
                    "permutation_importance/repeat_importance.parquet"
                ]
                if (
                    importance["n_folds"] != training["n_folds"]
                    or importance["n_features"] != experiment["n_features"]
                    or len(feature) != importance["n_features"]
                    or len(fold)
                    != importance["n_folds"] * importance["n_features"]
                    or len(repeat) != importance["n_repeat_evaluations"]
                    or importance["model_name"] != training["model_name"]
                    or not _same_json(
                        importance["resolved_model_parameters"],
                        training["resolved_model_parameters"],
                    )
                    or importance["first_prediction_date"]
                    != training["first_prediction_date"]
                    or importance["last_prediction_date"]
                    != training["last_prediction_date"]
                ):
                    raise MLArtifactIntegrityError(
                        "permutation importance artifacts are inconsistent"
                    )
        except MLArtifactIntegrityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise MLArtifactIntegrityError(
                "artifact files have incomplete integrity fields"
            ) from exc
