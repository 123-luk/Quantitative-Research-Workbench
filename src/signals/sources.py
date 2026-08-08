"""Load predictions only from one explicit, validated native ML Artifact."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml import MLArtifactError, MLExperimentArtifactStore


NATIVE_ML_PREDICTIONS_FILENAME = "predictions.parquet"
_NATIVE_ML_PREDICTION_COLUMNS = (
    "trade_date",
    "ts_code",
    "entry_trade_date",
    "exit_trade_date",
    "target",
    "prediction",
    "fold_id",
)


class PredictionSourceError(ValueError):
    """Raised when an explicit native ML Artifact cannot supply predictions."""


@dataclass(frozen=True)
class PredictionSourceProvenance:
    """Immutable native Artifact identity for later audit and persistence."""

    artifact_dir: Path
    prediction_path: Path
    artifact_schema_version: str
    experiment_id: str
    model_name: str
    prediction_sha256: str

    def __post_init__(self) -> None:
        artifact_dir = self.artifact_dir.resolve()
        prediction_path = self.prediction_path.resolve()
        if (
            not artifact_dir.is_dir()
            or artifact_dir.is_symlink()
            or not prediction_path.is_file()
            or prediction_path.is_symlink()
            or prediction_path.parent != artifact_dir
            or prediction_path.name != NATIVE_ML_PREDICTIONS_FILENAME
        ):
            raise PredictionSourceError("prediction provenance paths are invalid.")
        for field_name in (
            "artifact_schema_version",
            "experiment_id",
            "model_name",
            "prediction_sha256",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise PredictionSourceError(
                    f"prediction provenance {field_name} is invalid."
                )
        if len(self.prediction_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.prediction_sha256
        ):
            raise PredictionSourceError(
                "prediction provenance prediction_sha256 is invalid."
            )
        object.__setattr__(self, "artifact_dir", artifact_dir)
        object.__setattr__(self, "prediction_path", prediction_path)

    def as_dict(self) -> dict[str, str]:
        """Return a detached JSON-safe provenance mapping."""
        return {
            "artifact_dir": str(self.artifact_dir),
            "prediction_path": str(self.prediction_path),
            "artifact_schema_version": self.artifact_schema_version,
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "prediction_sha256": self.prediction_sha256,
        }


class PredictionSourceResult:
    """Defensively expose validated predictions and immutable provenance."""

    __slots__ = ("_predictions", "_provenance")

    def __init__(
        self,
        predictions: pd.DataFrame,
        provenance: PredictionSourceProvenance,
    ) -> None:
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            raise PredictionSourceError("validated predictions must be non-empty.")
        if not isinstance(provenance, PredictionSourceProvenance):
            raise PredictionSourceError("prediction provenance is invalid.")
        self._predictions = predictions.copy(deep=True)
        self._provenance = provenance

    @property
    def predictions(self) -> pd.DataFrame:
        """Return a deep defensive copy of the validated native predictions."""
        return self._predictions.copy(deep=True)

    @property
    def provenance(self) -> PredictionSourceProvenance:
        """Return immutable native Artifact provenance."""
        return self._provenance


class PredictionSourceAdapter:
    """Validate one explicit native ML Artifact and load its fixed prediction file."""

    def load_native_ml_artifact(
        self, artifact_dir: str | os.PathLike[str]
    ) -> PredictionSourceResult:
        """Return validated native predictions without searching or fallback."""
        directory = self._explicit_directory(artifact_dir)
        store = MLExperimentArtifactStore()
        try:
            report = store.validate(directory)
            manifest = store.read_manifest(directory)
        except MLArtifactError as exc:
            raise PredictionSourceError(
                "native ML Artifact validation failed."
            ) from exc
        if NATIVE_ML_PREDICTIONS_FILENAME not in report.validated_artifacts:
            raise PredictionSourceError(
                "validated native ML Artifact has no predictions payload."
            )
        prediction_record = next(
            (
                record
                for record in manifest.artifacts
                if record.relative_path == NATIVE_ML_PREDICTIONS_FILENAME
            ),
            None,
        )
        if prediction_record is None:
            raise PredictionSourceError(
                "native ML Artifact manifest has no predictions record."
            )
        prediction_path = directory / NATIVE_ML_PREDICTIONS_FILENAME
        try:
            predictions = pd.read_parquet(prediction_path, engine="pyarrow")
        except Exception as exc:
            raise PredictionSourceError(
                "validated predictions payload could not be read."
            ) from exc
        self._validate_native_predictions(predictions)
        provenance = PredictionSourceProvenance(
            artifact_dir=directory,
            prediction_path=prediction_path,
            artifact_schema_version=report.schema_version,
            experiment_id=report.experiment_id,
            model_name=manifest.model_name,
            prediction_sha256=prediction_record.sha256,
        )
        return PredictionSourceResult(predictions, provenance)

    @staticmethod
    def _explicit_directory(value: object) -> Path:
        if not isinstance(value, (str, os.PathLike)):
            raise PredictionSourceError(
                "artifact_dir must be an explicit path to a native ML Artifact."
            )
        try:
            raw = os.fspath(value)
        except TypeError as exc:
            raise PredictionSourceError("artifact_dir must be path-like.") from exc
        if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
            raise PredictionSourceError(
                "artifact_dir must be a non-empty trimmed path."
            )
        configured = Path(raw)
        if configured.name.lower() == NATIVE_ML_PREDICTIONS_FILENAME:
            raise PredictionSourceError(
                "a bare predictions.parquet path is not OOS proof; "
                "provide its native ML Artifact directory."
            )
        directory = Path(os.path.abspath(configured))
        if (
            not directory.exists()
            or not directory.is_dir()
            or directory.is_symlink()
        ):
            raise PredictionSourceError(
                "artifact_dir must be an existing non-symlink directory."
            )
        return directory

    @staticmethod
    def _validate_native_predictions(predictions: pd.DataFrame) -> None:
        if (
            not isinstance(predictions, pd.DataFrame)
            or predictions.empty
            or not predictions.columns.is_unique
            or tuple(predictions.columns) != _NATIVE_ML_PREDICTION_COLUMNS
        ):
            raise PredictionSourceError(
                "native predictions payload schema is invalid."
            )
        if (
            not predictions.index.is_unique
            or predictions.index.name != "dataset_index"
        ):
            raise PredictionSourceError(
                "native predictions payload index is invalid."
            )
        dates = predictions["trade_date"]
        if (
            not pd.api.types.is_datetime64_ns_dtype(dates.dtype)
            or getattr(dates.dt, "tz", None) is not None
            or dates.isna().any()
            or not dates.eq(dates.dt.normalize()).all()
        ):
            raise PredictionSourceError(
                "native predictions trade_date contract is invalid."
            )
        codes = predictions["ts_code"]
        if (
            codes.isna().any()
            or not codes.map(lambda value: isinstance(value, (str, np.str_))).all()
            or codes.astype("string").str.strip().eq("").any()
            or predictions.duplicated(["trade_date", "ts_code"]).any()
        ):
            raise PredictionSourceError(
                "native predictions key contract is invalid."
            )
        for column in ("target", "prediction"):
            series = predictions[column]
            if (
                pd.api.types.is_bool_dtype(series.dtype)
                or not pd.api.types.is_numeric_dtype(series.dtype)
                or pd.api.types.is_complex_dtype(series.dtype)
            ):
                raise PredictionSourceError(
                    f"native predictions {column} must be real numeric."
                )
            try:
                values = series.to_numpy(dtype=np.float64, na_value=np.nan)
            except (TypeError, ValueError) as exc:
                raise PredictionSourceError(
                    f"native predictions {column} must be real numeric."
                ) from exc
            if not np.isfinite(values).all():
                raise PredictionSourceError(
                    f"native predictions {column} must be finite."
                )
        fold_ids = predictions["fold_id"]
        if (
            not pd.api.types.is_integer_dtype(fold_ids.dtype)
            or bool((fold_ids < 0).any())
        ):
            raise PredictionSourceError(
                "native predictions fold_id must be non-negative integer."
            )