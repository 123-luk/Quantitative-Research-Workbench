"""Execution boundary for the optional V3 ML Pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import pandas as pd

from src.ml import (
    MLArtifactConfig,
    MLArtifactWriteResult,
    MLExperimentArtifactStore,
    MLExperimentResult,
    MLExperimentRunner,
)
from src.pipeline.ml_config import (
    MLExperimentPipelineConfig,
    MLPipelineError,
)


class MLPipelinePanelError(MLPipelineError):
    """Raised when the merged modeling panel cannot be consumed."""


class MLPipelineExecutionError(MLPipelineError):
    """Raised when the public ML experiment runner fails."""


class MLPipelineArtifactError(MLPipelineError):
    """Raised when ML artifact configuration or persistence fails."""


class MLPipelineIntegrityError(MLPipelineError):
    """Raised when public ML outputs disagree across stage boundaries."""


_BASE_NON_FEATURE_COLUMNS = {
    "trade_date",
    "ts_code",
    "entry_trade_date",
    "exit_trade_date",
    "entry_price",
    "exit_price",
}


def _directory(value: str | Path, field_name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise MLPipelinePanelError(
            f"{field_name} must be a str or pathlib.Path"
        )
    if isinstance(value, str) and not value.strip():
        raise MLPipelinePanelError(f"{field_name} must not be empty")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise MLPipelinePanelError(f"{field_name} does not exist: {path}")
    if not path.is_dir():
        raise MLPipelinePanelError(f"{field_name} must be a directory: {path}")
    return path


def read_ml_modeling_panel(
    path: str | Path,
    *,
    project_root: str | Path,
    label_col: str,
) -> pd.DataFrame:
    """Read one pre-merged Parquet without transforming its rows or dtypes."""
    root = _directory(project_root, "project_root")
    if not isinstance(path, (str, Path)):
        raise MLPipelinePanelError(
            "modeling panel path must be a str or pathlib.Path"
        )
    if isinstance(path, str) and not path.strip():
        raise MLPipelinePanelError("modeling panel path must not be empty")
    if not isinstance(label_col, str) or not label_col.strip():
        raise MLPipelinePanelError("label_col must be a non-empty string")
    normalized_label = label_col.strip()

    configured = Path(path).expanduser()
    candidate = configured if configured.is_absolute() else root / configured
    if candidate.is_symlink():
        raise MLPipelinePanelError(
            f"modeling panel must not be a symlink: {candidate.resolve()}"
        )
    resolved = candidate.resolve()
    if not resolved.exists():
        raise MLPipelinePanelError(
            f"modeling panel does not exist: {resolved}"
        )
    if not resolved.is_file():
        raise MLPipelinePanelError(
            f"modeling panel must be a regular file: {resolved}"
        )
    if resolved.suffix.lower() != ".parquet":
        raise MLPipelinePanelError(
            f"modeling panel must use .parquet: {resolved}"
        )
    try:
        frame = pd.read_parquet(resolved, engine="pyarrow")
    except Exception as exc:
        raise MLPipelinePanelError(
            f"failed to read modeling panel Parquet: {resolved}"
        ) from exc
    if not isinstance(frame, pd.DataFrame):
        raise MLPipelinePanelError(
            f"modeling panel reader returned {type(frame).__name__}: {resolved}"
        )
    if frame.empty:
        raise MLPipelinePanelError(f"modeling panel is empty: {resolved}")
    if not frame.columns.is_unique:
        raise MLPipelinePanelError(
            f"modeling panel column names must be unique: {resolved}"
        )
    required = {
        "trade_date",
        "ts_code",
        "entry_trade_date",
        "exit_trade_date",
        normalized_label,
    }
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise MLPipelinePanelError(
            f"modeling panel is missing required columns {sorted(missing)!r}: "
            f"{resolved}"
        )
    reserved = {*_BASE_NON_FEATURE_COLUMNS, normalized_label}
    if not any(column not in reserved for column in frame.columns):
        raise MLPipelinePanelError(
            f"modeling panel contains no candidate feature columns: {resolved}"
        )
    return frame.copy(deep=True)


def _finite_optional(field_name: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise MLPipelineIntegrityError(
            f"{field_name} must be a finite number or None"
        )
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise MLPipelineIntegrityError(
            f"{field_name} must be a finite number or None"
        ) from exc
    if not math.isfinite(normalized):
        raise MLPipelineIntegrityError(f"{field_name} must be finite")
    return normalized


@dataclass(frozen=True)
class MLExperimentPipelineResult:
    """Compact immutable Pipeline summary for one successful ML experiment."""

    enabled: bool
    model_name: str | None
    n_folds: int
    n_prediction_rows: int
    n_prediction_dates: int
    mae: float | None
    rmse: float | None
    r2: float | None
    r2_valid: bool
    r2_invalid_reason: str | None
    pearson_ic_mean: float | None
    rank_ic_mean: float | None
    permutation_importance_enabled: bool
    permutation_importance_completed: bool
    artifacts_saved: bool
    artifact_dir: str | None

    def __post_init__(self) -> None:
        if self.enabled is not True:
            raise MLPipelineIntegrityError(
                "MLExperimentPipelineResult represents only enabled runs"
            )
        if not isinstance(self.model_name, str) or not self.model_name:
            raise MLPipelineIntegrityError("model_name must be non-empty")
        for field_name in (
            "n_folds",
            "n_prediction_rows",
            "n_prediction_dates",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise MLPipelineIntegrityError(
                    f"{field_name} must be a positive integer"
                )
        for field_name in (
            "mae",
            "rmse",
            "r2",
            "pearson_ic_mean",
            "rank_ic_mean",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_optional(field_name, getattr(self, field_name)),
            )
        for field_name in (
            "r2_valid",
            "permutation_importance_enabled",
            "permutation_importance_completed",
            "artifacts_saved",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise MLPipelineIntegrityError(
                    f"{field_name} must be a bool"
                )
        if self.r2_valid:
            if self.r2 is None or self.r2_invalid_reason is not None:
                raise MLPipelineIntegrityError("R-squared status is inconsistent")
        elif self.r2 is not None or not isinstance(
            self.r2_invalid_reason, str
        ) or not self.r2_invalid_reason:
            raise MLPipelineIntegrityError("R-squared status is inconsistent")
        if (
            self.permutation_importance_enabled
            != self.permutation_importance_completed
        ):
            raise MLPipelineIntegrityError(
                "permutation importance status is inconsistent"
            )
        if self.artifacts_saved != (self.artifact_dir is not None):
            raise MLPipelineIntegrityError(
                "artifact saved state and directory are inconsistent"
            )
        json.dumps(self.to_dict(), allow_nan=False)

    def to_dict(self) -> dict[str, object]:
        """Return a compact JSON-safe summary without tables or parameters."""
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "n_folds": self.n_folds,
            "n_prediction_rows": self.n_prediction_rows,
            "n_prediction_dates": self.n_prediction_dates,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "r2_valid": self.r2_valid,
            "r2_invalid_reason": self.r2_invalid_reason,
            "pearson_ic_mean": self.pearson_ic_mean,
            "rank_ic_mean": self.rank_ic_mean,
            "permutation_importance_enabled": (
                self.permutation_importance_enabled
            ),
            "permutation_importance_completed": (
                self.permutation_importance_completed
            ),
            "artifacts_saved": self.artifacts_saved,
            "artifact_dir": self.artifact_dir,
        }


class MLExperimentPipelineExecutor:
    """Read a merged panel and invoke only public V3-G/V3-H APIs."""

    def __init__(
        self,
        config: MLExperimentPipelineConfig,
        *,
        project_root: str | Path | None = None,
    ) -> None:
        if not isinstance(config, MLExperimentPipelineConfig):
            raise MLPipelineExecutionError(
                "config must be MLExperimentPipelineConfig"
            )
        if not config.enabled:
            raise MLPipelineExecutionError(
                "MLExperimentPipelineExecutor requires enabled=True"
            )
        root_value = (
            Path(__file__).resolve().parents[2]
            if project_root is None
            else project_root
        )
        self.config = config
        self.project_root = _directory(root_value, "project_root")

    def execute(
        self,
        run_dir: str | Path,
    ) -> MLExperimentPipelineResult:
        """Run one independent ML experiment in an existing Pipeline run."""
        if not isinstance(run_dir, (str, Path)):
            raise MLPipelineExecutionError(
                "run_dir must be a str or pathlib.Path"
            )
        configured_run = Path(run_dir).expanduser()
        if configured_run.is_symlink():
            raise MLPipelineExecutionError("run_dir must not be a symlink")
        run_path = configured_run.resolve()
        if not run_path.exists():
            raise MLPipelineExecutionError(
                f"run_dir does not exist: {run_path}"
            )
        if not run_path.is_dir():
            raise MLPipelineExecutionError(
                f"run_dir must be a directory: {run_path}"
            )
        experiment_config = self.config.experiment
        if experiment_config is None or self.config.panel_path is None:
            raise MLPipelineIntegrityError(
                "enabled ML configuration is incomplete"
            )
        panel = read_ml_modeling_panel(
            self.config.panel_path,
            project_root=self.project_root,
            label_col=experiment_config.dataset_config.label_col,
        )
        try:
            experiment_result = MLExperimentRunner().run(
                frame=panel,
                config=experiment_config,
            )
        except Exception as exc:
            raise MLPipelineExecutionError(
                f"ML experiment execution failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(experiment_result, MLExperimentResult):
            raise MLPipelineExecutionError(
                "MLExperimentRunner returned an invalid result type"
            )

        artifact_dir: str | None = None
        if self.config.save_artifacts:
            artifact_dir = self._save_artifacts(
                experiment_result, run_path
            )
        return self._build_result(
            experiment_result,
            run_path=run_path,
            artifact_dir=artifact_dir,
        )

    def _save_artifacts(
        self,
        experiment_result: MLExperimentResult,
        run_path: Path,
    ) -> str:
        artifact_root = (run_path / self.config.artifact_root).resolve()
        try:
            artifact_root.relative_to(run_path)
        except ValueError as exc:
            raise MLPipelineIntegrityError(
                "artifact_root must remain inside run_dir"
            ) from exc
        if artifact_root == run_path:
            raise MLPipelineIntegrityError(
                "artifact_root must be a child of run_dir"
            )
        try:
            artifact_config = MLArtifactConfig(
                artifact_root=artifact_root,
                experiment_id=self.config.experiment_id,
                parquet_compression=self.config.parquet_compression,
            )
            write_result = MLExperimentArtifactStore().write(
                result=experiment_result,
                config=artifact_config,
            )
        except Exception as exc:
            raise MLPipelineArtifactError(
                f"ML artifact persistence failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(write_result, MLArtifactWriteResult):
            raise MLPipelineArtifactError(
                "MLExperimentArtifactStore returned an invalid result type"
            )
        formal_dir = write_result.experiment_dir.resolve()
        try:
            formal_dir.relative_to(run_path)
        except ValueError as exc:
            raise MLPipelineIntegrityError(
                "formal ML artifact directory escaped run_dir"
            ) from exc
        return str(formal_dir)

    def _build_result(
        self,
        experiment_result: MLExperimentResult,
        *,
        run_path: Path,
        artifact_dir: str | None,
    ) -> MLExperimentPipelineResult:
        audit = experiment_result.audit
        evaluation = experiment_result.evaluation_result
        evaluation_audit = evaluation.audit
        regression = evaluation.regression_metrics
        importance = experiment_result.permutation_importance_result
        if (
            not audit.model_name
            or audit.n_folds <= 0
            or audit.n_prediction_rows <= 0
            or audit.n_prediction_dates <= 0
            or not audit.evaluation_completed
            or regression.n_obs != audit.n_prediction_rows
            or evaluation_audit.n_rows != audit.n_prediction_rows
            or evaluation_audit.n_dates != audit.n_prediction_dates
            or evaluation_audit.n_folds != audit.n_folds
            or evaluation_audit.row_coverage != 1.0
            or evaluation_audit.date_coverage != 1.0
            or audit.permutation_importance_enabled
            != audit.permutation_importance_completed
            or audit.permutation_importance_enabled != (importance is not None)
        ):
            raise MLPipelineIntegrityError(
                "ML experiment result and evaluation audit are inconsistent"
            )
        artifacts_saved = artifact_dir is not None
        if artifacts_saved != self.config.save_artifacts:
            raise MLPipelineIntegrityError(
                "artifact result state differs from configuration"
            )
        if artifact_dir is not None:
            try:
                Path(artifact_dir).resolve().relative_to(run_path)
            except ValueError as exc:
                raise MLPipelineIntegrityError(
                    "artifact result path escaped run_dir"
                ) from exc
        return MLExperimentPipelineResult(
            enabled=True,
            model_name=audit.model_name,
            n_folds=audit.n_folds,
            n_prediction_rows=audit.n_prediction_rows,
            n_prediction_dates=audit.n_prediction_dates,
            mae=regression.mae,
            rmse=regression.rmse,
            r2=regression.r2,
            r2_valid=regression.r2_valid,
            r2_invalid_reason=regression.r2_invalid_reason,
            pearson_ic_mean=evaluation.pearson_ic_summary.mean,
            rank_ic_mean=evaluation.rank_ic_summary.mean,
            permutation_importance_enabled=(
                audit.permutation_importance_enabled
            ),
            permutation_importance_completed=(
                audit.permutation_importance_completed
            ),
            artifacts_saved=artifacts_saved,
            artifact_dir=artifact_dir,
        )
