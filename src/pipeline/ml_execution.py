"""Execution boundary for the optional V3 ML Pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
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
    """Compact immutable Pipeline summary for one optional ML experiment."""

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
    panel_path: str | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise MLPipelineIntegrityError("enabled must be a bool")
        if not self.enabled:
            disabled_values = (
                self.model_name,
                self.n_folds,
                self.n_prediction_rows,
                self.n_prediction_dates,
                self.mae,
                self.rmse,
                self.r2,
                self.r2_valid,
                self.r2_invalid_reason,
                self.pearson_ic_mean,
                self.rank_ic_mean,
                self.permutation_importance_enabled,
                self.permutation_importance_completed,
                self.artifacts_saved,
                self.artifact_dir,
                self.panel_path,
            )
            if disabled_values != (
                None,
                0,
                0,
                0,
                None,
                None,
                None,
                False,
                None,
                None,
                None,
                False,
                False,
                False,
                None,
                None,
            ):
                raise MLPipelineIntegrityError(
                    "disabled ML result fields must use empty defaults"
                )
            json.dumps(self.to_dict(), allow_nan=False)
            return
        if not isinstance(self.panel_path, str) or not self.panel_path:
            raise MLPipelineIntegrityError(
                "enabled ML result requires panel_path"
            )
        panel_path = Path(self.panel_path)
        if (
            not panel_path.is_absolute()
            or panel_path.is_symlink()
            or not panel_path.is_file()
            or panel_path.suffix.lower() != ".parquet"
        ):
            raise MLPipelineIntegrityError(
                "enabled ML result panel_path is invalid"
            )
        object.__setattr__(self, "panel_path", str(panel_path.resolve()))
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

    @classmethod
    def disabled(cls) -> MLExperimentPipelineResult:
        """Return the stable empty result for a disabled ML executor."""
        return cls(
            enabled=False,
            model_name=None,
            n_folds=0,
            n_prediction_rows=0,
            n_prediction_dates=0,
            mae=None,
            rmse=None,
            r2=None,
            r2_valid=False,
            r2_invalid_reason=None,
            pearson_ic_mean=None,
            rank_ic_mean=None,
            permutation_importance_enabled=False,
            permutation_importance_completed=False,
            artifacts_saved=False,
            artifact_dir=None,
            panel_path=None,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a compact JSON-safe summary without tables or parameters."""
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "panel_path": self.panel_path,
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
        root_value = (
            Path(__file__).resolve().parents[2]
            if project_root is None
            else project_root
        )
        if not isinstance(root_value, (str, Path)):
            raise MLPipelineExecutionError(
                "project_root must be a str or pathlib.Path"
            )
        self.config = config
        self.project_root = (
            _directory(root_value, "project_root")
            if config.enabled
            else Path(root_value)
        )

    def execute(
        self,
        run_dir: str | Path,
        *,
        panel_path_override: str | os.PathLike[str] | None = None,
    ) -> MLExperimentPipelineResult:
        """Run one independent ML experiment in an existing Pipeline run."""
        if not self.config.enabled:
            if panel_path_override is not None:
                raise MLPipelineExecutionError(
                    "disabled ML execution does not accept panel_path_override"
                )
            return MLExperimentPipelineResult.disabled()
        panel_source = self._select_panel_path(panel_path_override)
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
        if experiment_config is None:
            raise MLPipelineIntegrityError(
                "enabled ML configuration is incomplete"
            )
        panel = read_ml_modeling_panel(
            panel_source,
            project_root=self.project_root,
            label_col=experiment_config.dataset_config.label_col,
        )
        configured_panel = Path(panel_source).expanduser()
        actual_panel_path = (
            configured_panel
            if configured_panel.is_absolute()
            else self.project_root / configured_panel
        ).resolve()
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
            panel_path=actual_panel_path,
        )

    def _select_panel_path(
        self,
        override: str | os.PathLike[str] | None,
    ) -> str | Path:
        configured = self.config.panel_path
        if override is not None and not isinstance(override, (str, os.PathLike)):
            raise MLPipelineExecutionError(
                "panel_path_override must be a str, os.PathLike, or None"
            )
        if configured is not None and override is not None:
            raise MLPipelineExecutionError(
                "ML panel source conflict: config panel_path and override "
                "are both configured"
            )
        if configured is None and override is None:
            raise MLPipelineExecutionError(
                "ML execution requires exactly one panel path source"
            )
        return configured if configured is not None else Path(os.fspath(override))

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
        panel_path: Path,
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
            panel_path=str(panel_path),
        )
