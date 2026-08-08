"""Independent execution boundary for the optional V5 Signal stage."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from src.pipeline.ml_execution import MLExperimentPipelineResult
from src.pipeline.signal_config import SignalPipelineConfig
from src.signals import (
    SIGNAL_ARTIFACT_SCHEMA_VERSION,
    PredictionSourceAdapter,
    PredictionSourceError,
    SignalArtifactConfig,
    SignalArtifactError,
    SignalArtifactStore,
    SignalBuilder,
    SignalDataError,
)


class SignalPipelineExecutionError(Exception):
    """Raised when Signal execution or an explicit upstream handoff fails."""


def _absolute(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(value))


@dataclass(frozen=True)
class SignalPipelineResult:
    """Compact immutable Signal stage summary without DataFrames."""

    enabled: bool
    source_mode: str | None = None
    source_artifact_dir: Path | None = None
    artifact_dir: Path | None = None
    signal_path: Path | None = None
    manifest_path: Path | None = None
    rows: int = 0
    trade_date_count: int = 0
    prediction_column: str | None = None
    signal_direction: str | None = None
    schema_version: str | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise SignalPipelineExecutionError("enabled must be a bool.")
        for name in ("rows", "trade_date_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SignalPipelineExecutionError(
                    f"{name} must be a non-negative integer."
                )
        if not self.enabled:
            if (
                self.source_mode is not None
                or self.source_artifact_dir is not None
                or self.artifact_dir is not None
                or self.signal_path is not None
                or self.manifest_path is not None
                or self.rows
                or self.trade_date_count
                or self.prediction_column is not None
                or self.signal_direction is not None
                or self.schema_version is not None
            ):
                raise SignalPipelineExecutionError(
                    "disabled result fields must use empty defaults."
                )
        else:
            if self.source_mode not in {"files", "ml"}:
                raise SignalPipelineExecutionError("source_mode is invalid.")
            if self.rows <= 0 or self.trade_date_count <= 0:
                raise SignalPipelineExecutionError(
                    "enabled result requires positive row/date counts."
                )
            if (
                not isinstance(self.prediction_column, str)
                or not self.prediction_column
                or self.signal_direction not in {"ascending", "descending"}
                or self.schema_version != SIGNAL_ARTIFACT_SCHEMA_VERSION
            ):
                raise SignalPipelineExecutionError(
                    "enabled result metadata is invalid."
                )
            if (
                self.source_artifact_dir is None
                or self.artifact_dir is None
                or self.signal_path is None
                or self.manifest_path is None
            ):
                raise SignalPipelineExecutionError(
                    "enabled result requires source and Artifact paths."
                )
            source = _absolute(self.source_artifact_dir)
            artifact = _absolute(self.artifact_dir)
            signal = _absolute(self.signal_path)
            manifest = _absolute(self.manifest_path)
            if (
                not source.is_dir()
                or source.is_symlink()
                or not artifact.is_dir()
                or artifact.is_symlink()
                or not signal.is_file()
                or signal.is_symlink()
                or not manifest.is_file()
                or manifest.is_symlink()
                or signal.parent != artifact
                or manifest.parent != artifact
                or signal.name != "signals.parquet"
                or manifest.name != "manifest.json"
            ):
                raise SignalPipelineExecutionError(
                    "enabled result Artifact paths are invalid."
                )
            object.__setattr__(self, "source_artifact_dir", source)
            object.__setattr__(self, "artifact_dir", artifact)
            object.__setattr__(self, "signal_path", signal)
            object.__setattr__(self, "manifest_path", manifest)
        json.dumps(self.as_dict(), allow_nan=False)

    @classmethod
    def disabled(cls) -> SignalPipelineResult:
        return cls(enabled=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "source_mode": self.source_mode,
            "source_artifact_dir": (
                None if self.source_artifact_dir is None
                else str(self.source_artifact_dir)
            ),
            "artifact_dir": None if self.artifact_dir is None else str(self.artifact_dir),
            "signal_path": None if self.signal_path is None else str(self.signal_path),
            "manifest_path": None if self.manifest_path is None else str(self.manifest_path),
            "rows": self.rows,
            "trade_date_count": self.trade_date_count,
            "prediction_column": self.prediction_column,
            "signal_direction": self.signal_direction,
            "schema_version": self.schema_version,
        }


class SignalPipelineExecutor:
    """Build and persist Signals from one explicit native ML Artifact."""

    def __init__(self, config: SignalPipelineConfig) -> None:
        if not isinstance(config, SignalPipelineConfig):
            raise SignalPipelineExecutionError(
                "config must be SignalPipelineConfig."
            )
        self.config = config

    def execute(
        self,
        run_dir: str | Path,
        *,
        ml_result: MLExperimentPipelineResult | None = None,
    ) -> SignalPipelineResult:
        if not self.config.enabled:
            return SignalPipelineResult.disabled()
        run_path = self._run_dir(run_dir)
        source_dir = self._source_artifact_dir(ml_result)
        try:
            source = PredictionSourceAdapter().load_native_ml_artifact(source_dir)
            built = SignalBuilder().build(
                source.predictions,
                prediction_column=self.config.prediction_column,
                signal_direction=self.config.signal_direction,
            )
            artifact_dir = _absolute(run_path / self.config.artifact_subdir)
            if artifact_dir.parent != run_path:
                raise SignalPipelineExecutionError(
                    "artifact_dir must be a direct child of run_dir."
                )
            written = SignalArtifactStore().write(
                built,
                source.provenance,
                SignalArtifactConfig(artifact_dir),
            )
        except SignalPipelineExecutionError:
            raise
        except PredictionSourceError as exc:
            raise SignalPipelineExecutionError(
                f"Signal source validation failed: {type(exc).__name__}."
            ) from exc
        except SignalDataError as exc:
            raise SignalPipelineExecutionError(
                f"Signal build failed: {type(exc).__name__}."
            ) from exc
        except SignalArtifactError as exc:
            raise SignalPipelineExecutionError(
                f"Signal Artifact write failed: {type(exc).__name__}."
            ) from exc
        if not written.validation.is_valid:
            raise SignalPipelineExecutionError(
                "Signal Artifact validation is invalid."
            )
        return SignalPipelineResult(
            enabled=True,
            source_mode=self.config.source.mode,
            source_artifact_dir=source.provenance.artifact_dir,
            artifact_dir=written.artifact_dir,
            signal_path=written.signal_path,
            manifest_path=written.manifest_path,
            rows=built.audit.output_rows,
            trade_date_count=built.audit.trade_date_count,
            prediction_column=built.audit.prediction_column,
            signal_direction=built.audit.signal_direction,
            schema_version=written.schema_version,
        )

    @staticmethod
    def _run_dir(value: object) -> Path:
        if not isinstance(value, (str, os.PathLike)):
            raise SignalPipelineExecutionError(
                "run_dir must be a str or os.PathLike."
            )
        path = Path(os.fspath(value))
        if path.is_symlink():
            raise SignalPipelineExecutionError("run_dir must not be a symlink.")
        resolved = _absolute(path)
        if not resolved.exists() or not resolved.is_dir():
            raise SignalPipelineExecutionError(
                "run_dir must be an existing directory."
            )
        return resolved

    def _source_artifact_dir(
        self, ml_result: MLExperimentPipelineResult | None
    ) -> Path:
        if self.config.source.mode == "files":
            if ml_result is not None:
                raise SignalPipelineExecutionError(
                    "files source does not accept ml_result."
                )
            if self.config.source.artifact_dir is None:
                raise SignalPipelineExecutionError(
                    "files source artifact_dir is missing."
                )
            return _absolute(self.config.source.artifact_dir)
        if not isinstance(ml_result, MLExperimentPipelineResult):
            raise SignalPipelineExecutionError(
                "ml source requires MLExperimentPipelineResult."
            )
        if (
            not ml_result.enabled
            or not ml_result.artifacts_saved
            or not isinstance(ml_result.artifact_dir, str)
            or not ml_result.artifact_dir
        ):
            raise SignalPipelineExecutionError(
                "ml source requires an enabled ML result with artifact_dir."
            )
        return _absolute(ml_result.artifact_dir)
