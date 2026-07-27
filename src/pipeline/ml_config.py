"""Strict configuration for the optional V3 ML pipeline stage."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from src.ml import MLExperimentConfig


class MLPipelineError(Exception):
    """Base error for optional ML pipeline integration."""


class MLPipelineConfigError(MLPipelineError):
    """Raised when ML pipeline configuration is invalid."""


def _optional_text(field_name: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MLPipelineConfigError(
            f"{field_name} must be a non-empty string or None"
        )
    return value.strip()


def _safe_artifact_root(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MLPipelineConfigError(
            "artifact_root must be a non-empty relative path string"
        )
    text = value.strip()
    windows_path = PureWindowsPath(text)
    path = Path(text)
    if (
        text in {".", ".."}
        or text.startswith(("/", "\\"))
        or windows_path.drive
        or windows_path.is_absolute()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in windows_path.parts)
    ):
        raise MLPipelineConfigError(
            "artifact_root must be a safe relative path without traversal"
        )
    return path.as_posix()


@dataclass(frozen=True)
class MLExperimentPipelineConfig:
    """Configure one opt-in ML experiment inside a Pipeline run directory."""

    enabled: bool = False
    panel_path: str | None = None
    save_artifacts: bool = False
    artifact_root: str = "ml_artifacts"
    experiment_id: str | None = None
    parquet_compression: str = "zstd"
    experiment: MLExperimentConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise MLPipelineConfigError("enabled must be a bool")
        if not isinstance(self.save_artifacts, bool):
            raise MLPipelineConfigError("save_artifacts must be a bool")

        panel_path = _optional_text("panel_path", self.panel_path)
        experiment_id = _optional_text("experiment_id", self.experiment_id)
        artifact_root = _safe_artifact_root(self.artifact_root)
        object.__setattr__(self, "panel_path", panel_path)
        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "artifact_root", artifact_root)

        compression = self.parquet_compression
        if not isinstance(compression, str):
            raise MLPipelineConfigError(
                "parquet_compression must be a string"
            )
        compression = compression.strip().lower()
        if compression not in {"zstd", "snappy", "none"}:
            raise MLPipelineConfigError(
                "parquet_compression must be zstd, snappy, or none"
            )
        object.__setattr__(self, "parquet_compression", compression)

        experiment = self.experiment
        if isinstance(experiment, Mapping):
            try:
                experiment = MLExperimentConfig.from_dict(experiment)
            except Exception as exc:
                raise MLPipelineConfigError(
                    "experiment configuration is invalid"
                ) from exc
            object.__setattr__(self, "experiment", experiment)
        elif experiment is not None and not isinstance(
            experiment, MLExperimentConfig
        ):
            raise MLPipelineConfigError(
                "experiment must be MLExperimentConfig, Mapping, or None"
            )

        if self.enabled:
            if panel_path is None:
                raise MLPipelineConfigError(
                    "panel_path is required when ML is enabled"
                )
            if experiment is None:
                raise MLPipelineConfigError(
                    "experiment is required when ML is enabled"
                )
        if self.save_artifacts:
            if not self.enabled:
                raise MLPipelineConfigError(
                    "save_artifacts=True requires enabled=True"
                )
            if experiment_id is None:
                raise MLPipelineConfigError(
                    "experiment_id is required when saving artifacts"
                )

    @classmethod
    def from_dict(
        cls,
        values: (
            Mapping[str, object]
            | MLExperimentPipelineConfig
            | None
        ),
    ) -> MLExperimentPipelineConfig:
        """Build a detached strict configuration from a mapping or None."""
        if values is None:
            return cls()
        if isinstance(values, cls):
            return values
        if not isinstance(values, Mapping):
            raise MLPipelineConfigError(
                "ml_experiment configuration must be a Mapping, "
                "MLExperimentPipelineConfig, or None"
            )
        if any(not isinstance(key, str) for key in values):
            raise MLPipelineConfigError(
                "ml_experiment field names must be strings"
            )
        allowed = {
            "enabled",
            "panel_path",
            "save_artifacts",
            "artifact_root",
            "experiment_id",
            "parquet_compression",
            "experiment",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise MLPipelineConfigError(
                "unknown ml_experiment field(s): "
                + ", ".join(unknown)
            )
        detached = deepcopy(dict(values))
        try:
            return cls(**detached)
        except MLPipelineConfigError:
            raise
        except (TypeError, ValueError) as exc:
            raise MLPipelineConfigError(
                "ml_experiment configuration is invalid"
            ) from exc

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-safe configuration mapping."""
        return {
            "enabled": self.enabled,
            "panel_path": self.panel_path,
            "save_artifacts": self.save_artifacts,
            "artifact_root": self.artifact_root,
            "experiment_id": self.experiment_id,
            "parquet_compression": self.parquet_compression,
            "experiment": (
                None
                if self.experiment is None
                else self.experiment.as_dict()
            ),
        }
