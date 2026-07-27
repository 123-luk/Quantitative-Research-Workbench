"""Configuration contracts for the optional Modeling Panel Pipeline stage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from src.modeling_panel import ModelingPanelConfig


class ModelingPanelPipelineError(Exception):
    """Base error for Modeling Panel Pipeline integration."""


class ModelingPanelPipelineConfigError(ModelingPanelPipelineError):
    """Raised when Modeling Panel Pipeline configuration is invalid."""


class ModelingPanelPipelineExecutionError(ModelingPanelPipelineError):
    """Raised when Modeling Panel Pipeline execution fails."""


def _strict_mapping(
    value: object, allowed: set[str], context: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ModelingPanelPipelineConfigError(f"{context} must be a Mapping.")
    values = dict(value)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ModelingPanelPipelineConfigError(
            f"{context} contains unknown fields: {unknown!r}."
        )
    return values


def _parquet_path(value: object, field_name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise ModelingPanelPipelineConfigError(
            f"{field_name} must be a str, os.PathLike, or None."
        )
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ModelingPanelPipelineConfigError(
            f"{field_name} must be path-like."
        ) from exc
    if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
        raise ModelingPanelPipelineConfigError(
            f"{field_name} must be a non-empty trimmed path."
        )
    path = Path(raw)
    if path.suffix.lower() != ".parquet":
        raise ModelingPanelPipelineConfigError(
            f"{field_name} must have a .parquet suffix."
        )
    return path


@dataclass(frozen=True)
class ModelingPanelSourceConfig:
    """Choose explicit files or this run's Factor Research published outputs."""

    mode: str = "files"
    factor_panel_path: Path | None = None
    forward_returns_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise ModelingPanelPipelineConfigError("source mode must be a string.")
        mode = self.mode.strip().lower()
        if mode not in {"files", "factor_research"}:
            raise ModelingPanelPipelineConfigError(
                "source mode must be 'files' or 'factor_research'."
            )
        factor = _parquet_path(self.factor_panel_path, "factor_panel_path")
        returns = _parquet_path(
            self.forward_returns_path, "forward_returns_path"
        )
        if mode == "factor_research" and (factor is not None or returns is not None):
            raise ModelingPanelPipelineConfigError(
                "factor_research source must not configure file paths."
            )
        if mode == "files":
            if (factor is None) != (returns is None):
                raise ModelingPanelPipelineConfigError(
                    "files source requires both factor and returns paths."
                )
            if factor is not None and factor == returns:
                raise ModelingPanelPipelineConfigError(
                    "files source paths must be different."
                )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "factor_panel_path", factor)
        object.__setattr__(self, "forward_returns_path", returns)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | ModelingPanelSourceConfig | None
    ) -> ModelingPanelSourceConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value,
            {"mode", "factor_panel_path", "forward_returns_path"},
            "modeling_panel.source",
        )
        return cls(**values)  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "factor_panel_path": (
                None if self.factor_panel_path is None else str(self.factor_panel_path)
            ),
            "forward_returns_path": (
                None
                if self.forward_returns_path is None
                else str(self.forward_returns_path)
            ),
        }


def _safe_subdir(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ModelingPanelPipelineConfigError(
            "artifact_subdir must be a non-empty trimmed string."
        )
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
        or "://" in value
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
    ):
        raise ModelingPanelPipelineConfigError(
            "artifact_subdir must be one safe relative directory name."
        )
    return value


@dataclass(frozen=True)
class ModelingPanelOutputConfig:
    """Configure the required Modeling Panel Artifact publication."""

    save_artifact: bool = True
    artifact_subdir: str = "modeling_panel"
    parquet_compression: str = "zstd"
    verify_after_write: bool = True

    def __post_init__(self) -> None:
        if self.save_artifact is not True:
            raise ModelingPanelPipelineConfigError(
                "save_artifact must be True for this Pipeline stage."
            )
        subdir = _safe_subdir(self.artifact_subdir)
        if not isinstance(self.parquet_compression, str):
            raise ModelingPanelPipelineConfigError(
                "parquet_compression must be a string."
            )
        compression = self.parquet_compression.strip().lower()
        if compression not in {"zstd", "snappy"}:
            raise ModelingPanelPipelineConfigError(
                "parquet_compression must be 'zstd' or 'snappy'."
            )
        if type(self.verify_after_write) is not bool:
            raise ModelingPanelPipelineConfigError(
                "verify_after_write must be a bool."
            )
        object.__setattr__(self, "artifact_subdir", subdir)
        object.__setattr__(self, "parquet_compression", compression)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | ModelingPanelOutputConfig | None
    ) -> ModelingPanelOutputConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value,
            {
                "save_artifact",
                "artifact_subdir",
                "parquet_compression",
                "verify_after_write",
            },
            "modeling_panel.output",
        )
        return cls(**values)  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, object]:
        return {
            "save_artifact": self.save_artifact,
            "artifact_subdir": self.artifact_subdir,
            "parquet_compression": self.parquet_compression,
            "verify_after_write": self.verify_after_write,
        }


@dataclass(frozen=True)
class ModelingPanelPipelineConfig:
    """Hold one optional, immutable Modeling Panel Pipeline configuration."""

    enabled: bool = False
    source: ModelingPanelSourceConfig = field(
        default_factory=ModelingPanelSourceConfig
    )
    builder: ModelingPanelConfig = field(default_factory=ModelingPanelConfig)
    output: ModelingPanelOutputConfig = field(
        default_factory=ModelingPanelOutputConfig
    )

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ModelingPanelPipelineConfigError("enabled must be a bool.")
        source = ModelingPanelSourceConfig.from_dict(self.source)
        try:
            builder = ModelingPanelConfig.from_dict(self.builder)
        except Exception as exc:
            raise ModelingPanelPipelineConfigError(
                f"builder configuration is invalid: {exc}"
            ) from exc
        output = ModelingPanelOutputConfig.from_dict(self.output)
        if self.enabled and source.mode == "files" and (
            source.factor_panel_path is None
            or source.forward_returns_path is None
        ):
            raise ModelingPanelPipelineConfigError(
                "enabled files source requires factor_panel_path and "
                "forward_returns_path."
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "builder", builder)
        object.__setattr__(self, "output", output)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | ModelingPanelPipelineConfig | None
    ) -> ModelingPanelPipelineConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value, {"enabled", "source", "builder", "output"}, "modeling_panel"
        )
        if "source" in values:
            values["source"] = ModelingPanelSourceConfig.from_dict(
                values["source"]  # type: ignore[arg-type]
            )
        if "builder" in values:
            try:
                values["builder"] = ModelingPanelConfig.from_dict(
                    values["builder"]  # type: ignore[arg-type]
                )
            except Exception as exc:
                raise ModelingPanelPipelineConfigError(
                    f"builder configuration is invalid: {exc}"
                ) from exc
        if "output" in values:
            values["output"] = ModelingPanelOutputConfig.from_dict(
                values["output"]  # type: ignore[arg-type]
            )
        return cls(**values)  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source": self.source.as_dict(),
            "builder": self.builder.as_dict(),
            "output": self.output.as_dict(),
        }
