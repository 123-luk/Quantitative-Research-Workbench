"""Strict static configuration for the optional V5 Signal stage."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path, PurePosixPath, PureWindowsPath


class SignalConfigError(ValueError):
    """Raised when Signal pipeline configuration is invalid."""


def _strict_mapping(
    value: object, allowed: frozenset[str], context: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SignalConfigError(f"{context} must be a Mapping.")
    if any(not isinstance(key, str) for key in value):
        raise SignalConfigError(f"{context} field names must be strings.")
    values = deepcopy(dict(value))
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise SignalConfigError(
            f"{context} contains unknown fields: {unknown!r}."
        )
    return values


def _optional_artifact_dir(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise SignalConfigError(
            "artifact_dir must be a str, os.PathLike, or None."
        )
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise SignalConfigError("artifact_dir must be path-like.") from exc
    if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
        raise SignalConfigError(
            "artifact_dir must be a non-empty trimmed path."
        )
    return Path(raw)


def _safe_artifact_subdir(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SignalConfigError(
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
        raise SignalConfigError(
            "artifact_subdir must be one safe relative directory name."
        )
    return value


@dataclass(frozen=True)
class PredictionSourceConfig:
    """Select this run's ML result or one explicit native ML Artifact."""

    mode: str = "ml"
    artifact_dir: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise SignalConfigError("source mode must be a string.")
        mode = self.mode.strip().lower()
        if mode not in {"ml", "files"}:
            raise SignalConfigError("source mode must be 'ml' or 'files'.")
        artifact_dir = _optional_artifact_dir(self.artifact_dir)
        if mode == "ml" and artifact_dir is not None:
            raise SignalConfigError(
                "ml source must not configure artifact_dir."
            )
        if mode == "files" and artifact_dir is None:
            raise SignalConfigError(
                "files source requires an explicit native ML artifact_dir."
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "artifact_dir", artifact_dir)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | PredictionSourceConfig | None
    ) -> PredictionSourceConfig:
        """Build a detached source config without reading the Artifact."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value, frozenset({"mode", "artifact_dir"}), "signal.source"
        )
        try:
            return cls(**values)  # type: ignore[arg-type]
        except SignalConfigError:
            raise
        except (TypeError, ValueError) as exc:
            raise SignalConfigError("signal.source configuration is invalid.") from exc

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-safe source mapping."""
        return {
            "mode": self.mode,
            "artifact_dir": (
                None if self.artifact_dir is None else str(self.artifact_dir)
            ),
        }


@dataclass(frozen=True)
class SignalPipelineConfig:
    """Configure the optional V5 Signal stage without executing it."""

    enabled: bool = False
    source: PredictionSourceConfig = field(default_factory=PredictionSourceConfig)
    prediction_column: str = "prediction"
    signal_direction: str = "descending"
    artifact_subdir: str = "signal"

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise SignalConfigError("enabled must be a bool.")
        source = PredictionSourceConfig.from_dict(self.source)
        if not isinstance(self.prediction_column, str):
            raise SignalConfigError("prediction_column must be a string.")
        prediction_column = self.prediction_column.strip()
        if not prediction_column:
            raise SignalConfigError("prediction_column must be non-empty.")
        if not isinstance(self.signal_direction, str):
            raise SignalConfigError("signal_direction must be a string.")
        direction = self.signal_direction.strip().lower()
        if direction not in {"descending", "ascending"}:
            raise SignalConfigError(
                "signal_direction must be 'descending' or 'ascending'."
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "prediction_column", prediction_column)
        object.__setattr__(self, "signal_direction", direction)
        object.__setattr__(
            self, "artifact_subdir", _safe_artifact_subdir(self.artifact_subdir)
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | SignalPipelineConfig | None
    ) -> SignalPipelineConfig:
        """Build a detached strict Signal config from a mapping or None."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value,
            frozenset(
                {
                    "enabled",
                    "source",
                    "prediction_column",
                    "signal_direction",
                    "artifact_subdir",
                }
            ),
            "signal",
        )
        if "source" in values:
            values["source"] = PredictionSourceConfig.from_dict(
                values["source"]  # type: ignore[arg-type]
            )
        try:
            return cls(**values)  # type: ignore[arg-type]
        except SignalConfigError:
            raise
        except (TypeError, ValueError) as exc:
            raise SignalConfigError("signal configuration is invalid.") from exc

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-safe Signal configuration mapping."""
        return {
            "enabled": self.enabled,
            "source": self.source.to_dict(),
            "prediction_column": self.prediction_column,
            "signal_direction": self.signal_direction,
            "artifact_subdir": self.artifact_subdir,
        }
