"""Configuration-only bridge between the V1 pipeline and V2 factor research.

The YAML section is named ``factor_research`` and contains input path strings
plus nested G2, D1, D2, E1, E2, F1, F2, G1, and G3 configuration mappings.
This module validates and serializes configuration only. It does not read
Parquet files, run research, create directories, or persist artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any

from src.factors.composition import FactorCompositionConfig
from src.factors.dynamic_composition import RollingICWeightConfig
from src.factors.evaluation import FactorEvaluationConfig
from src.factors.forward_returns import ForwardReturnConfig
from src.factors.neutralization import NeutralizationConfig
from src.factors.preprocessing import PreprocessingConfig
from src.factors.quantile_evaluation import QuantileEvaluationConfig
from src.factors.research_artifacts import ResearchArtifactConfig
from src.factors.research_pipeline import FactorResearchConfig


_CONFIG_FIELDS = frozenset(
    {
        "enabled",
        "factor_input_path",
        "score_panel_path",
        "price_panel_path",
        "exposure_panel_path",
        "artifact_subdir",
        "research",
        "preprocessing",
        "neutralization",
        "evaluation",
        "quantile",
        "composition",
        "rolling",
        "forward_returns",
        "artifacts",
    }
)


def _json_safe(value: Any) -> Any:
    """Recursively convert immutable tuples to detached JSON-safe lists."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _parse_nested(
    value: object,
    expected_type: type,
    field_name: str,
    *,
    conversions: Mapping[str, Any] | None = None,
) -> Any:
    """Build one nested config from a mapping without mutating caller data."""
    if isinstance(value, expected_type):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{field_name} must be a Mapping or {expected_type.__name__}."
        )
    values = dict(value)
    for key, converter in (conversions or {}).items():
        if key in values:
            values[key] = converter(values[key])
    try:
        return expected_type(**values)
    except TypeError as exc:
        raise TypeError(f"Invalid {field_name} configuration: {exc}") from exc


def _tuple_value(value: object) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("Expected a list or tuple, not a string.")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("Expected a list or tuple.") from exc


def _fixed_weights(value: object) -> tuple[tuple[Any, ...], ...]:
    return tuple(_tuple_value(pair) for pair in _tuple_value(value))


def _safe_artifact_subdir(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("artifact_subdir must be a non-empty string.")
    text = value.strip()
    if (
        text in {".", ".."}
        or text.startswith(("/", "\\"))
        or PurePosixPath(text).is_absolute()
        or PureWindowsPath(text).is_absolute()
        or PureWindowsPath(text).drive
        or ":" in text
    ):
        raise ValueError("artifact_subdir must be a safe relative directory.")
    parts = re.split(r"[\\/]", text)
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact_subdir must not contain path traversal.")
    return text


@dataclass(frozen=True)
class FactorResearchPipelineConfig:
    """Hold validated pipeline inputs and all factor-research component configs."""

    enabled: bool = False
    factor_input_path: str | None = None
    score_panel_path: str | None = None
    price_panel_path: str | None = None
    exposure_panel_path: str | None = None
    artifact_subdir: str = "factor_research"
    research: FactorResearchConfig | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    neutralization: NeutralizationConfig = field(
        default_factory=NeutralizationConfig
    )
    evaluation: FactorEvaluationConfig = field(
        default_factory=FactorEvaluationConfig
    )
    quantile: QuantileEvaluationConfig = field(
        default_factory=QuantileEvaluationConfig
    )
    composition: FactorCompositionConfig = field(
        default_factory=FactorCompositionConfig
    )
    rolling: RollingICWeightConfig = field(default_factory=RollingICWeightConfig)
    forward_returns: ForwardReturnConfig = field(
        default_factory=ForwardReturnConfig
    )
    artifacts: ResearchArtifactConfig = field(
        default_factory=ResearchArtifactConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        object.__setattr__(
            self, "artifact_subdir", _safe_artifact_subdir(self.artifact_subdir)
        )

        for field_name in (
            "factor_input_path",
            "score_panel_path",
            "price_panel_path",
            "exposure_panel_path",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a non-empty string or None.")
            if isinstance(value, str):
                normalized = value.strip()
                if not normalized:
                    raise ValueError(
                        f"{field_name} must be a non-empty string or None."
                    )
                object.__setattr__(self, field_name, normalized)

        expected_configs = {
            "preprocessing": PreprocessingConfig,
            "neutralization": NeutralizationConfig,
            "evaluation": FactorEvaluationConfig,
            "quantile": QuantileEvaluationConfig,
            "composition": FactorCompositionConfig,
            "rolling": RollingICWeightConfig,
            "forward_returns": ForwardReturnConfig,
            "artifacts": ResearchArtifactConfig,
        }
        for field_name, expected_type in expected_configs.items():
            if not isinstance(getattr(self, field_name), expected_type):
                raise TypeError(
                    f"{field_name} must be a {expected_type.__name__}."
                )
        if self.research is not None and not isinstance(
            self.research, FactorResearchConfig
        ):
            raise TypeError("research must be a FactorResearchConfig or None.")

        if self.enabled:
            if self.research is None:
                raise ValueError("research is required when factor research is enabled.")
            for field_name in (
                "factor_input_path",
                "score_panel_path",
                "price_panel_path",
            ):
                if getattr(self, field_name) is None:
                    raise ValueError(
                        f"{field_name} is required when factor research is enabled."
                    )
            if (
                self.research.use_neutralization
                and self.exposure_panel_path is None
            ):
                raise ValueError(
                    "exposure_panel_path is required when neutralization is enabled."
                )

        return_columns = {
            self.evaluation.return_col,
            self.quantile.return_col,
            self.forward_returns.return_col,
        }
        if len(return_columns) != 1:
            raise ValueError(
                "evaluation, quantile, and forward_returns return_col values "
                "must match."
            )
        self._validate_composition_consistency()

    def _validate_composition_consistency(self) -> None:
        if self.research is None:
            return
        method = self.research.composition_method
        if method == "equal" and self.composition.method != "equal":
            raise ValueError(
                "composition_method='equal' requires composition.method='equal'."
            )
        if method == "fixed" and self.composition.method != "fixed":
            raise ValueError(
                "composition_method='fixed' requires composition.method='fixed'."
            )
        if method == "rolling_ic" and self.rolling.metric != "ic":
            raise ValueError(
                "composition_method='rolling_ic' requires rolling.metric='ic'."
            )
        if method == "rolling_rank_ic" and self.rolling.metric != "rank_ic":
            raise ValueError(
                "composition_method='rolling_rank_ic' requires "
                "rolling.metric='rank_ic'."
            )
        if method == "none" and self.research.evaluate_composite:
            raise ValueError(
                "composition_method='none' requires evaluate_composite=False."
            )
        if method in {"rolling_ic", "rolling_rank_ic"} and not (
            self.research.evaluate_components
        ):
            raise ValueError(
                "Rolling composition requires evaluate_components=True."
            )

    @property
    def score_col(self) -> str:
        """Return the score column G4B should consume for the active method."""
        if self.research is not None and self.research.composition_method in {
            "rolling_ic",
            "rolling_rank_ic",
        }:
            return self.rolling.score_col
        return self.composition.score_col

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FactorResearchPipelineConfig":
        """Parse a detached nested mapping into existing component configs."""
        if not isinstance(data, Mapping):
            raise TypeError("factor_research configuration must be a Mapping.")
        values = dict(data)
        unknown = sorted(set(values) - _CONFIG_FIELDS)
        if unknown:
            raise ValueError(
                "Unknown factor_research configuration keys: "
                + ", ".join(unknown)
                + "."
            )

        if "research" in values and values["research"] is not None:
            values["research"] = _parse_nested(
                values["research"],
                FactorResearchConfig,
                "research",
                conversions={"factor_names": _tuple_value},
            )
        nested = {
            "preprocessing": (PreprocessingConfig, None),
            "neutralization": (
                NeutralizationConfig,
                {"size_exempt_factors": _tuple_value},
            ),
            "evaluation": (FactorEvaluationConfig, None),
            "quantile": (QuantileEvaluationConfig, None),
            "composition": (
                FactorCompositionConfig,
                {"fixed_weights": _fixed_weights},
            ),
            "rolling": (RollingICWeightConfig, None),
            "forward_returns": (ForwardReturnConfig, None),
            "artifacts": (ResearchArtifactConfig, None),
        }
        for field_name, (expected_type, conversions) in nested.items():
            if field_name in values:
                values[field_name] = _parse_nested(
                    values[field_name],
                    expected_type,
                    field_name,
                    conversions=conversions,
                )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached YAML/JSON-safe dictionary with stable lists."""
        result = {
            "enabled": self.enabled,
            "factor_input_path": self.factor_input_path,
            "score_panel_path": self.score_panel_path,
            "price_panel_path": self.price_panel_path,
            "exposure_panel_path": self.exposure_panel_path,
            "artifact_subdir": self.artifact_subdir,
            "research": self.research.to_dict() if self.research else None,
            "preprocessing": self.preprocessing.to_dict(),
            "neutralization": self.neutralization.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "quantile": self.quantile.to_dict(),
            "composition": self.composition.to_dict(),
            "rolling": self.rolling.to_dict(),
            "forward_returns": self.forward_returns.to_dict(),
            "artifacts": self.artifacts.to_dict(),
        }
        return _json_safe(result)
