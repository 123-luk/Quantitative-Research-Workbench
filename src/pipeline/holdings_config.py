"""Strict static configuration for the optional V5 Holdings stage."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath

from src.portfolio_construction import (
    PortfolioConstructionConfig,
    PortfolioConstructionConfigError,
)


class HoldingsConfigError(ValueError):
    """Raised when Holdings pipeline configuration is invalid."""


def _strict_mapping(
    value: object, allowed: frozenset[str], context: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise HoldingsConfigError(f"{context} must be a Mapping.")
    if any(not isinstance(key, str) for key in value):
        raise HoldingsConfigError(f"{context} field names must be strings.")
    values = deepcopy(dict(value))
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise HoldingsConfigError(
            f"{context} contains unknown fields: {unknown!r}."
        )
    return values


def _safe_artifact_subdir(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise HoldingsConfigError(
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
        raise HoldingsConfigError(
            "artifact_subdir must be one safe relative directory name."
        )
    return value


@dataclass(frozen=True)
class HoldingsPipelineConfig:
    """Configure V5 Top-N long-only equal-weight target holdings."""

    enabled: bool = False
    top_n: int = 20
    insufficient_universe_policy: str = "error"
    weighting: str = "equal_weight"
    artifact_subdir: str = "holdings"
    portfolio_construction: PortfolioConstructionConfig = field(
        default_factory=lambda: PortfolioConstructionConfig(
            "equal_weight", {}
        )
    )

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise HoldingsConfigError("enabled must be a bool.")
        if type(self.top_n) is not int:
            raise HoldingsConfigError("top_n must be a strict int.")
        if self.top_n < 1:
            raise HoldingsConfigError("top_n must be >= 1.")
        if not isinstance(self.insufficient_universe_policy, str):
            raise HoldingsConfigError(
                "insufficient_universe_policy must be a string."
            )
        policy = self.insufficient_universe_policy.strip().lower()
        if policy not in {"error", "allow_partial"}:
            raise HoldingsConfigError(
                "insufficient_universe_policy must be 'error' or "
                "'allow_partial'."
            )
        if not isinstance(self.weighting, str):
            raise HoldingsConfigError("weighting must be a string.")
        weighting = self.weighting.strip().lower()
        if weighting != "equal_weight":
            raise HoldingsConfigError(
                "weighting must be 'equal_weight' for V5."
            )
        object.__setattr__(self, "insufficient_universe_policy", policy)
        object.__setattr__(self, "weighting", weighting)
        object.__setattr__(
            self, "artifact_subdir", _safe_artifact_subdir(self.artifact_subdir)
        )
        try:
            portfolio = (
                self.portfolio_construction
                if isinstance(
                    self.portfolio_construction, PortfolioConstructionConfig
                )
                else PortfolioConstructionConfig.from_dict(
                    self.portfolio_construction
                )
            )
        except PortfolioConstructionConfigError as exc:
            raise HoldingsConfigError(
                "portfolio_construction configuration is invalid."
            ) from exc
        object.__setattr__(self, "portfolio_construction", portfolio)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | HoldingsPipelineConfig | None
    ) -> HoldingsPipelineConfig:
        """Build a detached strict Holdings config from a mapping or None."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value,
            frozenset(
                {
                    "enabled",
                    "top_n",
                    "insufficient_universe_policy",
                    "weighting",
                    "artifact_subdir",
                    "portfolio_construction",
                }
            ),
            "holdings",
        )
        try:
            return cls(**values)  # type: ignore[arg-type]
        except HoldingsConfigError:
            raise
        except (TypeError, ValueError) as exc:
            raise HoldingsConfigError(
                "holdings configuration is invalid."
            ) from exc

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-safe Holdings configuration mapping."""
        return {
            "enabled": self.enabled,
            "top_n": self.top_n,
            "insufficient_universe_policy": self.insufficient_universe_policy,
            "weighting": self.weighting,
            "artifact_subdir": self.artifact_subdir,
            "portfolio_construction": self.portfolio_construction.to_dict(),
        }
