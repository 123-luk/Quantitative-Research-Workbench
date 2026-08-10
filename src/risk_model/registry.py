"""Fresh-instance risk-estimator registry."""

from __future__ import annotations

from .contracts import RiskEstimator
from .errors import RiskModelRegistryError


class RiskEstimatorRegistry:
    def __init__(self) -> None:
        self._estimators: dict[str, RiskEstimator] = {}

    def register(self, name: str, estimator: RiskEstimator) -> None:
        if not isinstance(name, str) or not name or name != name.strip() or getattr(estimator, "name", None) != name:
            raise RiskModelRegistryError("estimator registration requires an exact canonical name.")
        if not callable(getattr(estimator, "parse_params", None)) or not callable(getattr(estimator, "estimate", None)):
            raise RiskModelRegistryError("estimator must provide parse_params and estimate.")
        if name in self._estimators:
            raise RiskModelRegistryError(f"estimator {name!r} is already registered.")
        self._estimators[name] = estimator

    def resolve(self, name: object) -> RiskEstimator:
        if not isinstance(name, str) or not name or name != name.strip():
            raise RiskModelRegistryError("estimator name must be a non-empty trimmed string.")
        try:
            return self._estimators[name]
        except KeyError as exc:
            raise RiskModelRegistryError(f"unknown risk estimator {name!r}.") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._estimators))


def build_default_risk_estimator_registry() -> RiskEstimatorRegistry:
    from .estimators import LedoitWolfEstimator, SampleCovarianceEstimator

    registry = RiskEstimatorRegistry()
    registry.register("sample_covariance", SampleCovarianceEstimator())
    registry.register("ledoit_wolf", LedoitWolfEstimator())
    return registry
