"""Read-only capability catalog backed exclusively by backend registries."""

from __future__ import annotations

from src.factors.registry import FactorRegistry, create_default_registry
from src.ml.models.base import ModelParameterSpec
from src.ml.models.registry import ModelRegistry, create_default_model_registry
from src.portfolio_construction.registry import (
    ConstraintRegistry,
    PortfolioConstructionRegistry,
    build_default_constraint_registry,
    build_default_portfolio_construction_registry,
)
from src.risk_model.registry import (
    RiskEstimatorRegistry,
    build_default_risk_estimator_registry,
)


class CapabilityCatalogService:
    """Expose fresh registry snapshots without owning duplicate capability lists."""

    def __init__(
        self,
        *,
        factor_registry: FactorRegistry | None = None,
        model_registry: ModelRegistry | None = None,
        portfolio_registry: PortfolioConstructionRegistry | None = None,
        risk_registry: RiskEstimatorRegistry | None = None,
        constraint_registry: ConstraintRegistry | None = None,
    ) -> None:
        self._factors = factor_registry or create_default_registry()
        self._models = model_registry or create_default_model_registry()
        self._portfolios = (
            portfolio_registry or build_default_portfolio_construction_registry()
        )
        self._risk = risk_registry or build_default_risk_estimator_registry()
        self._constraints = constraint_registry or build_default_constraint_registry()

    def list_factor_names(self) -> tuple[str, ...]:
        return tuple(self._factors.list_names())

    def list_model_names(self) -> tuple[str, ...]:
        return self._models.list_models()

    def get_model_parameter_schema(
        self, model_name: str
    ) -> tuple[ModelParameterSpec, ...]:
        return self._models.get_parameter_schema(model_name)

    def validate_model_parameters(
        self, model_name: str, parameters: dict[str, object]
    ) -> dict[str, object]:
        """Return fully resolved parameters after backend-owned validation."""
        return dict(self._models.create(model_name, parameters).config.as_dict())

    def list_portfolio_methods(self) -> tuple[str, ...]:
        return self._portfolios.names()

    def list_risk_estimators(self) -> tuple[str, ...]:
        return self._risk.names()

    def list_constraints(self) -> tuple[str, ...]:
        return self._constraints.names()

