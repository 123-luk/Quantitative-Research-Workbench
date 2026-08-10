"""Extensible portfolio-construction core public API."""

from .allocation import capped_proportional_allocation
from .constraints import MaxWeightConstraint, MaxWeightParams
from .contracts import (
    CANDIDATE_COLUMNS,
    RETURN_COLUMNS,
    WEIGHT_ABSOLUTE_TOLERANCE,
    WEIGHT_COLUMNS,
    ConstraintSpec,
    HistoricalReturnService,
    HistoricalReturnWindow,
    PortfolioConstructionConfig,
    PortfolioConstructionRequest,
    PortfolioConstructionResult,
    PortfolioConstructionServices,
    StrategyConstructionOutput,
)
from .engine import PortfolioConstructionEngine
from .errors import (
    PortfolioConstructionConfigError,
    PortfolioConstructionConstraintError,
    PortfolioConstructionDataError,
    PortfolioConstructionError,
    PortfolioConstructionRegistryError,
    PortfolioConstructionValidationError,
)
from .registry import (
    ConstraintRegistry,
    PortfolioConstructionRegistry,
    ResolvedConstraint,
    build_default_constraint_registry,
    build_default_portfolio_construction_registry,
)
from .risk import SampleVolatilityEstimator, VolatilityEstimate
from .strategies import (
    EqualWeightStrategy,
    InverseVolatilityParams,
    InverseVolatilityStrategy,
    MinimumVarianceConstructor,
    RankWeightStrategy,
)

__all__ = [
    "CANDIDATE_COLUMNS",
    "RETURN_COLUMNS",
    "WEIGHT_ABSOLUTE_TOLERANCE",
    "WEIGHT_COLUMNS",
    "ConstraintRegistry",
    "ConstraintSpec",
    "EqualWeightStrategy",
    "HistoricalReturnService",
    "HistoricalReturnWindow",
    "InverseVolatilityParams",
    "InverseVolatilityStrategy",
    "MaxWeightConstraint",
    "MaxWeightParams",
    "MinimumVarianceConstructor",
    "PortfolioConstructionConfig",
    "PortfolioConstructionConfigError",
    "PortfolioConstructionConstraintError",
    "PortfolioConstructionDataError",
    "PortfolioConstructionEngine",
    "PortfolioConstructionError",
    "PortfolioConstructionRegistry",
    "PortfolioConstructionRegistryError",
    "PortfolioConstructionRequest",
    "PortfolioConstructionResult",
    "PortfolioConstructionServices",
    "PortfolioConstructionValidationError",
    "RankWeightStrategy",
    "ResolvedConstraint",
    "SampleVolatilityEstimator",
    "StrategyConstructionOutput",
    "VolatilityEstimate",
    "build_default_constraint_registry",
    "build_default_portfolio_construction_registry",
    "capped_proportional_allocation",
]
