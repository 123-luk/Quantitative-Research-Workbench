"""Fail-closed error hierarchy for portfolio construction."""


class PortfolioConstructionError(ValueError):
    """Base error for portfolio-construction operations."""


class PortfolioConstructionConfigError(PortfolioConstructionError):
    """Raised when public or strategy-specific configuration is invalid."""


class PortfolioConstructionRegistryError(PortfolioConstructionError):
    """Raised when a registry operation is invalid."""


class PortfolioConstructionConstraintError(PortfolioConstructionError):
    """Raised when a constraint is invalid, unsupported, or violated."""


class PortfolioConstructionDataError(PortfolioConstructionError):
    """Raised when required risk data is missing or invalid."""


class PortfolioConstructionValidationError(PortfolioConstructionError):
    """Raised when a constructor result violates the engine contract."""
