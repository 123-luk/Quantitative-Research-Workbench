"""Fail-closed portfolio-optimization errors."""


class PortfolioOptimizationError(ValueError):
    """Base optimization error."""


class PortfolioOptimizationValidationError(PortfolioOptimizationError):
    """Raised when a problem or backend result violates its contract."""


class PortfolioOptimizationSolveError(PortfolioOptimizationError):
    """Raised when a numerical solver does not succeed."""
