"""Backend-neutral portfolio optimization API."""

from .contracts import MinimumVarianceProblem, OptimizationResult, OptimizerBackend
from .errors import PortfolioOptimizationError, PortfolioOptimizationSolveError, PortfolioOptimizationValidationError
from .scipy_slsqp import ScipySLSQPBackend

__all__ = [
    "MinimumVarianceProblem", "OptimizationResult", "OptimizerBackend",
    "PortfolioOptimizationError", "PortfolioOptimizationSolveError",
    "PortfolioOptimizationValidationError", "ScipySLSQPBackend",
]
