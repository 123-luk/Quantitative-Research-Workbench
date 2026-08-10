"""Canonical SciPy SLSQP backend."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .contracts import MinimumVarianceProblem, OptimizationResult
from .errors import PortfolioOptimizationValidationError


class ScipySLSQPBackend:
    METHOD = "SLSQP"
    FTOL = 1e-12
    MAXITER = 1000

    def solve(self, problem: MinimumVarianceProblem) -> OptimizationResult:
        if not isinstance(problem, MinimumVarianceProblem):
            raise PortfolioOptimizationValidationError("problem must be MinimumVarianceProblem.")
        result = minimize(
            problem.objective,
            problem.initial_weights,
            jac=problem.gradient,
            method=self.METHOD,
            bounds=list(zip(problem.lower_bounds, problem.upper_bounds, strict=True)),
            constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0), "jac": lambda weights: np.ones_like(weights)},
            options={"ftol": self.FTOL, "maxiter": self.MAXITER, "disp": False},
        )
        return OptimizationResult(
            weights=result.x,
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            objective_value=float(result.fun),
            iterations=int(result.nit),
            diagnostics={"method": self.METHOD, "ftol": self.FTOL, "maxiter": self.MAXITER, "disp": False},
        )
