"""Long-only fully-invested constrained minimum-variance strategy."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.portfolio_optimization import (
    MinimumVarianceProblem,
    OptimizationResult,
    OptimizerBackend,
    ScipySLSQPBackend,
)
from src.risk_model import RiskModelConfig, RiskModelRequest, RiskModelResult

from ..constraints.max_weight import MaxWeightParams
from ..contracts import (
    WEIGHT_ABSOLUTE_TOLERANCE,
    PortfolioConstructionRequest,
    PortfolioConstructionServices,
    StrategyConstructionOutput,
)
from ..errors import (
    PortfolioConstructionConfigError,
    PortfolioConstructionConstraintError,
    PortfolioConstructionDataError,
    PortfolioConstructionValidationError,
)
from ..registry import ResolvedConstraint
from .common import max_weight_value


class MinimumVarianceConstructor:
    name = "minimum_variance"
    supported_constraint_types = frozenset({"max_weight"})
    required_services = frozenset({"risk_model"})

    def __init__(self, backend: OptimizerBackend | None = None) -> None:
        self._backend = ScipySLSQPBackend() if backend is None else backend
        if not callable(getattr(self._backend, "solve", None)):
            raise PortfolioConstructionValidationError("backend must provide solve.")

    def parse_params(self, raw_params: Mapping[str, object]) -> RiskModelConfig:
        if not isinstance(raw_params, Mapping) or set(raw_params) != {"risk_model"}:
            raise PortfolioConstructionConfigError(
                "minimum_variance params must contain exactly risk_model."
            )
        try:
            return RiskModelConfig.from_dict(raw_params["risk_model"])
        except ValueError as exc:
            raise PortfolioConstructionConfigError(str(exc)) from exc

    def construct(
        self,
        request: PortfolioConstructionRequest,
        parsed_params: object,
        constraints: tuple[ResolvedConstraint, ...],
        services: PortfolioConstructionServices,
    ) -> StrategyConstructionOutput:
        if not isinstance(parsed_params, RiskModelConfig):
            raise PortfolioConstructionConfigError("parsed minimum_variance params are invalid.")
        count = len(request.ts_codes)
        if count < 2:
            raise PortfolioConstructionDataError("minimum_variance requires at least two assets.")
        service = services.risk_model
        if service is None or not callable(getattr(service, "estimate", None)):
            raise PortfolioConstructionDataError("minimum_variance requires RiskModelService.")
        cap = max_weight_value(constraints)
        if cap is not None and count * cap < 1.0 - WEIGHT_ABSOLUTE_TOLERANCE:
            raise PortfolioConstructionConstraintError("max_weight is infeasible for the candidate count.")
        try:
            risk = service.estimate(RiskModelRequest(request.formation_date, request.ts_codes, parsed_params))
        except ValueError:
            raise
        except Exception as exc:
            raise PortfolioConstructionDataError("risk model service failed.") from exc
        if not isinstance(risk, RiskModelResult) or risk.assets != request.ts_codes:
            raise PortfolioConstructionDataError("risk result assets must exactly equal candidates in canonical order.")
        upper = np.full(count, 1.0 if cap is None else cap, dtype=np.float64)
        problem = MinimumVarianceProblem(
            covariance=risk.covariance,
            initial_weights=np.full(count, 1.0 / count, dtype=np.float64),
            lower_bounds=np.zeros(count, dtype=np.float64),
            upper_bounds=upper,
        )
        result = self._backend.solve(problem)
        weights = self._validate_result(result, count, upper)
        return StrategyConstructionOutput(
            pd.DataFrame({"ts_code": list(request.ts_codes), "target_weight": weights}),
            diagnostics={
                "risk_cutoff": risk.risk_cutoff.strftime("%Y-%m-%d"),
                "risk_estimator": risk.estimator,
                "risk_observation_count": risk.observation_count,
                "risk": risk.diagnostics,
                "optimizer": {
                    "status": result.status,
                    "message": result.message,
                    "objective_value": result.objective_value,
                    "iterations": result.iterations,
                    "diagnostics": result.diagnostics,
                },
            },
        )

    @staticmethod
    def _validate_result(result: object, count: int, upper: np.ndarray) -> np.ndarray:
        if not isinstance(result, OptimizationResult):
            raise PortfolioConstructionValidationError("backend must return OptimizationResult.")
        if not result.success:
            raise PortfolioConstructionValidationError("optimizer did not succeed.")
        weights = result.weights
        if weights.shape != (count,) or not np.isfinite(weights).all():
            raise PortfolioConstructionValidationError("optimizer weights have invalid shape or values.")
        if bool((weights < -WEIGHT_ABSOLUTE_TOLERANCE).any()):
            raise PortfolioConstructionValidationError("optimizer weights violate long-only bounds.")
        if bool((weights > upper + WEIGHT_ABSOLUTE_TOLERANCE).any()):
            raise PortfolioConstructionValidationError("optimizer weights violate upper bounds.")
        if not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=WEIGHT_ABSOLUTE_TOLERANCE):
            raise PortfolioConstructionValidationError("optimizer weights must sum to one.")
        if not np.isfinite(result.objective_value):
            raise PortfolioConstructionValidationError("optimizer objective must be finite.")
        return weights
