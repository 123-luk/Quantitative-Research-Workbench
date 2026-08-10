"""Registry-driven portfolio-construction orchestration and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import (
    WEIGHT_ABSOLUTE_TOLERANCE,
    WEIGHT_COLUMNS,
    PortfolioConstructionConfig,
    PortfolioConstructionRequest,
    PortfolioConstructionResult,
    PortfolioConstructionServices,
    StrategyConstructionOutput,
)
from .errors import (
    PortfolioConstructionConstraintError,
    PortfolioConstructionValidationError,
)
from .registry import (
    ConstraintRegistry,
    PortfolioConstructionRegistry,
    ResolvedConstraint,
    build_default_constraint_registry,
    build_default_portfolio_construction_registry,
)


class PortfolioConstructionEngine:
    """Resolve plugins, call one constructor, and fail closed on invalid output."""

    def __init__(
        self,
        strategy_registry: PortfolioConstructionRegistry | None = None,
        constraint_registry: ConstraintRegistry | None = None,
        services: PortfolioConstructionServices | None = None,
    ) -> None:
        self._strategies = (
            build_default_portfolio_construction_registry()
            if strategy_registry is None
            else strategy_registry
        )
        self._constraints = (
            build_default_constraint_registry()
            if constraint_registry is None
            else constraint_registry
        )
        self._services = (
            PortfolioConstructionServices() if services is None else services
        )
        if not isinstance(self._strategies, PortfolioConstructionRegistry):
            raise PortfolioConstructionValidationError(
                "strategy_registry must be PortfolioConstructionRegistry."
            )
        if not isinstance(self._constraints, ConstraintRegistry):
            raise PortfolioConstructionValidationError(
                "constraint_registry must be ConstraintRegistry."
            )
        if not isinstance(self._services, PortfolioConstructionServices):
            raise PortfolioConstructionValidationError(
                "services must be PortfolioConstructionServices."
            )

    def construct(
        self,
        request: PortfolioConstructionRequest,
        config: PortfolioConstructionConfig,
    ) -> PortfolioConstructionResult:
        if not isinstance(request, PortfolioConstructionRequest):
            raise PortfolioConstructionValidationError(
                "request must be PortfolioConstructionRequest."
            )
        if not isinstance(config, PortfolioConstructionConfig):
            raise PortfolioConstructionValidationError(
                "config must be PortfolioConstructionConfig."
            )
        strategy = self._strategies.resolve(config.method)
        parsed_params = strategy.parse_params(config.params)
        resolved = tuple(self._constraints.parse(item) for item in config.constraints)
        self._supported(strategy.supported_constraint_types, resolved)
        output = strategy.construct(
            request, parsed_params, resolved, self._services
        )
        weights = self._validated_weights(request, output)
        for constraint in resolved:
            constraint.plugin.validate(request, weights, constraint.params)
        diagnostics = {
            "method": config.method,
            "candidate_count": len(request.ts_codes),
            "constraints": [item.to_dict() for item in config.constraints],
            "strategy": output.diagnostics,
        }
        return PortfolioConstructionResult(weights, diagnostics)

    @staticmethod
    def _supported(
        supported: frozenset[str], constraints: tuple[ResolvedConstraint, ...]
    ) -> None:
        unsupported = tuple(
            item.type for item in constraints if item.type not in supported
        )
        if unsupported:
            raise PortfolioConstructionConstraintError(
                f"strategy does not support constraints {unsupported!r}."
            )

    @staticmethod
    def _validated_weights(
        request: PortfolioConstructionRequest,
        output: object,
    ) -> pd.DataFrame:
        if not isinstance(output, StrategyConstructionOutput):
            raise PortfolioConstructionValidationError(
                "strategy must return StrategyConstructionOutput."
            )
        frame = output.weights
        if (
            isinstance(frame.columns, pd.MultiIndex)
            or tuple(frame.columns) != WEIGHT_COLUMNS
        ):
            raise PortfolioConstructionValidationError(
                f"strategy weights must contain exactly {WEIGHT_COLUMNS!r}."
            )
        if len(frame) != len(request.ts_codes):
            raise PortfolioConstructionValidationError(
                "strategy must return exactly one row per candidate."
            )
        codes = frame["ts_code"]
        if (
            codes.isna().any()
            or not codes.map(lambda item: isinstance(item, (str, np.str_))).all()
            or codes.duplicated().any()
        ):
            raise PortfolioConstructionValidationError(
                "strategy ts_code values must be unique strings."
            )
        actual = tuple(str(item) for item in codes)
        if set(actual) != set(request.ts_codes):
            raise PortfolioConstructionValidationError(
                "strategy security set must exactly equal the candidate set."
            )
        values = frame["target_weight"]
        if pd.api.types.is_bool_dtype(
            values.dtype
        ) or not pd.api.types.is_numeric_dtype(values.dtype):
            raise PortfolioConstructionValidationError(
                "target_weight must be real numeric data."
            )
        try:
            numeric = values.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise PortfolioConstructionValidationError(
                "target_weight must be real numeric data."
            ) from exc
        if not np.isfinite(numeric).all() or bool((numeric < 0.0).any()):
            raise PortfolioConstructionValidationError(
                "target_weight must be finite and nonnegative."
            )
        if not np.isclose(
            float(numeric.sum()),
            1.0,
            rtol=0.0,
            atol=WEIGHT_ABSOLUTE_TOLERANCE,
        ):
            raise PortfolioConstructionValidationError(
                "target weights must sum to one."
            )
        normalized = pd.DataFrame(
            {"ts_code": list(actual), "target_weight": numeric},
            columns=list(WEIGHT_COLUMNS),
        )
        order = {code: position for position, code in enumerate(request.ts_codes)}
        normalized["_order"] = normalized["ts_code"].map(order)
        return normalized.sort_values(
            "_order", kind="mergesort", ignore_index=True
        ).drop(columns="_order")
