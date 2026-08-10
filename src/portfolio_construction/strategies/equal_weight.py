"""Equal-weight constructor for an already selected candidate set."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..contracts import (
    PortfolioConstructionRequest,
    PortfolioConstructionServices,
    StrategyConstructionOutput,
)
from ..registry import ResolvedConstraint
from .common import parse_empty_params, proportional_output


class EqualWeightStrategy:
    name = "equal_weight"
    supported_constraint_types = frozenset({"max_weight"})

    def parse_params(self, raw_params: Mapping[str, object]) -> None:
        return parse_empty_params(raw_params)

    def construct(
        self,
        request: PortfolioConstructionRequest,
        parsed_params: object,
        constraints: tuple[ResolvedConstraint, ...],
        services: PortfolioConstructionServices,
    ) -> StrategyConstructionOutput:
        del parsed_params, services
        return proportional_output(
            request, np.ones(len(request.ts_codes), dtype=np.float64), constraints
        )
