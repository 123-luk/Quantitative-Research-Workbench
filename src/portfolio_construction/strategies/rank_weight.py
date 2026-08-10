"""Ordinal rank-weight constructor using selected-set position only."""

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


class RankWeightStrategy:
    name = "rank_weight"
    supported_constraint_types = frozenset({"max_weight"})
    required_services = frozenset()

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
        candidates = request.candidates
        count = len(candidates)
        positions = candidates["selection_position"].to_numpy(dtype=np.float64)
        raw = count - positions + 1.0
        return proportional_output(request, raw, constraints)
