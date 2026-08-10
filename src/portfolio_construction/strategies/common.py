"""Shared constructor helpers without strategy dispatch."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..allocation import capped_proportional_allocation
from ..constraints.max_weight import MaxWeightParams
from ..contracts import PortfolioConstructionRequest, StrategyConstructionOutput
from ..errors import (
    PortfolioConstructionConfigError,
    PortfolioConstructionConstraintError,
)
from ..registry import ResolvedConstraint


def parse_empty_params(raw_params: Mapping[str, object]) -> None:
    if not isinstance(raw_params, Mapping) or raw_params:
        raise PortfolioConstructionConfigError(
            "strategy params must be an empty object."
        )
    return None


def max_weight_value(constraints: tuple[ResolvedConstraint, ...]) -> float | None:
    matches = tuple(item for item in constraints if item.type == "max_weight")
    if not matches:
        return None
    if len(matches) != 1 or not isinstance(matches[0].params, MaxWeightParams):
        raise PortfolioConstructionConstraintError(
            "resolved max_weight constraint is invalid."
        )
    return matches[0].params.max_weight


def proportional_output(
    request: PortfolioConstructionRequest,
    raw: np.ndarray,
    constraints: tuple[ResolvedConstraint, ...],
    *,
    diagnostics: Mapping[str, object] | None = None,
) -> StrategyConstructionOutput:
    weights = capped_proportional_allocation(raw, max_weight_value(constraints))
    return StrategyConstructionOutput(
        pd.DataFrame(
            {"ts_code": list(request.ts_codes), "target_weight": weights},
            columns=["ts_code", "target_weight"],
        ),
        diagnostics=diagnostics,
    )
