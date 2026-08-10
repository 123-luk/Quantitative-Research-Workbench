"""Typed max-weight constraint parser and validator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from ..contracts import WEIGHT_ABSOLUTE_TOLERANCE, PortfolioConstructionRequest
from ..errors import PortfolioConstructionConstraintError


@dataclass(frozen=True)
class MaxWeightParams:
    max_weight: float


class MaxWeightConstraint:
    """Validate and verify a hard upper bound on every target weight."""

    name = "max_weight"

    def parse_params(self, raw_params: Mapping[str, object]) -> MaxWeightParams:
        if not isinstance(raw_params, Mapping) or set(raw_params) != {"max_weight"}:
            raise PortfolioConstructionConstraintError(
                "max_weight params must contain exactly 'max_weight'."
            )
        value = raw_params["max_weight"]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise PortfolioConstructionConstraintError(
                "max_weight must be finite numeric data."
            )
        normalized = float(value)
        if not np.isfinite(normalized) or not 0.0 < normalized <= 1.0:
            raise PortfolioConstructionConstraintError(
                "max_weight must satisfy 0 < max_weight <= 1."
            )
        return MaxWeightParams(normalized)

    def validate(
        self,
        request: PortfolioConstructionRequest,
        weights: pd.DataFrame,
        parsed_params: object,
    ) -> None:
        if not isinstance(parsed_params, MaxWeightParams):
            raise PortfolioConstructionConstraintError(
                "parsed max_weight params are invalid."
            )
        if (
            len(request.ts_codes) * parsed_params.max_weight
            < 1.0 - WEIGHT_ABSOLUTE_TOLERANCE
        ):
            raise PortfolioConstructionConstraintError(
                "max_weight is infeasible for the candidate count."
            )
        if (
            float(weights["target_weight"].max())
            > parsed_params.max_weight + WEIGHT_ABSOLUTE_TOLERANCE
        ):
            raise PortfolioConstructionConstraintError(
                "constructor result violates max_weight."
            )
