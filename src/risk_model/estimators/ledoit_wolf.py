"""Canonical sklearn Ledoit-Wolf covariance estimator."""

from collections.abc import Mapping

import numpy as np
from sklearn.covariance import LedoitWolf

from ..contracts import RiskEstimate
from ..errors import RiskModelConfigError


class LedoitWolfEstimator:
    name = "ledoit_wolf"

    def parse_params(self, raw_params: Mapping[str, object]) -> None:
        if not isinstance(raw_params, Mapping) or raw_params:
            raise RiskModelConfigError("ledoit_wolf params must be empty.")
        return None

    def estimate(self, aligned_returns: np.ndarray, parsed_params: object) -> RiskEstimate:
        if parsed_params is not None:
            raise RiskModelConfigError("parsed ledoit_wolf params are invalid.")
        fitted = LedoitWolf(assume_centered=False, store_precision=False).fit(
            np.asarray(aligned_returns, dtype=np.float64)
        )
        return RiskEstimate(
            covariance=fitted.covariance_,
            diagnostics={"shrinkage": float(fitted.shrinkage_)},
        )
