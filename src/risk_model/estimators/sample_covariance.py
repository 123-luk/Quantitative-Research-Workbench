"""Unannualized sample covariance over a common complete-case matrix."""

from collections.abc import Mapping

import numpy as np

from ..contracts import RiskEstimate
from ..errors import RiskModelConfigError


class SampleCovarianceEstimator:
    name = "sample_covariance"

    def parse_params(self, raw_params: Mapping[str, object]) -> None:
        if not isinstance(raw_params, Mapping) or raw_params:
            raise RiskModelConfigError("sample_covariance params must be empty.")
        return None

    def estimate(self, aligned_returns: np.ndarray, parsed_params: object) -> RiskEstimate:
        if parsed_params is not None:
            raise RiskModelConfigError("parsed sample_covariance params are invalid.")
        matrix = np.asarray(aligned_returns, dtype=np.float64)
        covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
        return RiskEstimate(covariance=covariance, diagnostics={})
