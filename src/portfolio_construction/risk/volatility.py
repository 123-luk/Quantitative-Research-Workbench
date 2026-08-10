"""Strict sample-volatility estimation over resolved historical returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import HistoricalReturnWindow
from ..errors import PortfolioConstructionDataError


@dataclass(frozen=True)
class VolatilityEstimate:
    """Deterministic per-security sample volatility and observation count."""

    volatilities: tuple[tuple[str, float], ...]
    observation_counts: tuple[tuple[str, int], ...]

    def volatility_dict(self) -> dict[str, float]:
        return dict(self.volatilities)

    def observation_count_dict(self) -> dict[str, int]:
        return dict(self.observation_counts)


class SampleVolatilityEstimator:
    """Estimate sample standard deviation with ``ddof=1`` and no floor."""

    def estimate(
        self,
        window: HistoricalReturnWindow,
        ts_codes: tuple[str, ...],
        *,
        min_observations: int,
    ) -> VolatilityEstimate:
        if not isinstance(window, HistoricalReturnWindow):
            raise PortfolioConstructionDataError(
                "window must be HistoricalReturnWindow."
            )
        if (
            not isinstance(ts_codes, tuple)
            or not ts_codes
            or len(ts_codes) != len(set(ts_codes))
        ):
            raise PortfolioConstructionDataError(
                "ts_codes must be a non-empty unique tuple."
            )
        if type(min_observations) is not int or min_observations < 2:
            raise PortfolioConstructionDataError(
                "min_observations must be an int >= 2."
            )
        frame = window.returns
        volatilities: list[tuple[str, float]] = []
        counts: list[tuple[str, int]] = []
        for code in ts_codes:
            values = frame.loc[frame["ts_code"].eq(code), "return"].to_numpy(
                dtype=np.float64
            )
            count = len(values)
            if count < min_observations:
                raise PortfolioConstructionDataError(
                    f"insufficient observations for ts_code={code!r}: "
                    f"required={min_observations}, observed={count}."
                )
            volatility = float(np.std(values, ddof=1))
            if not np.isfinite(volatility) or volatility <= 0.0:
                raise PortfolioConstructionDataError(
                    "sample volatility must be finite and positive for "
                    f"ts_code={code!r}."
                )
            volatilities.append((code, volatility))
            counts.append((code, count))
        return VolatilityEstimate(tuple(volatilities), tuple(counts))
