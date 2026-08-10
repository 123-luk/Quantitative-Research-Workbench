"""Inverse-sample-volatility constructor over injected resolved returns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..contracts import (
    HistoricalReturnWindow,
    PortfolioConstructionRequest,
    PortfolioConstructionServices,
    StrategyConstructionOutput,
)
from ..errors import PortfolioConstructionConfigError, PortfolioConstructionDataError
from ..registry import ResolvedConstraint
from ..risk.volatility import SampleVolatilityEstimator
from .common import proportional_output


@dataclass(frozen=True)
class InverseVolatilityParams:
    lookback_trading_days: int
    min_observations: int


class InverseVolatilityStrategy:
    name = "inverse_volatility"
    supported_constraint_types = frozenset({"max_weight"})

    def parse_params(
        self, raw_params: Mapping[str, object]
    ) -> InverseVolatilityParams:
        expected = {"lookback_trading_days", "min_observations"}
        if not isinstance(raw_params, Mapping) or set(raw_params) != expected:
            raise PortfolioConstructionConfigError(
                "inverse_volatility params must contain exactly "
                "lookback_trading_days and min_observations."
            )
        lookback = raw_params["lookback_trading_days"]
        minimum = raw_params["min_observations"]
        if type(lookback) is not int or type(minimum) is not int:
            raise PortfolioConstructionConfigError(
                "inverse_volatility params must be strict ints."
            )
        if lookback < 2 or minimum < 2 or minimum > lookback:
            raise PortfolioConstructionConfigError(
                "inverse_volatility requires lookback >= 2 and "
                "2 <= min_observations <= lookback."
            )
        return InverseVolatilityParams(lookback, minimum)

    def construct(
        self,
        request: PortfolioConstructionRequest,
        parsed_params: object,
        constraints: tuple[ResolvedConstraint, ...],
        services: PortfolioConstructionServices,
    ) -> StrategyConstructionOutput:
        if not isinstance(parsed_params, InverseVolatilityParams):
            raise PortfolioConstructionConfigError(
                "parsed inverse_volatility params are invalid."
            )
        service = services.historical_returns
        if service is None or not callable(getattr(service, "load_window", None)):
            raise PortfolioConstructionDataError(
                "inverse_volatility requires HistoricalReturnService."
            )
        try:
            window = service.load_window(
                request.ts_codes,
                request.formation_date,
                parsed_params.lookback_trading_days,
            )
        except PortfolioConstructionDataError:
            raise
        except Exception as exc:
            raise PortfolioConstructionDataError(
                "historical return service failed."
            ) from exc
        if not isinstance(window, HistoricalReturnWindow):
            raise PortfolioConstructionDataError(
                "historical return service must return HistoricalReturnWindow."
            )
        if window.risk_cutoff > request.formation_date:
            raise PortfolioConstructionDataError(
                "risk_cutoff must not exceed formation_date."
            )
        returns = window.returns
        if (
            not returns.empty
            and bool((returns["trade_date"] > window.risk_cutoff).any())
        ) or (
            not returns.empty
            and bool((returns["trade_date"] > request.formation_date).any())
        ):
            raise PortfolioConstructionDataError(
                "historical return window contains future rows."
            )
        observed_codes = set(str(item) for item in returns["ts_code"])
        if not observed_codes.issubset(set(request.ts_codes)):
            raise PortfolioConstructionDataError(
                "historical return window contains unrequested securities."
            )
        counts = returns.groupby("ts_code", sort=False).size()
        if not counts.empty and bool(
            (counts > parsed_params.lookback_trading_days).any()
        ):
            raise PortfolioConstructionDataError(
                "historical return window exceeds lookback_trading_days."
            )
        estimate = SampleVolatilityEstimator().estimate(
            window,
            request.ts_codes,
            min_observations=parsed_params.min_observations,
        )
        volatilities = estimate.volatility_dict()
        raw = np.asarray(
            [1.0 / volatilities[code] for code in request.ts_codes],
            dtype=np.float64,
        )
        diagnostics = {
            "risk_cutoff": window.risk_cutoff.strftime("%Y-%m-%d"),
            "observation_counts": estimate.observation_count_dict(),
            "volatility": volatilities,
        }
        return proportional_output(
            request, raw, constraints, diagnostics=diagnostics
        )
