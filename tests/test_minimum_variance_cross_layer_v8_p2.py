"""Gate H no-network cross-layer tests for minimum-variance Holdings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.holdings import HOLDINGS_OUTPUT_COLUMNS, HoldingsBuilder
from src.pipeline.runner import _portfolio_engine
from src.portfolio_construction import (
    ConstraintSpec,
    PortfolioConstructionConfig,
)
from src.portfolio_construction.adapters import ResearchBacktestHistoricalReturnService
from src.risk_model import (
    HistoricalCovarianceRiskModelService,
    RiskModelConfig,
    RiskModelRequest,
    RiskModelDataError,
)


@dataclass
class MarketClient:
    returns: dict[str, dict[str, float]]
    list_dates: dict[str, str]
    suspensions: list[dict[str, object]] = field(default_factory=list)
    daily_ends: list[str] = field(default_factory=list)

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.date_range(start_date, end_date, freq="D")
        return pd.DataFrame(
            {
                "cal_date": dates.strftime("%Y%m%d"),
                "is_open": [int(date.weekday() < 5) for date in dates],
            }
        )

    def get_daily(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
        del trade_date
        self.daily_ends.append(end_date)
        return pd.DataFrame(
            [
                {"trade_date": date, "ts_code": ts_code, "pct_chg": value}
                for date, value in self.returns.get(ts_code, {}).items()
                if start_date <= date <= end_date
            ],
            columns=["trade_date", "ts_code", "pct_chg"],
        )

    def get_stock_basic(self, list_status="L") -> pd.DataFrame:
        if list_status != "L":
            return pd.DataFrame(
                columns=["ts_code", "list_status", "list_date", "delist_date"]
            )
        return pd.DataFrame(
            [
                {
                    "ts_code": code,
                    "list_status": "L",
                    "list_date": date,
                    "delist_date": None,
                }
                for code, date in self.list_dates.items()
            ]
        )

    def get_suspend_d(self, ts_code=None, trade_date=None, start_date=None, end_date=None, suspend_type=None):
        del trade_date, suspend_type
        return pd.DataFrame(
            [
                row for row in self.suspensions
                if row["ts_code"] == ts_code
                and start_date <= row["trade_date"] <= end_date
            ]
        )


DATES = ("20240104", "20240105", "20240108", "20240109", "20240110")


def return_data(count: int, *, future: float = 100.0) -> dict[str, dict[str, float]]:
    result = {}
    for asset in range(1, count + 1):
        code = f"S{asset:02d}.SZ"
        values = [
            float(((asset + day) % 5) - 2) + 0.1 * asset * day
            for day in range(1, 6)
        ]
        result[code] = dict(zip((*DATES, "20240111"), (*values, future), strict=True))
    return result


def signals(count: int, formation: str = "2024-01-10") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.Series([pd.Timestamp(formation)] * count, dtype="datetime64[ns]"),
            "ts_code": pd.Series([f"S{i:02d}.SZ" for i in range(1, count + 1)], dtype="string"),
            "score": np.arange(count, 0, -1, dtype=np.float64),
            "rank": np.arange(1, count + 1, dtype=np.int64),
        }
    )


def portfolio(*, minimum: int = 5, cap: float | None = None) -> PortfolioConstructionConfig:
    constraints = () if cap is None else (
        ConstraintSpec("max_weight", {"max_weight": cap}),
    )
    return PortfolioConstructionConfig(
        "minimum_variance",
        {"risk_model": {
            "estimator": "ledoit_wolf",
            "params": {},
            "lookback_trading_days": 5,
            "min_observations": minimum,
        }},
        constraints,
    )


def holdings(client: MarketClient, *, top_n: int, cap: float | None = None, formation: str = "2024-01-10"):
    engine, actual = _portfolio_engine(
        "minimum_variance", lambda: client, shared_client=None
    )
    assert actual is client
    return HoldingsBuilder(engine).build(
        signals(max(10, top_n), formation),
        top_n=top_n,
        insufficient_universe_policy="error",
        weighting="equal_weight",
        portfolio_construction=portfolio(cap=cap),
    ).holdings


@pytest.mark.parametrize("top_n", [5, 10])
def test_top_n_owns_selection_and_minimum_variance_only_maps_weights(top_n: int) -> None:
    client = MarketClient(
        return_data(10), {f"S{i:02d}.SZ": "20200101" for i in range(1, 11)}
    )
    result = holdings(client, top_n=top_n)
    assert tuple(result.columns) == HOLDINGS_OUTPUT_COLUMNS
    assert tuple(result.ts_code) == tuple(signals(10).head(top_n).ts_code)
    assert result.target_weight.sum() == pytest.approx(1.0, abs=1e-12)
    assert bool((result.target_weight >= 0.0).all())


def test_max_weight_is_respected_without_changing_selected_set() -> None:
    client = MarketClient(
        return_data(10), {f"S{i:02d}.SZ": "20200101" for i in range(1, 11)}
    )
    result = holdings(client, top_n=5, cap=0.25)
    assert tuple(result.ts_code) == tuple(signals(10).head(5).ts_code)
    assert result.target_weight.max() <= 0.25 + 1e-12


def test_t_plus_one_perturbation_cannot_change_holdings() -> None:
    clients = [
        MarketClient(
            return_data(5, future=future),
            {f"S{i:02d}.SZ": "20200101" for i in range(1, 6)},
        )
        for future in (100.0, -100.0)
    ]
    results = [holdings(client, top_n=5) for client in clients]
    pdt.assert_frame_equal(results[0], results[1])
    assert all(end == "20240110" for client in clients for end in client.daily_ends)


def test_prelisting_uses_common_intersection_and_insufficient_rows_fail() -> None:
    data = return_data(2)
    data["S02.SZ"] = {
        date: value for date, value in data["S02.SZ"].items() if date >= "20240108"
    }
    client = MarketClient(data, {"S01.SZ": "20200101", "S02.SZ": "20240108"})
    historical = ResearchBacktestHistoricalReturnService(client)
    risk = HistoricalCovarianceRiskModelService(historical)
    result = risk.estimate(RiskModelRequest(
        "2024-01-10", ("S01.SZ", "S02.SZ"),
        RiskModelConfig("ledoit_wolf", {}, 5, 3),
    ))
    assert result.observation_count == 3
    with pytest.raises(ValueError, match="insufficient common"):
        risk.estimate(RiskModelRequest(
            "2024-01-10", ("S01.SZ", "S02.SZ"),
            RiskModelConfig("ledoit_wolf", {}, 5, 4),
        ))


def test_suspension_zero_remains_common_observation() -> None:
    data = return_data(2)
    del data["S01.SZ"]["20240109"]
    client = MarketClient(
        data,
        {"S01.SZ": "20200101", "S02.SZ": "20200101"},
        [{"trade_date": "20240109", "ts_code": "S01.SZ", "suspend_type": "S", "suspend_timing": None}],
    )
    result = HistoricalCovarianceRiskModelService(
        ResearchBacktestHistoricalReturnService(client)
    ).estimate(RiskModelRequest(
        "2024-01-10", ("S01.SZ", "S02.SZ"),
        RiskModelConfig("ledoit_wolf", {}, 5, 5),
    ))
    assert result.observation_count == 5


def test_active_unknown_missing_fails_closed_through_holdings() -> None:
    data = return_data(5)
    del data["S01.SZ"]["20240109"]
    client = MarketClient(
        data, {f"S{i:02d}.SZ": "20200101" for i in range(1, 6)}
    )
    with pytest.raises(RiskModelDataError):
        holdings(client, top_n=5)


def test_weekend_formation_uses_previous_open_cutoff() -> None:
    data = return_data(5)
    for code, values in data.items():
        values["20240112"] = values["20240110"] + 0.5
    client = MarketClient(
        data, {f"S{i:02d}.SZ": "20200101" for i in range(1, 6)}
    )
    result = holdings(client, top_n=5, formation="2024-01-13")
    assert len(result) == 5
    assert all(end == "20240112" for end in client.daily_ends)
