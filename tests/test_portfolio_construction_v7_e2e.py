"""Release-level V7 portfolio-construction E2E and compatibility gates."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

from src.holdings import HOLDINGS_OUTPUT_COLUMNS, HoldingsBuilder, HoldingsDataError
from src.pipeline import PipelineConfig
from src.pipeline.holdings_config import HoldingsPipelineConfig
from src.portfolio_construction import (
    ConstraintSpec,
    PortfolioConstructionConfig,
    PortfolioConstructionConstraintError,
    PortfolioConstructionDataError,
    PortfolioConstructionEngine,
    PortfolioConstructionRequest,
    PortfolioConstructionServices,
)
from src.portfolio_construction.adapters import (
    ResearchBacktestHistoricalReturnService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _signals(count: int = 25, *, trade_date: str = "2024-01-10") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.Series(
                [pd.Timestamp(trade_date)] * count, dtype="datetime64[ns]"
            ),
            "ts_code": pd.Series(
                [f"S{index:02d}.SZ" for index in range(1, count + 1)],
                dtype="string",
            ),
            "score": np.arange(count, 0, -1, dtype=np.float64),
            "rank": np.arange(1, count + 1, dtype=np.int64),
        }
    )


def _old_holdings_config(top_n: int) -> HoldingsPipelineConfig:
    return HoldingsPipelineConfig.from_dict(
        {
            "enabled": True,
            "top_n": top_n,
            "insufficient_universe_policy": "error",
            "weighting": "equal_weight",
            "artifact_subdir": "holdings",
        }
    )


@pytest.mark.parametrize("top_n", [1, 5, 10, 20])
def test_v07_holdings_config_is_exact_equal_weight_compatible(top_n: int) -> None:
    config = _old_holdings_config(top_n)
    actual = HoldingsBuilder().build(
        _signals(),
        top_n=config.top_n,
        insufficient_universe_policy=config.insufficient_universe_policy,
        weighting=config.weighting,
        portfolio_construction=config.portfolio_construction,
    ).holdings
    expected = _signals().iloc[:top_n].copy(deep=True)
    expected.insert(
        2,
        "target_weight",
        np.full(top_n, 1.0 / top_n, dtype=np.float64),
    )
    expected = expected.loc[:, list(HOLDINGS_OUTPUT_COLUMNS)]
    pdt.assert_frame_equal(actual, expected, atol=1e-12, rtol=0.0)
    assert config.portfolio_construction.to_dict() == {
        "method": "equal_weight",
        "params": {},
        "constraints": [],
    }


def test_v07_insufficient_universe_still_fails_without_partial_result() -> None:
    with pytest.raises(HoldingsDataError, match="insufficient universe"):
        HoldingsBuilder().build(
            _signals(4),
            top_n=5,
            insufficient_universe_policy="error",
            weighting="equal_weight",
        )


def test_example_and_all_method_configs_roundtrip_without_mutation() -> None:
    path = PROJECT_ROOT / "config" / "portfolio_construction_pipeline.example.yaml"
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    original = deepcopy(source)
    parsed = PipelineConfig.from_dict(source)
    assert source == original
    assert parsed.holdings.portfolio_construction.to_dict() == {
        "method": "inverse_volatility",
        "params": {
            "lookback_trading_days": 60,
            "min_observations": 40,
        },
        "constraints": [
            {"type": "max_weight", "params": {"max_weight": 0.2}}
        ],
    }
    assert PipelineConfig.from_dict(parsed.to_dict()) == parsed
    json.dumps(parsed.to_dict(), allow_nan=False)
    assert "portfolio_construction" not in parsed.to_dict()

    old_values = deepcopy(original)
    del old_values["holdings"]["portfolio_construction"]
    old_snapshot = deepcopy(old_values)
    old_pipeline = PipelineConfig.from_dict(old_values)
    assert old_values == old_snapshot
    assert old_pipeline.holdings.portfolio_construction.to_dict() == {
        "method": "equal_weight",
        "params": {},
        "constraints": [],
    }
    json.dumps(old_pipeline.to_dict(), allow_nan=False)

    for method in ("equal_weight", "rank_weight"):
        values = deepcopy(original)
        values["holdings"]["portfolio_construction"] = {
            "method": method,
            "params": {},
            "constraints": [],
        }
        roundtrip = PipelineConfig.from_dict(values)
        assert roundtrip.holdings.portfolio_construction.method == method
        assert dict(roundtrip.holdings.portfolio_construction.params) == {}
        assert PipelineConfig.from_dict(roundtrip.to_dict()) == roundtrip


def _request(
    *, score_scale: float = 1.0, ranks: tuple[int, ...] = (2, 7, 20, 50, 99)
) -> PortfolioConstructionRequest:
    count = len(ranks)
    return PortfolioConstructionRequest(
        "2024-01-10",
        pd.DataFrame(
            {
                "ts_code": [f"S{index}" for index in range(1, count + 1)],
                "score": score_scale * np.arange(count, 0, -1, dtype=float),
                "rank": ranks,
                "selection_position": np.arange(1, count + 1, dtype=np.int64),
            }
        ),
    )


def test_rank_weight_uses_selected_ordinal_not_score_scale_or_raw_rank() -> None:
    engine = PortfolioConstructionEngine()
    config = PortfolioConstructionConfig("rank_weight", {})
    first = engine.construct(_request(), config).weights
    second = engine.construct(_request(score_scale=1_000_000.0), config).weights
    pdt.assert_frame_equal(first, second)
    np.testing.assert_allclose(
        first["target_weight"], np.asarray([5, 4, 3, 2, 1]) / 15.0
    )


def test_max_weight_release_numerics_and_infeasibility() -> None:
    engine = PortfolioConstructionEngine()
    equal = engine.construct(
        _request(),
        PortfolioConstructionConfig(
            "equal_weight",
            {},
            (ConstraintSpec("max_weight", {"max_weight": 0.25}),),
        ),
    ).weights["target_weight"].to_numpy()
    np.testing.assert_allclose(equal, np.full(5, 0.2))

    ranked = engine.construct(
        _request(),
        PortfolioConstructionConfig(
            "rank_weight",
            {},
            (ConstraintSpec("max_weight", {"max_weight": 0.25}),),
        ),
    ).weights["target_weight"].to_numpy()
    np.testing.assert_allclose(ranked, [0.25, 0.25, 0.25, 1.0 / 6.0, 1.0 / 12.0])
    assert ranked.sum() == pytest.approx(1.0, abs=1e-12)
    assert ranked.min() >= 0.0 and ranked.max() <= 0.25
    assert ranked[-2] / ranked[-1] == pytest.approx(2.0)

    with pytest.raises(PortfolioConstructionConstraintError, match="infeasible"):
        engine.construct(
            _request(),
            PortfolioConstructionConfig(
                "rank_weight",
                {},
                (ConstraintSpec("max_weight", {"max_weight": 0.19}),),
            ),
        )


class _MarketClient:
    def __init__(
        self,
        returns: dict[str, dict[str, float]],
        *,
        delisted: frozenset[str] = frozenset(),
    ) -> None:
        self.returns = returns
        self.delisted = delisted
        self.daily_ends: list[str] = []

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.date_range(start_date, end_date, freq="D")
        return pd.DataFrame(
            {
                "cal_date": dates.strftime("%Y%m%d"),
                "is_open": [int(item.weekday() < 5) for item in dates],
            }
        )

    def get_daily(
        self,
        ts_code=None,
        trade_date=None,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
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
        codes = sorted(self.returns)
        if list_status == "L":
            codes = [code for code in codes if code not in self.delisted]
        elif list_status == "D":
            codes = [code for code in codes if code in self.delisted]
        else:
            codes = []
        return pd.DataFrame(
            [
                {
                    "ts_code": code,
                    "list_status": list_status,
                    "list_date": "20200101",
                    "delist_date": "20240108" if list_status == "D" else None,
                }
                for code in codes
            ],
            columns=["ts_code", "list_status", "list_date", "delist_date"],
        )

    def get_suspend_d(self, **kwargs) -> pd.DataFrame:
        del kwargs
        return pd.DataFrame(
            columns=["ts_code", "trade_date", "suspend_timing", "suspend_type"]
        )


def _risk_returns(t_plus_one: float) -> dict[str, dict[str, float]]:
    dates = ("20240104", "20240105", "20240108", "20240109", "20240110")
    a = (1.0, -1.0, 2.0, -2.0, 1.0)
    b = tuple(2.0 * item for item in a)
    return {
        "A": dict(zip((*dates, "20240111"), (*a, t_plus_one), strict=True)),
        "B": dict(zip(dates, b, strict=True)),
    }


def _inverse_holdings(client: _MarketClient) -> pd.DataFrame:
    engine = PortfolioConstructionEngine(
        services=PortfolioConstructionServices(
            historical_returns=ResearchBacktestHistoricalReturnService(client)
        )
    )
    signals = _signals(2)
    signals["ts_code"] = pd.Series(["A", "B"], dtype="string")
    return HoldingsBuilder(engine).build(
        signals,
        top_n=2,
        insufficient_universe_policy="error",
        weighting="equal_weight",
        portfolio_construction=PortfolioConstructionConfig(
            "inverse_volatility",
            {"lookback_trading_days": 5, "min_observations": 5},
        ),
    ).holdings


def test_t_plus_one_cannot_change_inverse_volatility_holdings() -> None:
    clients = [_MarketClient(_risk_returns(value)) for value in (100.0, -100.0)]
    holdings = [_inverse_holdings(client) for client in clients]
    pdt.assert_frame_equal(holdings[0], holdings[1])
    np.testing.assert_allclose(holdings[0]["target_weight"], [2.0 / 3.0, 1.0 / 3.0])
    assert all(end == "20240110" for client in clients for end in client.daily_ends)


def test_concrete_service_fails_closed_for_unresolved_post_delist() -> None:
    returns = _risk_returns(0.0)
    returns["A"] = {
        date: value for date, value in returns["A"].items() if date <= "20240108"
    }
    service = ResearchBacktestHistoricalReturnService(
        _MarketClient(returns, delisted=frozenset({"A"}))
    )
    with pytest.raises(PortfolioConstructionDataError):
        service.load_window(("A", "B"), pd.Timestamp("2024-01-10"), 5)


def test_factor_context_is_projected_away_before_portfolio_construction() -> None:
    base = _signals(5)
    small_context = base.assign(value_factor=1.0)
    extra = pd.DataFrame(
        {
            f"factor_{index}": np.full(len(base), float(index))
            for index in range(100)
        }
    )
    large_context = pd.concat([base, extra], axis=1)
    columns = ["ts_code", "score", "rank"]
    snapshots = []
    for context in (small_context, large_context):
        snapshot = context.loc[:, columns].copy(deep=True)
        snapshot["selection_position"] = np.arange(1, 6, dtype=np.int64)
        snapshots.append(snapshot)
    engine = PortfolioConstructionEngine()
    results = [
        engine.construct(
            PortfolioConstructionRequest("2024-01-10", snapshot),
            PortfolioConstructionConfig("rank_weight", {}),
        ).weights
        for snapshot in snapshots
    ]
    pdt.assert_frame_equal(results[0], results[1])
    json.dumps(
        engine.construct(
            PortfolioConstructionRequest("2024-01-10", snapshots[0]),
            PortfolioConstructionConfig("rank_weight", {}),
        ).diagnostics,
        allow_nan=False,
    )
