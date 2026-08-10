"""Gate C tests for historical-risk contracts and inverse volatility."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from src.portfolio_construction import (
    ConstraintSpec,
    HistoricalReturnWindow,
    PortfolioConstructionConfig,
    PortfolioConstructionConfigError,
    PortfolioConstructionDataError,
    PortfolioConstructionEngine,
    PortfolioConstructionRequest,
    PortfolioConstructionServices,
    SampleVolatilityEstimator,
)


def request() -> PortfolioConstructionRequest:
    return PortfolioConstructionRequest(
        "2024-01-10",
        pd.DataFrame(
            {
                "ts_code": ["A", "B"],
                "score": [9.0, 1.0],
                "rank": [1, 99],
                "selection_position": [1, 2],
            }
        ),
    )


def returns_frame(
    *,
    a: tuple[float, ...] = (0.01, -0.01, 0.02, -0.02),
    b: tuple[float, ...] = (0.02, -0.02, 0.04, -0.04),
) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-02", periods=max(len(a), len(b)), freq="D")
    for code, values in (("A", a), ("B", b)):
        rows.extend(
            {"trade_date": date, "ts_code": code, "return": value}
            for date, value in zip(dates, values)
        )
    return pd.DataFrame(rows, columns=["trade_date", "ts_code", "return"])


@dataclass
class FakeService:
    window: object
    calls: list[tuple[tuple[str, ...], pd.Timestamp, int]] = field(default_factory=list)

    def load_window(self, ts_codes, formation_date, lookback_trading_days):
        self.calls.append((tuple(ts_codes), formation_date, lookback_trading_days))
        return self.window


def inverse_config(*, cap: float | None = None) -> PortfolioConstructionConfig:
    constraints = () if cap is None else (
        ConstraintSpec("max_weight", {"max_weight": cap}),
    )
    return PortfolioConstructionConfig(
        "inverse_volatility",
        {"lookback_trading_days": 5, "min_observations": 4},
        constraints,
    )


def test_return_window_is_sparse_sorted_defensive_and_finite() -> None:
    source = returns_frame().iloc[::-1].reset_index(drop=True)
    window = HistoricalReturnWindow("2024-01-09", source)
    assert window.risk_cutoff == pd.Timestamp("2024-01-09")
    assert window.returns.equals(
        window.returns.sort_values(
            ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
        )
    )
    exposed = window.returns
    exposed.loc[0, "return"] = 99.0
    assert window.returns.loc[0, "return"] != 99.0


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"trade_date": ["2024-01-02"], "ts_code": ["A"]}),
        pd.DataFrame(
            {
                "trade_date": ["2024-01-02", "2024-01-02"],
                "ts_code": ["A", "A"],
                "return": [0.1, 0.2],
            }
        ),
        pd.DataFrame(
            {"trade_date": ["2024-01-02"], "ts_code": ["A"], "return": [np.nan]}
        ),
    ],
)
def test_return_window_rejects_invalid_schema_keys_or_values(
    frame: pd.DataFrame,
) -> None:
    with pytest.raises(PortfolioConstructionDataError):
        HistoricalReturnWindow("2024-01-09", frame)


def test_sample_volatility_uses_ddof_one() -> None:
    window = HistoricalReturnWindow("2024-01-09", returns_frame())
    estimate = SampleVolatilityEstimator().estimate(
        window, ("A", "B"), min_observations=4
    )
    assert estimate.volatility_dict()["A"] == pytest.approx(
        np.std([0.01, -0.01, 0.02, -0.02], ddof=1)
    )
    assert estimate.observation_count_dict() == {"A": 4, "B": 4}


def test_inverse_volatility_requests_exact_scope_and_weights() -> None:
    service = FakeService(HistoricalReturnWindow("2024-01-09", returns_frame()))
    result = PortfolioConstructionEngine(
        services=PortfolioConstructionServices(historical_returns=service)
    ).construct(request(), inverse_config())
    assert service.calls == [(request().ts_codes, pd.Timestamp("2024-01-10"), 5)]
    # B has exactly twice A's sample volatility.
    np.testing.assert_allclose(result.weights["target_weight"], [2.0 / 3.0, 1.0 / 3.0])
    assert result.diagnostics["strategy"]["risk_cutoff"] == "2024-01-09"
    assert result.diagnostics["strategy"]["observation_counts"] == {"A": 4, "B": 4}


def test_inverse_volatility_combines_with_max_weight() -> None:
    service = FakeService(HistoricalReturnWindow("2024-01-09", returns_frame()))
    result = PortfolioConstructionEngine(
        services=PortfolioConstructionServices(historical_returns=service)
    ).construct(request(), inverse_config(cap=0.6))
    np.testing.assert_allclose(result.weights["target_weight"], [0.6, 0.4])


def test_inverse_volatility_requires_service() -> None:
    with pytest.raises(PortfolioConstructionDataError):
        PortfolioConstructionEngine().construct(request(), inverse_config())


@pytest.mark.parametrize(
    "params",
    [
        {"lookback_trading_days": 1, "min_observations": 2},
        {"lookback_trading_days": 5, "min_observations": 1},
        {"lookback_trading_days": 5, "min_observations": 6},
        {"lookback_trading_days": True, "min_observations": 2},
        {"lookback_trading_days": 5, "min_observations": 2, "abc": 1},
    ],
)
def test_inverse_volatility_params_are_strict(params: dict[str, object]) -> None:
    with pytest.raises(PortfolioConstructionConfigError):
        PortfolioConstructionEngine().construct(
            request(), PortfolioConstructionConfig("inverse_volatility", params)
        )


def test_inverse_volatility_rejects_insufficient_observations() -> None:
    frame = returns_frame(a=(0.01, -0.01, 0.02))
    service = FakeService(HistoricalReturnWindow("2024-01-09", frame))
    with pytest.raises(PortfolioConstructionDataError, match="insufficient"):
        PortfolioConstructionEngine(
            services=PortfolioConstructionServices(historical_returns=service)
        ).construct(request(), inverse_config())


def test_inverse_volatility_rejects_zero_volatility_without_floor_or_fallback() -> None:
    frame = returns_frame(a=(0.01, 0.01, 0.01, 0.01))
    service = FakeService(HistoricalReturnWindow("2024-01-09", frame))
    with pytest.raises(PortfolioConstructionDataError, match="positive"):
        PortfolioConstructionEngine(
            services=PortfolioConstructionServices(historical_returns=service)
        ).construct(request(), inverse_config())


@pytest.mark.parametrize(
    "cutoff,row_date",
    [("2024-01-11", "2024-01-09"), ("2024-01-09", "2024-01-10")],
)
def test_no_lookahead_guard_rejects_future_cutoff_or_row(
    cutoff: str, row_date: str
) -> None:
    frame = returns_frame()
    frame.loc[0, "trade_date"] = pd.Timestamp(row_date)
    with pytest.raises(PortfolioConstructionDataError):
        service = FakeService(HistoricalReturnWindow(cutoff, frame))
        PortfolioConstructionEngine(
            services=PortfolioConstructionServices(historical_returns=service)
        ).construct(request(), inverse_config())


def test_inverse_volatility_rejects_unrequested_security() -> None:
    frame = returns_frame()
    extra = pd.DataFrame(
        [{"trade_date": "2024-01-02", "ts_code": "C", "return": 0.1}]
    )
    service = FakeService(
        HistoricalReturnWindow(
            "2024-01-09", pd.concat([frame, extra], ignore_index=True)
        )
    )
    with pytest.raises(PortfolioConstructionDataError, match="unrequested"):
        PortfolioConstructionEngine(
            services=PortfolioConstructionServices(historical_returns=service)
        ).construct(request(), inverse_config())


def test_empty_sparse_window_is_valid_but_estimator_fails_observations() -> None:
    empty = pd.DataFrame(columns=["trade_date", "ts_code", "return"])
    service = FakeService(HistoricalReturnWindow("2024-01-09", empty))
    with pytest.raises(PortfolioConstructionDataError, match="insufficient"):
        PortfolioConstructionEngine(
            services=PortfolioConstructionServices(historical_returns=service)
        ).construct(request(), inverse_config())
