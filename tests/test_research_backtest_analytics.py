from __future__ import annotations

import json
from math import sqrt

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.pipeline.research_backtest_config import BenchmarkConfig, PerformanceConfig
from src.research_backtest import (
    BENCHMARK_DAILY_COLUMNS,
    BENCHMARK_DAILY_RETURN_COLUMNS,
    DAILY_PORTFOLIO_COLUMNS,
    PERFORMANCE_METRIC_KEYS,
    BenchmarkCalendarAlignmentError,
    BenchmarkSeriesError,
    PerformanceAnalyticsEngine,
    PerformanceAnalyticsInputError,
    PerformanceMetricError,
    PortfolioDailyAccountingResult,
)


def _portfolio(
    *,
    dates: list[str] | None = None,
    gross_returns: list[float] | None = None,
    costs: list[float] | None = None,
    rebalances: list[bool] | None = None,
    turnovers: list[float] | None = None,
    notionals: list[float] | None = None,
    initial_nav: float = 100.0,
    metadata_count: int | None = None,
) -> PortfolioDailyAccountingResult:
    dates = dates or ["2024-01-03", "2024-01-04", "2024-01-05"]
    gross_returns = gross_returns or [0.0, 0.1, -0.05]
    costs = costs or [0.001, 0.0, 0.002]
    rebalances = rebalances or [True, False, True]
    turnovers = turnovers or [1.0, 0.0, 0.5]
    notionals = notionals or [1.0, 0.0, 1.0]
    gross_nav = initial_nav
    net_nav = initial_nav
    rows: list[dict[str, object]] = []
    for date, gross, cost, flag, turnover, notional in zip(
        dates,
        gross_returns,
        costs,
        rebalances,
        turnovers,
        notionals,
    ):
        net_return = (1.0 + gross) * (1.0 - cost) - 1.0
        gross_nav *= 1.0 + gross
        net_nav *= 1.0 + net_return
        rows.append(
            {
                "trade_date": date,
                "gross_return": gross,
                "transaction_cost": cost,
                "net_return": net_return,
                "gross_nav": gross_nav,
                "net_nav": net_nav,
                "is_rebalance": flag,
                "turnover": turnover,
                "traded_notional": notional,
            }
        )
    frame = pd.DataFrame(rows, columns=list(DAILY_PORTFOLIO_COLUMNS))
    count = sum(rebalances) if metadata_count is None else metadata_count
    return PortfolioDailyAccountingResult(
        frame,
        start_date=pd.Timestamp(dates[0]),
        end_date=pd.Timestamp(dates[-1]),
        rebalance_count=count,
        initial_nav=initial_nav,
        cost_bps=10.0,
    )


def _benchmark(
    rows: list[tuple[str, str, float]] | None = None,
) -> pd.DataFrame:
    rows = rows or [
        ("2024-01-03", "TEST.IDX", 0.2),
        ("2024-01-04", "TEST.IDX", 0.05),
        ("2024-01-05", "TEST.IDX", -0.02),
    ]
    return pd.DataFrame(rows, columns=list(BENCHMARK_DAILY_RETURN_COLUMNS))


def _run(
    portfolio: PortfolioDailyAccountingResult | None = None,
    benchmark: pd.DataFrame | None = None,
    *,
    annualization_days: int = 252,
    risk_free: float = 0.0,
    code: str = "TEST.IDX",
):
    return PerformanceAnalyticsEngine(
        BenchmarkConfig(benchmark_code=code),
        PerformanceConfig(
            annual_risk_free_rate=risk_free,
            annualization_days=annualization_days,
        ),
    ).run(
        portfolio=_portfolio() if portfolio is None else portfolio,
        benchmark_returns=_benchmark() if benchmark is None else benchmark,
    )


def test_benchmark_fair_start_schema_and_nav() -> None:
    result = _run()
    daily = result.benchmark_daily
    assert tuple(daily.columns) == BENCHMARK_DAILY_COLUMNS
    assert list(daily["benchmark_return"]) == [0.0, 0.05, -0.02]
    assert daily.loc[0, "benchmark_nav"] == 100.0
    assert daily.loc[1, "benchmark_nav"] == pytest.approx(105.0)
    assert daily.loc[2, "benchmark_nav"] == pytest.approx(102.9)
    assert result.benchmark_code == "TEST.IDX"


def test_first_raw_benchmark_return_is_ignored_without_input_mutation() -> None:
    benchmark = _benchmark()
    before = benchmark.copy(deep=True)
    result = _run(benchmark=benchmark)
    assert result.benchmark_daily.loc[0, "benchmark_return"] == 0.0
    pdt.assert_frame_equal(benchmark, before)


def test_unordered_benchmark_is_canonicalized() -> None:
    expected = _run().benchmark_daily
    actual = _run(benchmark=_benchmark().iloc[::-1]).benchmark_daily
    pdt.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("missing_date", ["2024-01-04", "2024-01-05"])
def test_missing_benchmark_date_fails_closed(missing_date: str) -> None:
    rows = _benchmark()
    rows = rows.loc[~rows["trade_date"].eq(missing_date)]
    with pytest.raises(BenchmarkCalendarAlignmentError):
        _run(benchmark=rows)


def test_extra_date_inside_evaluation_window_fails_closed() -> None:
    portfolio = _portfolio(
        dates=["2024-01-03", "2024-01-05"],
        gross_returns=[0.0, 0.0],
        costs=[0.0, 0.0],
        rebalances=[True, False],
        turnovers=[1.0, 0.0],
        notionals=[1.0, 0.0],
    )
    benchmark = _benchmark(
        [
            ("2024-01-03", "TEST.IDX", 0.0),
            ("2024-01-04", "TEST.IDX", 0.0),
            ("2024-01-05", "TEST.IDX", 0.0),
        ]
    )
    with pytest.raises(BenchmarkCalendarAlignmentError):
        _run(portfolio, benchmark)


def test_dates_outside_evaluation_window_are_explicitly_cropped() -> None:
    rows = [
        ("2024-01-02", "TEST.IDX", 9.0),
        *list(_benchmark().itertuples(index=False, name=None)),
        ("2024-01-08", "TEST.IDX", 9.0),
    ]
    result = _run(benchmark=_benchmark(rows))
    assert result.observation_count == 3
    assert result.start_date == pd.Timestamp("2024-01-03")
    assert result.end_date == pd.Timestamp("2024-01-05")


def test_duplicate_benchmark_date_and_code_fails() -> None:
    benchmark = pd.concat([_benchmark(), _benchmark().iloc[[0]]], ignore_index=True)
    with pytest.raises(PerformanceAnalyticsInputError, match="unique"):
        _run(benchmark=benchmark)


@pytest.mark.parametrize("codes", [["WRONG.IDX"] * 3, ["TEST.IDX", "X.IDX", "TEST.IDX"]])
def test_wrong_or_mixed_benchmark_code_fails(codes: list[str]) -> None:
    benchmark = _benchmark()
    benchmark["benchmark_code"] = codes
    with pytest.raises(BenchmarkSeriesError):
        _run(benchmark=benchmark)


@pytest.mark.parametrize("value", [-1.0, -1.01])
def test_later_benchmark_total_loss_or_impossible_return_fails(value: float) -> None:
    benchmark = _benchmark()
    benchmark.loc[1, "return"] = value
    with pytest.raises(BenchmarkSeriesError):
        _run(benchmark=benchmark)


def test_total_returns_come_from_nav_and_custom_initial_value() -> None:
    result = _run()
    metrics = result.metrics
    portfolio = _portfolio()
    assert metrics["gross_total_return"] == pytest.approx(
        portfolio.daily_portfolio.iloc[-1]["gross_nav"] / 100.0 - 1.0
    )
    assert metrics["net_total_return"] == pytest.approx(
        portfolio.daily_portfolio.iloc[-1]["net_nav"] / 100.0 - 1.0
    )
    assert metrics["benchmark_total_return"] == pytest.approx(0.029)


@pytest.mark.parametrize("annualization_days", [252, 365])
def test_annualized_returns_use_n_observations_and_config(
    annualization_days: int,
) -> None:
    metrics = _run(annualization_days=annualization_days).metrics
    expected = (1.0 + metrics["net_total_return"]) ** (annualization_days / 3) - 1.0
    assert metrics["net_annualized_return"] == pytest.approx(expected)


def test_gross_and_benchmark_annualized_returns_are_exact() -> None:
    metrics = _run(annualization_days=12).metrics
    assert metrics["gross_annualized_return"] == pytest.approx(
        (1.0 + metrics["gross_total_return"]) ** 4 - 1.0
    )
    assert metrics["benchmark_annualized_return"] == pytest.approx(
        (1.0 + metrics["benchmark_total_return"]) ** 4 - 1.0
    )


def test_net_volatility_uses_sample_standard_deviation() -> None:
    portfolio = _portfolio()
    returns = portfolio.daily_portfolio["net_return"].to_numpy()
    metrics = _run(portfolio).metrics
    assert metrics["net_annualized_volatility"] == pytest.approx(
        np.std(returns, ddof=1) * sqrt(252)
    )


def test_single_observation_undefined_metrics_are_none() -> None:
    portfolio = _portfolio(
        dates=["2024-01-03"],
        gross_returns=[0.0],
        costs=[0.001],
        rebalances=[True],
        turnovers=[1.0],
        notionals=[1.0],
    )
    benchmark = _benchmark([("2024-01-03", "TEST.IDX", 0.2)])
    metrics = _run(portfolio, benchmark).metrics
    assert metrics["observation_count"] == 1
    assert metrics["net_annualized_volatility"] is None
    assert metrics["net_sharpe_ratio"] is None
    assert metrics["tracking_error"] is None
    assert metrics["information_ratio"] is None


@pytest.mark.parametrize("risk_free", [0.0, 0.05, -0.05])
def test_sharpe_uses_simple_daily_risk_free_conversion(risk_free: float) -> None:
    portfolio = _portfolio()
    returns = portfolio.daily_portfolio["net_return"].to_numpy()
    metrics = _run(portfolio, risk_free=risk_free).metrics
    expected = (
        np.mean(returns - risk_free / 252)
        / np.std(returns, ddof=1)
        * sqrt(252)
    )
    assert metrics["net_sharpe_ratio"] == pytest.approx(expected)


def test_zero_volatility_sharpe_is_none() -> None:
    portfolio = _portfolio(
        gross_returns=[0.0, 0.0, 0.0],
        costs=[0.0, 0.0, 0.0],
    )
    metrics = _run(portfolio).metrics
    assert metrics["net_annualized_volatility"] == 0.0
    assert metrics["net_sharpe_ratio"] is None


def test_max_drawdown_includes_initial_nav_high_water_mark() -> None:
    portfolio = _portfolio(
        dates=["2024-01-03"],
        gross_returns=[0.0],
        costs=[0.01],
        rebalances=[True],
        turnovers=[1.0],
        notionals=[1.0],
    )
    benchmark = _benchmark([("2024-01-03", "TEST.IDX", 0.5)])
    assert _run(portfolio, benchmark).metrics["net_max_drawdown"] == pytest.approx(
        -0.01
    )


def test_no_drawdown_is_zero() -> None:
    portfolio = _portfolio(
        gross_returns=[0.0, 0.1, 0.1],
        costs=[0.0, 0.0, 0.0],
    )
    assert _run(portfolio).metrics["net_max_drawdown"] == 0.0


def test_relative_total_and_annualized_definitions_are_differences() -> None:
    metrics = _run().metrics
    assert metrics["excess_total_return"] == pytest.approx(
        metrics["net_total_return"] - metrics["benchmark_total_return"]
    )
    assert metrics["annualized_excess_return"] == pytest.approx(
        metrics["net_annualized_return"] - metrics["benchmark_annualized_return"]
    )


def test_tracking_error_uses_daily_active_sample_deviation() -> None:
    portfolio = _portfolio()
    net = portfolio.daily_portfolio["net_return"].to_numpy()
    benchmark = np.array([0.0, 0.05, -0.02])
    expected = np.std(net - benchmark, ddof=1) * sqrt(252)
    assert _run(portfolio).metrics["tracking_error"] == pytest.approx(expected)


def test_information_ratio_uses_annualized_mean_active_return() -> None:
    portfolio = _portfolio()
    net = portfolio.daily_portfolio["net_return"].to_numpy()
    active = net - np.array([0.0, 0.05, -0.02])
    metrics = _run(portfolio).metrics
    expected = np.mean(active) * 252 / metrics["tracking_error"]
    assert metrics["information_ratio"] == pytest.approx(expected)


def test_zero_tracking_error_is_zero_and_ir_is_none() -> None:
    portfolio = _portfolio(
        gross_returns=[0.0, 0.05, -0.02],
        costs=[0.0, 0.0, 0.0],
    )
    metrics = _run(portfolio).metrics
    assert metrics["tracking_error"] == 0.0
    assert metrics["information_ratio"] is None


def test_trading_metrics_use_only_rebalance_rows_and_keep_units_distinct() -> None:
    metrics = _run().metrics
    assert metrics["rebalance_count"] == 2
    assert metrics["total_turnover"] == 1.5
    assert metrics["average_turnover"] == 0.75
    assert metrics["total_traded_notional"] == 2.0
    assert metrics["total_transaction_cost"] == 0.003


def test_transaction_cost_return_drag_is_cumulative_return_difference() -> None:
    metrics = _run().metrics
    assert metrics["transaction_cost_return_drag"] == pytest.approx(
        metrics["gross_total_return"] - metrics["net_total_return"]
    )
    assert metrics["transaction_cost_return_drag"] != pytest.approx(
        metrics["total_transaction_cost"]
    )


def test_zero_cost_trading_summary_and_drag_are_zero() -> None:
    portfolio = _portfolio(costs=[0.0, 0.0, 0.0])
    metrics = _run(portfolio).metrics
    assert metrics["total_transaction_cost"] == 0.0
    assert metrics["transaction_cost_return_drag"] == pytest.approx(0.0)


def test_metrics_are_deterministic_json_safe_builtins() -> None:
    metrics = _run().metrics
    assert tuple(metrics) == PERFORMANCE_METRIC_KEYS
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics
    for value in metrics.values():
        assert value is None or type(value) in (int, float)


def test_result_is_defensive_and_repeated_runs_are_identical() -> None:
    first = _run()
    second = _run()
    pdt.assert_frame_equal(first.benchmark_daily, second.benchmark_daily)
    assert first.metrics == second.metrics
    leaked = first.benchmark_daily
    leaked.loc[:, "benchmark_nav"] = -1.0
    assert first.benchmark_daily.loc[0, "benchmark_nav"] > 0.0
    leaked_metrics = first.metrics
    leaked_metrics["observation_count"] = 999
    assert first.observation_count == 3


def test_daily_portfolio_input_is_not_mutated() -> None:
    portfolio = _portfolio()
    before = portfolio.daily_portfolio
    _run(portfolio)
    pdt.assert_frame_equal(portfolio.daily_portfolio, before)


def test_rebalance_count_metadata_mismatch_fails() -> None:
    with pytest.raises(PerformanceAnalyticsInputError, match="rebalance_count"):
        _run(_portfolio(metadata_count=1))


def test_non_rebalance_cost_turnover_or_notional_fails() -> None:
    for column in ("transaction_cost", "turnover", "traded_notional"):
        portfolio = _portfolio()
        frame = portfolio.daily_portfolio
        frame.loc[1, column] = 0.1
        tampered = PortfolioDailyAccountingResult(
            frame,
            start_date=portfolio.start_date,
            end_date=portfolio.end_date,
            rebalance_count=portfolio.rebalance_count,
            initial_nav=portfolio.initial_nav,
            cost_bps=portfolio.cost_bps,
        )
        with pytest.raises(PerformanceAnalyticsInputError, match="non-rebalance"):
            _run(tampered)


@pytest.mark.parametrize("column", ["gross_nav", "net_nav"])
def test_nonpositive_nav_fails(column: str) -> None:
    portfolio = _portfolio()
    frame = portfolio.daily_portfolio
    frame.loc[0, column] = 0.0
    tampered = PortfolioDailyAccountingResult(
        frame,
        start_date=portfolio.start_date,
        end_date=portfolio.end_date,
        rebalance_count=portfolio.rebalance_count,
        initial_nav=portfolio.initial_nav,
        cost_bps=portfolio.cost_bps,
    )
    with pytest.raises(PerformanceAnalyticsInputError, match="positive"):
        _run(tampered)


def test_is_rebalance_requires_strict_bool() -> None:
    portfolio = _portfolio()
    frame = portfolio.daily_portfolio.astype({"is_rebalance": "object"})
    frame.loc[0, "is_rebalance"] = 1
    tampered = PortfolioDailyAccountingResult(
        frame,
        start_date=portfolio.start_date,
        end_date=portfolio.end_date,
        rebalance_count=portfolio.rebalance_count,
        initial_nav=portfolio.initial_nav,
        cost_bps=portfolio.cost_bps,
    )
    with pytest.raises(PerformanceAnalyticsInputError, match="strict bool"):
        _run(tampered)


def test_nonfinite_annualized_metric_fails_closed() -> None:
    portfolio = _portfolio(
        dates=["2024-01-03"],
        gross_returns=[1e100],
        costs=[0.0],
        rebalances=[True],
        turnovers=[1.0],
        notionals=[1.0],
    )
    benchmark = _benchmark([("2024-01-03", "TEST.IDX", 0.0)])
    with pytest.raises(PerformanceMetricError):
        _run(portfolio, benchmark)
