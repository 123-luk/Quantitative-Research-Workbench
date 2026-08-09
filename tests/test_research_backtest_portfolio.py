from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.holdings import HOLDINGS_OUTPUT_COLUMNS
from src.pipeline.research_backtest_config import (
    PortfolioAccountingConfig,
    TransactionCostConfig,
)
from src.research_backtest import (
    AVAILABLE,
    DAILY_PORTFOLIO_COLUMNS,
    POST_DELIST,
    PRE_LISTING,
    SECURITY_DAILY_RETURN_COLUMNS,
    SECURITY_STATUS_COLUMNS,
    SUSPENDED,
    UNKNOWN_MISSING,
    MissingSecurityReturnError,
    PortfolioConsistencyError,
    PortfolioDailyAccountingEngine,
    PortfolioDailyInputError,
    PortfolioTransactionCostError,
    PortfolioValueError,
    RebalanceAccountingEngine,
    TradingCalendar,
)


def _calendar(end: str = "2024-02-29") -> TradingCalendar:
    dates = pd.date_range("2024-01-01", end, freq="D")
    return TradingCalendar.from_frame(
        pd.DataFrame(
            {
                "cal_date": dates,
                "is_open": [int(item.weekday() < 5) for item in dates],
            }
        ),
        start_date="2024-01-01",
        end_date=end,
    )


def _holdings(events: list[tuple[str, dict[str, float]]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trade_date, weights in events:
        for rank, (code, weight) in enumerate(weights.items(), start=1):
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "target_weight": weight,
                    "score": float(100 - rank),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows, columns=list(HOLDINGS_OUTPUT_COLUMNS))


def _returns(rows: list[tuple[str, str, float]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows or [], columns=list(SECURITY_DAILY_RETURN_COLUMNS))


def _statuses(rows: list[tuple[str, str, str]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows or [], columns=list(SECURITY_STATUS_COLUMNS))


def _rebalance(
    events: list[tuple[str, dict[str, float]]],
    returns: pd.DataFrame | None = None,
    statuses: pd.DataFrame | None = None,
):
    return RebalanceAccountingEngine(_calendar()).run(
        holdings=_holdings(events),
        security_returns=_returns() if returns is None else returns,
        security_status=_statuses() if statuses is None else statuses,
    )


def _daily(
    rebalances,
    *,
    returns: pd.DataFrame | None = None,
    statuses: pd.DataFrame | None = None,
    end_date: object = "2024-01-03",
    cost_bps: float = 0.0,
    initial_nav: float = 1.0,
):
    return PortfolioDailyAccountingEngine(
        _calendar(),
        PortfolioAccountingConfig(initial_nav=initial_nav),
        TransactionCostConfig(cost_bps=cost_bps),
    ).run(
        rebalances=rebalances,
        security_returns=_returns() if returns is None else returns,
        security_status=_statuses() if statuses is None else statuses,
        end_date=end_date,
    )


def test_first_event_timing_cost_and_schema() -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    result = _daily(
        rebalances,
        returns=_returns([("2024-01-03", "A.SZ", 0.5)]),
        statuses=_statuses([("2024-01-03", "A.SZ", UNKNOWN_MISSING)]),
        cost_bps=10.0,
    )
    row = result.daily_portfolio.iloc[0]
    assert tuple(result.daily_portfolio.columns) == DAILY_PORTFOLIO_COLUMNS
    assert row["trade_date"] == pd.Timestamp("2024-01-03")
    assert row["gross_return"] == 0.0
    assert row["gross_nav"] == 1.0
    assert row["traded_notional"] == 1.0
    assert row["turnover"] == 1.0
    assert row["transaction_cost"] == 0.001
    assert row["net_return"] == pytest.approx(-0.001)
    assert row["net_nav"] == pytest.approx(0.999)
    assert bool(row["is_rebalance"])


def test_target_starts_return_on_next_open_date() -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    result = _daily(
        rebalances,
        returns=_returns([("2024-01-04", "A.SZ", 0.1)]),
        statuses=_statuses([("2024-01-04", "A.SZ", AVAILABLE)]),
        end_date="2024-01-04",
    )
    daily = result.daily_portfolio.set_index("trade_date")
    assert daily.loc[pd.Timestamp("2024-01-03"), "gross_return"] == 0.0
    assert daily.loc[pd.Timestamp("2024-01-04"), "gross_return"] == 0.1
    assert not bool(daily.loc[pd.Timestamp("2024-01-04"), "is_rebalance"])


def test_later_effective_return_belongs_to_old_portfolio() -> None:
    events = [("2024-01-02", {"A.SZ": 1.0}), ("2024-01-04", {"B.SZ": 1.0})]
    returns = _returns(
        [
            ("2024-01-04", "A.SZ", 0.0),
            ("2024-01-05", "A.SZ", 0.1),
            ("2024-01-05", "B.SZ", 1.0),
            ("2024-01-08", "B.SZ", 0.2),
        ]
    )
    statuses = _statuses(
        [
            ("2024-01-04", "A.SZ", AVAILABLE),
            ("2024-01-05", "A.SZ", AVAILABLE),
            ("2024-01-08", "B.SZ", AVAILABLE),
        ]
    )
    rebalances = _rebalance(events, returns, statuses)
    daily = _daily(
        rebalances,
        returns=returns,
        statuses=statuses,
        end_date="2024-01-08",
        cost_bps=10.0,
    ).daily_portfolio.set_index("trade_date")
    assert daily.loc[pd.Timestamp("2024-01-05"), "gross_return"] == 0.1
    assert daily.loc[pd.Timestamp("2024-01-05"), "traded_notional"] == 2.0
    assert daily.loc[pd.Timestamp("2024-01-05"), "transaction_cost"] == 0.002
    assert daily.loc[pd.Timestamp("2024-01-08"), "gross_return"] == 0.2


def test_daily_return_and_weight_drift_use_same_returns() -> None:
    events = [
        ("2024-01-02", {"A.SZ": 0.6, "B.SZ": 0.4}),
        ("2024-01-03", {"A.SZ": 0.6, "B.SZ": 0.4}),
    ]
    returns = _returns(
        [("2024-01-04", "A.SZ", 0.1), ("2024-01-04", "B.SZ", -0.05)]
    )
    statuses = _statuses(
        [
            ("2024-01-04", "A.SZ", AVAILABLE),
            ("2024-01-04", "B.SZ", AVAILABLE),
        ]
    )
    rebalances = _rebalance(events, returns, statuses)
    row = _daily(
        rebalances,
        returns=returns,
        statuses=statuses,
        end_date="2024-01-04",
    ).daily_portfolio.iloc[-1]
    assert row["gross_return"] == pytest.approx(0.04)
    assert row["traded_notional"] > 0.0


def test_zero_cost_makes_gross_and_net_nav_identical() -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    returns = _returns([("2024-01-04", "A.SZ", 0.1)])
    statuses = _statuses([("2024-01-04", "A.SZ", AVAILABLE)])
    daily = _daily(
        rebalances,
        returns=returns,
        statuses=statuses,
        end_date="2024-01-04",
    ).daily_portfolio
    assert set(daily["transaction_cost"]) == {0.0}
    pdt.assert_series_equal(daily["gross_nav"], daily["net_nav"], check_names=False)


def test_net_return_uses_multiplicative_cost_accounting() -> None:
    events = [
        ("2024-01-02", {"A.SZ": 0.5, "B.SZ": 0.5}),
        ("2024-01-03", {"A.SZ": 0.5, "B.SZ": 0.5}),
    ]
    returns = _returns(
        [("2024-01-04", "A.SZ", 0.1), ("2024-01-04", "B.SZ", 0.0)]
    )
    statuses = _statuses(
        [
            ("2024-01-04", "A.SZ", AVAILABLE),
            ("2024-01-04", "B.SZ", AVAILABLE),
        ]
    )
    rebalances = _rebalance(events, returns, statuses)
    row = _daily(
        rebalances,
        returns=returns,
        statuses=statuses,
        end_date="2024-01-04",
        cost_bps=100.0,
    ).daily_portfolio.iloc[-1]
    expected = (1.0 + row["gross_return"]) * (1.0 - row["transaction_cost"]) - 1.0
    assert row["net_return"] == pytest.approx(expected)
    assert row["net_return"] != pytest.approx(
        row["gross_return"] - row["transaction_cost"], abs=1e-10
    )


@pytest.mark.parametrize("initial_nav", [1.0, 100.0])
def test_nav_compounds_from_explicit_initial_value(initial_nav: float) -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    returns = _returns(
        [("2024-01-04", "A.SZ", 0.1), ("2024-01-05", "A.SZ", -0.2)]
    )
    statuses = _statuses(
        [
            ("2024-01-04", "A.SZ", AVAILABLE),
            ("2024-01-05", "A.SZ", AVAILABLE),
        ]
    )
    result = _daily(
        rebalances,
        returns=returns,
        statuses=statuses,
        end_date="2024-01-05",
        initial_nav=initial_nav,
    )
    assert result.daily_portfolio.iloc[-1]["gross_nav"] == pytest.approx(
        initial_nav * 1.1 * 0.8
    )
    assert result.initial_nav == initial_nav


@pytest.mark.parametrize(
    ("cost_bps", "expected"),
    [(0.5, 0.00005), (10.0, 0.001), (9999.0, 0.9999)],
)
def test_fractional_and_large_valid_bps_have_no_arbitrary_cap(
    cost_bps: float, expected: float
) -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    row = _daily(rebalances, cost_bps=cost_bps).daily_portfolio.iloc[0]
    assert row["transaction_cost"] == pytest.approx(expected)
    assert row["net_nav"] > 0.0


@pytest.mark.parametrize("cost_bps", [10000.0, 20000.0])
def test_cost_at_or_above_one_fails_closed(cost_bps: float) -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    with pytest.raises(PortfolioTransactionCostError):
        _daily(rebalances, cost_bps=cost_bps)


def test_non_rebalance_days_have_zero_cost_turnover_and_notional() -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    returns = _returns([("2024-01-04", "A.SZ", 0.0)])
    statuses = _statuses([("2024-01-04", "A.SZ", AVAILABLE)])
    row = _daily(
        rebalances,
        returns=returns,
        statuses=statuses,
        end_date="2024-01-04",
        cost_bps=100.0,
    ).daily_portfolio.iloc[-1]
    assert not bool(row["is_rebalance"])
    assert row["transaction_cost"] == 0.0
    assert row["turnover"] == 0.0
    assert row["traded_notional"] == 0.0


def test_same_target_without_drift_has_zero_event_notional() -> None:
    events = [("2024-01-02", {"A.SZ": 1.0}), ("2024-01-03", {"A.SZ": 1.0})]
    returns = _returns([("2024-01-04", "A.SZ", 0.0)])
    statuses = _statuses([("2024-01-04", "A.SZ", AVAILABLE)])
    rebalances = _rebalance(events, returns, statuses)
    row = _daily(
        rebalances,
        returns=returns,
        statuses=statuses,
        end_date="2024-01-04",
        cost_bps=10.0,
    ).daily_portfolio.iloc[-1]
    assert row["turnover"] == 0.0
    assert row["traded_notional"] == 0.0
    assert row["transaction_cost"] == 0.0


@pytest.mark.parametrize(
    ("end_date", "error"),
    [
        ("2024-01-02", PortfolioDailyInputError),
        ("2024-01-06", PortfolioDailyInputError),
        ("2024-03-01", PortfolioDailyInputError),
        ("bad-date", PortfolioDailyInputError),
    ],
)
def test_invalid_end_dates_fail(end_date: object, error: type[Exception]) -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    with pytest.raises(error):
        _daily(rebalances, end_date=end_date)


def test_end_date_equal_first_effective_date_is_valid() -> None:
    result = _daily(_rebalance([("2024-01-02", {"A.SZ": 1.0})]))
    assert result.row_count == 1
    assert result.start_date == result.end_date == pd.Timestamp("2024-01-03")


def test_explicit_end_date_ignores_later_extra_returns() -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    returns = _returns(
        [("2024-01-04", "A.SZ", 0.1), ("2024-01-31", "A.SZ", 9.0)]
    )
    statuses = _statuses([("2024-01-04", "A.SZ", AVAILABLE)])
    result = _daily(
        rebalances,
        returns=returns,
        statuses=statuses,
        end_date="2024-01-04",
    )
    assert result.row_count == 2
    assert result.daily_portfolio.iloc[-1]["gross_return"] == 0.1


def test_final_target_drifts_to_explicit_end_date_and_missing_fails() -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    with pytest.raises(MissingSecurityReturnError):
        _daily(
            rebalances,
            statuses=_statuses([("2024-01-04", "A.SZ", UNKNOWN_MISSING)]),
            end_date="2024-01-04",
        )


def test_suspension_resolves_held_missing_return_to_zero() -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    result = _daily(
        rebalances,
        statuses=_statuses([("2024-01-04", "A.SZ", SUSPENDED)]),
        end_date="2024-01-04",
    )
    row = result.daily_portfolio.iloc[-1]
    assert row["gross_return"] == 0.0
    assert row["gross_nav"] == 1.0


@pytest.mark.parametrize("status", [UNKNOWN_MISSING, PRE_LISTING, POST_DELIST, AVAILABLE])
def test_unexplained_held_missing_return_fails(status: str) -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    with pytest.raises(MissingSecurityReturnError):
        _daily(
            rebalances,
            statuses=_statuses([("2024-01-04", "A.SZ", status)]),
            end_date="2024-01-04",
        )


def test_closed_calendar_dates_never_appear() -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    dates = ["2024-01-04", "2024-01-05", "2024-01-08"]
    result = _daily(
        rebalances,
        returns=_returns([(item, "A.SZ", 0.0) for item in dates]),
        statuses=_statuses([(item, "A.SZ", AVAILABLE) for item in dates]),
        end_date="2024-01-08",
    )
    assert tuple(result.daily_portfolio["trade_date"]) == tuple(
        pd.to_datetime(["2024-01-03", *dates])
    )


def _two_asset_rebalances():
    events = [
        ("2024-01-02", {"A.SZ": 0.5, "B.SZ": 0.5}),
        ("2024-01-03", {"A.SZ": 0.5, "B.SZ": 0.5}),
    ]
    returns = _returns(
        [("2024-01-04", "A.SZ", 0.0), ("2024-01-04", "B.SZ", 0.0)]
    )
    statuses = _statuses(
        [
            ("2024-01-04", "A.SZ", AVAILABLE),
            ("2024-01-04", "B.SZ", AVAILABLE),
        ]
    )
    return _rebalance(events, returns, statuses), returns, statuses


def test_tampered_later_pre_weights_fail_c_d_consistency() -> None:
    result, returns, statuses = _two_asset_rebalances()
    rows = result.rebalances
    later = rows["effective_date"].eq(pd.Timestamp("2024-01-04"))
    rows.loc[later & rows["ts_code"].eq("A.SZ"), "pre_rebalance_weight"] = 0.6
    rows.loc[later & rows["ts_code"].eq("B.SZ"), "pre_rebalance_weight"] = 0.4
    rows.loc[later & rows["ts_code"].eq("A.SZ"), "weight_change"] = -0.1
    rows.loc[later & rows["ts_code"].eq("B.SZ"), "weight_change"] = 0.1
    rows.loc[later, "turnover"] = 0.1
    with pytest.raises(PortfolioConsistencyError):
        _daily(rows, returns=returns, statuses=statuses, end_date="2024-01-04")


def test_tampered_first_pre_state_fails_c_d_consistency() -> None:
    rows = _rebalance([("2024-01-02", {"A.SZ": 1.0})]).rebalances
    rows.loc[:, "pre_rebalance_weight"] = 0.1
    rows.loc[:, "pre_cash_weight"] = 0.9
    rows.loc[:, "weight_change"] = 0.9
    rows.loc[:, "cash_weight_change"] = -0.9
    rows.loc[:, "turnover"] = 0.9
    with pytest.raises(PortfolioConsistencyError):
        _daily(rows)


@pytest.mark.parametrize("field", ["turnover", "target_weight"])
def test_intrinsically_tampered_event_fails_input_validation(field: str) -> None:
    rows = _rebalance([("2024-01-02", {"A.SZ": 1.0})]).rebalances
    rows.loc[:, field] = 0.5
    with pytest.raises(PortfolioDailyInputError):
        _daily(rows)


def test_tampered_pre_cash_with_valid_event_math_fails_consistency() -> None:
    events = [("2024-01-02", {"A.SZ": 1.0}), ("2024-01-03", {"B.SZ": 1.0})]
    returns = _returns([("2024-01-04", "A.SZ", 0.0)])
    statuses = _statuses([("2024-01-04", "A.SZ", AVAILABLE)])
    rows = _rebalance(events, returns, statuses).rebalances
    later = rows["effective_date"].eq(pd.Timestamp("2024-01-04"))
    a_row = later & rows["ts_code"].eq("A.SZ")
    rows.loc[a_row, "pre_rebalance_weight"] = 0.9
    rows.loc[a_row, "weight_change"] = -0.9
    rows.loc[later, "pre_cash_weight"] = 0.1
    rows.loc[later, "cash_weight_change"] = -0.1
    with pytest.raises(PortfolioConsistencyError):
        _daily(rows, returns=returns, statuses=statuses, end_date="2024-01-04")


def test_return_minus_one_that_zeroes_portfolio_fails() -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    with pytest.raises(PortfolioValueError):
        _daily(
            rebalances,
            returns=_returns([("2024-01-04", "A.SZ", -1.0)]),
            statuses=_statuses([("2024-01-04", "A.SZ", AVAILABLE)]),
            end_date="2024-01-04",
        )


def test_return_below_minus_one_fails() -> None:
    rebalances = _rebalance([("2024-01-02", {"A.SZ": 1.0})])
    with pytest.raises(PortfolioValueError):
        _daily(
            rebalances,
            returns=_returns([("2024-01-04", "A.SZ", -1.01)]),
            statuses=_statuses([("2024-01-04", "A.SZ", AVAILABLE)]),
            end_date="2024-01-04",
        )


@pytest.mark.parametrize(
    "trade_dates",
    [
        ["2024-01-02", "2024-01-03"],
        ["2024-01-02", "2024-01-09"],
        ["2024-01-02", "2024-01-30"],
    ],
)
def test_daily_weekly_and_monthly_like_events_use_one_engine(trade_dates) -> None:
    calendar = _calendar()
    events = [(item, {"A.SZ": 1.0}) for item in trade_dates]
    effective = [calendar.next_trading_day(item) for item in trade_dates]
    required_dates = [
        item for item in calendar.open_dates if effective[0] < item <= effective[-1]
    ]
    returns = _returns(
        [(item.strftime("%Y-%m-%d"), "A.SZ", 0.0) for item in required_dates]
    )
    statuses = _statuses(
        [(item.strftime("%Y-%m-%d"), "A.SZ", AVAILABLE) for item in required_dates]
    )
    rebalances = _rebalance(events, returns, statuses)
    result = _daily(
        rebalances,
        returns=returns,
        statuses=statuses,
        end_date=effective[-1],
    )
    assert result.rebalance_count == len(trade_dates)


def test_shuffled_inputs_are_deterministic_and_not_mutated() -> None:
    result, returns, statuses = _two_asset_rebalances()
    rows = result.rebalances.sample(frac=1.0, random_state=3)
    returns = returns.sample(frac=1.0, random_state=4)
    statuses = statuses.sample(frac=1.0, random_state=5)
    before = (rows.copy(deep=True), returns.copy(deep=True), statuses.copy(deep=True))
    first = _daily(
        rows,
        returns=returns,
        statuses=statuses,
        end_date="2024-01-04",
    ).daily_portfolio
    second = _daily(
        rows.iloc[::-1],
        returns=returns.iloc[::-1],
        statuses=statuses.iloc[::-1],
        end_date="2024-01-04",
    ).daily_portfolio
    pdt.assert_frame_equal(first, second)
    pdt.assert_frame_equal(rows, before[0])
    pdt.assert_frame_equal(returns, before[1])
    pdt.assert_frame_equal(statuses, before[2])


def test_result_is_defensive_and_has_audit_properties() -> None:
    result = _daily(
        _rebalance([("2024-01-02", {"A.SZ": 1.0})]),
        cost_bps=2.5,
        initial_nav=100.0,
    )
    leaked = result.daily_portfolio
    leaked.loc[:, "net_nav"] = -1.0
    assert result.daily_portfolio.loc[0, "net_nav"] > 0.0
    assert result.row_count == 1
    assert result.rebalance_count == 1
    assert result.initial_nav == 100.0
    assert result.cost_bps == 2.5
