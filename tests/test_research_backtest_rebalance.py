from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.holdings import HOLDINGS_OUTPUT_COLUMNS
from src.research_backtest import (
    AVAILABLE,
    DELIST_DATE,
    POST_DELIST,
    PRE_LISTING,
    REBALANCE_OUTPUT_COLUMNS,
    SECURITY_DAILY_RETURN_COLUMNS,
    SECURITY_STATUS_COLUMNS,
    SUSPENDED,
    UNKNOWN_MISSING,
    MissingSecurityReturnError,
    RebalanceAccountingEngine,
    RebalanceInputError,
    RebalanceScheduleError,
    TradingCalendar,
    TradingCalendarCoverageError,
    WeightDriftError,
)


def _calendar(end: str = "2024-01-31") -> TradingCalendar:
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


def _run(
    holdings: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    statuses: pd.DataFrame | None = None,
):
    return RebalanceAccountingEngine(_calendar()).run(
        holdings=holdings,
        security_returns=_returns() if returns is None else returns,
        security_status=_statuses() if statuses is None else statuses,
    )


@pytest.mark.parametrize("count", [1, 2, 10])
def test_first_fully_invested_event_has_complete_cash_turnover(count: int) -> None:
    weights = {f"{index:06d}.SZ": 1.0 / count for index in range(count)}
    ledger = _run(_holdings([("2024-01-02", weights)])).rebalances

    assert tuple(ledger.columns) == REBALANCE_OUTPUT_COLUMNS
    assert len(ledger) == count
    assert set(ledger["pre_rebalance_weight"]) == {0.0}
    assert set(ledger["pre_cash_weight"]) == {1.0}
    assert set(ledger["target_cash_weight"]) == {0.0}
    assert set(ledger["cash_weight_change"]) == {-1.0}
    assert set(ledger["turnover"]) == {1.0}
    assert "CASH" not in set(ledger["ts_code"])


def test_first_event_ignores_its_effective_date_return() -> None:
    ledger = _run(
        _holdings([("2024-01-02", {"A.SZ": 1.0})]),
        _returns([("2024-01-03", "A.SZ", -1.5)]),
        _statuses([("2024-01-03", "A.SZ", UNKNOWN_MISSING)]),
    ).rebalances
    assert ledger.loc[0, "effective_date"] == pd.Timestamp("2024-01-03")
    assert ledger.loc[0, "turnover"] == 1.0


def test_equal_weights_drift_before_next_rebalance() -> None:
    holdings = _holdings(
        [
            ("2024-01-02", {"A.SZ": 0.5, "B.SZ": 0.5}),
            ("2024-01-03", {"A.SZ": 0.5, "B.SZ": 0.5}),
        ]
    )
    ledger = _run(
        holdings,
        _returns(
            [("2024-01-04", "A.SZ", 0.1), ("2024-01-04", "B.SZ", 0.0)]
        ),
        _statuses(
            [
                ("2024-01-04", "A.SZ", AVAILABLE),
                ("2024-01-04", "B.SZ", AVAILABLE),
            ]
        ),
    ).rebalances
    second = ledger[ledger["effective_date"].eq(pd.Timestamp("2024-01-04"))]
    actual = dict(zip(second["ts_code"], second["pre_rebalance_weight"]))
    assert actual["A.SZ"] == pytest.approx(0.55 / 1.05)
    assert actual["B.SZ"] == pytest.approx(0.50 / 1.05)
    assert second["turnover"].iloc[0] == pytest.approx(abs(actual["A.SZ"] - 0.5))


def test_single_security_drift_remains_fully_invested() -> None:
    result = _run(
        _holdings(
            [("2024-01-02", {"A.SZ": 1.0}), ("2024-01-03", {"A.SZ": 1.0})]
        ),
        _returns([("2024-01-04", "A.SZ", 0.25)]),
        _statuses([("2024-01-04", "A.SZ", AVAILABLE)]),
    )
    second = result.rebalances.iloc[-1]
    assert second["pre_rebalance_weight"] == 1.0
    assert second["turnover"] == 0.0


def test_drift_compounds_across_every_open_date_through_effective_close() -> None:
    holdings = _holdings(
        [
            ("2024-01-02", {"A.SZ": 0.5, "B.SZ": 0.5}),
            ("2024-01-05", {"A.SZ": 0.5, "B.SZ": 0.5}),
        ]
    )
    daily = [
        (date, code, value)
        for date in ("2024-01-04", "2024-01-05", "2024-01-08")
        for code, value in (("A.SZ", 0.1), ("B.SZ", 0.0))
    ]
    statuses = [(date, code, AVAILABLE) for date, code, _ in daily]
    ledger = _run(holdings, _returns(daily), _statuses(statuses)).rebalances
    second = ledger[ledger["effective_date"].eq(pd.Timestamp("2024-01-08"))]
    actual = second.set_index("ts_code")["pre_rebalance_weight"]
    assert actual["A.SZ"] == pytest.approx(0.5 * 1.1**3 / (0.5 * 1.1**3 + 0.5))


def test_new_target_does_not_receive_effective_close_return() -> None:
    ledger = _run(
        _holdings(
            [("2024-01-02", {"A.SZ": 1.0}), ("2024-01-03", {"B.SZ": 1.0})]
        ),
        _returns([("2024-01-04", "A.SZ", 0.1), ("2024-01-04", "B.SZ", 1.0)]),
        _statuses([("2024-01-04", "A.SZ", AVAILABLE)]),
    ).rebalances
    second = ledger[ledger["effective_date"].eq(pd.Timestamp("2024-01-04"))]
    assert second.set_index("ts_code").loc["B.SZ", "pre_rebalance_weight"] == 0.0
    assert set(second["turnover"]) == {1.0}


def test_new_target_starts_receiving_return_on_next_open_date() -> None:
    holdings = _holdings(
        [
            ("2024-01-02", {"A.SZ": 1.0}),
            ("2024-01-03", {"B.SZ": 1.0}),
            ("2024-01-04", {"A.SZ": 0.5, "B.SZ": 0.5}),
        ]
    )
    returns = _returns(
        [
            ("2024-01-04", "A.SZ", 0.0),
            ("2024-01-05", "B.SZ", 0.5),
        ]
    )
    statuses = _statuses(
        [("2024-01-04", "A.SZ", AVAILABLE), ("2024-01-05", "B.SZ", AVAILABLE)]
    )
    ledger = _run(holdings, returns, statuses).rebalances
    third = ledger[ledger["effective_date"].eq(pd.Timestamp("2024-01-05"))]
    assert third.set_index("ts_code").loc["B.SZ", "pre_rebalance_weight"] == 1.0


def test_suspension_resolves_missing_positive_weight_return_to_zero() -> None:
    ledger = _run(
        _holdings(
            [("2024-01-02", {"A.SZ": 1.0}), ("2024-01-03", {"A.SZ": 1.0})]
        ),
        statuses=_statuses([("2024-01-04", "A.SZ", SUSPENDED)]),
    ).rebalances
    assert ledger.iloc[-1]["pre_rebalance_weight"] == 1.0
    assert ledger.iloc[-1]["turnover"] == 0.0


@pytest.mark.parametrize(
    "status", [AVAILABLE, PRE_LISTING, DELIST_DATE, POST_DELIST, UNKNOWN_MISSING]
)
def test_unresolvable_missing_positive_weight_return_fails(status: str) -> None:
    with pytest.raises(MissingSecurityReturnError):
        _run(
            _holdings(
                [("2024-01-02", {"A.SZ": 1.0}), ("2024-01-03", {"A.SZ": 1.0})]
            ),
            statuses=_statuses([("2024-01-04", "A.SZ", status)]),
        )


def test_missing_status_for_positive_weight_fails_closed() -> None:
    with pytest.raises(RebalanceInputError, match="lacks a required"):
        _run(
            _holdings(
                [("2024-01-02", {"A.SZ": 1.0}), ("2024-01-03", {"A.SZ": 1.0})]
            ),
            _returns([("2024-01-04", "A.SZ", 0.0)]),
        )


def test_strict_zero_weight_does_not_require_return_or_status() -> None:
    ledger = _run(
        _holdings(
            [
                ("2024-01-02", {"A.SZ": 1.0, "Z.SZ": 0.0}),
                ("2024-01-03", {"A.SZ": 1.0}),
            ]
        ),
        _returns([("2024-01-04", "A.SZ", 0.0)]),
        _statuses([("2024-01-04", "A.SZ", AVAILABLE)]),
    ).rebalances
    assert "Z.SZ" in set(ledger["ts_code"])


def test_total_loss_is_allowed_when_another_asset_survives() -> None:
    ledger = _run(
        _holdings(
            [
                ("2024-01-02", {"A.SZ": 0.5, "B.SZ": 0.5}),
                ("2024-01-03", {"B.SZ": 1.0}),
            ]
        ),
        _returns([("2024-01-04", "A.SZ", -1.0), ("2024-01-04", "B.SZ", 0.0)]),
        _statuses(
            [
                ("2024-01-04", "A.SZ", AVAILABLE),
                ("2024-01-04", "B.SZ", AVAILABLE),
            ]
        ),
    ).rebalances
    second = ledger[ledger["effective_date"].eq(pd.Timestamp("2024-01-04"))]
    assert second.set_index("ts_code").loc["A.SZ", "pre_rebalance_weight"] == 0.0


def test_all_assets_total_loss_makes_normalization_impossible() -> None:
    with pytest.raises(WeightDriftError, match="strictly positive"):
        _run(
            _holdings(
                [("2024-01-02", {"A.SZ": 1.0}), ("2024-01-03", {"A.SZ": 1.0})]
            ),
            _returns([("2024-01-04", "A.SZ", -1.0)]),
            _statuses([("2024-01-04", "A.SZ", AVAILABLE)]),
        )


def test_return_below_minus_one_fails() -> None:
    with pytest.raises(WeightDriftError, match="below -1"):
        _run(
            _holdings(
                [("2024-01-02", {"A.SZ": 1.0}), ("2024-01-03", {"A.SZ": 1.0})]
            ),
            _returns([("2024-01-04", "A.SZ", -1.01)]),
            _statuses([("2024-01-04", "A.SZ", AVAILABLE)]),
        )


def test_union_contains_entering_exiting_and_continuing_securities() -> None:
    holdings = _holdings(
        [
            ("2024-01-02", {"A.SZ": 0.5, "B.SZ": 0.5}),
            ("2024-01-03", {"B.SZ": 0.25, "C.SZ": 0.75}),
        ]
    )
    daily = _returns([("2024-01-04", "A.SZ", 0.0), ("2024-01-04", "B.SZ", 0.0)])
    status = _statuses(
        [("2024-01-04", "A.SZ", AVAILABLE), ("2024-01-04", "B.SZ", AVAILABLE)]
    )
    second = _run(holdings, daily, status).rebalances
    second = second[second["effective_date"].eq(pd.Timestamp("2024-01-04"))]
    assert list(second["ts_code"]) == ["A.SZ", "B.SZ", "C.SZ"]
    assert set(second["turnover"]) == {0.75}
    complete_change = (
        second["weight_change"].sum() + second["cash_weight_change"].iloc[0]
    )
    assert complete_change == pytest.approx(0.0)


def test_effective_date_collision_fails_closed() -> None:
    with pytest.raises(RebalanceScheduleError, match="multiple holdings"):
        _run(
            _holdings(
                [("2024-01-05", {"A.SZ": 1.0}), ("2024-01-06", {"B.SZ": 1.0})]
            )
        )


def test_insufficient_future_calendar_coverage_is_propagated() -> None:
    with pytest.raises(TradingCalendarCoverageError):
        RebalanceAccountingEngine(_calendar("2024-01-05")).run(
            holdings=_holdings([("2024-01-05", {"A.SZ": 1.0})]),
            security_returns=_returns(),
            security_status=_statuses(),
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.iloc[0:0],
        lambda frame: frame.assign(target_weight=-1.0),
        lambda frame: frame.assign(target_weight=np.nan),
        lambda frame: frame.assign(target_weight=np.inf),
        lambda frame: frame.assign(target_weight=True),
        lambda frame: frame.assign(target_weight="1.0"),
        lambda frame: frame.assign(ts_code=""),
        lambda frame: frame.assign(trade_date="not-a-date"),
        lambda frame: pd.concat([frame, frame], ignore_index=True),
        lambda frame: frame.rename(columns={"rank": "not_rank"}),
    ],
)
def test_invalid_holdings_fail_closed(mutator) -> None:
    with pytest.raises(RebalanceInputError):
        _run(mutator(_holdings([("2024-01-02", {"A.SZ": 1.0})])))


def test_holdings_weight_sum_uses_tight_tolerance() -> None:
    with pytest.raises(RebalanceInputError, match="sum to 1"):
        _run(_holdings([("2024-01-02", {"A.SZ": 0.999999})]))


@pytest.mark.parametrize("kind", ["returns", "status"])
def test_daily_inputs_require_exact_canonical_schema(kind: str) -> None:
    kwargs = {
        "holdings": _holdings([("2024-01-02", {"A.SZ": 1.0})]),
        "security_returns": _returns(),
        "security_status": _statuses(),
    }
    key = "security_returns" if kind == "returns" else "security_status"
    kwargs[key] = kwargs[key].assign(extra=1)
    with pytest.raises(RebalanceInputError, match="columns"):
        RebalanceAccountingEngine(_calendar()).run(**kwargs)


def test_duplicate_daily_keys_fail_closed() -> None:
    with pytest.raises(RebalanceInputError, match="keys must be unique"):
        _run(
            _holdings([("2024-01-02", {"A.SZ": 1.0})]),
            _returns(
                [("2024-01-03", "A.SZ", 0.0), ("2024-01-03", "A.SZ", 0.1)]
            ),
        )


def test_shuffled_inputs_are_deterministic_and_not_mutated() -> None:
    holdings = _holdings(
        [("2024-01-02", {"B.SZ": 0.5, "A.SZ": 0.5})]
    ).sample(frac=1.0, random_state=7)
    returns = _returns([("2024-01-03", "A.SZ", 0.1)])
    statuses = _statuses([("2024-01-03", "A.SZ", AVAILABLE)])
    before = (holdings.copy(deep=True), returns.copy(deep=True), statuses.copy(deep=True))
    first = _run(holdings, returns, statuses).rebalances
    second = _run(holdings.iloc[::-1], returns, statuses).rebalances
    pdt.assert_frame_equal(first, second)
    pdt.assert_frame_equal(holdings, before[0])
    pdt.assert_frame_equal(returns, before[1])
    pdt.assert_frame_equal(statuses, before[2])
    assert list(first["ts_code"]) == ["A.SZ", "B.SZ"]


def test_result_is_defensive_and_reports_event_bounds() -> None:
    result = _run(
        _holdings(
            [("2024-01-02", {"A.SZ": 1.0}), ("2024-01-03", {"A.SZ": 1.0})]
        ),
        _returns([("2024-01-04", "A.SZ", 0.0)]),
        _statuses([("2024-01-04", "A.SZ", AVAILABLE)]),
    )
    leaked = result.rebalances
    leaked.loc[:, "turnover"] = 999.0
    assert 999.0 not in set(result.rebalances["turnover"])
    assert result.event_count == 2
    assert result.first_effective_date == pd.Timestamp("2024-01-03")
    assert result.last_effective_date == pd.Timestamp("2024-01-04")


@pytest.mark.parametrize(
    "trade_dates",
    [
        ["2024-01-02", "2024-01-09"],
        ["2024-01-02", "2024-01-16"],
        ["2024-01-02", "2024-01-30"],
    ],
)
def test_schedule_frequency_is_inferred_only_from_holdings_dates(trade_dates) -> None:
    holdings = _holdings([(item, {"A.SZ": 1.0}) for item in trade_dates])
    effective = [_calendar().next_trading_day(item) for item in trade_dates]
    drift_dates = [item for item in _calendar().open_dates if effective[0] < item <= effective[1]]
    returns = _returns([(item.strftime("%Y-%m-%d"), "A.SZ", 0.0) for item in drift_dates])
    statuses = _statuses(
        [(item.strftime("%Y-%m-%d"), "A.SZ", AVAILABLE) for item in drift_dates]
    )
    result = _run(holdings, returns, statuses)
    assert result.event_count == 2
    assert result.last_effective_date == effective[1]
