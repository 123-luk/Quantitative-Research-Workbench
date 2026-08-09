"""Tests for V6 security availability and explicit missing-return policy."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.research_backtest import (
    AVAILABLE,
    DELIST_DATE,
    POST_DELIST,
    PRE_LISTING,
    SECURITY_AVAILABILITY_STATUSES,
    SECURITY_LIFECYCLE_COLUMNS,
    SECURITY_STATUS_COLUMNS,
    SECURITY_SUSPENSION_COLUMNS,
    SUSPENDED,
    UNKNOWN_MISSING,
    MissingSecurityReturnError,
    SecurityAvailabilityError,
    SecurityAvailabilityProviderError,
    SecurityLifecycleError,
    SecuritySuspensionDataError,
    TradingCalendar,
    TushareSecurityLifecycleAdapter,
    TushareSecuritySuspensionAdapter,
    build_security_daily_returns,
    build_security_lifecycle,
    build_security_status,
    build_security_suspensions,
    resolve_security_return,
)


def _calendar() -> TradingCalendar:
    dates = pd.date_range("2024-01-01", "2024-01-12", freq="D")
    rows = pd.DataFrame(
        {
            "cal_date": dates,
            "is_open": [int(item.weekday() < 5) for item in dates],
        }
    )
    return TradingCalendar.from_frame(
        rows, start_date="2024-01-01", end_date="2024-01-12"
    )


def _raw_lifecycle() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["ACTIVE", "NEW", "OLD", "OLD"],
            "list_status": ["L", "L", "L", "D"],
            "list_date": ["20200101", "20240105", "20100101", "20100101"],
            "delist_date": [None, None, None, "20240105"],
            "name": ["active", "new", "old", "old"],
        }
    )


def _lifecycle() -> pd.DataFrame:
    return build_security_lifecycle(_raw_lifecycle())


def _empty_suspensions() -> pd.DataFrame:
    return pd.DataFrame(columns=list(SECURITY_SUSPENSION_COLUMNS))


def _returns(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    raw = pd.DataFrame(rows, columns=["trade_date", "ts_code", "pct_chg"])
    if raw.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code", "return"])
    return build_security_daily_returns(raw)


def test_status_vocabulary_and_schemas_are_exact_immutable_tuples() -> None:
    assert SECURITY_AVAILABILITY_STATUSES == (
        "AVAILABLE",
        "SUSPENDED",
        "PRE_LISTING",
        "DELIST_DATE",
        "POST_DELIST",
        "UNKNOWN_MISSING",
    )
    assert SECURITY_STATUS_COLUMNS == ("trade_date", "ts_code", "status")
    assert SECURITY_LIFECYCLE_COLUMNS == ("ts_code", "list_date", "delist_date")
    assert SECURITY_SUSPENSION_COLUMNS == (
        "trade_date",
        "ts_code",
        "suspend_type",
        "suspend_timing",
    )


def test_lifecycle_reconciles_distinct_status_rows_without_mutating_input() -> None:
    source = _raw_lifecycle().sample(frac=1.0, random_state=7).reset_index(drop=True)
    original = source.copy(deep=True)
    result = build_security_lifecycle(source)
    assert tuple(result.columns) == SECURITY_LIFECYCLE_COLUMNS
    assert result["ts_code"].tolist() == ["ACTIVE", "NEW", "OLD"]
    old = result.loc[result["ts_code"].eq("OLD")].iloc[0]
    assert old["list_date"] == pd.Timestamp("2010-01-01")
    assert old["delist_date"] == pd.Timestamp("2024-01-05")
    pdt.assert_frame_equal(source, original)


@pytest.mark.parametrize(
    ("trade_date", "expected"),
    [
        ("2024-01-04", PRE_LISTING),
        ("2024-01-05", UNKNOWN_MISSING),
        ("2024-01-08", UNKNOWN_MISSING),
    ],
)
def test_listing_boundary_is_active_from_list_date(
    trade_date: str, expected: str
) -> None:
    status = build_security_status(
        ts_codes=["NEW"],
        evaluation_dates=[trade_date],
        lifecycle=_lifecycle(),
        suspensions=_empty_suspensions(),
        security_returns=_returns([]),
        trading_calendar=_calendar(),
    )
    assert status.loc[0, "status"] == expected


@pytest.mark.parametrize(
    ("trade_date", "expected"),
    [
        ("2024-01-04", UNKNOWN_MISSING),
        ("2024-01-05", DELIST_DATE),
        ("2024-01-08", POST_DELIST),
    ],
)
def test_delist_boundary_is_explicit_and_post_is_strictly_after(
    trade_date: str, expected: str
) -> None:
    status = build_security_status(
        ts_codes=["OLD"],
        evaluation_dates=[trade_date],
        lifecycle=_lifecycle(),
        suspensions=_empty_suspensions(),
        security_returns=_returns([]),
        trading_calendar=_calendar(),
    )
    assert status.loc[0, "status"] == expected


def test_no_delist_date_remains_active_but_missing_is_unknown() -> None:
    result = build_security_status(
        ts_codes=["ACTIVE"],
        evaluation_dates=["2024-01-12"],
        lifecycle=_lifecycle(),
        suspensions=_empty_suspensions(),
        security_returns=_returns([]),
        trading_calendar=_calendar(),
    )
    assert result.loc[0, "status"] == UNKNOWN_MISSING


@pytest.mark.parametrize(
    "updates",
    [
        {"list_date": "bad"},
        {"delist_date": "bad"},
        {"list_date": "20240102", "delist_date": "20240101"},
        {"ts_code": ""},
        {"list_status": "G"},
    ],
)
def test_malformed_lifecycle_rows_are_rejected(updates: dict[str, object]) -> None:
    row: dict[str, object] = {
        "ts_code": "A",
        "list_status": "L",
        "list_date": "20200101",
        "delist_date": None,
    }
    row.update(updates)
    with pytest.raises(SecurityLifecycleError):
        build_security_lifecycle(pd.DataFrame([row]))


def test_delisted_status_requires_delist_date() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["A"],
            "list_status": ["D"],
            "list_date": ["20200101"],
            "delist_date": [None],
        }
    )
    with pytest.raises(SecurityLifecycleError, match="requires delist_date"):
        build_security_lifecycle(source)


def test_duplicate_same_status_and_conflicting_cross_status_fail_closed() -> None:
    duplicate = pd.DataFrame(
        {
            "ts_code": ["A", "A"],
            "list_status": ["L", "L"],
            "list_date": ["20200101", "20200101"],
            "delist_date": [None, None],
        }
    )
    with pytest.raises(SecurityLifecycleError, match="duplicate"):
        build_security_lifecycle(duplicate)
    conflict = duplicate.copy(deep=True)
    conflict["list_status"] = ["L", "P"]
    conflict["list_date"] = ["20200101", "20200102"]
    with pytest.raises(SecurityLifecycleError, match="conflicting"):
        build_security_lifecycle(conflict)


def test_suspension_rows_are_daily_events_sorted_and_input_is_unchanged() -> None:
    source = pd.DataFrame(
        {
            "trade_date": ["20240108", "20240103", "20240105"],
            "ts_code": ["A", "A", "A"],
            "suspend_type": ["S", "S", "R"],
            "suspend_timing": [None, "09:30-10:30", None],
            "extra": [1, 2, 3],
        }
    )
    original = source.copy(deep=True)
    result = build_security_suspensions(source)
    assert tuple(result.columns) == SECURITY_SUSPENSION_COLUMNS
    assert result["trade_date"].tolist() == list(
        pd.to_datetime(["2024-01-03", "2024-01-05", "2024-01-08"])
    )
    assert result["suspend_type"].tolist() == ["S", "R", "S"]
    pdt.assert_frame_equal(source, original)


def test_resume_event_and_dates_outside_suspension_event_are_not_suspended() -> None:
    suspensions = build_security_suspensions(
        pd.DataFrame(
            {
                "trade_date": ["20240103", "20240105"],
                "ts_code": ["ACTIVE", "ACTIVE"],
                "suspend_type": ["S", "R"],
                "suspend_timing": [None, None],
            }
        )
    )
    status = build_security_status(
        ts_codes=["ACTIVE"],
        evaluation_dates=["2024-01-03", "2024-01-04", "2024-01-05"],
        lifecycle=_lifecycle(),
        suspensions=suspensions,
        security_returns=_returns([]),
        trading_calendar=_calendar(),
    )
    assert status["status"].tolist() == [
        SUSPENDED,
        UNKNOWN_MISSING,
        UNKNOWN_MISSING,
    ]


def test_intraday_suspension_segment_does_not_prove_full_day_zero() -> None:
    suspensions = pd.DataFrame(
        {
            "trade_date": ["20240103"],
            "ts_code": ["ACTIVE"],
            "suspend_type": ["S"],
            "suspend_timing": ["09:30-10:30"],
        }
    )
    missing_status = build_security_status(
        ts_codes=["ACTIVE"],
        evaluation_dates=["2024-01-03"],
        lifecycle=_lifecycle(),
        suspensions=suspensions,
        security_returns=_returns([]),
        trading_calendar=_calendar(),
    )
    assert missing_status.loc[0, "status"] == UNKNOWN_MISSING
    observed_status = build_security_status(
        ts_codes=["ACTIVE"],
        evaluation_dates=["2024-01-03"],
        lifecycle=_lifecycle(),
        suspensions=suspensions,
        security_returns=_returns([("20240103", "ACTIVE", 1.0)]),
        trading_calendar=_calendar(),
    )
    assert observed_status.loc[0, "status"] == AVAILABLE


@pytest.mark.parametrize(
    "updates",
    [
        {"trade_date": "bad"},
        {"ts_code": ""},
        {"suspend_type": "X"},
        {"suspend_timing": 1},
    ],
)
def test_malformed_suspension_records_are_rejected(
    updates: dict[str, object]
) -> None:
    row: dict[str, object] = {
        "trade_date": "20240103",
        "ts_code": "A",
        "suspend_type": "S",
        "suspend_timing": None,
    }
    row.update(updates)
    with pytest.raises(SecuritySuspensionDataError):
        build_security_suspensions(pd.DataFrame([row]))


def test_duplicate_or_conflicting_suspension_key_is_rejected() -> None:
    source = pd.DataFrame(
        {
            "trade_date": ["20240103", "20240103"],
            "ts_code": ["A", "A"],
            "suspend_type": ["S", "R"],
            "suspend_timing": [None, None],
        }
    )
    with pytest.raises(SecuritySuspensionDataError, match="unique"):
        build_security_suspensions(source)


def test_existing_return_is_available_and_preserved_by_resolver() -> None:
    returns = _returns([("20240103", "ACTIVE", 1.25)])
    status = build_security_status(
        ts_codes=["ACTIVE"],
        evaluation_dates=["2024-01-03"],
        lifecycle=_lifecycle(),
        suspensions=_empty_suspensions(),
        security_returns=returns,
        trading_calendar=_calendar(),
    )
    assert status.loc[0, "status"] == AVAILABLE
    assert resolve_security_return(
        trade_date="2024-01-03",
        ts_code="ACTIVE",
        status=AVAILABLE,
        observed_return=returns.loc[0, "return"],
    ) == pytest.approx(0.0125)


def test_proven_same_date_suspension_is_the_only_missing_zero_path() -> None:
    suspensions = build_security_suspensions(
        pd.DataFrame(
            {
                "trade_date": ["20240103"],
                "ts_code": ["ACTIVE"],
                "suspend_type": ["S"],
                "suspend_timing": [None],
            }
        )
    )
    status = build_security_status(
        ts_codes=["ACTIVE"],
        evaluation_dates=["2024-01-03"],
        lifecycle=_lifecycle(),
        suspensions=suspensions,
        security_returns=_returns([]),
        trading_calendar=_calendar(),
    )
    assert status.loc[0, "status"] == SUSPENDED
    assert resolve_security_return(
        trade_date="2024-01-03",
        ts_code="ACTIVE",
        status=SUSPENDED,
        observed_return=None,
    ) == 0.0


@pytest.mark.parametrize(
    "status",
    [AVAILABLE, PRE_LISTING, DELIST_DATE, POST_DELIST, UNKNOWN_MISSING],
)
def test_all_other_missing_statuses_fail_closed_with_identity(status: str) -> None:
    with pytest.raises(MissingSecurityReturnError) as exc:
        resolve_security_return(
            trade_date="2024-01-03",
            ts_code="ACTIVE",
            status=status,
            observed_return=None,
        )
    assert "2024-01-03" in str(exc.value)
    assert "ACTIVE" in str(exc.value)
    assert status in str(exc.value)


@pytest.mark.parametrize("value", [True, "0.1", np.nan, np.inf])
def test_resolver_does_not_bypass_b2_return_validation(value: object) -> None:
    with pytest.raises(MissingSecurityReturnError, match="finite real"):
        resolve_security_return(
            trade_date="2024-01-03",
            ts_code="ACTIVE",
            status=AVAILABLE,
            observed_return=value,
        )


def test_return_and_suspended_status_conflict_fails_closed() -> None:
    suspensions = pd.DataFrame(
        {
            "trade_date": ["20240103"],
            "ts_code": ["ACTIVE"],
            "suspend_type": ["S"],
            "suspend_timing": [None],
        }
    )
    with pytest.raises(SecuritySuspensionDataError, match="conflict"):
        build_security_status(
            ts_codes=["ACTIVE"],
            evaluation_dates=["2024-01-03"],
            lifecycle=_lifecycle(),
            suspensions=suspensions,
            security_returns=_returns([("20240103", "ACTIVE", 1.0)]),
            trading_calendar=_calendar(),
        )
    with pytest.raises(MissingSecurityReturnError, match="conflicts"):
        resolve_security_return(
            trade_date="2024-01-03",
            ts_code="ACTIVE",
            status=SUSPENDED,
            observed_return=0.01,
        )


def test_closed_day_never_generates_a_return_status_or_zero() -> None:
    with pytest.raises(SecurityAvailabilityError, match="closed date"):
        build_security_status(
            ts_codes=["ACTIVE"],
            evaluation_dates=["2024-01-06"],
            lifecycle=_lifecycle(),
            suspensions=_empty_suspensions(),
            security_returns=_returns([]),
            trading_calendar=_calendar(),
        )


def test_status_builder_is_bounded_sorted_and_frequency_agnostic() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-12"]
    result = build_security_status(
        ts_codes=["NEW", "ACTIVE"],
        evaluation_dates=reversed(dates),
        lifecycle=_lifecycle(),
        suspensions=_empty_suspensions(),
        security_returns=_returns([]),
        trading_calendar=_calendar(),
    )
    assert tuple(result.columns) == SECURITY_STATUS_COLUMNS
    assert len(result) == 8
    expected = result.sort_values(
        ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
    )
    pdt.assert_frame_equal(result, expected)


class _FakeAvailabilityClient:
    def __init__(
        self,
        lifecycle_frames: dict[str, pd.DataFrame] | None = None,
        suspension_frames: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.lifecycle_frames = lifecycle_frames or {}
        self.suspension_frames = suspension_frames or {}
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get_stock_basic(self, list_status: str = "L") -> pd.DataFrame:
        self.calls.append(("stock_basic", {"list_status": list_status}))
        return self.lifecycle_frames.get(list_status, pd.DataFrame())

    def get_suspend_d(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("suspend_d", kwargs))
        return self.suspension_frames.get(str(kwargs["ts_code"]), pd.DataFrame())


def test_lifecycle_adapter_calls_each_explicit_status_and_reconciles() -> None:
    client = _FakeAvailabilityClient(
        lifecycle_frames={
            "L": _raw_lifecycle().loc[lambda item: item["list_status"].eq("L")],
            "D": _raw_lifecycle().loc[lambda item: item["list_status"].eq("D")],
            "P": pd.DataFrame(),
        }
    )
    result = TushareSecurityLifecycleAdapter(client).load(
        list_statuses=["L", "D", "P"]
    )
    assert result.equals(_lifecycle())
    assert client.calls == [
        ("stock_basic", {"list_status": "L"}),
        ("stock_basic", {"list_status": "D"}),
        ("stock_basic", {"list_status": "P"}),
    ]


def test_suspension_adapter_calls_explicit_codes_and_all_event_types() -> None:
    client = _FakeAvailabilityClient(
        suspension_frames={
            "A": pd.DataFrame(
                {
                    "trade_date": ["20240103"],
                    "ts_code": ["A"],
                    "suspend_type": ["S"],
                    "suspend_timing": [None],
                }
            )
        }
    )
    result = TushareSecuritySuspensionAdapter(client).load(
        ts_codes=["A", "B"], start_date="20240102", end_date="20240105"
    )
    assert result["ts_code"].tolist() == ["A"]
    assert client.calls == [
        (
            "suspend_d",
            {
                "ts_code": "A",
                "trade_date": None,
                "start_date": "20240102",
                "end_date": "20240105",
                "suspend_type": None,
            },
        ),
        (
            "suspend_d",
            {
                "ts_code": "B",
                "trade_date": None,
                "start_date": "20240102",
                "end_date": "20240105",
                "suspend_type": None,
            },
        ),
    ]


@pytest.mark.parametrize("adapter", ["lifecycle", "suspension"])
def test_provider_adapters_wrap_failures_without_network(adapter: str) -> None:
    class BrokenClient(_FakeAvailabilityClient):
        def get_stock_basic(self, list_status: str = "L") -> pd.DataFrame:
            raise RuntimeError("offline")

        def get_suspend_d(self, **kwargs: object) -> pd.DataFrame:
            raise RuntimeError("offline")

    with pytest.raises(SecurityAvailabilityProviderError) as exc:
        if adapter == "lifecycle":
            TushareSecurityLifecycleAdapter(BrokenClient()).load(
                list_statuses=["L"]
            )
        else:
            TushareSecuritySuspensionAdapter(BrokenClient()).load(
                ts_codes=["A"], start_date="20240102", end_date="20240103"
            )
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_status_builder_does_not_mutate_b2_returns_or_status_inputs() -> None:
    lifecycle = _lifecycle()
    suspensions = _empty_suspensions()
    returns = _returns([("20240103", "ACTIVE", 1.0)])
    originals = tuple(
        item.copy(deep=True) for item in (lifecycle, suspensions, returns)
    )
    build_security_status(
        ts_codes=["ACTIVE"],
        evaluation_dates=["20240103"],
        lifecycle=lifecycle,
        suspensions=suspensions,
        security_returns=returns,
        trading_calendar=_calendar(),
    )
    for actual, original in zip((lifecycle, suspensions, returns), originals):
        pdt.assert_frame_equal(actual, original)


def test_timezone_aware_dates_are_rejected() -> None:
    source = _raw_lifecycle().iloc[[0]].copy(deep=True)
    source["list_date"] = source["list_date"].astype(object)
    source.at[source.index[0], "list_date"] = datetime(
        2020, 1, 1, tzinfo=timezone.utc
    )
    with pytest.raises(SecurityLifecycleError, match="timezone-naive"):
        build_security_lifecycle(source)
