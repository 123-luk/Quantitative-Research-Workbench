"""Tests for the V6 canonical trading calendar and TuShare adapter."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pandas.testing as pdt
import pytest

from src.research_backtest import (
    TradingCalendar,
    TradingCalendarCoverageError,
    TradingCalendarDataError,
    TradingCalendarProviderError,
    TushareTradingCalendarAdapter,
)


def _calendar_rows(
    start: str = "2024-01-01",
    end: str = "2024-01-10",
    *,
    closed: set[str] | None = None,
) -> pd.DataFrame:
    closed_dates = closed or {"2024-01-01", "2024-01-06", "2024-01-07"}
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame(
        {
            "exchange": ["SSE"] * len(dates),
            "cal_date": [item.strftime("%Y%m%d") for item in dates],
            "is_open": [
                0 if item.strftime("%Y-%m-%d") in closed_dates else 1
                for item in dates
            ],
            "pretrade_date": [None] * len(dates),
        }
    )


def _calendar() -> TradingCalendar:
    return TradingCalendar.from_frame(
        _calendar_rows(), start_date="2024-01-01", end_date="2024-01-10"
    )


def test_open_dates_are_filtered_sorted_and_canonical() -> None:
    source = _calendar_rows().sample(frac=1.0, random_state=17).reset_index(drop=True)
    original = source.copy(deep=True)
    calendar = TradingCalendar.from_frame(
        source, start_date="20240101", end_date=date(2024, 1, 10)
    )
    assert calendar.start_date == pd.Timestamp("2024-01-01")
    assert calendar.end_date == pd.Timestamp("2024-01-10")
    assert calendar.open_dates == tuple(
        pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
            ]
        )
    )
    pdt.assert_frame_equal(source, original)


def test_is_trading_day_distinguishes_open_weekend_and_holiday() -> None:
    calendar = _calendar()
    assert calendar.is_trading_day("2024-01-02") is True
    assert calendar.is_trading_day(pd.Timestamp("2024-01-06")) is False
    assert calendar.is_trading_day(date(2024, 1, 1)) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-01-04", "2024-01-05"),
        ("2024-01-05", "2024-01-08"),
        ("2024-01-06", "2024-01-08"),
        ("2024-01-01", "2024-01-02"),
    ],
)
def test_next_trading_day_is_strict_for_open_weekend_and_holiday(
    value: object, expected: str
) -> None:
    result = _calendar().next_trading_day(value)
    assert result == pd.Timestamp(expected)
    assert result > pd.Timestamp(value)


def test_pre_holiday_date_skips_all_closed_natural_dates() -> None:
    rows = _calendar_rows(
        "2024-02-08",
        "2024-02-19",
        closed={
            "2024-02-09",
            "2024-02-10",
            "2024-02-11",
            "2024-02-12",
            "2024-02-13",
            "2024-02-14",
            "2024-02-15",
            "2024-02-16",
            "2024-02-17",
            "2024-02-18",
        },
    )
    calendar = TradingCalendar.from_frame(
        rows, start_date="2024-02-08", end_date="2024-02-19"
    )
    assert calendar.next_trading_day("2024-02-08") == pd.Timestamp("2024-02-19")


def test_no_natural_day_or_business_day_fallback() -> None:
    rows = _calendar_rows(
        "2024-01-05",
        "2024-01-09",
        closed={"2024-01-06", "2024-01-07", "2024-01-08"},
    )
    calendar = TradingCalendar.from_frame(
        rows, start_date="2024-01-05", end_date="2024-01-09"
    )
    assert calendar.next_trading_day("2024-01-05") == pd.Timestamp("2024-01-09")


@pytest.mark.parametrize("value", ["2023-12-31", "2024-01-11"])
def test_is_trading_day_fails_closed_outside_coverage(value: str) -> None:
    with pytest.raises(TradingCalendarCoverageError, match="outside calendar coverage"):
        _calendar().is_trading_day(value)


@pytest.mark.parametrize("value", ["2024-01-10", "2024-01-09"])
def test_next_trading_day_requires_sufficient_future_coverage(value: str) -> None:
    calendar = TradingCalendar.from_frame(
        _calendar_rows(end="2024-01-10", closed={"2024-01-10"}),
        start_date="2024-01-01",
        end_date="2024-01-10",
    )
    with pytest.raises(TradingCalendarCoverageError, match="insufficient future"):
        calendar.next_trading_day(value)


def test_empty_provider_rows_fail_closed() -> None:
    with pytest.raises(TradingCalendarCoverageError, match="must not be empty"):
        TradingCalendar.from_frame(
            pd.DataFrame(columns=["cal_date", "is_open"]),
            start_date="2024-01-01",
            end_date="2024-01-02",
        )


def test_provider_rows_must_cover_every_requested_natural_date() -> None:
    rows = _calendar_rows().loc[lambda item: item["cal_date"] != "20240106"]
    with pytest.raises(TradingCalendarCoverageError, match="2024-01-06"):
        TradingCalendar.from_frame(
            rows, start_date="2024-01-01", end_date="2024-01-10"
        )


def test_start_date_may_be_closed_when_response_covers_it() -> None:
    calendar = _calendar()
    assert not calendar.is_trading_day(calendar.start_date)
    assert calendar.next_trading_day(calendar.start_date) == pd.Timestamp("2024-01-02")


def test_duplicate_calendar_date_is_rejected() -> None:
    rows = pd.concat([_calendar_rows(), _calendar_rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(TradingCalendarDataError, match="duplicate cal_date"):
        TradingCalendar.from_frame(
            rows, start_date="2024-01-01", end_date="2024-01-10"
        )


@pytest.mark.parametrize("value", [2, -1, 0.0, 1.0, True, "1", None])
def test_invalid_is_open_values_are_rejected(value: object) -> None:
    rows = _calendar_rows()
    rows["is_open"] = rows["is_open"].astype(object)
    rows.at[0, "is_open"] = value
    with pytest.raises(TradingCalendarDataError, match="integer 0 or 1"):
        TradingCalendar.from_frame(
            rows, start_date="2024-01-01", end_date="2024-01-10"
        )


@pytest.mark.parametrize(
    "value",
    [None, pd.NaT, "", " 2024-01-01 ", "2024/01/01", "not-a-date"],
)
def test_malformed_calendar_dates_are_rejected(value: object) -> None:
    rows = _calendar_rows()
    rows.loc[0, "cal_date"] = value
    with pytest.raises(TradingCalendarDataError):
        TradingCalendar.from_frame(
            rows, start_date="2024-01-01", end_date="2024-01-10"
        )


@pytest.mark.parametrize(
    "value",
    [
        pd.Timestamp("2024-01-02", tz="Asia/Shanghai"),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    ],
)
def test_timezone_aware_public_dates_are_rejected(value: object) -> None:
    with pytest.raises(TradingCalendarDataError, match="timezone-naive"):
        _calendar().is_trading_day(value)


def test_naive_datetime_is_normalized_to_calendar_date() -> None:
    assert _calendar().is_trading_day(datetime(2024, 1, 2, 18, 30))


def test_invalid_range_and_missing_columns_are_rejected() -> None:
    with pytest.raises(TradingCalendarCoverageError, match="start_date"):
        TradingCalendar.from_frame(
            _calendar_rows(), start_date="2024-01-10", end_date="2024-01-01"
        )
    for column in ("cal_date", "is_open"):
        with pytest.raises(TradingCalendarDataError, match="missing required"):
            TradingCalendar.from_frame(
                _calendar_rows().drop(columns=column),
                start_date="2024-01-01",
                end_date="2024-01-10",
            )


def test_frequency_like_holdings_dates_share_one_strict_next_primitive() -> None:
    dates = pd.date_range("2024-01-01", "2024-03-04", freq="D")
    rows = pd.DataFrame(
        {
            "cal_date": dates,
            "is_open": [int(item.weekday() < 5) for item in dates],
        }
    )
    calendar = TradingCalendar.from_frame(
        rows, start_date="2024-01-01", end_date="2024-03-04"
    )
    samples = {
        "monthly_like": ["2024-01-31", "2024-02-29"],
        "weekly_like": ["2024-01-05", "2024-01-12", "2024-01-19"],
        "daily_like": ["2024-01-02", "2024-01-03", "2024-01-04"],
    }
    for values in samples.values():
        for value in values:
            assert calendar.next_trading_day(value) > pd.Timestamp(value)
    assert calendar.next_trading_day("2024-01-02") == pd.Timestamp("2024-01-03")
    assert calendar.next_trading_day("2024-01-03") == pd.Timestamp("2024-01-04")


class _FakeClient:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[dict[str, str]] = []

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append({"start_date": start_date, "end_date": end_date})
        return self.frame


def test_tushare_adapter_calls_existing_method_with_exact_provider_dates() -> None:
    frame = _calendar_rows()
    original = frame.copy(deep=True)
    client = _FakeClient(frame)
    calendar = TushareTradingCalendarAdapter(client).load(
        start_date=date(2024, 1, 1), end_date=pd.Timestamp("2024-01-10")
    )
    assert client.calls == [{"start_date": "20240101", "end_date": "20240110"}]
    assert calendar == _calendar()
    pdt.assert_frame_equal(frame, original)


def test_tushare_adapter_wraps_provider_exception() -> None:
    class BrokenClient:
        def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
            raise RuntimeError("offline")

    with pytest.raises(
        TradingCalendarProviderError, match="provider call failed"
    ) as exc:
        TushareTradingCalendarAdapter(BrokenClient()).load(
            start_date="2024-01-01", end_date="2024-01-10"
        )
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_adapter_requires_get_trade_cal_and_never_constructs_global_client() -> None:
    with pytest.raises(TypeError, match="get_trade_cal"):
        TushareTradingCalendarAdapter(object())  # type: ignore[arg-type]
