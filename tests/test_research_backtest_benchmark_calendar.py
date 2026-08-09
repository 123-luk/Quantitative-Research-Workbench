"""Tests for strict V6 strategy/benchmark calendar alignment."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.research_backtest import (
    BenchmarkCalendarAlignmentError,
    BenchmarkCalendarError,
    validate_strict_common_calendar,
)


def test_exact_same_calendar_returns_sorted_canonical_dates() -> None:
    result = validate_strict_common_calendar(
        ["2024-01-02", date(2024, 1, 3), pd.Timestamp("2024-01-04")],
        ["20240102", datetime(2024, 1, 3), "2024-01-04"],
    )
    assert result == tuple(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))


def test_same_set_in_different_order_passes_after_canonical_sort() -> None:
    expected = tuple(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert validate_strict_common_calendar(
        reversed(expected), [expected[1], expected[2], expected[0]]
    ) == expected


@pytest.mark.parametrize("side", ["strategy", "benchmark"])
def test_duplicate_dates_are_rejected_before_alignment(side: str) -> None:
    unique = ["2024-01-02", "2024-01-03"]
    duplicate = ["2024-01-02", "2024-01-02"]
    strategy, benchmark = (
        (duplicate, unique) if side == "strategy" else (unique, duplicate)
    )
    with pytest.raises(BenchmarkCalendarError, match=f"{side}_dates.*unique"):
        validate_strict_common_calendar(strategy, benchmark)


def test_missing_and_extra_dates_are_reported_deterministically() -> None:
    with pytest.raises(BenchmarkCalendarAlignmentError) as exc:
        validate_strict_common_calendar(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            ["2024-01-02", "2024-01-05"],
        )
    assert exc.value.missing_in_benchmark == tuple(
        pd.to_datetime(["2024-01-03", "2024-01-04"])
    )
    assert exc.value.extra_in_benchmark == (pd.Timestamp("2024-01-05"),)
    assert "count=2" in str(exc.value)
    assert "count=1" in str(exc.value)


def test_missing_benchmark_observation_never_silently_intersects() -> None:
    with pytest.raises(BenchmarkCalendarAlignmentError) as exc:
        validate_strict_common_calendar(
            ["2024-01-02", "2024-01-03"], ["2024-01-02"]
        )
    assert exc.value.missing_in_benchmark == (pd.Timestamp("2024-01-03"),)
    assert exc.value.extra_in_benchmark == ()


def test_extra_benchmark_observation_never_silently_drops() -> None:
    with pytest.raises(BenchmarkCalendarAlignmentError) as exc:
        validate_strict_common_calendar(
            ["2024-01-02"], ["2024-01-02", "2024-01-03"]
        )
    assert exc.value.missing_in_benchmark == ()
    assert exc.value.extra_in_benchmark == (pd.Timestamp("2024-01-03"),)


@pytest.mark.parametrize(
    ("strategy", "benchmark"),
    [([], ["2024-01-02"]), (["2024-01-02"], []), ([], [])],
)
def test_empty_calendars_are_rejected(
    strategy: list[str], benchmark: list[str]
) -> None:
    with pytest.raises(BenchmarkCalendarError, match="must not be empty"):
        validate_strict_common_calendar(strategy, benchmark)


@pytest.mark.parametrize(
    "value",
    [None, pd.NaT, "", "2024/01/02", "bad-date", 20240102],
)
def test_malformed_dates_are_rejected(value: object) -> None:
    with pytest.raises(BenchmarkCalendarError):
        validate_strict_common_calendar([value], ["2024-01-02"])


@pytest.mark.parametrize(
    "value",
    [
        pd.Timestamp("2024-01-02", tz="Asia/Shanghai"),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    ],
)
def test_timezone_aware_dates_are_rejected(value: object) -> None:
    with pytest.raises(BenchmarkCalendarError, match="timezone-naive"):
        validate_strict_common_calendar([value], [value])


def test_weekend_observation_is_a_set_member_not_an_inferred_market_error() -> None:
    weekend = ["2024-01-06", "2024-01-07"]
    assert validate_strict_common_calendar(weekend, weekend) == tuple(
        pd.to_datetime(weekend)
    )


def test_multiple_same_month_dates_are_fully_preserved() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-12"]
    result = validate_strict_common_calendar(dates, reversed(dates))
    assert len(result) == 4
    assert result == tuple(pd.to_datetime(dates))


def test_non_iterable_and_scalar_text_inputs_are_rejected() -> None:
    for value in ("2024-01-02", 20240102, None):
        with pytest.raises(BenchmarkCalendarError, match="iterable"):
            validate_strict_common_calendar(value, ["2024-01-02"])


def test_public_alignment_result_is_immutable_and_deterministic() -> None:
    strategy = ["2024-01-03", "2024-01-02"]
    benchmark = ["2024-01-02", "2024-01-03"]
    first = validate_strict_common_calendar(strategy, benchmark)
    second = validate_strict_common_calendar(strategy, benchmark)
    assert isinstance(first, tuple)
    assert first == second
    assert strategy == ["2024-01-03", "2024-01-02"]
    assert benchmark == ["2024-01-02", "2024-01-03"]
