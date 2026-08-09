"""Strict strategy/benchmark observation-calendar compatibility for V6."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.research_backtest.calendar import TradingCalendarDataError, _calendar_date


class BenchmarkCalendarError(ValueError):
    """Base error for benchmark calendar normalization and alignment."""


class BenchmarkCalendarAlignmentError(BenchmarkCalendarError):
    """Report deterministic strategy/benchmark date-set differences."""

    def __init__(
        self,
        *,
        missing_in_benchmark: tuple[pd.Timestamp, ...],
        extra_in_benchmark: tuple[pd.Timestamp, ...],
    ) -> None:
        self.missing_in_benchmark = missing_in_benchmark
        self.extra_in_benchmark = extra_in_benchmark
        missing_text = tuple(item.strftime("%Y-%m-%d") for item in missing_in_benchmark)
        extra_text = tuple(item.strftime("%Y-%m-%d") for item in extra_in_benchmark)
        super().__init__(
            "strategy and benchmark calendars must match exactly; "
            f"missing_in_benchmark={missing_text!r} (count={len(missing_text)}), "
            f"extra_in_benchmark={extra_text!r} (count={len(extra_text)})."
        )


def _canonical_dates(values: object, *, field_name: str) -> tuple[pd.Timestamp, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise BenchmarkCalendarError(f"{field_name} must be an iterable of dates.")
    materialized = tuple(values)
    if not materialized:
        raise BenchmarkCalendarError(f"{field_name} must not be empty.")
    normalized: list[pd.Timestamp] = []
    for index, value in enumerate(materialized):
        try:
            normalized.append(
                _calendar_date(value, field_name=f"{field_name}[{index}]")
            )
        except TradingCalendarDataError as exc:
            raise BenchmarkCalendarError(str(exc)) from exc
    if len(normalized) != len(set(normalized)):
        duplicates = tuple(
            item.strftime("%Y-%m-%d")
            for item in sorted(
                {item for item in normalized if normalized.count(item) > 1}
            )
        )
        raise BenchmarkCalendarError(
            f"{field_name} must contain unique dates; duplicates={duplicates!r}."
        )
    return tuple(sorted(normalized))


def validate_strict_common_calendar(
    strategy_dates: object,
    benchmark_dates: object,
) -> tuple[pd.Timestamp, ...]:
    """Return canonical dates only when both evaluation calendars match exactly.

    Input order is canonicalized because observation row order is not calendar
    meaning. Duplicates remain invalid. No intersection, fill, or unmatched-row
    dropping is performed.
    """
    strategy = _canonical_dates(strategy_dates, field_name="strategy_dates")
    benchmark = _canonical_dates(benchmark_dates, field_name="benchmark_dates")
    if strategy != benchmark:
        strategy_set = set(strategy)
        benchmark_set = set(benchmark)
        raise BenchmarkCalendarAlignmentError(
            missing_in_benchmark=tuple(sorted(strategy_set - benchmark_set)),
            extra_in_benchmark=tuple(sorted(benchmark_set - strategy_set)),
        )
    return strategy
