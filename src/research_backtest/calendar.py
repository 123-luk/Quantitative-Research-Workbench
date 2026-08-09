"""Validated market calendar and TuShare calendar adapter for V6 research."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral
from typing import Protocol

import pandas as pd


class TradingCalendarError(ValueError):
    """Base error for canonical trading-calendar operations."""


class TradingCalendarDataError(TradingCalendarError):
    """Raised when provider calendar rows violate the data contract."""


class TradingCalendarCoverageError(TradingCalendarError):
    """Raised when known calendar coverage cannot answer a date request."""


class TradingCalendarProviderError(TradingCalendarError):
    """Raised when the injected provider cannot return calendar rows."""


class TradeCalendarClient(Protocol):
    """Structural protocol implemented by the existing TuShare wrapper."""

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return provider calendar rows for an inclusive YYYYMMDD range."""


def _calendar_date(
    value: object,
    *,
    field_name: str,
    error_type: type[TradingCalendarError] = TradingCalendarDataError,
) -> pd.Timestamp:
    """Normalize one supported date-like value to a naive midnight Timestamp."""
    if value is None or value is pd.NaT:
        raise error_type(f"{field_name} must be a valid date and cannot be null.")
    if isinstance(value, str):
        text = value.strip()
        if text != value or not text:
            raise error_type(
                f"{field_name} must use YYYY-MM-DD or YYYYMMDD format."
            )
        date_format = "%Y%m%d" if len(text) == 8 and text.isdigit() else "%Y-%m-%d"
        try:
            timestamp = pd.Timestamp(datetime.strptime(text, date_format).date())
        except ValueError as exc:
            raise error_type(
                f"{field_name} must use YYYY-MM-DD or YYYYMMDD format."
            ) from exc
        return timestamp
    if not isinstance(value, (pd.Timestamp, datetime, date)):
        raise error_type(
            f"{field_name} must be an ISO date string, Timestamp, datetime, or date."
        )
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise error_type(f"{field_name} must be a valid date.") from exc
    if pd.isna(timestamp):
        raise error_type(f"{field_name} must be a valid date and cannot be NaT.")
    if timestamp.tz is not None:
        raise error_type(f"{field_name} must be timezone-naive.")
    return timestamp.normalize()


def _provider_open_flag(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TradingCalendarDataError("is_open must contain only integer 0 or 1.")
    normalized = int(value)
    if normalized not in (0, 1):
        raise TradingCalendarDataError("is_open must contain only integer 0 or 1.")
    return normalized


@dataclass(frozen=True)
class TradingCalendar:
    """Immutable open dates with explicit, inclusive natural-date coverage."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    open_dates: tuple[pd.Timestamp, ...]

    def __post_init__(self) -> None:
        start = _calendar_date(self.start_date, field_name="start_date")
        end = _calendar_date(self.end_date, field_name="end_date")
        if start > end:
            raise TradingCalendarCoverageError(
                "start_date must be earlier than or equal to end_date."
            )
        if not isinstance(self.open_dates, tuple):
            raise TradingCalendarDataError("open_dates must be a tuple.")
        dates = tuple(
            _calendar_date(item, field_name="open_dates") for item in self.open_dates
        )
        if not dates:
            raise TradingCalendarCoverageError(
                "calendar coverage must contain at least one open trading date."
            )
        if dates != tuple(sorted(dates)) or len(dates) != len(set(dates)):
            raise TradingCalendarDataError(
                "open_dates must be strictly increasing and unique."
            )
        if dates[0] < start or dates[-1] > end:
            raise TradingCalendarCoverageError(
                "open_dates must stay within the declared calendar coverage."
            )
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)
        object.__setattr__(self, "open_dates", dates)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        start_date: object,
        end_date: object,
    ) -> TradingCalendar:
        """Validate provider rows, prove coverage, and retain only open dates.

        Raw row order is not trusted because the existing TuShare wrapper does
        not promise sorting. Duplicate dates and gaps in the inclusive provider
        response fail closed; unsorted valid rows are sorted deterministically.
        """
        start = _calendar_date(start_date, field_name="start_date")
        end = _calendar_date(end_date, field_name="end_date")
        if start > end:
            raise TradingCalendarCoverageError(
                "start_date must be earlier than or equal to end_date."
            )
        if not isinstance(frame, pd.DataFrame):
            raise TradingCalendarDataError("calendar rows must be a pandas DataFrame.")
        missing_columns = [
            name for name in ("cal_date", "is_open") if name not in frame.columns
        ]
        if missing_columns:
            raise TradingCalendarDataError(
                f"calendar rows are missing required columns: {missing_columns!r}."
            )
        if frame.empty:
            raise TradingCalendarCoverageError(
                "provider calendar rows must not be empty."
            )

        rows = frame.loc[:, ["cal_date", "is_open"]].copy(deep=True)
        normalized_dates: list[pd.Timestamp] = []
        open_flags: list[int] = []
        for index, row in rows.iterrows():
            normalized_dates.append(
                _calendar_date(row["cal_date"], field_name=f"cal_date[{index!r}]")
            )
            open_flags.append(_provider_open_flag(row["is_open"]))
        rows["cal_date"] = normalized_dates
        rows["is_open"] = open_flags
        if rows["cal_date"].duplicated().any():
            duplicates = tuple(
                item.strftime("%Y-%m-%d")
                for item in sorted(
                    rows.loc[rows["cal_date"].duplicated(False), "cal_date"].unique()
                )
            )
            raise TradingCalendarDataError(
                f"calendar rows contain duplicate cal_date values: {duplicates!r}."
            )

        requested = pd.date_range(start, end, freq="D")
        available = set(rows["cal_date"])
        missing_coverage = tuple(
            item.strftime("%Y-%m-%d") for item in requested if item not in available
        )
        if missing_coverage:
            raise TradingCalendarCoverageError(
                "provider calendar rows do not cover every requested natural date; "
                f"missing: {missing_coverage!r}."
            )
        within = rows.loc[rows["cal_date"].between(start, end)]
        open_dates = tuple(
            sorted(within.loc[within["is_open"].eq(1), "cal_date"].tolist())
        )
        return cls(start_date=start, end_date=end, open_dates=open_dates)

    def _known_date(self, value: object) -> pd.Timestamp:
        timestamp = _calendar_date(value, field_name="date")
        if timestamp < self.start_date or timestamp > self.end_date:
            raise TradingCalendarCoverageError(
                f"date {timestamp.strftime('%Y-%m-%d')} is outside calendar coverage "
                f"[{self.start_date.strftime('%Y-%m-%d')}, "
                f"{self.end_date.strftime('%Y-%m-%d')}]."
            )
        return timestamp

    def is_trading_day(self, value: object) -> bool:
        """Return open/closed status only when the date is within coverage."""
        return self._known_date(value) in self.open_dates

    def next_trading_day(self, value: object) -> pd.Timestamp:
        """Return the first known open date strictly greater than the input."""
        timestamp = self._known_date(value)
        position = bisect_right(self.open_dates, timestamp)
        if position >= len(self.open_dates):
            raise TradingCalendarCoverageError(
                "calendar has insufficient future coverage to determine the next "
                f"trading day after {timestamp.strftime('%Y-%m-%d')}."
            )
        result = self.open_dates[position]
        if result <= timestamp:  # Defensive no-look-ahead assertion.
            raise TradingCalendarCoverageError(
                "next trading day must be strictly greater than the input date."
            )
        return result


@dataclass(frozen=True)
class TushareTradingCalendarAdapter:
    """Load a canonical calendar through an injected existing TuShare client."""

    client: TradeCalendarClient

    def __post_init__(self) -> None:
        method = getattr(self.client, "get_trade_cal", None)
        if not isinstance(method, Callable):
            raise TypeError(
                "client must provide callable get_trade_cal(start_date, end_date)."
            )

    def load(self, *, start_date: object, end_date: object) -> TradingCalendar:
        """Fetch one explicit range and validate it without network-side globals."""
        start = _calendar_date(start_date, field_name="start_date")
        end = _calendar_date(end_date, field_name="end_date")
        if start > end:
            raise TradingCalendarCoverageError(
                "start_date must be earlier than or equal to end_date."
            )
        provider_start = start.strftime("%Y%m%d")
        provider_end = end.strftime("%Y%m%d")
        try:
            frame = self.client.get_trade_cal(
                start_date=provider_start,
                end_date=provider_end,
            )
        except Exception as exc:
            raise TradingCalendarProviderError(
                "TuShare trade calendar provider call failed."
            ) from exc
        return TradingCalendar.from_frame(
            frame,
            start_date=start,
            end_date=end,
        )
