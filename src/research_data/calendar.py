"""Canonical research formation calendar and typed history windows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import pandas as pd

from src.data.contracts import ResearchFrequency, canonical_date, formation_dates


class ResearchCalendarError(ValueError):
    """Raised when canonical calendar evidence cannot prove a requested window."""


class HistoryKind(str, Enum):
    TRADING_DAYS = "TRADING_DAYS"
    CALENDAR_MONTHS = "CALENDAR_MONTHS"
    LATEST_AS_OF = "LATEST_AS_OF"


@dataclass(frozen=True)
class HistoryRequirement:
    kind: HistoryKind
    count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, HistoryKind):
            raise TypeError("kind must be a HistoryKind.")
        if self.kind is HistoryKind.LATEST_AS_OF:
            if self.count is not None:
                raise ValueError("LATEST_AS_OF does not accept a count.")
        elif type(self.count) is not int or self.count < 1:
            raise ValueError(f"{self.kind.value} count must be a strict positive integer.")

    @classmethod
    def trading_days(cls, count: int) -> "HistoryRequirement":
        return cls(HistoryKind.TRADING_DAYS, count)

    @classmethod
    def calendar_months(cls, count: int) -> "HistoryRequirement":
        return cls(HistoryKind.CALENDAR_MONTHS, count)

    @classmethod
    def latest_as_of(cls) -> "HistoryRequirement":
        return cls(HistoryKind.LATEST_AS_OF)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "HistoryRequirement":
        if not isinstance(value, Mapping) or set(value) != {"kind", "count"}:
            raise ValueError("HistoryRequirement requires only kind and count.")
        try:
            kind = HistoryKind(value["kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown history requirement kind: {value.get('kind')!r}.") from exc
        return cls(kind, value["count"])  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "count": self.count}


@dataclass(frozen=True)
class HistoryWindow:
    start_date: str
    end_date: str
    open_dates: tuple[str, ...]

    def __post_init__(self) -> None:
        start = canonical_date(self.start_date)
        end = canonical_date(self.end_date)
        dates = tuple(canonical_date(item) for item in self.open_dates)
        if start > end or not dates or dates != tuple(sorted(set(dates))):
            raise ResearchCalendarError("history window must contain ordered unique open dates.")
        if dates[0] != start or dates[-1] != end:
            raise ResearchCalendarError("history window boundaries must equal its first and last open dates.")
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)
        object.__setattr__(self, "open_dates", dates)


class ResearchCalendar:
    """Use canonical trade_cal rows as the sole formation/history calendar truth."""

    def __init__(self, trade_calendar: pd.DataFrame) -> None:
        if not isinstance(trade_calendar, pd.DataFrame):
            raise TypeError("trade_calendar must be a pandas DataFrame.")
        required = {"cal_date", "is_open"}
        missing = sorted(required - set(trade_calendar.columns))
        if missing or trade_calendar.empty:
            raise ResearchCalendarError(f"trade_calendar is empty or missing fields: {missing!r}.")
        rows = trade_calendar.loc[:, ["cal_date", "is_open"]].copy(deep=True)
        try:
            rows["cal_date"] = [canonical_date(item) for item in rows["cal_date"]]
        except ValueError as exc:
            raise ResearchCalendarError("trade_calendar contains an invalid cal_date.") from exc
        if rows["cal_date"].duplicated().any():
            raise ResearchCalendarError("trade_calendar contains duplicate cal_date rows.")
        if rows["is_open"].map(lambda value: isinstance(value, bool)).any():
            raise ResearchCalendarError("trade_calendar is_open cannot contain bool values.")
        numeric = pd.to_numeric(rows["is_open"], errors="coerce")
        if numeric.isna().any() or not numeric.isin((0, 1)).all():
            raise ResearchCalendarError("trade_calendar is_open must contain only 0 or 1.")
        rows["is_open"] = numeric.astype(int)
        self._rows = rows.sort_values("cal_date", kind="mergesort", ignore_index=True)
        self._all_dates = tuple(self._rows["cal_date"])
        self._open_dates = tuple(self._rows.loc[self._rows["is_open"].eq(1), "cal_date"])

    def _covered(self, start: object, end: object) -> tuple[str, str]:
        first = canonical_date(start)
        last = canonical_date(end)
        if first > last:
            raise ResearchCalendarError("start_date must not be after end_date.")
        expected = tuple(day.strftime("%Y-%m-%d") for day in pd.date_range(first, last, freq="D"))
        observed = tuple(item for item in self._all_dates if first <= item <= last)
        if observed != expected:
            raise ResearchCalendarError("canonical trade calendar does not completely cover the requested interval.")
        return first, last

    def formation_dates(self, frequency: ResearchFrequency, start_date: object, end_date: object) -> tuple[str, ...]:
        if not isinstance(frequency, ResearchFrequency):
            raise TypeError("frequency must be a ResearchFrequency.")
        start, end = self._covered(start_date, end_date)
        open_dates = tuple(item for item in self._open_dates if start <= item <= end)
        if frequency is ResearchFrequency.DAILY:
            if not open_dates:
                raise ResearchCalendarError("requested interval has no open trading date.")
            return formation_dates(frequency, open_dates)
        result: list[str] = []
        for month in pd.period_range(pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M"):
            month_start = month.start_time.date().isoformat()
            month_end = month.end_time.date().isoformat()
            expected = tuple(day.strftime("%Y-%m-%d") for day in pd.date_range(month_start, month_end, freq="D"))
            observed = tuple(item for item in self._all_dates if month_start <= item <= month_end)
            if observed != expected:
                if month_end > end:
                    continue  # An unproven partial final month has no formation yet.
                raise ResearchCalendarError(f"canonical trade calendar cannot prove month-end formation for {month!s}.")
            month_open = tuple(item for item in self._open_dates if month_start <= item <= month_end)
            if not month_open:
                raise ResearchCalendarError(f"calendar month has no open trading date: {month!s}.")
            formation = month_open[-1]
            if start <= formation <= end:
                result.append(formation)
        return tuple(result)

    def resolve_history(self, formation_date: object, requirement: HistoryRequirement) -> HistoryWindow:
        formation = canonical_date(formation_date)
        if not isinstance(requirement, HistoryRequirement):
            raise TypeError("requirement must be a HistoryRequirement.")
        if formation not in set(self._open_dates):
            raise ResearchCalendarError("formation_date must be a proven open trading day.")
        through = tuple(item for item in self._open_dates if item <= formation)
        if requirement.kind is HistoryKind.TRADING_DAYS:
            count = requirement.count or 0
            if len(through) < count:
                raise ResearchCalendarError("canonical trade calendar has insufficient trading-day history.")
            selected = through[-count:]
        elif requirement.kind is HistoryKind.CALENDAR_MONTHS:
            count = requirement.count or 0
            formation_month = pd.Period(formation, freq="M")
            first_month = formation_month - (count - 1)
            boundary = first_month.start_time.date().isoformat()
            selected = tuple(item for item in through if item >= boundary)
            if not selected or selected[0][:7] != str(first_month):
                raise ResearchCalendarError("canonical trade calendar has insufficient calendar-month history.")
        else:
            selected = (formation,)
        return HistoryWindow(selected[0], formation, selected)

    def shift_open_date(self, anchor_date: object, periods: int) -> str:
        """Return the exact open date ``periods`` positions after an open anchor."""
        anchor = canonical_date(anchor_date)
        if type(periods) is not int or periods < 0:
            raise ResearchCalendarError("periods must be a strict non-negative integer.")
        try:
            position = self._open_dates.index(anchor)
        except ValueError as exc:
            raise ResearchCalendarError("anchor_date must be a proven open trading day.") from exc
        target = position + periods
        if target >= len(self._open_dates):
            raise ResearchCalendarError("canonical trade calendar has insufficient future open dates.")
        return self._open_dates[target]

    @property
    def open_dates(self) -> tuple[str, ...]:
        return self._open_dates
