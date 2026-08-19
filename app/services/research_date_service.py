"""Shared Shanghai-business-date validation for Workbench submissions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo


SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class ResearchDateValidation:
    start: date
    end: date
    today: date
    code: str | None = None

    @property
    def valid(self) -> bool:
        return self.code is None


class ResearchDateError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def shanghai_today(now: datetime | None = None) -> date:
    value = now or datetime.now(SHANGHAI_TIMEZONE)
    if value.tzinfo is None:
        value = value.replace(tzinfo=SHANGHAI_TIMEZONE)
    return value.astimezone(SHANGHAI_TIMEZONE).date()


def validate_research_dates(
    start: date | str,
    end: date | str,
    *,
    today: date | None = None,
) -> ResearchDateValidation:
    start_date = date.fromisoformat(start) if isinstance(start, str) else start
    end_date = date.fromisoformat(end) if isinstance(end, str) else end
    if not isinstance(start_date, date) or not isinstance(end_date, date):
        raise TypeError("research dates must be date or ISO date strings")
    business_today = today or shanghai_today()
    code = None
    if start_date > business_today or end_date > business_today:
        code = "future"
    elif end_date <= start_date:
        code = "order"
    return ResearchDateValidation(start_date, end_date, business_today, code)


def require_valid_research_dates(start: date | str, end: date | str) -> None:
    result = validate_research_dates(start, end)
    if not result.valid:
        raise ResearchDateError(result.code or "invalid")
