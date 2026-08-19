"""Auditable security availability and missing-return policy for V6 research.

Only a same-date TuShare ``suspend_d`` suspension record without an intraday
``suspend_timing`` segment can resolve a missing daily holding return to zero.
That zero is a research valuation convention; it does not assert that the
security can be bought, sold, or rebalanced.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

import numpy as np
import pandas as pd

from src.research_backtest.calendar import (
    TradingCalendar,
    TradingCalendarDataError,
    _calendar_date,
)


AVAILABLE = "AVAILABLE"
SUSPENDED = "SUSPENDED"
PRE_LISTING = "PRE_LISTING"
DELIST_DATE = "DELIST_DATE"
POST_DELIST = "POST_DELIST"
UNKNOWN_MISSING = "UNKNOWN_MISSING"
SECURITY_AVAILABILITY_STATUSES = (
    AVAILABLE,
    SUSPENDED,
    PRE_LISTING,
    DELIST_DATE,
    POST_DELIST,
    UNKNOWN_MISSING,
)
SECURITY_STATUS_COLUMNS = ("trade_date", "ts_code", "status")
SECURITY_LIFECYCLE_COLUMNS = ("ts_code", "list_date", "delist_date")
SECURITY_SUSPENSION_COLUMNS = (
    "trade_date",
    "ts_code",
    "suspend_type",
    "suspend_timing",
)
SUPPORTED_LIST_STATUSES = ("L", "D", "P")
SUPPORTED_SUSPEND_TYPES = ("S", "R")


class SecurityAvailabilityError(ValueError):
    """Base error for security status construction and resolution."""


class SecurityLifecycleError(SecurityAvailabilityError):
    """Raised when stock lifecycle data is missing or contradictory."""


class SecuritySuspensionDataError(SecurityAvailabilityError):
    """Raised when daily suspension/resumption evidence is invalid."""


class MissingSecurityReturnError(SecurityAvailabilityError):
    """Raised when a required security return has no permitted resolution."""


class SecurityAvailabilityProviderError(SecurityAvailabilityError):
    """Raised when an injected lifecycle or suspension provider call fails."""


class AvailabilityClient(Protocol):
    """Structural TuShare methods used by the B3 provider adapters."""

    def get_stock_basic(self, list_status: str = "L") -> pd.DataFrame:
        """Return raw stock basics for one listing status."""

    def get_suspend_d(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        suspend_type: str | None = None,
    ) -> pd.DataFrame:
        """Return raw daily suspension/resumption records."""


def _date_value(
    value: object,
    *,
    field_name: str,
    error_type: type[SecurityAvailabilityError],
) -> pd.Timestamp:
    try:
        return _calendar_date(value, field_name=field_name)
    except TradingCalendarDataError as exc:
        raise error_type(str(exc)) from exc


def _optional_date(
    value: object,
    *,
    field_name: str,
    error_type: type[SecurityAvailabilityError],
) -> pd.Timestamp | None:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, str) and value == "":
        return None
    try:
        if not isinstance(value, (str, bytes)) and bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return _date_value(value, field_name=field_name, error_type=error_type)


def _code(
    value: object,
    *,
    field_name: str,
    error_type: type[SecurityAvailabilityError],
) -> str:
    if not isinstance(value, str):
        raise error_type(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise error_type(f"{field_name} must be non-empty.")
    return normalized


def _choice(
    value: object,
    *,
    field_name: str,
    allowed: tuple[str, ...],
    error_type: type[SecurityAvailabilityError],
) -> str:
    if not isinstance(value, str):
        raise error_type(f"{field_name} must be a string.")
    normalized = value.strip().upper()
    if normalized not in allowed:
        raise error_type(f"{field_name} must be one of {allowed!r}.")
    return normalized


def _frame(
    value: object,
    *,
    required: tuple[str, ...],
    context: str,
    error_type: type[SecurityAvailabilityError],
    allow_empty: bool,
) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise error_type(f"{context} must be a pandas DataFrame.")
    missing = [column for column in required if column not in value.columns]
    if missing:
        raise error_type(f"{context} is missing required columns: {missing!r}.")
    if value.empty and not allow_empty:
        raise error_type(f"{context} must not be empty.")
    return value.loc[:, list(required)].copy(deep=True)


def build_security_lifecycle(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconcile explicit L/D/P stock-basic rows into one lifecycle per code.

    Exact lifecycle facts may repeat across distinct listing statuses. Repeated
    rows for the same status or conflicting list/delist dates fail closed.
    """
    rows = _frame(
        frame,
        required=("ts_code", "list_status", "list_date", "delist_date"),
        context="stock lifecycle rows",
        error_type=SecurityLifecycleError,
        allow_empty=False,
    )
    rows["ts_code"] = [
        _code(
            value,
            field_name=f"ts_code[{index!r}]",
            error_type=SecurityLifecycleError,
        )
        for index, value in rows["ts_code"].items()
    ]
    rows["list_status"] = [
        _choice(
            value,
            field_name=f"list_status[{index!r}]",
            allowed=SUPPORTED_LIST_STATUSES,
            error_type=SecurityLifecycleError,
        )
        for index, value in rows["list_status"].items()
    ]
    rows["list_date"] = [
        _date_value(
            value,
            field_name=f"list_date[{index!r}]",
            error_type=SecurityLifecycleError,
        )
        for index, value in rows["list_date"].items()
    ]
    rows["delist_date"] = [
        _optional_date(
            value,
            field_name=f"delist_date[{index!r}]",
            error_type=SecurityLifecycleError,
        )
        for index, value in rows["delist_date"].items()
    ]

    canonical: list[dict[str, object]] = []
    for code, group in rows.groupby("ts_code", sort=True):
        if group["list_status"].duplicated().any():
            raise SecurityLifecycleError(
                f"duplicate lifecycle list_status for ts_code={code!r}."
            )
        list_dates = tuple(sorted(set(group["list_date"])))
        delist_dates = tuple(
            sorted({item for item in group["delist_date"] if not pd.isna(item)})
        )
        if len(list_dates) != 1 or len(delist_dates) > 1:
            raise SecurityLifecycleError(
                f"conflicting lifecycle dates for ts_code={code!r}."
            )
        list_date = list_dates[0]
        delist_date = delist_dates[0] if delist_dates else None
        if "D" in set(group["list_status"]) and delist_date is None:
            raise SecurityLifecycleError(
                f"delisted ts_code={code!r} requires delist_date."
            )
        if delist_date is not None and delist_date < list_date:
            raise SecurityLifecycleError(
                f"delist_date cannot precede list_date for ts_code={code!r}."
            )
        canonical.append(
            {
                "ts_code": code,
                "list_date": list_date,
                "delist_date": delist_date,
            }
        )
    return pd.DataFrame(canonical, columns=list(SECURITY_LIFECYCLE_COLUMNS))


def build_security_suspensions(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize same-date TuShare suspend/resume evidence without intervals."""
    rows = _frame(
        frame,
        required=SECURITY_SUSPENSION_COLUMNS,
        context="security suspension rows",
        error_type=SecuritySuspensionDataError,
        allow_empty=True,
    )
    if rows.empty:
        return rows.astype(
            {
                "trade_date": "datetime64[ns]",
                "ts_code": "object",
                "suspend_type": "object",
                "suspend_timing": "object",
            }
        )
    rows["trade_date"] = [
        _date_value(
            value,
            field_name=f"trade_date[{index!r}]",
            error_type=SecuritySuspensionDataError,
        )
        for index, value in rows["trade_date"].items()
    ]
    rows["ts_code"] = [
        _code(
            value,
            field_name=f"ts_code[{index!r}]",
            error_type=SecuritySuspensionDataError,
        )
        for index, value in rows["ts_code"].items()
    ]
    rows["suspend_type"] = [
        _choice(
            value,
            field_name=f"suspend_type[{index!r}]",
            allowed=SUPPORTED_SUSPEND_TYPES,
            error_type=SecuritySuspensionDataError,
        )
        for index, value in rows["suspend_type"].items()
    ]
    timing: list[str | None] = []
    for index, value in rows["suspend_timing"].items():
        if value is None or (not isinstance(value, str) and bool(pd.isna(value))):
            timing.append(None)
        elif isinstance(value, str):
            timing.append(value.strip() or None)
        else:
            raise SecuritySuspensionDataError(
                f"suspend_timing[{index!r}] must be a string or null."
            )
    rows["suspend_timing"] = timing
    if rows.duplicated(["trade_date", "ts_code"]).any():
        raise SecuritySuspensionDataError(
            "suspension rows must have unique (trade_date, ts_code) keys."
        )
    return rows.sort_values(
        ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
    )


def _codes(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise SecurityAvailabilityError("ts_codes must be an iterable of strings.")
    codes = tuple(
        _code(
            item,
            field_name="ts_codes",
            error_type=SecurityAvailabilityError,
        )
        for item in values
    )
    if not codes or len(codes) != len(set(codes)):
        raise SecurityAvailabilityError("ts_codes must be non-empty and unique.")
    return codes


def _dates(values: object) -> tuple[pd.Timestamp, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise SecurityAvailabilityError(
            "evaluation_dates must be an iterable of dates."
        )
    dates = tuple(
        _date_value(
            item,
            field_name="evaluation_dates",
            error_type=SecurityAvailabilityError,
        )
        for item in values
    )
    if not dates or len(dates) != len(set(dates)):
        raise SecurityAvailabilityError(
            "evaluation_dates must be non-empty and unique."
        )
    return tuple(sorted(dates))


def _canonical_lifecycle(frame: pd.DataFrame) -> pd.DataFrame:
    rows = _frame(
        frame,
        required=SECURITY_LIFECYCLE_COLUMNS,
        context="canonical lifecycle",
        error_type=SecurityLifecycleError,
        allow_empty=False,
    )
    rows["ts_code"] = [
        _code(value, field_name="ts_code", error_type=SecurityLifecycleError)
        for value in rows["ts_code"]
    ]
    rows["list_date"] = [
        _date_value(
            value, field_name="list_date", error_type=SecurityLifecycleError
        )
        for value in rows["list_date"]
    ]
    rows["delist_date"] = [
        _optional_date(
            value, field_name="delist_date", error_type=SecurityLifecycleError
        )
        for value in rows["delist_date"]
    ]
    if rows["ts_code"].duplicated().any():
        raise SecurityLifecycleError("canonical lifecycle ts_code must be unique.")
    for row in rows.itertuples(index=False):
        if not pd.isna(row.delist_date) and row.delist_date < row.list_date:
            raise SecurityLifecycleError(
                f"delist_date cannot precede list_date for ts_code={row.ts_code!r}."
            )
    return rows.sort_values("ts_code", kind="mergesort", ignore_index=True)


def _canonical_returns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = _frame(
        frame,
        required=("trade_date", "ts_code", "return"),
        context="canonical security returns",
        error_type=SecurityAvailabilityError,
        allow_empty=True,
    )
    if rows.empty:
        return rows
    rows["trade_date"] = [
        _date_value(
            value, field_name="trade_date", error_type=SecurityAvailabilityError
        )
        for value in rows["trade_date"]
    ]
    rows["ts_code"] = [
        _code(value, field_name="ts_code", error_type=SecurityAvailabilityError)
        for value in rows["ts_code"]
    ]
    for value in rows["return"]:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise SecurityAvailabilityError("return must contain finite real values.")
        if not np.isfinite(float(value)):
            raise SecurityAvailabilityError("return must contain finite real values.")
    if rows.duplicated(["trade_date", "ts_code"]).any():
        raise SecurityAvailabilityError(
            "canonical security return keys must be unique."
        )
    return rows


def build_security_status(
    *,
    ts_codes: object,
    evaluation_dates: object,
    lifecycle: pd.DataFrame,
    suspensions: pd.DataFrame,
    security_returns: pd.DataFrame,
    trading_calendar: TradingCalendar,
) -> pd.DataFrame:
    """Build bounded daily status rows from lifecycle, events, and return presence."""
    if not isinstance(trading_calendar, TradingCalendar):
        raise SecurityAvailabilityError(
            "trading_calendar must be a canonical TradingCalendar."
        )
    codes = _codes(ts_codes)
    dates = _dates(evaluation_dates)
    for trade_date in dates:
        if not trading_calendar.is_trading_day(trade_date):
            raise SecurityAvailabilityError(
                f"closed date {trade_date.strftime('%Y-%m-%d')} cannot be a "
                "daily return evaluation date."
            )
    lifecycle_rows = _canonical_lifecycle(lifecycle)
    missing_lifecycle = tuple(sorted(set(codes) - set(lifecycle_rows["ts_code"])))
    if missing_lifecycle:
        raise SecurityLifecycleError(
            f"requested ts_codes lack lifecycle data: {missing_lifecycle!r}."
        )
    lifecycle_rows = lifecycle_rows.set_index("ts_code")
    suspension_rows = build_security_suspensions(suspensions)
    return_rows = _canonical_returns(security_returns)
    requested_keys = {(trade_date, code) for trade_date in dates for code in codes}
    suspension_keys = set(
        zip(suspension_rows["trade_date"], suspension_rows["ts_code"])
    )
    return_keys = set(zip(return_rows["trade_date"], return_rows["ts_code"]))
    if not suspension_keys.issubset(requested_keys):
        raise SecuritySuspensionDataError(
            "suspension rows must stay within the explicit evaluation scope."
        )
    if not return_keys.issubset(requested_keys):
        raise SecurityAvailabilityError(
            "security return rows must stay within the explicit evaluation scope."
        )
    suspension_events = {
        (row.trade_date, row.ts_code): (row.suspend_type, row.suspend_timing)
        for row in suspension_rows.itertuples(index=False)
    }

    output: list[dict[str, object]] = []
    for trade_date in dates:
        for code in sorted(codes):
            lifecycle_row = lifecycle_rows.loc[code]
            list_date = lifecycle_row["list_date"]
            raw_delist_date = lifecycle_row["delist_date"]
            delist_date = None if pd.isna(raw_delist_date) else raw_delist_date
            has_return = (trade_date, code) in return_keys
            suspend_event = suspension_events.get((trade_date, code))
            proven_full_day_suspension = suspend_event == ("S", None)
            if trade_date < list_date:
                status = PRE_LISTING
            elif delist_date is not None and trade_date > delist_date:
                status = POST_DELIST
            elif delist_date is not None and trade_date == delist_date:
                status = DELIST_DATE
            elif has_return and proven_full_day_suspension:
                raise SecuritySuspensionDataError(
                    "return/status conflict for "
                    f"trade_date={trade_date.strftime('%Y-%m-%d')}, "
                    f"ts_code={code!r}, status={SUSPENDED}."
                )
            elif has_return:
                status = AVAILABLE
            elif proven_full_day_suspension:
                status = SUSPENDED
            else:
                status = UNKNOWN_MISSING
            if status in (PRE_LISTING, POST_DELIST) and (
                has_return or suspend_event is not None
            ):
                raise SecurityAvailabilityError(
                    "lifecycle/data conflict for "
                    f"trade_date={trade_date.strftime('%Y-%m-%d')}, "
                    f"ts_code={code!r}, status={status}."
                )
            output.append(
                {"trade_date": trade_date, "ts_code": code, "status": status}
            )
    return pd.DataFrame(output, columns=list(SECURITY_STATUS_COLUMNS))


def resolve_security_return(
    *,
    trade_date: object,
    ts_code: object,
    status: object,
    observed_return: object | None,
    suspension_mode: str = "STRICT_EVENT",
) -> float:
    """Use an observation or resolve only a proven suspension to research zero.

    A suspended zero is daily holding-value accounting only. It conveys no
    tradability, execution, order, or rebalance assumption.
    """
    date_value = _date_value(
        trade_date,
        field_name="trade_date",
        error_type=MissingSecurityReturnError,
    )
    code = _code(
        ts_code, field_name="ts_code", error_type=MissingSecurityReturnError
    )
    status_value = _choice(
        status,
        field_name="status",
        allowed=SECURITY_AVAILABILITY_STATUSES,
        error_type=MissingSecurityReturnError,
    )
    identity = (
        f"trade_date={date_value.strftime('%Y-%m-%d')}, "
        f"ts_code={code!r}, status={status_value}"
    )
    if observed_return is not None:
        if isinstance(observed_return, bool) or not isinstance(observed_return, Real):
            raise MissingSecurityReturnError(
                f"observed return must be finite real data for {identity}."
            )
        result = float(observed_return)
        if not np.isfinite(result):
            raise MissingSecurityReturnError(
                f"observed return must be finite real data for {identity}."
            )
        if status_value not in (AVAILABLE, DELIST_DATE):
            raise MissingSecurityReturnError(
                f"observed return conflicts with classified status for {identity}."
            )
        return result
    if status_value == SUSPENDED or (
        suspension_mode == "STANDARD_ROBUST" and status_value == UNKNOWN_MISSING
    ):
        return 0.0
    raise MissingSecurityReturnError(
        f"missing security return has no permitted resolution for {identity}."
    )


@dataclass(frozen=True)
class TushareSecurityLifecycleAdapter:
    """Load explicit stock-basic listing statuses through dependency injection."""

    client: AvailabilityClient

    def __post_init__(self) -> None:
        if not isinstance(getattr(self.client, "get_stock_basic", None), Callable):
            raise TypeError("client must provide callable get_stock_basic(...).")

    def load(self, *, list_statuses: object) -> pd.DataFrame:
        if isinstance(list_statuses, (str, bytes)) or not isinstance(
            list_statuses, Iterable
        ):
            raise SecurityLifecycleError(
                "list_statuses must be an iterable containing L, D, or P."
            )
        statuses = tuple(
            _choice(
                item,
                field_name="list_statuses",
                allowed=SUPPORTED_LIST_STATUSES,
                error_type=SecurityLifecycleError,
            )
            for item in list_statuses
        )
        if not statuses or len(statuses) != len(set(statuses)):
            raise SecurityLifecycleError(
                "list_statuses must be non-empty and unique."
            )
        frames: list[pd.DataFrame] = []
        for status in statuses:
            try:
                frame = self.client.get_stock_basic(list_status=status)
            except Exception as exc:
                raise SecurityAvailabilityProviderError(
                    "TuShare stock_basic provider call failed for "
                    f"list_status={status!r}."
                ) from exc
            if not isinstance(frame, pd.DataFrame):
                raise SecurityLifecycleError(
                    "stock_basic provider result must be a pandas DataFrame."
                )
            if not frame.empty:
                frames.append(frame.copy(deep=True))
        if not frames:
            raise SecurityLifecycleError(
                "stock_basic provider returned no lifecycle rows."
            )
        combined = pd.concat(frames, ignore_index=True, sort=False)
        observed = set(combined.get("list_status", pd.Series(dtype=object)))
        if not observed.issubset(set(statuses)):
            raise SecurityLifecycleError(
                "stock_basic provider returned statuses outside the explicit scope."
            )
        return build_security_lifecycle(combined)


@dataclass(frozen=True)
class TushareSecuritySuspensionAdapter:
    """Load bounded same-date suspend/resume evidence for explicit securities."""

    client: AvailabilityClient

    def __post_init__(self) -> None:
        if not isinstance(getattr(self.client, "get_suspend_d", None), Callable):
            raise TypeError("client must provide callable get_suspend_d(...).")

    def load(
        self,
        *,
        ts_codes: object,
        start_date: object,
        end_date: object,
    ) -> pd.DataFrame:
        codes = _codes(ts_codes)
        start = _date_value(
            start_date,
            field_name="start_date",
            error_type=SecuritySuspensionDataError,
        )
        end = _date_value(
            end_date,
            field_name="end_date",
            error_type=SecuritySuspensionDataError,
        )
        if start > end:
            raise SecuritySuspensionDataError(
                "start_date must be earlier than or equal to end_date."
            )
        frames: list[pd.DataFrame] = []
        for code in codes:
            try:
                frame = self.client.get_suspend_d(
                    ts_code=code,
                    trade_date=None,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    suspend_type=None,
                )
            except Exception as exc:
                raise SecurityAvailabilityProviderError(
                    f"TuShare suspend_d provider call failed for ts_code={code!r}."
                ) from exc
            if not isinstance(frame, pd.DataFrame):
                raise SecuritySuspensionDataError(
                    "suspend_d provider result must be a pandas DataFrame."
                )
            if not frame.empty:
                frames.append(frame.copy(deep=True))
        if not frames:
            return build_security_suspensions(
                pd.DataFrame(columns=list(SECURITY_SUSPENSION_COLUMNS))
            )
        result = build_security_suspensions(
            pd.concat(frames, ignore_index=True, sort=False)
        )
        if not set(result["ts_code"]).issubset(set(codes)):
            raise SecuritySuspensionDataError(
                "suspend_d provider returned codes outside the explicit scope."
            )
        if not result["trade_date"].between(start, end).all():
            raise SecuritySuspensionDataError(
                "suspend_d provider returned dates outside the explicit scope."
            )
        return result
