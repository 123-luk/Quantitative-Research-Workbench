"""Canonical TuShare-reported daily returns for V6 research backtesting.

Security returns use ``daily.pct_chg / 100`` and benchmark returns use
``index_daily.pct_chg / 100``. This module intentionally does not reconstruct
prices, fill sparse security observations, or decide suspension/listing/delist
policy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from src.research_backtest.calendar import TradingCalendarDataError, _calendar_date


SECURITY_DAILY_RETURN_COLUMNS = ("trade_date", "ts_code", "return")
BENCHMARK_DAILY_RETURN_COLUMNS = ("trade_date", "benchmark_code", "return")
SECURITY_RETURN_SOURCE_NAME = "tushare.daily"
BENCHMARK_RETURN_SOURCE_NAME = "tushare.index_daily"
RAW_RETURN_FIELD = "pct_chg"
RETURN_UNIT = "decimal"
RETURN_CONVENTION = "adjusted_close_to_close"


class MarketReturnError(ValueError):
    """Base error for canonical market-return operations."""


class SecurityReturnDataError(MarketReturnError):
    """Raised when raw security daily returns violate the contract."""


class BenchmarkReturnDataError(MarketReturnError):
    """Raised when raw benchmark daily returns violate the contract."""


class MarketReturnProviderError(MarketReturnError):
    """Raised when an injected daily-return provider call fails."""


class DailyReturnClient(Protocol):
    """Structural subset of the existing TuShare wrapper used by B2."""

    def get_daily(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Return raw security daily rows."""

    def get_index_daily(
        self,
        ts_code: str,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Return raw benchmark daily rows."""


def _date_value(
    value: object,
    *,
    field_name: str,
    error_type: type[MarketReturnError],
) -> pd.Timestamp:
    try:
        return _calendar_date(value, field_name=field_name)
    except TradingCalendarDataError as exc:
        raise error_type(str(exc)) from exc


def _required_frame(
    frame: object,
    *,
    required: tuple[str, ...],
    context: str,
    error_type: type[MarketReturnError],
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise error_type(f"{context} must be a pandas DataFrame.")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise error_type(f"{context} is missing required columns: {missing!r}.")
    if frame.empty:
        raise error_type(f"{context} must not be empty.")
    return frame.loc[:, list(required)].copy(deep=True)


def _strict_code(
    value: object,
    *,
    field_name: str,
    error_type: type[MarketReturnError],
) -> str:
    if not isinstance(value, str):
        raise error_type(f"{field_name} must be a string.")
    code = value.strip()
    if not code:
        raise error_type(f"{field_name} must be non-empty.")
    return code


def _decimal_returns(
    values: pd.Series,
    *,
    error_type: type[MarketReturnError],
) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype) or not pd.api.types.is_numeric_dtype(
        values.dtype
    ):
        raise error_type("pct_chg must contain real numeric values, not bool or text.")
    try:
        numeric = values.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise error_type("pct_chg must contain real numeric values.") from exc
    if not np.isfinite(numeric).all():
        raise error_type("pct_chg must contain only finite, non-null values.")
    return numeric / 100.0


def build_security_daily_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert sparse TuShare security ``pct_chg`` rows to decimal returns."""
    normalized = _required_frame(
        frame,
        required=("trade_date", "ts_code", RAW_RETURN_FIELD),
        context="security daily rows",
        error_type=SecurityReturnDataError,
    )
    normalized["trade_date"] = [
        _date_value(
            value,
            field_name=f"trade_date[{index!r}]",
            error_type=SecurityReturnDataError,
        )
        for index, value in normalized["trade_date"].items()
    ]
    normalized["ts_code"] = [
        _strict_code(
            value,
            field_name=f"ts_code[{index!r}]",
            error_type=SecurityReturnDataError,
        )
        for index, value in normalized["ts_code"].items()
    ]
    normalized["return"] = _decimal_returns(
        normalized[RAW_RETURN_FIELD], error_type=SecurityReturnDataError
    )
    if normalized.duplicated(["trade_date", "ts_code"]).any():
        raise SecurityReturnDataError(
            "security daily rows must have unique (trade_date, ts_code) keys."
        )
    result = normalized.loc[:, list(SECURITY_DAILY_RETURN_COLUMNS)]
    return result.sort_values(
        ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
    )


def build_benchmark_daily_returns(
    frame: pd.DataFrame,
    *,
    benchmark_code: object,
) -> pd.DataFrame:
    """Convert one explicit TuShare index ``pct_chg`` series to decimals."""
    code = _strict_code(
        benchmark_code,
        field_name="benchmark_code",
        error_type=BenchmarkReturnDataError,
    )
    required = ("trade_date", RAW_RETURN_FIELD)
    normalized = _required_frame(
        frame,
        required=required,
        context="benchmark daily rows",
        error_type=BenchmarkReturnDataError,
    )
    if "ts_code" in frame.columns:
        raw_codes = tuple(
            _strict_code(
                value,
                field_name=f"ts_code[{index!r}]",
                error_type=BenchmarkReturnDataError,
            )
            for index, value in frame["ts_code"].items()
        )
        observed = tuple(sorted(set(raw_codes)))
        if observed != (code,):
            raise BenchmarkReturnDataError(
                "raw benchmark ts_code values must match the explicit "
                f"benchmark_code {code!r}; observed={observed!r}."
            )
    normalized["trade_date"] = [
        _date_value(
            value,
            field_name=f"trade_date[{index!r}]",
            error_type=BenchmarkReturnDataError,
        )
        for index, value in normalized["trade_date"].items()
    ]
    normalized["return"] = _decimal_returns(
        normalized[RAW_RETURN_FIELD], error_type=BenchmarkReturnDataError
    )
    if normalized["trade_date"].duplicated().any():
        raise BenchmarkReturnDataError(
            "benchmark daily rows must have unique trade_date keys."
        )
    normalized["benchmark_code"] = code
    result = normalized.loc[:, list(BENCHMARK_DAILY_RETURN_COLUMNS)]
    return result.sort_values("trade_date", kind="mergesort", ignore_index=True)


def _security_codes(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise SecurityReturnDataError("ts_codes must be an iterable of strings.")
    codes = tuple(
        _strict_code(
            item,
            field_name="ts_codes",
            error_type=SecurityReturnDataError,
        )
        for item in values
    )
    if not codes:
        raise SecurityReturnDataError("ts_codes must not be empty.")
    if len(codes) != len(set(codes)):
        raise SecurityReturnDataError("ts_codes must be unique.")
    return codes


def _date_range(
    start_date: object,
    end_date: object,
    *,
    error_type: type[MarketReturnError],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = _date_value(start_date, field_name="start_date", error_type=error_type)
    end = _date_value(end_date, field_name="end_date", error_type=error_type)
    if start > end:
        raise error_type("start_date must be earlier than or equal to end_date.")
    return start, end


@dataclass(frozen=True)
class TushareSecurityDailyReturnAdapter:
    """Load explicit security scopes without filling missing provider rows."""

    client: DailyReturnClient

    def __post_init__(self) -> None:
        if not isinstance(getattr(self.client, "get_daily", None), Callable):
            raise TypeError("client must provide callable get_daily(...).")

    def load(
        self,
        *,
        ts_codes: object,
        start_date: object,
        end_date: object,
    ) -> pd.DataFrame:
        """Fetch each explicit code and return the observed sparse panel."""
        codes = _security_codes(ts_codes)
        start, end = _date_range(
            start_date, end_date, error_type=SecurityReturnDataError
        )
        frames: list[pd.DataFrame] = []
        for code in codes:
            try:
                frame = self.client.get_daily(
                    ts_code=code,
                    trade_date=None,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                )
            except Exception as exc:
                raise MarketReturnProviderError(
                    f"TuShare daily provider call failed for {code!r}."
                ) from exc
            if not isinstance(frame, pd.DataFrame):
                raise SecurityReturnDataError(
                    "security daily provider result must be a pandas DataFrame."
                )
            if not frame.empty:
                frames.append(frame.copy(deep=True))
        if not frames:
            raise SecurityReturnDataError(
                "security daily provider returned no rows for the explicit scope."
            )
        result = build_security_daily_returns(
            pd.concat(frames, ignore_index=True, sort=False)
        )
        observed_codes = set(result["ts_code"])
        unexpected = tuple(sorted(observed_codes - set(codes)))
        if unexpected:
            raise SecurityReturnDataError(
                "security provider returned codes outside the explicit scope: "
                f"{unexpected!r}."
            )
        if not result["trade_date"].between(start, end).all():
            raise SecurityReturnDataError(
                "security provider returned dates outside the explicit scope."
            )
        return result


@dataclass(frozen=True)
class TushareBenchmarkDailyReturnAdapter:
    """Load one explicit benchmark return series without calendar alignment."""

    client: DailyReturnClient

    def __post_init__(self) -> None:
        if not isinstance(getattr(self.client, "get_index_daily", None), Callable):
            raise TypeError("client must provide callable get_index_daily(...).")

    def load(
        self,
        *,
        benchmark_code: object,
        start_date: object,
        end_date: object,
    ) -> pd.DataFrame:
        """Fetch and canonicalize one explicitly named benchmark."""
        code = _strict_code(
            benchmark_code,
            field_name="benchmark_code",
            error_type=BenchmarkReturnDataError,
        )
        start, end = _date_range(
            start_date, end_date, error_type=BenchmarkReturnDataError
        )
        try:
            frame = self.client.get_index_daily(
                ts_code=code,
                trade_date=None,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
        except Exception as exc:
            raise MarketReturnProviderError(
                f"TuShare index_daily provider call failed for {code!r}."
            ) from exc
        result = build_benchmark_daily_returns(frame, benchmark_code=code)
        if not result["trade_date"].between(start, end).all():
            raise BenchmarkReturnDataError(
                "benchmark provider returned dates outside the explicit scope."
            )
        return result
