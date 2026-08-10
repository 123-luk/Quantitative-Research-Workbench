"""Concrete historical-return service reusing canonical V6 B1/B2/B3 semantics."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from src.research_backtest.availability import (
    PRE_LISTING,
    SUPPORTED_LIST_STATUSES,
    TushareSecurityLifecycleAdapter,
    TushareSecuritySuspensionAdapter,
    build_security_status,
    resolve_security_return,
)
from src.research_backtest.calendar import TushareTradingCalendarAdapter
from src.research_backtest.returns import TushareSecurityDailyReturnAdapter

from ..contracts import HistoricalReturnWindow, normalize_date
from ..errors import PortfolioConstructionDataError


class ResearchBacktestHistoricalReturnService:
    """Resolve exact open-date return windows through injected V6 adapters."""

    _MAX_EXPANSIONS = 10

    def __init__(self, client: object) -> None:
        self._calendar = TushareTradingCalendarAdapter(client)  # type: ignore[arg-type]
        self._returns = TushareSecurityDailyReturnAdapter(client)  # type: ignore[arg-type]
        self._lifecycle = TushareSecurityLifecycleAdapter(client)  # type: ignore[arg-type]
        self._suspensions = TushareSecuritySuspensionAdapter(client)  # type: ignore[arg-type]
        self._cache: dict[
            tuple[tuple[str, ...], pd.Timestamp, int], HistoricalReturnWindow
        ] = {}

    def load_window(
        self,
        ts_codes: tuple[str, ...],
        formation_date: pd.Timestamp,
        lookback_trading_days: int,
    ) -> HistoricalReturnWindow:
        codes = self._codes(ts_codes)
        formation = normalize_date(formation_date, field_name="formation_date")
        if type(lookback_trading_days) is not int or lookback_trading_days < 2:
            raise PortfolioConstructionDataError(
                "lookback_trading_days must be a strict int >= 2."
            )
        key = (codes, formation, lookback_trading_days)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            open_dates, calendar = self._open_window(
                formation, lookback_trading_days
            )
            risk_cutoff = open_dates[-1]
            start = open_dates[0]
            observed = self._returns.load(
                ts_codes=codes, start_date=start, end_date=risk_cutoff
            )
            lifecycle = self._lifecycle.load(
                list_statuses=SUPPORTED_LIST_STATUSES
            )
            suspensions = self._suspensions.load(
                ts_codes=codes, start_date=start, end_date=risk_cutoff
            )
            statuses = build_security_status(
                ts_codes=codes,
                evaluation_dates=open_dates,
                lifecycle=lifecycle,
                suspensions=suspensions,
                security_returns=observed,
                trading_calendar=calendar,
            )
            observed_map = {
                (trade_date, ts_code): return_value
                for trade_date, ts_code, return_value in zip(
                    observed["trade_date"],
                    observed["ts_code"],
                    observed["return"],
                    strict=True,
                )
            }
            rows: list[dict[str, object]] = []
            for row in statuses.itertuples(index=False):
                if row.status == PRE_LISTING:
                    continue
                value = resolve_security_return(
                    trade_date=row.trade_date,
                    ts_code=row.ts_code,
                    status=row.status,
                    observed_return=observed_map.get(
                        (row.trade_date, row.ts_code)
                    ),
                )
                rows.append(
                    {
                        "trade_date": row.trade_date,
                        "ts_code": row.ts_code,
                        "return": value,
                    }
                )
            frame = pd.DataFrame(
                rows, columns=["trade_date", "ts_code", "return"]
            )
            window = HistoricalReturnWindow(risk_cutoff, frame)
        except PortfolioConstructionDataError:
            raise
        except Exception as exc:
            raise PortfolioConstructionDataError(
                "canonical historical-return resolution failed."
            ) from exc
        self._cache[key] = window
        return window

    def _open_window(
        self, formation: pd.Timestamp, lookback: int
    ) -> tuple[tuple[pd.Timestamp, ...], object]:
        span_days = max(32, lookback * 2)
        for _ in range(self._MAX_EXPANSIONS):
            start = formation - timedelta(days=span_days)
            calendar = self._calendar.load(
                start_date=start, end_date=formation
            )
            if len(calendar.open_dates) >= lookback:
                return calendar.open_dates[-lookback:], calendar
            span_days *= 2
        raise PortfolioConstructionDataError(
            "calendar expansion could not resolve the requested open-date window."
        )

    @staticmethod
    def _codes(value: object) -> tuple[str, ...]:
        if not isinstance(value, tuple) or not value:
            raise PortfolioConstructionDataError(
                "ts_codes must be a non-empty tuple."
            )
        codes = tuple(value)
        if (
            len(codes) != len(set(codes))
            or any(
                not isinstance(code, str)
                or not code
                or code != code.strip()
                for code in codes
            )
        ):
            raise PortfolioConstructionDataError(
                "ts_codes must contain unique non-empty trimmed strings."
            )
        return codes
