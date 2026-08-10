"""Gate E tests for the concrete V6-backed historical-return service."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from src.portfolio_construction import (
    PortfolioConstructionDataError,
)
from src.portfolio_construction.adapters import (
    ResearchBacktestHistoricalReturnService,
)


@dataclass
class FakeMarketClient:
    returns: dict[str, dict[str, float]]
    list_dates: dict[str, str]
    suspensions: list[dict[str, object]] = field(default_factory=list)
    forced_closed: frozenset[str] = frozenset()
    calendar_calls: list[tuple[str, str]] = field(default_factory=list)
    daily_calls: list[tuple[str, str, str]] = field(default_factory=list)

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        self.calendar_calls.append((start_date, end_date))
        dates = pd.date_range(start_date, end_date, freq="D")
        return pd.DataFrame(
            {
                "cal_date": dates.strftime("%Y%m%d"),
                "is_open": [
                    int(date.weekday() < 5 and date.strftime("%Y%m%d") not in self.forced_closed)
                    for date in dates
                ],
            }
        )

    def get_daily(
        self,
        ts_code=None,
        trade_date=None,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        del trade_date
        self.daily_calls.append((ts_code, start_date, end_date))
        rows = [
            {
                "trade_date": date,
                "ts_code": ts_code,
                "pct_chg": value,
            }
            for date, value in self.returns.get(ts_code, {}).items()
            if start_date <= date <= end_date
        ]
        return pd.DataFrame(rows)

    def get_stock_basic(self, list_status="L") -> pd.DataFrame:
        if list_status != "L":
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "ts_code": code,
                    "list_status": "L",
                    "list_date": list_date,
                    "delist_date": None,
                }
                for code, list_date in self.list_dates.items()
            ]
        )

    def get_suspend_d(
        self,
        ts_code=None,
        trade_date=None,
        start_date=None,
        end_date=None,
        suspend_type=None,
    ) -> pd.DataFrame:
        del trade_date, suspend_type
        rows = [
            row
            for row in self.suspensions
            if row["ts_code"] == ts_code
            and start_date <= row["trade_date"] <= end_date
        ]
        return pd.DataFrame(rows)


def weekday_returns(
    start: str = "20240101", end: str = "20240112"
) -> dict[str, dict[str, float]]:
    dates = pd.bdate_range(start, end)
    return {
        "A": {date.strftime("%Y%m%d"): float(index + 1) for index, date in enumerate(dates)},
        "B": {date.strftime("%Y%m%d"): float(2 * index + 2) for index, date in enumerate(dates)},
    }


def test_service_resolves_exact_open_window_cutoff_and_decimal_returns() -> None:
    client = FakeMarketClient(
        weekday_returns(), {"A": "20200101", "B": "20200101"}
    )
    service = ResearchBacktestHistoricalReturnService(client)
    window = service.load_window(("A", "B"), pd.Timestamp("2024-01-10"), 5)
    assert window.risk_cutoff == pd.Timestamp("2024-01-10")
    assert tuple(window.returns["trade_date"].drop_duplicates()) == tuple(
        pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])
    )
    assert window.returns["return"].max() < 1.0
    assert set(window.returns["ts_code"]) == {"A", "B"}
    assert all(end == "20240110" for _, _, end in client.daily_calls)


def test_weekend_formation_uses_previous_open_cutoff() -> None:
    client = FakeMarketClient(
        weekday_returns(), {"A": "20200101", "B": "20200101"}
    )
    window = ResearchBacktestHistoricalReturnService(client).load_window(
        ("A", "B"), pd.Timestamp("2024-01-13"), 5
    )
    assert window.risk_cutoff == pd.Timestamp("2024-01-12")
    assert window.returns["trade_date"].max() == pd.Timestamp("2024-01-12")


def test_calendar_expands_deterministically_across_long_closure() -> None:
    all_dates = pd.date_range("2023-11-01", "2024-01-10", freq="D")
    closed = frozenset(
        date.strftime("%Y%m%d")
        for date in all_dates
        if pd.Timestamp("2023-12-01") <= date <= pd.Timestamp("2024-01-08")
    )
    open_dates = [
        date
        for date in all_dates
        if date.weekday() < 5 and date.strftime("%Y%m%d") not in closed
    ]
    data = {
        code: {date.strftime("%Y%m%d"): 1.0 + index for index, date in enumerate(open_dates)}
        for code in ("A", "B")
    }
    client = FakeMarketClient(
        data, {"A": "20200101", "B": "20200101"}, forced_closed=closed
    )
    window = ResearchBacktestHistoricalReturnService(client).load_window(
        ("A", "B"), pd.Timestamp("2024-01-10"), 5
    )
    assert len(client.calendar_calls) >= 2
    assert window.returns.groupby("ts_code").size().to_dict() == {"A": 5, "B": 5}


def test_pre_listing_dates_are_excluded_not_zero_filled() -> None:
    data = weekday_returns()
    data["B"] = {
        date: value for date, value in data["B"].items() if date >= "20240108"
    }
    client = FakeMarketClient(data, {"A": "20200101", "B": "20240108"})
    window = ResearchBacktestHistoricalReturnService(client).load_window(
        ("A", "B"), pd.Timestamp("2024-01-10"), 5
    )
    assert window.returns.groupby("ts_code").size().to_dict() == {"A": 5, "B": 3}


def test_proven_full_day_suspension_resolves_zero_and_counts() -> None:
    data = weekday_returns()
    del data["A"]["20240109"]
    client = FakeMarketClient(
        data,
        {"A": "20200101", "B": "20200101"},
        suspensions=[
            {
                "trade_date": "20240109",
                "ts_code": "A",
                "suspend_type": "S",
                "suspend_timing": None,
            }
        ],
    )
    window = ResearchBacktestHistoricalReturnService(client).load_window(
        ("A", "B"), pd.Timestamp("2024-01-10"), 5
    )
    row = window.returns.loc[
        window.returns["trade_date"].eq("2024-01-09")
        & window.returns["ts_code"].eq("A")
    ]
    assert row["return"].tolist() == [0.0]
    assert len(window.returns.loc[window.returns["ts_code"].eq("A")]) == 5


def test_active_unknown_missing_fails_closed() -> None:
    data = weekday_returns()
    del data["A"]["20240109"]
    client = FakeMarketClient(data, {"A": "20200101", "B": "20200101"})
    with pytest.raises(PortfolioConstructionDataError):
        ResearchBacktestHistoricalReturnService(client).load_window(
            ("A", "B"), pd.Timestamp("2024-01-10"), 5
        )


def test_t_plus_one_perturbation_cannot_change_formation_window() -> None:
    base = weekday_returns()
    first = {code: dict(values) for code, values in base.items()}
    second = {code: dict(values) for code, values in base.items()}
    first["A"]["20240111"] = 100.0
    second["A"]["20240111"] = -100.0
    services = [
        ResearchBacktestHistoricalReturnService(
            FakeMarketClient(data, {"A": "20200101", "B": "20200101"})
        )
        for data in (first, second)
    ]
    windows = [
        service.load_window(("A", "B"), pd.Timestamp("2024-01-10"), 5)
        for service in services
    ]
    pd.testing.assert_frame_equal(windows[0].returns, windows[1].returns)


def test_run_scoped_memoization_avoids_duplicate_provider_calls() -> None:
    client = FakeMarketClient(
        weekday_returns(), {"A": "20200101", "B": "20200101"}
    )
    service = ResearchBacktestHistoricalReturnService(client)
    first = service.load_window(("A", "B"), pd.Timestamp("2024-01-10"), 5)
    call_counts = (len(client.calendar_calls), len(client.daily_calls))
    second = service.load_window(("A", "B"), pd.Timestamp("2024-01-10"), 5)
    assert second is first
    assert (len(client.calendar_calls), len(client.daily_calls)) == call_counts
