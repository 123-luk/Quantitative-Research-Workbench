"""Offline delegation tests for the minimal TuShare client methods used by V6."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.tushare_client import TushareClient


class _FakePro:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.stock_frame = pd.DataFrame({"raw": ["stock_basic"]})
        self.daily_frame = pd.DataFrame({"raw": ["daily"]})
        self.index_frame = pd.DataFrame({"raw": ["index_daily"]})
        self.suspend_frame = pd.DataFrame({"raw": ["suspend_d"]})

    def stock_basic(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("stock_basic", kwargs))
        return self.stock_frame

    def daily(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("daily", kwargs))
        return self.daily_frame

    def index_daily(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("index_daily", kwargs))
        return self.index_frame

    def suspend_d(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("suspend_d", kwargs))
        return self.suspend_frame


def _client(pro: _FakePro) -> TushareClient:
    client = object.__new__(TushareClient)
    client.pro = pro
    return client


@pytest.mark.parametrize("list_status", ["L", "D", "P"])
def test_get_stock_basic_passes_explicit_lifecycle_status(list_status: str) -> None:
    pro = _FakePro()
    result = _client(pro).get_stock_basic(list_status=list_status)
    assert result is pro.stock_frame
    assert pro.calls == [
        (
            "stock_basic",
            {
                "exchange": "",
                "list_status": list_status,
                "fields": (
                    "ts_code,symbol,name,area,industry,market,list_status,"
                    "list_date,delist_date"
                ),
            },
        )
    ]


def test_get_stock_basic_preserves_listed_default_for_backward_compatibility() -> None:
    pro = _FakePro()
    _client(pro).get_stock_basic()
    assert pro.calls[0][1]["list_status"] == "L"


def test_get_daily_delegates_exact_scope_and_returns_raw_frame() -> None:
    pro = _FakePro()
    result = _client(pro).get_daily(
        ts_code="000001.SZ",
        trade_date="20240102",
        start_date="20240101",
        end_date="20240131",
    )
    assert result is pro.daily_frame
    assert pro.calls == [
        (
            "daily",
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "start_date": "20240101",
                "end_date": "20240131",
                "fields": (
                    "ts_code,trade_date,open,high,low,close,pre_close,change,"
                    "pct_chg,vol,amount"
                ),
            },
        )
    ]


def test_get_index_daily_requires_explicit_code_and_returns_raw_frame() -> None:
    pro = _FakePro()
    result = _client(pro).get_index_daily(
        ts_code="000905.SH",
        trade_date=None,
        start_date="20240101",
        end_date="20240131",
    )
    assert result is pro.index_frame
    assert pro.calls == [
        (
            "index_daily",
            {
                "ts_code": "000905.SH",
                "trade_date": None,
                "start_date": "20240101",
                "end_date": "20240131",
                "fields": (
                    "ts_code,trade_date,open,high,low,close,pre_close,change,"
                    "pct_chg,vol,amount"
                ),
            },
        )
    ]


def test_get_index_daily_has_no_hidden_benchmark_default() -> None:
    with pytest.raises(TypeError, match="ts_code"):
        _client(_FakePro()).get_index_daily()  # type: ignore[call-arg]


def test_get_suspend_d_delegates_exact_raw_event_scope() -> None:
    pro = _FakePro()
    result = _client(pro).get_suspend_d(
        ts_code="000001.SZ",
        trade_date="20240103",
        start_date="20240101",
        end_date="20240131",
        suspend_type="S",
    )
    assert result is pro.suspend_frame
    assert pro.calls == [
        (
            "suspend_d",
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240103",
                "start_date": "20240101",
                "end_date": "20240131",
                "suspend_type": "S",
                "fields": "ts_code,trade_date,suspend_timing,suspend_type",
            },
        )
    ]


def test_provider_exception_is_not_hidden_by_raw_wrapper() -> None:
    class BrokenPro(_FakePro):
        def daily(self, **kwargs: object) -> pd.DataFrame:
            raise RuntimeError("provider error")

    try:
        _client(BrokenPro()).get_daily()
    except RuntimeError as exc:
        assert str(exc) == "provider error"
    else:
        raise AssertionError("raw provider exception must propagate")
