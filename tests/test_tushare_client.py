"""Offline delegation tests for the minimal TuShare client methods used by V6."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.tushare_client import TushareClient


class _FakePro:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.daily_frame = pd.DataFrame({"raw": ["daily"]})
        self.index_frame = pd.DataFrame({"raw": ["index_daily"]})

    def daily(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("daily", kwargs))
        return self.daily_frame

    def index_daily(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("index_daily", kwargs))
        return self.index_frame


def _client(pro: _FakePro) -> TushareClient:
    client = object.__new__(TushareClient)
    client.pro = pro
    return client


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
