"""Tests for canonical V6 security and benchmark daily-return adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.research_backtest import (
    BENCHMARK_DAILY_RETURN_COLUMNS,
    BENCHMARK_RETURN_SOURCE_NAME,
    RAW_RETURN_FIELD,
    RETURN_CONVENTION,
    RETURN_UNIT,
    SECURITY_DAILY_RETURN_COLUMNS,
    SECURITY_RETURN_SOURCE_NAME,
    BenchmarkCalendarAlignmentError,
    BenchmarkReturnDataError,
    MarketReturnProviderError,
    SecurityReturnDataError,
    TradingCalendar,
    TushareBenchmarkDailyReturnAdapter,
    TushareSecurityDailyReturnAdapter,
    build_benchmark_daily_returns,
    build_security_daily_returns,
    validate_strict_common_calendar,
)


def _security_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000002.SZ", "000001.SZ", "000001.SZ", "000002.SZ"],
            "trade_date": ["20240103", "20240102", "20240103", "20240102"],
            "pct_chg": [-3.0, 1.25, 0.0, 0.125],
            "close": [8.0, 10.0, 10.5, 8.1],
        }
    )


def _benchmark_rows(code: str = "000905.SH") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [code, code, code],
            "trade_date": ["20240104", "20240102", "20240103"],
            "pct_chg": [-1.5, 2.0, 0.0],
            "close": [5000.0, 5100.0, 5100.0],
        }
    )


def test_return_source_identity_is_explicit_and_frozen() -> None:
    assert SECURITY_RETURN_SOURCE_NAME == "tushare.daily"
    assert BENCHMARK_RETURN_SOURCE_NAME == "tushare.index_daily"
    assert RAW_RETURN_FIELD == "pct_chg"
    assert RETURN_UNIT == "decimal"
    assert RETURN_CONVENTION == "adjusted_close_to_close"


def test_security_pct_chg_is_converted_from_percent_to_decimal() -> None:
    result = build_security_daily_returns(_security_rows())
    assert tuple(result.columns) == SECURITY_DAILY_RETURN_COLUMNS
    assert result.to_dict("records") == [
        {
            "trade_date": pd.Timestamp("2024-01-02"),
            "ts_code": "000001.SZ",
            "return": 0.0125,
        },
        {
            "trade_date": pd.Timestamp("2024-01-02"),
            "ts_code": "000002.SZ",
            "return": 0.00125,
        },
        {
            "trade_date": pd.Timestamp("2024-01-03"),
            "ts_code": "000001.SZ",
            "return": 0.0,
        },
        {
            "trade_date": pd.Timestamp("2024-01-03"),
            "ts_code": "000002.SZ",
            "return": -0.03,
        },
    ]


def test_security_sort_is_deterministic_and_input_is_not_mutated() -> None:
    source = _security_rows().sample(frac=1.0, random_state=31).reset_index(drop=True)
    original = source.copy(deep=True)
    first = build_security_daily_returns(source)
    second = build_security_daily_returns(source)
    pdt.assert_frame_equal(first, second)
    pdt.assert_frame_equal(source, original)


def test_security_codes_are_stripped_without_changing_identity() -> None:
    source = pd.DataFrame(
        {"trade_date": ["2024-01-02"], "ts_code": [" 000001.SZ "], "pct_chg": [1]}
    )
    result = build_security_daily_returns(source)
    assert result.loc[0, "ts_code"] == "000001.SZ"


def test_security_panel_remains_sparse_without_grid_creation_or_fill() -> None:
    source = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-03"],
            "ts_code": ["A", "A", "B"],
            "pct_chg": [1.0, 2.0, -1.0],
        }
    )
    result = build_security_daily_returns(source)
    assert len(result) == 3
    missing_key = (result["trade_date"] == pd.Timestamp("2024-01-02")) & (
        result["ts_code"] == "B"
    )
    assert not bool(
        missing_key.any()
    )
    assert result["return"].tolist() == [0.01, 0.02, -0.01]


def test_multiple_same_month_security_dates_are_preserved() -> None:
    source = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-05", "2024-01-12"],
            "ts_code": ["A", "A", "A"],
            "pct_chg": [1.0, 2.0, 3.0],
        }
    )
    assert len(build_security_daily_returns(source)) == 3


def test_security_return_has_no_arbitrary_bounds() -> None:
    source = pd.DataFrame(
        {"trade_date": ["2024-01-02"], "ts_code": ["A"], "pct_chg": [1000.0]}
    )
    assert build_security_daily_returns(source).loc[0, "return"] == 10.0


def test_duplicate_security_key_is_rejected_after_normalization() -> None:
    source = pd.DataFrame(
        {
            "trade_date": ["20240102", "2024-01-02"],
            "ts_code": ["A", " A "],
            "pct_chg": [1.0, 2.0],
        }
    )
    with pytest.raises(SecurityReturnDataError, match="unique"):
        build_security_daily_returns(source)


@pytest.mark.parametrize(
    "source",
    [
        pd.DataFrame(),
        pd.DataFrame(columns=["trade_date", "ts_code", "pct_chg"]),
        pd.DataFrame({"trade_date": ["2024-01-02"], "ts_code": ["A"]}),
        pd.DataFrame({"trade_date": ["2024-01-02"], "pct_chg": [1.0]}),
        pd.DataFrame({"ts_code": ["A"], "pct_chg": [1.0]}),
    ],
)
def test_empty_or_incomplete_security_frame_is_rejected(source: pd.DataFrame) -> None:
    with pytest.raises(SecurityReturnDataError):
        build_security_daily_returns(source)


@pytest.mark.parametrize(
    "value",
    [None, pd.NaT, "", "2024/01/02", "bad-date", pd.Timestamp("2024-01-02", tz="UTC")],
)
def test_invalid_security_dates_are_rejected(value: object) -> None:
    source = pd.DataFrame(
        {"trade_date": [value], "ts_code": ["A"], "pct_chg": [1.0]}
    )
    with pytest.raises(SecurityReturnDataError):
        build_security_daily_returns(source)


@pytest.mark.parametrize("value", [None, "", "   ", 1, True])
def test_invalid_security_codes_are_rejected(value: object) -> None:
    source = pd.DataFrame(
        {"trade_date": ["2024-01-02"], "ts_code": [value], "pct_chg": [1.0]}
    )
    with pytest.raises(SecurityReturnDataError):
        build_security_daily_returns(source)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "1.25", True, None])
def test_invalid_security_pct_chg_is_rejected(value: object) -> None:
    source = pd.DataFrame(
        {"trade_date": ["2024-01-02"], "ts_code": ["A"], "pct_chg": [value]}
    )
    with pytest.raises(SecurityReturnDataError, match="pct_chg"):
        build_security_daily_returns(source)


def test_benchmark_pct_chg_is_decimal_sorted_and_exact_schema() -> None:
    result = build_benchmark_daily_returns(
        _benchmark_rows(), benchmark_code="000905.SH"
    )
    assert tuple(result.columns) == BENCHMARK_DAILY_RETURN_COLUMNS
    assert result["trade_date"].tolist() == list(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    )
    assert result["benchmark_code"].tolist() == ["000905.SH"] * 3
    assert result["return"].tolist() == [0.02, 0.0, -0.015]


@pytest.mark.parametrize("value", [None, "", "  ", 1, True])
def test_benchmark_code_is_explicit_strict_and_nonempty(value: object) -> None:
    with pytest.raises(BenchmarkReturnDataError, match="benchmark_code"):
        build_benchmark_daily_returns(_benchmark_rows(), benchmark_code=value)


def test_benchmark_raw_code_must_match_requested_code() -> None:
    with pytest.raises(BenchmarkReturnDataError, match="must match"):
        build_benchmark_daily_returns(
            _benchmark_rows("000300.SH"), benchmark_code="000905.SH"
        )


def test_mixed_benchmark_raw_codes_are_rejected() -> None:
    source = _benchmark_rows()
    source.loc[0, "ts_code"] = "399006.SZ"
    with pytest.raises(BenchmarkReturnDataError, match="observed"):
        build_benchmark_daily_returns(source, benchmark_code="000905.SH")


def test_optional_raw_benchmark_code_column_may_be_absent() -> None:
    result = build_benchmark_daily_returns(
        _benchmark_rows().drop(columns="ts_code"), benchmark_code="000905.SH"
    )
    assert result["benchmark_code"].eq("000905.SH").all()


def test_duplicate_benchmark_date_is_rejected() -> None:
    source = _benchmark_rows()
    source.loc[0, "trade_date"] = "20240102"
    with pytest.raises(BenchmarkReturnDataError, match="unique"):
        build_benchmark_daily_returns(source, benchmark_code="000905.SH")


@pytest.mark.parametrize("value", [np.nan, np.inf, "2", True, None])
def test_invalid_benchmark_pct_chg_is_rejected(value: object) -> None:
    source = _benchmark_rows()
    source["pct_chg"] = source["pct_chg"].astype(object)
    source.at[0, "pct_chg"] = value
    with pytest.raises(BenchmarkReturnDataError, match="pct_chg"):
        build_benchmark_daily_returns(source, benchmark_code="000905.SH")


def test_benchmark_input_is_not_mutated() -> None:
    source = _benchmark_rows()
    original = source.copy(deep=True)
    build_benchmark_daily_returns(source, benchmark_code="000905.SH")
    pdt.assert_frame_equal(source, original)


@pytest.mark.parametrize(
    "source",
    [
        pd.DataFrame(columns=["trade_date", "pct_chg", "ts_code"]),
        pd.DataFrame({"trade_date": ["2024-01-02"]}),
        pd.DataFrame({"pct_chg": [1.0]}),
    ],
)
def test_empty_or_incomplete_benchmark_frame_is_rejected(
    source: pd.DataFrame,
) -> None:
    with pytest.raises(BenchmarkReturnDataError):
        build_benchmark_daily_returns(source, benchmark_code="000905.SH")


@pytest.mark.parametrize(
    "value",
    [None, pd.NaT, "", "2024/01/02", pd.Timestamp("2024-01-02", tz="UTC")],
)
def test_invalid_benchmark_dates_are_rejected(value: object) -> None:
    source = pd.DataFrame(
        {
            "trade_date": [value],
            "ts_code": ["000905.SH"],
            "pct_chg": [1.0],
        }
    )
    with pytest.raises(BenchmarkReturnDataError):
        build_benchmark_daily_returns(source, benchmark_code="000905.SH")


class _FakeReturnClient:
    def __init__(
        self,
        security_frames: dict[str, pd.DataFrame] | None = None,
        benchmark_frame: pd.DataFrame | None = None,
    ) -> None:
        self.security_frames = security_frames or {}
        self.benchmark_frame = (
            _benchmark_rows() if benchmark_frame is None else benchmark_frame
        )
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get_daily(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("daily", kwargs))
        return self.security_frames.get(
            str(kwargs["ts_code"]),
            pd.DataFrame(columns=["trade_date", "ts_code", "pct_chg"]),
        )

    def get_index_daily(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("index_daily", kwargs))
        return self.benchmark_frame


def test_security_adapter_calls_each_explicit_code_and_preserves_sparse_scope() -> None:
    client = _FakeReturnClient(
        security_frames={
            "A": pd.DataFrame(
                {
                    "trade_date": ["20240102", "20240103"],
                    "ts_code": ["A", "A"],
                    "pct_chg": [1.0, 2.0],
                }
            ),
            "B": pd.DataFrame(
                {"trade_date": ["20240103"], "ts_code": ["B"], "pct_chg": [-1.0]}
            ),
        }
    )
    result = TushareSecurityDailyReturnAdapter(client).load(
        ts_codes=["A", "B"], start_date="2024-01-02", end_date="2024-01-03"
    )
    assert len(result) == 3
    assert client.calls == [
        (
            "daily",
            {
                "ts_code": "A",
                "trade_date": None,
                "start_date": "20240102",
                "end_date": "20240103",
            },
        ),
        (
            "daily",
            {
                "ts_code": "B",
                "trade_date": None,
                "start_date": "20240102",
                "end_date": "20240103",
            },
        ),
    ]


def test_security_adapter_allows_one_explicit_code_to_have_no_rows() -> None:
    client = _FakeReturnClient(
        security_frames={
            "A": pd.DataFrame(
                {"trade_date": ["20240102"], "ts_code": ["A"], "pct_chg": [1.0]}
            )
        }
    )
    result = TushareSecurityDailyReturnAdapter(client).load(
        ts_codes=["A", "B"], start_date="20240102", end_date="20240103"
    )
    assert result["ts_code"].tolist() == ["A"]


def test_security_adapter_rejects_empty_total_scope_and_unexpected_rows() -> None:
    with pytest.raises(SecurityReturnDataError, match="no rows"):
        TushareSecurityDailyReturnAdapter(_FakeReturnClient()).load(
            ts_codes=["A"], start_date="20240102", end_date="20240103"
        )
    client = _FakeReturnClient(
        security_frames={
            "A": pd.DataFrame(
                {"trade_date": ["20240102"], "ts_code": ["X"], "pct_chg": [1.0]}
            )
        }
    )
    with pytest.raises(SecurityReturnDataError, match="outside the explicit scope"):
        TushareSecurityDailyReturnAdapter(client).load(
            ts_codes=["A"], start_date="20240102", end_date="20240103"
        )


def test_adapters_reject_provider_rows_outside_explicit_date_scope() -> None:
    security_client = _FakeReturnClient(
        security_frames={
            "A": pd.DataFrame(
                {
                    "trade_date": ["20240104"],
                    "ts_code": ["A"],
                    "pct_chg": [1.0],
                }
            )
        }
    )
    with pytest.raises(SecurityReturnDataError, match="dates outside"):
        TushareSecurityDailyReturnAdapter(security_client).load(
            ts_codes=["A"], start_date="20240102", end_date="20240103"
        )
    benchmark_client = _FakeReturnClient(
        benchmark_frame=pd.DataFrame(
            {
                "trade_date": ["20240104"],
                "ts_code": ["000905.SH"],
                "pct_chg": [1.0],
            }
        )
    )
    with pytest.raises(BenchmarkReturnDataError, match="dates outside"):
        TushareBenchmarkDailyReturnAdapter(benchmark_client).load(
            benchmark_code="000905.SH",
            start_date="20240102",
            end_date="20240103",
        )


@pytest.mark.parametrize("value", [[], ["A", "A"], "A", [1], None])
def test_security_adapter_requires_explicit_unique_string_codes(value: object) -> None:
    with pytest.raises(SecurityReturnDataError):
        TushareSecurityDailyReturnAdapter(_FakeReturnClient()).load(
            ts_codes=value, start_date="20240102", end_date="20240103"
        )


def test_benchmark_adapter_calls_exact_explicit_code_and_scope() -> None:
    client = _FakeReturnClient()
    result = TushareBenchmarkDailyReturnAdapter(client).load(
        benchmark_code="000905.SH", start_date="20240102", end_date="20240104"
    )
    assert result["benchmark_code"].eq("000905.SH").all()
    assert client.calls == [
        (
            "index_daily",
            {
                "ts_code": "000905.SH",
                "trade_date": None,
                "start_date": "20240102",
                "end_date": "20240104",
            },
        )
    ]


@pytest.mark.parametrize("adapter", ["security", "benchmark"])
def test_adapters_wrap_provider_failures_without_network(adapter: str) -> None:
    class BrokenClient(_FakeReturnClient):
        def get_daily(self, **kwargs: object) -> pd.DataFrame:
            raise RuntimeError("offline")

        def get_index_daily(self, **kwargs: object) -> pd.DataFrame:
            raise RuntimeError("offline")

    with pytest.raises(MarketReturnProviderError) as exc:
        if adapter == "security":
            TushareSecurityDailyReturnAdapter(BrokenClient()).load(
                ts_codes=["A"], start_date="20240102", end_date="20240103"
            )
        else:
            TushareBenchmarkDailyReturnAdapter(BrokenClient()).load(
                benchmark_code="000905.SH", start_date="20240102", end_date="20240103"
            )
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_security_dates_are_consumable_by_b1_trading_calendar() -> None:
    rows = pd.DataFrame(
        {
            "cal_date": pd.date_range("2024-01-01", "2024-01-04", freq="D"),
            "is_open": [0, 1, 1, 1],
        }
    )
    calendar = TradingCalendar.from_frame(
        rows, start_date="2024-01-01", end_date="2024-01-04"
    )
    returns = build_security_daily_returns(_security_rows())
    assert all(calendar.is_trading_day(item) for item in returns["trade_date"].unique())


def test_benchmark_dates_reuse_b1_strict_common_calendar() -> None:
    benchmark = build_benchmark_daily_returns(
        _benchmark_rows(), benchmark_code="000905.SH"
    )
    strategy_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    assert validate_strict_common_calendar(
        strategy_dates, benchmark["trade_date"]
    ) == tuple(strategy_dates)
    with pytest.raises(BenchmarkCalendarAlignmentError):
        validate_strict_common_calendar(
            strategy_dates, benchmark["trade_date"].iloc[:-1]
        )


def test_production_source_has_no_price_reconstruction_or_fill_behavior() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "research_backtest" / "returns.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "pct_change(",
        ".ffill(",
        ".bfill(",
        "fillna(0",
        "rebalance_frequency",
    )
    assert all(token not in source for token in forbidden)
