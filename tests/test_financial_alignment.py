"""Tests for financial announcement point-in-time alignment."""

from __future__ import annotations

import pandas as pd
import pytest

from src.factors.financial_alignment import FinancialPointInTimeAligner


def make_trading_panel(
    dates=None,
    codes=("000001.SZ",),
) -> pd.DataFrame:
    if dates is None:
        dates = pd.bdate_range("2024-04-01", "2024-04-12")
    return pd.DataFrame(
        [
            {"trade_date": trade_date, "ts_code": ts_code}
            for ts_code in codes
            for trade_date in pd.to_datetime(dates)
        ]
    )


def make_financial_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["2024-04-02"],
            "end_date": ["2024-03-31"],
            "roe": [10.0],
        }
    )


def align(
    trading_panel: pd.DataFrame,
    financial_data: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    return FinancialPointInTimeAligner().align(
        trading_panel,
        financial_data,
        value_columns=["roe"],
        availability_lag_trading_days=lag,
    )


def test_normal_single_stock_alignment_with_default_lag() -> None:
    result = align(make_trading_panel(), make_financial_data())
    values = result.set_index("trade_date")["roe"]

    assert pd.isna(values.loc[pd.Timestamp("2024-04-01")])
    assert pd.isna(values.loc[pd.Timestamp("2024-04-02")])
    assert values.loc[pd.Timestamp("2024-04-03")] == 10.0


def test_normal_multi_stock_alignment_is_independent() -> None:
    trading = make_trading_panel(codes=("000001.SZ", "000002.SZ"))
    financial = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "ann_date": ["2024-04-02", "2024-04-04"],
            "end_date": ["2024-03-31", "2024-03-31"],
            "roe": [10.0, 20.0],
        }
    )

    result = align(trading, financial)

    assert set(result.loc[result["ts_code"] == "000001.SZ", "roe"].dropna()) == {10.0}
    assert set(result.loc[result["ts_code"] == "000002.SZ", "roe"].dropna()) == {20.0}


def test_unsorted_inputs_produce_the_same_result() -> None:
    trading = make_trading_panel(codes=("000001.SZ", "000002.SZ"))
    financial = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "ann_date": ["2024-04-02", "2024-04-04"],
            "end_date": ["2024-03-31", "2024-03-31"],
            "roe": [10.0, 20.0],
        }
    )

    expected = align(trading, financial)
    actual = align(
        trading.sample(frac=1.0, random_state=2).reset_index(drop=True),
        financial.sample(frac=1.0, random_state=3).reset_index(drop=True),
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_output_shape_columns_and_stable_sort() -> None:
    trading = make_trading_panel(codes=("000002.SZ", "000001.SZ"))
    result = align(trading, make_financial_data())
    expected_keys = result[["trade_date", "ts_code"]].sort_values(
        ["trade_date", "ts_code"],
        kind="mergesort",
        ignore_index=True,
    )

    assert len(result) == len(trading)
    assert list(result.columns) == [
        "trade_date",
        "ts_code",
        "source_ann_date",
        "source_end_date",
        "roe",
    ]
    pd.testing.assert_frame_equal(result[["trade_date", "ts_code"]], expected_keys)


def test_lag_zero_uses_first_trade_date_on_or_after_announcement() -> None:
    result = align(make_trading_panel(), make_financial_data(), lag=0)
    values = result.set_index("trade_date")["roe"]

    assert pd.isna(values.loc[pd.Timestamp("2024-04-01")])
    assert values.loc[pd.Timestamp("2024-04-02")] == 10.0


def test_weekend_announcement_uses_actual_trading_rows_for_lag() -> None:
    trading = make_trading_panel(dates=["2024-04-26", "2024-04-29", "2024-04-30"])
    financial = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["2024-04-28"],
            "end_date": ["2024-03-31"],
            "roe": [10.0],
        }
    )

    lag_zero = align(trading, financial, lag=0).set_index("trade_date")["roe"]
    lag_one = align(trading, financial, lag=1).set_index("trade_date")["roe"]

    assert lag_zero.loc[pd.Timestamp("2024-04-29")] == 10.0
    assert pd.isna(lag_one.loc[pd.Timestamp("2024-04-29")])
    assert lag_one.loc[pd.Timestamp("2024-04-30")] == 10.0


def test_stock_without_financial_history_remains_missing() -> None:
    trading = make_trading_panel(codes=("000001.SZ", "000002.SZ"))

    result = align(trading, make_financial_data())
    missing_stock = result[result["ts_code"] == "000002.SZ"]

    assert missing_stock["roe"].isna().all()
    assert missing_stock["source_ann_date"].isna().all()
    assert missing_stock["source_end_date"].isna().all()


def test_multiple_reporting_periods_select_latest_effective_record() -> None:
    financial = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ann_date": ["2024-04-01", "2024-04-05"],
            "end_date": ["2023-12-31", "2024-03-31"],
            "roe": [8.0, 10.0],
        }
    )
    result = align(make_trading_panel(), financial, lag=0).set_index("trade_date")

    assert result.loc[pd.Timestamp("2024-04-04"), "roe"] == 8.0
    assert result.loc[pd.Timestamp("2024-04-05"), "roe"] == 10.0


def test_revision_switches_only_after_revision_becomes_effective() -> None:
    trading = make_trading_panel(dates=pd.bdate_range("2024-04-19", "2024-05-14"))
    financial = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ann_date": ["2024-04-20", "2024-05-10"],
            "end_date": ["2024-03-31", "2024-03-31"],
            "roe": [10.0, 12.0],
        }
    )
    result = align(trading, financial).set_index("trade_date")

    assert result.loc[pd.Timestamp("2024-05-10"), "roe"] == 10.0
    assert result.loc[pd.Timestamp("2024-05-13"), "roe"] == 12.0
    assert result.loc[pd.Timestamp("2024-05-10"), "source_ann_date"] == pd.Timestamp(
        "2024-04-20"
    )


def test_modifying_future_announcement_does_not_change_past_output() -> None:
    trading = make_trading_panel(dates=pd.bdate_range("2024-04-01", "2024-05-31"))
    financial = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ann_date": ["2024-04-02", "2024-05-15"],
            "end_date": ["2024-03-31", "2024-03-31"],
            "roe": [10.0, 12.0],
        }
    )
    changed = financial.copy(deep=True)
    changed.loc[1, "ann_date"] = "2024-05-20"
    changed.loc[1, "roe"] = 99.0

    before = align(trading, financial)
    after = align(trading, changed)
    past = before["trade_date"] < pd.Timestamp("2024-05-15")

    pd.testing.assert_frame_equal(
        before.loc[past].reset_index(drop=True),
        after.loc[past].reset_index(drop=True),
    )


def test_stock_a_announcement_never_matches_stock_b() -> None:
    trading = make_trading_panel(codes=("000001.SZ", "000002.SZ"))
    result = align(trading, make_financial_data())

    assert result.loc[result["ts_code"] == "000002.SZ", "roe"].isna().all()


def test_duplicate_trading_key_raises_error() -> None:
    trading = make_trading_panel()
    trading = pd.concat([trading, trading.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate trade_date.*ts_code"):
        align(trading, make_financial_data())


def test_duplicate_financial_key_raises_error() -> None:
    financial = make_financial_data()
    financial = pd.concat([financial, financial], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate ts_code.*ann_date.*end_date"):
        align(make_trading_panel(), financial)


@pytest.mark.parametrize("missing_field", ["trade_date", "ts_code"])
def test_missing_trading_key_raises_error(missing_field: str) -> None:
    trading = make_trading_panel().drop(columns=missing_field)

    with pytest.raises(ValueError, match=missing_field):
        align(trading, make_financial_data())


@pytest.mark.parametrize("missing_field", ["ts_code", "ann_date", "end_date"])
def test_missing_financial_key_raises_error(missing_field: str) -> None:
    financial = make_financial_data().drop(columns=missing_field)

    with pytest.raises(ValueError, match=missing_field):
        align(make_trading_panel(), financial)


def test_missing_requested_value_column_raises_error() -> None:
    with pytest.raises(ValueError, match="roe"):
        FinancialPointInTimeAligner().align(
            make_trading_panel(),
            make_financial_data().drop(columns="roe"),
            value_columns=["roe"],
        )


@pytest.mark.parametrize("invalid_lag", [-1, -5])
def test_negative_lag_raises_error(invalid_lag: int) -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        align(make_trading_panel(), make_financial_data(), lag=invalid_lag)


@pytest.mark.parametrize("invalid_lag", [1.5, "1", True])
def test_non_integer_lag_raises_error(invalid_lag) -> None:
    with pytest.raises(TypeError, match="integer"):
        align(make_trading_panel(), make_financial_data(), lag=invalid_lag)


@pytest.mark.parametrize("empty_target", ["trading", "financial"])
def test_empty_input_raises_clear_error(empty_target: str) -> None:
    trading = make_trading_panel()
    financial = make_financial_data()
    if empty_target == "trading":
        trading = trading.iloc[0:0]
    else:
        financial = financial.iloc[0:0]

    with pytest.raises(ValueError, match=f"{empty_target}.*cannot be empty"):
        align(trading, financial)


@pytest.mark.parametrize("bad_code", [None, "", "   "])
def test_empty_stock_code_raises_error(bad_code) -> None:
    financial = make_financial_data()
    financial.loc[0, "ts_code"] = bad_code

    with pytest.raises(ValueError, match="ts_code"):
        align(make_trading_panel(), financial)


@pytest.mark.parametrize("field_name", ["trade_date", "ann_date", "end_date"])
def test_empty_or_invalid_date_raises_error(field_name: str) -> None:
    trading = make_trading_panel()
    financial = make_financial_data()
    if field_name == "trade_date":
        trading[field_name] = trading[field_name].astype("object")
        trading.loc[0, field_name] = "not-a-date"
    else:
        financial.loc[0, field_name] = None

    with pytest.raises(ValueError, match=field_name):
        align(trading, financial)


def test_inputs_are_not_modified() -> None:
    trading = make_trading_panel()
    financial = make_financial_data()
    trading_before = trading.copy(deep=True)
    financial_before = financial.copy(deep=True)

    align(trading, financial)

    pd.testing.assert_frame_equal(trading, trading_before)
    pd.testing.assert_frame_equal(financial, financial_before)


def test_same_effective_date_uses_later_announcement_then_end_date() -> None:
    trading = make_trading_panel(dates=["2024-04-26", "2024-04-29", "2024-04-30"])
    financial = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 3,
            "ann_date": ["2024-04-27", "2024-04-28", "2024-04-28"],
            "end_date": ["2024-06-30", "2024-03-31", "2024-06-30"],
            "roe": [1.0, 2.0, 3.0],
        }
    )

    result = align(trading, financial, lag=0).set_index("trade_date")

    assert result.loc[pd.Timestamp("2024-04-29"), "roe"] == 3.0
    assert result.loc[pd.Timestamp("2024-04-29"), "source_ann_date"] == pd.Timestamp(
        "2024-04-28"
    )
    assert result.loc[pd.Timestamp("2024-04-29"), "source_end_date"] == pd.Timestamp(
        "2024-06-30"
    )


def test_end_date_does_not_make_record_available_before_announcement() -> None:
    trading = make_trading_panel(dates=pd.bdate_range("2024-04-01", "2024-04-30"))
    financial = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["2024-04-28"],
            "end_date": ["2024-03-31"],
            "roe": [10.0],
        }
    )
    result = align(trading, financial).set_index("trade_date")

    assert result.loc[: pd.Timestamp("2024-04-29"), "roe"].isna().all()
    assert result.loc[pd.Timestamp("2024-04-30"), "roe"] == 10.0


def test_source_announcement_dates_satisfy_lag_boundary() -> None:
    trading = make_trading_panel(dates=pd.bdate_range("2024-04-01", "2024-04-12"))
    financial = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ann_date": ["2024-04-02", "2024-04-06"],
            "end_date": ["2023-12-31", "2024-03-31"],
            "roe": [8.0, 10.0],
        }
    )
    result = align(trading, financial)
    trade_dates = sorted(result["trade_date"].unique())

    for _, row in result.dropna(subset=["source_ann_date"]).iterrows():
        first_position = int(
            pd.Index(trade_dates).searchsorted(row["source_ann_date"], side="left")
        )
        row_position = trade_dates.index(row["trade_date"])
        assert row_position >= first_position + 1


def test_empty_value_columns_raise_error() -> None:
    with pytest.raises(ValueError, match="value_columns"):
        FinancialPointInTimeAligner().align(
            make_trading_panel(),
            make_financial_data(),
            value_columns=[],
        )


@pytest.mark.parametrize("reserved_column", ["ts_code", "ann_date", "end_date"])
def test_value_columns_cannot_include_key_fields(reserved_column: str) -> None:
    with pytest.raises(ValueError, match="key or audit fields"):
        FinancialPointInTimeAligner().align(
            make_trading_panel(),
            make_financial_data(),
            value_columns=[reserved_column],
        )
