"""Tests for V2-B factor input contracts and multi-stock panel engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.contracts import normalize_factor_input
from src.factors.factor_engine import FactorEngine
from src.factors.registry import create_default_registry


FACTOR_NAMES = ["momentum_20d", "volatility_20d"]


def make_multi_stock_data(periods: int = 30) -> pd.DataFrame:
    """Create deterministic, deliberately unsorted daily data for two stocks."""
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    first = pd.DataFrame(
        {
            "trade_date": dates.astype(str),
            "ts_code": "000001.SZ",
            "close": 100.0 + np.arange(periods) * 1.5,
        }
    )
    second = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": "000002.SZ",
            "close": 200.0 + np.arange(periods) * 2.5,
        }
    )
    return pd.concat([first, second], ignore_index=True).sample(
        frac=1.0,
        random_state=7,
    ).reset_index(drop=True)


def make_engine() -> FactorEngine:
    """Create an engine with only the two V2-A example factors."""
    return FactorEngine(create_default_registry())


def test_valid_multi_stock_input_creates_expected_wide_panel() -> None:
    data = make_multi_stock_data()

    panel = make_engine().compute_factor_panel(data, FACTOR_NAMES)

    assert list(panel.columns) == [
        "trade_date",
        "ts_code",
        "momentum_20d",
        "volatility_20d",
    ]
    assert len(panel) == len(data.drop_duplicates(["trade_date", "ts_code"]))
    assert pd.api.types.is_datetime64_any_dtype(panel["trade_date"])


def test_unsorted_input_is_computed_and_output_in_stable_order() -> None:
    panel = make_engine().compute_factor_panel(make_multi_stock_data(), FACTOR_NAMES)
    expected_keys = panel[["trade_date", "ts_code"]].sort_values(
        ["trade_date", "ts_code"],
        kind="mergesort",
        ignore_index=True,
    )

    pd.testing.assert_frame_equal(panel[["trade_date", "ts_code"]], expected_keys)
    for _, group in panel.groupby("ts_code"):
        assert group["trade_date"].is_monotonic_increasing


def test_each_stock_has_its_own_twenty_row_warmup() -> None:
    panel = make_engine().compute_factor_panel(make_multi_stock_data(), FACTOR_NAMES)

    for _, group in panel.groupby("ts_code"):
        ordered = group.sort_values("trade_date")
        assert ordered["momentum_20d"].iloc[:20].isna().all()
        assert ordered["volatility_20d"].iloc[:20].isna().all()
        assert ordered["momentum_20d"].iloc[20:].notna().all()


def test_momentum_matches_manual_per_stock_calculation() -> None:
    data = normalize_factor_input(make_multi_stock_data(), required_fields=("close",))
    panel = make_engine().compute_factor_panel(data, FACTOR_NAMES)

    for ts_code, group in data.groupby("ts_code"):
        ordered = group.sort_values("trade_date")
        expected = ordered["close"] / ordered["close"].shift(20) - 1.0
        actual = panel.loc[panel["ts_code"] == ts_code].sort_values("trade_date")
        np.testing.assert_allclose(
            actual["momentum_20d"].to_numpy(),
            expected.to_numpy(),
            equal_nan=True,
        )


def test_volatility_matches_manual_per_stock_calculation() -> None:
    data = normalize_factor_input(make_multi_stock_data(), required_fields=("close",))
    panel = make_engine().compute_factor_panel(data, FACTOR_NAMES)

    for ts_code, group in data.groupby("ts_code"):
        ordered = group.sort_values("trade_date")
        expected = ordered["close"].pct_change(fill_method=None).rolling(20).std()
        actual = panel.loc[panel["ts_code"] == ts_code].sort_values("trade_date")
        np.testing.assert_allclose(
            actual["volatility_20d"].to_numpy(),
            expected.to_numpy(),
            equal_nan=True,
        )


@pytest.mark.parametrize(
    ("column", "message"),
    [("trade_date", "trade_date"), ("ts_code", "ts_code"), ("close", "close")],
)
def test_missing_required_column_raises_clear_error(column: str, message: str) -> None:
    data = make_multi_stock_data().drop(columns=column)

    with pytest.raises(ValueError, match=message):
        make_engine().compute_factor_panel(data, FACTOR_NAMES)


def test_duplicate_stock_date_key_raises_error() -> None:
    data = make_multi_stock_data()
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        make_engine().compute_factor_panel(data, FACTOR_NAMES)


@pytest.mark.parametrize("empty_code", [None, "", "   "])
def test_empty_ts_code_raises_error(empty_code: object) -> None:
    data = make_multi_stock_data()
    data.loc[0, "ts_code"] = empty_code

    with pytest.raises(ValueError, match="ts_code"):
        make_engine().compute_factor_panel(data, FACTOR_NAMES)


def test_unregistered_factor_name_raises_error() -> None:
    with pytest.raises(KeyError, match="not registered"):
        make_engine().compute_factor_panel(make_multi_stock_data(), ["missing"])


def test_empty_factor_names_raise_error() -> None:
    with pytest.raises(ValueError, match="factor_names"):
        make_engine().compute_factor_panel(make_multi_stock_data(), [])


def test_describe_requirements_returns_stable_unions_and_maxima() -> None:
    requirements = make_engine().describe_requirements(
        ["volatility_20d", "momentum_20d"]
    )

    assert requirements == {
        "factor_names": ["momentum_20d", "volatility_20d"],
        "required_datasets": ["daily"],
        "source_fields": ["close"],
        "max_lookback_days": 20,
        "max_availability_lag_days": 0,
        "categories": ["momentum", "volatility"],
    }


def test_compute_does_not_modify_original_input() -> None:
    data = make_multi_stock_data()
    before = data.copy(deep=True)

    make_engine().compute_factor_panel(data, FACTOR_NAMES)

    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("factor_name", FACTOR_NAMES)
def test_future_changes_do_not_affect_past_values(factor_name: str) -> None:
    data = normalize_factor_input(make_multi_stock_data(), required_fields=("close",))
    changed = data.copy(deep=True)
    target_code = "000001.SZ"
    target_dates = sorted(changed.loc[changed["ts_code"] == target_code, "trade_date"])
    changed_dates = target_dates[-5:]
    changed.loc[
        (changed["ts_code"] == target_code)
        & changed["trade_date"].isin(changed_dates),
        "close",
    ] *= 10

    before = make_engine().compute_factor_panel(data, FACTOR_NAMES)
    after = make_engine().compute_factor_panel(changed, FACTOR_NAMES)
    past_mask = (before["ts_code"] == target_code) & (
        before["trade_date"] < min(changed_dates)
    )

    pd.testing.assert_series_equal(
        before.loc[past_mask, factor_name].reset_index(drop=True),
        after.loc[past_mask, factor_name].reset_index(drop=True),
    )


def test_changing_one_stock_does_not_affect_another_stock() -> None:
    data = normalize_factor_input(make_multi_stock_data(), required_fields=("close",))
    changed = data.copy(deep=True)
    first_mask = changed["ts_code"] == "000001.SZ"
    changed.loc[first_mask, "close"] = np.square(changed.loc[first_mask, "close"])

    before = make_engine().compute_factor_panel(data, FACTOR_NAMES)
    after = make_engine().compute_factor_panel(changed, FACTOR_NAMES)
    second_mask = before["ts_code"] == "000002.SZ"

    pd.testing.assert_frame_equal(
        before.loc[second_mask, FACTOR_NAMES].reset_index(drop=True),
        after.loc[second_mask, FACTOR_NAMES].reset_index(drop=True),
    )
