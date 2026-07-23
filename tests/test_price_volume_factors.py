"""Tests for the V2-C1 core price-volume factor pack."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.factor_engine import FactorEngine
from src.factors.price_volume import (
    AMIHUD_20D,
    MOMENTUM_60D,
    MOMENTUM_120D,
    MOMENTUM_252_20D,
    PRICE_52W_HIGH,
    PRICE_VOLUME_FACTORS,
    SHORT_TERM_REVERSAL_5D,
    TURNOVER_MEAN_20D,
    VOLATILITY_60D,
    register_price_volume_factors,
)
from src.factors.registry import FactorRegistry, create_default_registry


NEW_FACTOR_NAMES = [factor.metadata.name for factor in PRICE_VOLUME_FACTORS]
ALL_PRICE_VOLUME_NAMES = ["momentum_20d", "volatility_20d"] + NEW_FACTOR_NAMES


def make_single_stock_data(periods: int = 340) -> pd.DataFrame:
    """Create deterministic local price, amount, and turnover observations."""
    index = pd.date_range("2023-01-01", periods=periods, freq="D")
    steps = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "close": 100.0 + steps * 0.35 + np.sin(steps / 9.0),
            "amount": 1_000_000.0 + steps * 1_000.0,
            "turnover_rate": 1.0 + (steps % 17) * 0.05,
        },
        index=index,
    )


def make_multi_stock_data(periods: int = 340) -> pd.DataFrame:
    """Create unsorted factor-engine input for two independent stocks."""
    first = make_single_stock_data(periods).reset_index(names="trade_date")
    first["ts_code"] = "000001.SZ"
    second = make_single_stock_data(periods).reset_index(names="trade_date")
    second["ts_code"] = "000002.SZ"
    second["close"] = second["close"] * 1.8 + np.arange(periods) * 0.1
    second["amount"] = second["amount"] * 2.2
    second["turnover_rate"] = second["turnover_rate"] * 0.7
    return pd.concat([first, second], ignore_index=True).sample(
        frac=1.0,
        random_state=11,
    ).reset_index(drop=True)


def make_full_registry() -> FactorRegistry:
    registry = create_default_registry()
    register_price_volume_factors(registry)
    return registry


def test_registers_exactly_eight_unique_new_factors() -> None:
    registry = FactorRegistry()

    returned = register_price_volume_factors(registry)

    assert returned is registry
    assert len(registry.list_names()) == 8
    assert registry.list_names() == sorted(NEW_FACTOR_NAMES)
    assert len(set(NEW_FACTOR_NAMES)) == 8


def test_default_registry_behavior_is_unchanged() -> None:
    assert create_default_registry().list_names() == [
        "momentum_20d",
        "volatility_20d",
    ]


def test_factor_metadata_matches_specification() -> None:
    expected = {
        "momentum_60d": ("momentum", 1, ("daily",), ("close",), 60),
        "momentum_120d": ("momentum", 1, ("daily",), ("close",), 120),
        "momentum_252_20d": ("momentum", 1, ("daily",), ("close",), 252),
        "short_term_reversal_5d": ("reversal", 1, ("daily",), ("close",), 5),
        "price_52w_high": ("momentum", 1, ("daily",), ("close",), 252),
        "volatility_60d": ("volatility", -1, ("daily",), ("close",), 60),
        "turnover_mean_20d": (
            "liquidity",
            1,
            ("daily_basic",),
            ("turnover_rate",),
            20,
        ),
        "amihud_20d": ("liquidity", -1, ("daily",), ("close", "amount"), 20),
    }

    for factor in PRICE_VOLUME_FACTORS:
        metadata = factor.metadata
        assert (
            metadata.category,
            metadata.direction,
            metadata.required_datasets,
            metadata.source_fields,
            metadata.lookback_days,
        ) == expected[metadata.name]


@pytest.mark.parametrize(
    ("factor", "expected"),
    [
        (MOMENTUM_60D, lambda data: data["close"] / data["close"].shift(60) - 1.0),
        (MOMENTUM_120D, lambda data: data["close"] / data["close"].shift(120) - 1.0),
        (
            MOMENTUM_252_20D,
            lambda data: data["close"].shift(20) / data["close"].shift(252) - 1.0,
        ),
        (
            SHORT_TERM_REVERSAL_5D,
            lambda data: -(data["close"] / data["close"].shift(5) - 1.0),
        ),
    ],
)
def test_shift_factors_match_manual_formula(factor, expected) -> None:
    data = make_single_stock_data()

    pd.testing.assert_series_equal(
        factor.compute(data),
        expected(data).rename(factor.metadata.name),
    )


def test_price_52w_high_matches_manual_formula() -> None:
    data = make_single_stock_data()
    expected = data["close"] / data["close"].rolling(252, min_periods=252).max()

    pd.testing.assert_series_equal(
        PRICE_52W_HIGH.compute(data),
        expected.rename("price_52w_high"),
    )


def test_volatility_60d_matches_manual_formula() -> None:
    data = make_single_stock_data()
    expected = data["close"].pct_change(fill_method=None).rolling(60).std()

    pd.testing.assert_series_equal(
        VOLATILITY_60D.compute(data),
        expected.rename("volatility_60d"),
    )


def test_turnover_mean_20d_matches_manual_formula_without_unit_conversion() -> None:
    data = make_single_stock_data()
    expected = data["turnover_rate"].rolling(20, min_periods=20).mean()

    pd.testing.assert_series_equal(
        TURNOVER_MEAN_20D.compute(data),
        expected.rename("turnover_mean_20d"),
    )


def test_amihud_20d_matches_manual_formula() -> None:
    data = make_single_stock_data()
    daily = data["close"].pct_change(fill_method=None).abs() / data["amount"]
    expected = daily.rolling(20, min_periods=20).mean()

    pd.testing.assert_series_equal(
        AMIHUD_20D.compute(data),
        expected.rename("amihud_20d"),
    )


@pytest.mark.parametrize("factor", PRICE_VOLUME_FACTORS)
def test_every_factor_preserves_input_length_and_index(factor) -> None:
    data = make_single_stock_data()

    result = factor.compute(data)

    assert len(result) == len(data)
    assert result.index.equals(data.index)


@pytest.mark.parametrize(
    ("factor", "leading_nan_count"),
    [
        (MOMENTUM_60D, 60),
        (MOMENTUM_120D, 120),
        (MOMENTUM_252_20D, 252),
        (SHORT_TERM_REVERSAL_5D, 5),
        (PRICE_52W_HIGH, 251),
        (VOLATILITY_60D, 60),
        (TURNOVER_MEAN_20D, 19),
        (AMIHUD_20D, 20),
    ],
)
def test_insufficient_history_remains_nan(factor, leading_nan_count: int) -> None:
    result = factor.compute(make_single_stock_data())

    assert result.iloc[:leading_nan_count].isna().all()
    assert pd.notna(result.iloc[leading_nan_count])


@pytest.mark.parametrize(
    ("factor", "missing_field"),
    [
        (MOMENTUM_60D, "close"),
        (TURNOVER_MEAN_20D, "turnover_rate"),
        (AMIHUD_20D, "amount"),
    ],
)
def test_missing_required_field_raises_clear_error(factor, missing_field: str) -> None:
    data = make_single_stock_data().drop(columns=missing_field)

    with pytest.raises(ValueError, match=missing_field):
        factor.compute(data)


def test_amihud_zero_amount_never_produces_infinity() -> None:
    data = make_single_stock_data()
    data.iloc[100, data.columns.get_loc("amount")] = 0.0

    result = AMIHUD_20D.compute(data)

    assert not np.isinf(result.dropna()).any()


def test_amihud_negative_amount_raises_error() -> None:
    data = make_single_stock_data()
    data.iloc[100, data.columns.get_loc("amount")] = -1.0

    with pytest.raises(ValueError, match="non-negative amount"):
        AMIHUD_20D.compute(data)


@pytest.mark.parametrize("factor", PRICE_VOLUME_FACTORS)
def test_future_changes_do_not_affect_past_factor_values(factor) -> None:
    data = make_single_stock_data()
    changed = data.copy(deep=True)
    changed.iloc[-20:, changed.columns.get_loc("close")] *= 7.0
    changed.iloc[-20:, changed.columns.get_loc("amount")] *= 3.0
    changed.iloc[-20:, changed.columns.get_loc("turnover_rate")] += 5.0

    before = factor.compute(data)
    after = factor.compute(changed)

    pd.testing.assert_series_equal(before.iloc[:-20], after.iloc[:-20])


def test_factor_engine_computes_all_ten_factors_for_two_stocks() -> None:
    data = make_multi_stock_data()
    panel = FactorEngine(make_full_registry()).compute_factor_panel(
        data,
        ALL_PRICE_VOLUME_NAMES,
    )

    assert list(panel.columns) == ["trade_date", "ts_code"] + ALL_PRICE_VOLUME_NAMES
    assert len(panel) == len(data)
    for _, group in panel.groupby("ts_code"):
        ordered = group.sort_values("trade_date")
        assert ordered["momentum_252_20d"].iloc[:252].isna().all()
        assert ordered["momentum_252_20d"].iloc[252:].notna().all()


def test_changing_one_stock_does_not_affect_the_other_stock() -> None:
    data = make_multi_stock_data()
    changed = data.copy(deep=True)
    first_mask = changed["ts_code"] == "000001.SZ"
    changed.loc[first_mask, "close"] = np.square(changed.loc[first_mask, "close"])
    changed.loc[first_mask, "amount"] *= 4.0
    changed.loc[first_mask, "turnover_rate"] += 8.0
    engine = FactorEngine(make_full_registry())

    before = engine.compute_factor_panel(data, ALL_PRICE_VOLUME_NAMES)
    after = engine.compute_factor_panel(changed, ALL_PRICE_VOLUME_NAMES)
    second_mask = before["ts_code"] == "000002.SZ"

    pd.testing.assert_frame_equal(
        before.loc[second_mask, ALL_PRICE_VOLUME_NAMES].reset_index(drop=True),
        after.loc[second_mask, ALL_PRICE_VOLUME_NAMES].reset_index(drop=True),
    )


def test_describe_requirements_covers_all_ten_factors() -> None:
    requirements = FactorEngine(make_full_registry()).describe_requirements(
        ALL_PRICE_VOLUME_NAMES
    )

    assert requirements["required_datasets"] == ["daily", "daily_basic"]
    assert requirements["source_fields"] == ["amount", "close", "turnover_rate"]
    assert requirements["max_lookback_days"] == 252
    assert requirements["max_availability_lag_days"] == 0
    assert requirements["categories"] == [
        "liquidity",
        "momentum",
        "reversal",
        "volatility",
    ]
