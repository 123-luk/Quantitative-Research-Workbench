"""Tests for the V2-C2A valuation and size factor pack."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.examples import register_example_factors
from src.factors.factor_engine import FactorEngine
from src.factors.price_volume import register_price_volume_factors
from src.factors.registry import FactorRegistry, create_default_registry
from src.factors.valuation import (
    BP,
    DIVIDEND_YIELD_TTM,
    EP_TTM,
    LOG_CIRC_MV,
    LOG_TOTAL_MV,
    SP_TTM,
    VALUATION_FACTORS,
    register_valuation_factors,
)


VALUATION_NAMES = [factor.metadata.name for factor in VALUATION_FACTORS]


def make_valuation_data(periods: int = 12) -> pd.DataFrame:
    """Create deterministic daily-basic values with a non-default index."""
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    steps = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "pe_ttm": 10.0 + steps,
            "pb": 1.5 + steps * 0.1,
            "ps_ttm": 2.0 + steps * 0.2,
            "dv_ttm": steps * 0.15,
            "total_mv": 1_000_000.0 + steps * 10_000.0,
            "circ_mv": 700_000.0 + steps * 8_000.0,
        },
        index=index,
    )


def make_multi_stock_data(periods: int = 12) -> pd.DataFrame:
    """Create unsorted multi-stock input for FactorEngine tests."""
    first = make_valuation_data(periods).reset_index(names="trade_date")
    first["ts_code"] = "000001.SZ"
    second = make_valuation_data(periods).reset_index(names="trade_date")
    second["ts_code"] = "000002.SZ"
    second["pe_ttm"] += 5.0
    second["pb"] += 0.8
    second["ps_ttm"] += 1.2
    second["dv_ttm"] += 0.4
    second["total_mv"] *= 2.0
    second["circ_mv"] *= 1.7
    return pd.concat([first, second], ignore_index=True).sample(
        frac=1.0,
        random_state=23,
    ).reset_index(drop=True)


def make_valuation_registry() -> FactorRegistry:
    registry = FactorRegistry()
    register_valuation_factors(registry)
    return registry


def test_registers_exactly_six_unique_valuation_factors() -> None:
    registry = FactorRegistry()

    returned = register_valuation_factors(registry)

    assert returned is registry
    assert len(registry.list_names()) == 6
    assert registry.list_names() == sorted(VALUATION_NAMES)
    assert len(set(VALUATION_NAMES)) == 6


def test_default_registry_is_not_modified() -> None:
    assert create_default_registry().list_names() == [
        "momentum_20d",
        "volatility_20d",
    ]


def test_metadata_matches_specification() -> None:
    expected = {
        "ep_ttm": ("valuation", 1, ("daily_basic",), ("pe_ttm",), 0),
        "bp": ("valuation", 1, ("daily_basic",), ("pb",), 0),
        "sp_ttm": ("valuation", 1, ("daily_basic",), ("ps_ttm",), 0),
        "dividend_yield_ttm": (
            "valuation",
            1,
            ("daily_basic",),
            ("dv_ttm",),
            0,
        ),
        "log_total_mv": ("size", -1, ("daily_basic",), ("total_mv",), 0),
        "log_circ_mv": ("size", -1, ("daily_basic",), ("circ_mv",), 0),
    }

    for factor in VALUATION_FACTORS:
        metadata = factor.metadata
        assert (
            metadata.category,
            metadata.direction,
            metadata.required_datasets,
            metadata.source_fields,
            metadata.lookback_days,
        ) == expected[metadata.name]
        assert metadata.frequency == "daily"
        assert metadata.availability_lag_days == 0


@pytest.mark.parametrize(
    ("factor", "field_name"),
    [(EP_TTM, "pe_ttm"), (BP, "pb"), (SP_TTM, "ps_ttm")],
)
def test_inverse_valuation_factors_match_manual_formula(factor, field_name: str) -> None:
    data = make_valuation_data()
    expected = 1.0 / data[field_name]

    pd.testing.assert_series_equal(
        factor.compute(data),
        expected.rename(factor.metadata.name),
    )


def test_dividend_yield_preserves_original_values_and_unit() -> None:
    data = make_valuation_data()

    pd.testing.assert_series_equal(
        DIVIDEND_YIELD_TTM.compute(data),
        data["dv_ttm"].rename("dividend_yield_ttm"),
    )


@pytest.mark.parametrize(
    ("factor", "field_name"),
    [(LOG_TOTAL_MV, "total_mv"), (LOG_CIRC_MV, "circ_mv")],
)
def test_log_size_factors_match_numpy_log(factor, field_name: str) -> None:
    data = make_valuation_data()
    expected = np.log(data[field_name])

    pd.testing.assert_series_equal(
        factor.compute(data),
        expected.rename(factor.metadata.name),
    )


@pytest.mark.parametrize("factor", VALUATION_FACTORS)
def test_outputs_preserve_length_index_and_numeric_dtype(factor) -> None:
    data = make_valuation_data()

    result = factor.compute(data)

    assert len(result) == len(data)
    assert result.index.equals(data.index)
    assert pd.api.types.is_numeric_dtype(result)


@pytest.mark.parametrize(
    ("factor", "field_name"),
    [(EP_TTM, "pe_ttm"), (BP, "pb"), (SP_TTM, "ps_ttm")],
)
@pytest.mark.parametrize("invalid_value", [0.0, -1.0])
def test_non_positive_valuation_denominators_become_nan(
    factor,
    field_name: str,
    invalid_value: float,
) -> None:
    data = make_valuation_data()
    data.iloc[0, data.columns.get_loc(field_name)] = invalid_value

    assert pd.isna(factor.compute(data).iloc[0])


def test_dividend_yield_masks_negative_and_preserves_zero() -> None:
    data = make_valuation_data()
    data.iloc[0, data.columns.get_loc("dv_ttm")] = -1.0
    data.iloc[1, data.columns.get_loc("dv_ttm")] = 0.0
    result = DIVIDEND_YIELD_TTM.compute(data)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 0.0


@pytest.mark.parametrize(
    ("factor", "field_name"),
    [(LOG_TOTAL_MV, "total_mv"), (LOG_CIRC_MV, "circ_mv")],
)
@pytest.mark.parametrize("invalid_value", [0.0, -1.0])
def test_non_positive_market_values_become_nan(
    factor,
    field_name: str,
    invalid_value: float,
) -> None:
    data = make_valuation_data()
    data.iloc[0, data.columns.get_loc(field_name)] = invalid_value

    assert pd.isna(factor.compute(data).iloc[0])


def test_outputs_never_contain_infinity() -> None:
    data = make_valuation_data()
    data.iloc[0] = [0.0, -1.0, 0.0, -1.0, 0.0, -1.0]

    for factor in VALUATION_FACTORS:
        result = factor.compute(data)
        assert not np.isinf(result.dropna()).any()


@pytest.mark.parametrize("factor", VALUATION_FACTORS)
def test_missing_required_field_raises_clear_error(factor) -> None:
    field_name = factor.metadata.source_fields[0]
    data = make_valuation_data().drop(columns=field_name)

    with pytest.raises(ValueError, match=field_name):
        factor.compute(data)


def test_factor_calculation_does_not_modify_input() -> None:
    data = make_valuation_data()
    before = data.copy(deep=True)

    for factor in VALUATION_FACTORS:
        factor.compute(data)

    pd.testing.assert_frame_equal(data, before)


def test_factor_engine_computes_all_six_factors_for_two_stocks() -> None:
    data = make_multi_stock_data()
    panel = FactorEngine(make_valuation_registry()).compute_factor_panel(
        data,
        VALUATION_NAMES,
    )

    assert list(panel.columns) == ["trade_date", "ts_code"] + VALUATION_NAMES
    assert len(panel) == len(data.drop_duplicates(["trade_date", "ts_code"]))
    assert set(panel["ts_code"]) == {"000001.SZ", "000002.SZ"}


def test_describe_requirements_for_valuation_pack() -> None:
    requirements = FactorEngine(make_valuation_registry()).describe_requirements(
        VALUATION_NAMES
    )

    assert requirements["required_datasets"] == ["daily_basic"]
    assert requirements["source_fields"] == [
        "circ_mv",
        "dv_ttm",
        "pb",
        "pe_ttm",
        "ps_ttm",
        "total_mv",
    ]
    assert requirements["max_lookback_days"] == 0
    assert requirements["max_availability_lag_days"] == 0
    assert requirements["categories"] == ["size", "valuation"]


def test_changing_one_stock_does_not_affect_the_other_stock() -> None:
    data = make_multi_stock_data()
    changed = data.copy(deep=True)
    first_mask = changed["ts_code"] == "000001.SZ"
    for field_name in ["pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv"]:
        changed.loc[first_mask, field_name] *= 4.0
    changed.loc[first_mask, "dv_ttm"] += 3.0
    engine = FactorEngine(make_valuation_registry())

    before = engine.compute_factor_panel(data, VALUATION_NAMES)
    after = engine.compute_factor_panel(changed, VALUATION_NAMES)
    second_mask = before["ts_code"] == "000002.SZ"

    pd.testing.assert_frame_equal(
        before.loc[second_mask, VALUATION_NAMES].reset_index(drop=True),
        after.loc[second_mask, VALUATION_NAMES].reset_index(drop=True),
    )


@pytest.mark.parametrize("factor", VALUATION_FACTORS)
def test_future_changes_do_not_affect_past_values(factor) -> None:
    data = make_valuation_data()
    changed = data.copy(deep=True)
    field_name = factor.metadata.source_fields[0]
    changed.iloc[-3:, changed.columns.get_loc(field_name)] *= 5.0

    before = factor.compute(data)
    after = factor.compute(changed)

    pd.testing.assert_series_equal(before.iloc[:-3], after.iloc[:-3])


def test_all_factor_packs_register_without_name_conflicts() -> None:
    registry = FactorRegistry()
    register_example_factors(registry)
    register_price_volume_factors(registry)
    register_valuation_factors(registry)

    names = registry.list_names()
    assert len(names) == 16
    assert len(names) == len(set(names))
