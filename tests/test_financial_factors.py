"""Tests for standardized PIT-aligned financial factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.examples import register_example_factors
from src.factors.factor_engine import FactorEngine
from src.factors.financial_alignment import FinancialPointInTimeAligner
from src.factors.financial_factors import (
    DEBT_TO_ASSETS,
    FINANCIAL_FACTORS,
    GROSS_MARGIN_TTM,
    NET_MARGIN_TTM,
    NET_PROFIT_YOY,
    OPERATING_CF_TO_ASSETS,
    ROA_TTM,
    ROE_TTM,
    REVENUE_YOY,
    register_financial_factors,
)
from src.factors.price_volume import register_price_volume_factors
from src.factors.registry import FactorRegistry, create_default_registry
from src.factors.valuation import register_valuation_factors


FIN_FIELD_MAP = {
    "roe_ttm": "fin_roe_ttm",
    "roa_ttm": "fin_roa_ttm",
    "gross_margin_ttm": "fin_gross_margin_ttm",
    "net_margin_ttm": "fin_net_margin_ttm",
    "revenue_yoy": "fin_revenue_yoy",
    "net_profit_yoy": "fin_net_profit_yoy",
    "debt_to_assets": "fin_debt_to_assets",
    "operating_cf_to_assets": "fin_operating_cf_to_assets",
}
FINANCIAL_NAMES = [factor.metadata.name for factor in FINANCIAL_FACTORS]
FINANCIAL_FIELDS = list(FIN_FIELD_MAP.values())


def make_financial_factor_data(periods: int = 10) -> pd.DataFrame:
    """Create standardized PIT-ready financial fields with finite negatives."""
    steps = np.arange(periods, dtype=float)
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "fin_roe_ttm": -5.0 + steps,
            "fin_roa_ttm": -2.0 + steps * 0.5,
            "fin_gross_margin_ttm": -10.0 + steps * 3.0,
            "fin_net_margin_ttm": -8.0 + steps * 2.0,
            "fin_revenue_yoy": -20.0 + steps * 4.0,
            "fin_net_profit_yoy": -30.0 + steps * 5.0,
            "fin_debt_to_assets": 65.0 + steps,
            "fin_operating_cf_to_assets": -4.0 + steps * 0.8,
        },
        index=index,
    )


def make_multi_stock_data(periods: int = 10) -> pd.DataFrame:
    first = make_financial_factor_data(periods).reset_index(names="trade_date")
    first["ts_code"] = "000001.SZ"
    second = make_financial_factor_data(periods).reset_index(names="trade_date")
    second["ts_code"] = "000002.SZ"
    second[FINANCIAL_FIELDS] = second[FINANCIAL_FIELDS] + 10.0
    return pd.concat([first, second], ignore_index=True).sample(
        frac=1.0,
        random_state=31,
    ).reset_index(drop=True)


def make_financial_registry() -> FactorRegistry:
    registry = FactorRegistry()
    register_financial_factors(registry)
    return registry


def make_financial_record(
    ts_code: str,
    ann_date: str,
    end_date: str,
    base: float,
) -> dict:
    record = {"ts_code": ts_code, "ann_date": ann_date, "end_date": end_date}
    record.update(
        {field_name: base + offset for offset, field_name in enumerate(FINANCIAL_FIELDS)}
    )
    return record


def test_registers_exactly_eight_unique_financial_factors() -> None:
    registry = FactorRegistry()

    returned = register_financial_factors(registry)

    assert returned is registry
    assert len(registry.list_names()) == 8
    assert registry.list_names() == sorted(FINANCIAL_NAMES)
    assert len(set(FINANCIAL_NAMES)) == 8


def test_default_registry_behavior_is_unchanged() -> None:
    assert create_default_registry().list_names() == [
        "momentum_20d",
        "volatility_20d",
    ]


def test_import_does_not_create_shared_registry_state() -> None:
    first = FactorRegistry()
    second = FactorRegistry()
    register_financial_factors(first)

    assert len(first.list_names()) == 8
    assert second.list_names() == []


def test_metadata_matches_standardized_field_specification() -> None:
    expected_categories = {
        "roe_ttm": ("profitability", 1),
        "roa_ttm": ("profitability", 1),
        "gross_margin_ttm": ("profitability", 1),
        "net_margin_ttm": ("profitability", 1),
        "revenue_yoy": ("growth", 1),
        "net_profit_yoy": ("growth", 1),
        "debt_to_assets": ("leverage", -1),
        "operating_cf_to_assets": ("quality", 1),
    }

    for factor in FINANCIAL_FACTORS:
        metadata = factor.metadata
        assert (metadata.category, metadata.direction) == expected_categories[metadata.name]
        assert metadata.required_datasets == ("financial_pit",)
        assert metadata.source_fields == (FIN_FIELD_MAP[metadata.name],)
        assert metadata.lookback_days == 0
        assert metadata.frequency == "daily"
        assert metadata.availability_lag_days == 0


@pytest.mark.parametrize("factor", FINANCIAL_FACTORS)
def test_factor_matches_standardized_input_without_unit_conversion(factor) -> None:
    data = make_financial_factor_data()
    source_field = FIN_FIELD_MAP[factor.metadata.name]

    pd.testing.assert_series_equal(
        factor.compute(data),
        data[source_field].rename(factor.metadata.name),
    )


@pytest.mark.parametrize("factor", FINANCIAL_FACTORS)
def test_output_preserves_length_and_index(factor) -> None:
    data = make_financial_factor_data()
    result = factor.compute(data)

    assert len(result) == len(data)
    assert result.index.equals(data.index)


@pytest.mark.parametrize("factor", FINANCIAL_FACTORS)
def test_nan_and_infinities_are_safely_normalized(factor) -> None:
    data = make_financial_factor_data()
    source_field = FIN_FIELD_MAP[factor.metadata.name]
    data.iloc[0, data.columns.get_loc(source_field)] = np.nan
    data.iloc[1, data.columns.get_loc(source_field)] = np.inf
    data.iloc[2, data.columns.get_loc(source_field)] = -np.inf

    result = factor.compute(data)

    assert result.iloc[:3].isna().all()
    assert not np.isinf(result.dropna()).any()


@pytest.mark.parametrize(
    "factor",
    [ROE_TTM, ROA_TTM, GROSS_MARGIN_TTM, NET_MARGIN_TTM, REVENUE_YOY, NET_PROFIT_YOY, OPERATING_CF_TO_ASSETS],
)
def test_economically_meaningful_negative_values_are_preserved(factor) -> None:
    data = make_financial_factor_data()
    source_field = FIN_FIELD_MAP[factor.metadata.name]
    data.iloc[0, data.columns.get_loc(source_field)] = -12.5

    assert factor.compute(data).iloc[0] == -12.5


def test_debt_to_assets_is_not_divided_or_clipped() -> None:
    data = make_financial_factor_data()
    data.iloc[0, data.columns.get_loc("fin_debt_to_assets")] = 150.0

    assert DEBT_TO_ASSETS.compute(data).iloc[0] == 150.0


@pytest.mark.parametrize("factor", FINANCIAL_FACTORS)
def test_missing_required_field_raises_clear_error(factor) -> None:
    source_field = FIN_FIELD_MAP[factor.metadata.name]
    data = make_financial_factor_data().drop(columns=source_field)

    with pytest.raises(ValueError, match=source_field):
        factor.compute(data)


def test_factor_calculation_does_not_modify_input() -> None:
    data = make_financial_factor_data()
    before = data.copy(deep=True)

    for factor in FINANCIAL_FACTORS:
        factor.compute(data)

    pd.testing.assert_frame_equal(data, before)


def test_factor_engine_computes_all_eight_for_two_stocks() -> None:
    data = make_multi_stock_data()
    panel = FactorEngine(make_financial_registry()).compute_factor_panel(
        data,
        FINANCIAL_NAMES,
    )

    assert list(panel.columns) == ["trade_date", "ts_code"] + FINANCIAL_NAMES
    assert len(panel) == len(data)
    assert set(panel["ts_code"]) == {"000001.SZ", "000002.SZ"}


def test_describe_requirements_for_financial_pack() -> None:
    requirements = FactorEngine(make_financial_registry()).describe_requirements(
        FINANCIAL_NAMES
    )

    assert requirements["required_datasets"] == ["financial_pit"]
    assert requirements["source_fields"] == sorted(FINANCIAL_FIELDS)
    assert requirements["max_lookback_days"] == 0
    assert requirements["max_availability_lag_days"] == 0
    assert requirements["categories"] == [
        "growth",
        "leverage",
        "profitability",
        "quality",
    ]


def test_changing_one_stock_does_not_affect_the_other() -> None:
    data = make_multi_stock_data()
    changed = data.copy(deep=True)
    first_mask = changed["ts_code"] == "000001.SZ"
    changed.loc[first_mask, FINANCIAL_FIELDS] *= 9.0
    engine = FactorEngine(make_financial_registry())

    before = engine.compute_factor_panel(data, FINANCIAL_NAMES)
    after = engine.compute_factor_panel(changed, FINANCIAL_NAMES)
    second_mask = before["ts_code"] == "000002.SZ"

    pd.testing.assert_frame_equal(
        before.loc[second_mask, FINANCIAL_NAMES].reset_index(drop=True),
        after.loc[second_mask, FINANCIAL_NAMES].reset_index(drop=True),
    )


@pytest.mark.parametrize("factor", FINANCIAL_FACTORS)
def test_future_rows_do_not_affect_past_values(factor) -> None:
    data = make_financial_factor_data()
    changed = data.copy(deep=True)
    source_field = FIN_FIELD_MAP[factor.metadata.name]
    changed.iloc[-3:, changed.columns.get_loc(source_field)] *= 11.0

    before = factor.compute(data)
    after = factor.compute(changed)

    pd.testing.assert_series_equal(before.iloc[:-3], after.iloc[:-3])


def test_all_factor_packs_register_without_name_conflicts() -> None:
    registry = FactorRegistry()
    register_example_factors(registry)
    register_price_volume_factors(registry)
    register_valuation_factors(registry)
    register_financial_factors(registry)

    names = registry.list_names()
    assert len(names) == 24
    assert len(names) == len(set(names))


def test_pit_alignment_then_factor_engine_handles_revision_and_audit_fields() -> None:
    dates = pd.bdate_range("2024-04-19", "2024-05-14")
    trading = pd.DataFrame(
        [
            {"trade_date": date, "ts_code": code}
            for code in ("000001.SZ", "000002.SZ")
            for date in dates
        ]
    )
    financial = pd.DataFrame(
        [
            make_financial_record("000001.SZ", "2024-04-20", "2024-03-31", 10.0),
            make_financial_record("000001.SZ", "2024-05-10", "2024-03-31", 20.0),
            make_financial_record("000002.SZ", "2024-04-22", "2024-03-31", 30.0),
        ]
    )

    aligned = FinancialPointInTimeAligner().align(
        trading,
        financial,
        value_columns=FINANCIAL_FIELDS,
    )
    panel = FactorEngine(make_financial_registry()).compute_factor_panel(
        aligned,
        FINANCIAL_NAMES,
    )
    stock_a = panel[panel["ts_code"] == "000001.SZ"].set_index("trade_date")
    stock_b = panel[panel["ts_code"] == "000002.SZ"].set_index("trade_date")

    assert {"source_ann_date", "source_end_date"}.issubset(aligned.columns)
    assert pd.isna(stock_a.loc[pd.Timestamp("2024-04-22"), "roe_ttm"])
    assert stock_a.loc[pd.Timestamp("2024-04-23"), "roe_ttm"] == 10.0
    assert stock_a.loc[pd.Timestamp("2024-05-10"), "roe_ttm"] == 10.0
    assert stock_a.loc[pd.Timestamp("2024-05-13"), "roe_ttm"] == 20.0
    assert stock_b.loc[pd.Timestamp("2024-04-23"), "roe_ttm"] == 30.0


def test_modifying_future_pit_announcement_does_not_change_past_factors() -> None:
    dates = pd.bdate_range("2024-04-19", "2024-05-20")
    trading = pd.DataFrame(
        [{"trade_date": date, "ts_code": "000001.SZ"} for date in dates]
    )
    financial = pd.DataFrame(
        [
            make_financial_record("000001.SZ", "2024-04-20", "2024-03-31", 10.0),
            make_financial_record("000001.SZ", "2024-05-10", "2024-03-31", 20.0),
        ]
    )
    changed = financial.copy(deep=True)
    changed.loc[1, "ann_date"] = "2024-05-15"
    changed.loc[1, FINANCIAL_FIELDS] = 99.0
    aligner = FinancialPointInTimeAligner()
    engine = FactorEngine(make_financial_registry())

    before = engine.compute_factor_panel(
        aligner.align(trading, financial, FINANCIAL_FIELDS),
        FINANCIAL_NAMES,
    )
    after = engine.compute_factor_panel(
        aligner.align(trading, changed, FINANCIAL_FIELDS),
        FINANCIAL_NAMES,
    )
    past = before["trade_date"] < pd.Timestamp("2024-05-10")

    pd.testing.assert_frame_equal(
        before.loc[past].reset_index(drop=True),
        after.loc[past].reset_index(drop=True),
    )
