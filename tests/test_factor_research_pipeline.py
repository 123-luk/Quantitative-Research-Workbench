"""Tests for pure in-memory orchestration of the V2 research components."""

import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.factors.composition import FactorCompositionConfig
from src.factors.dynamic_composition import (
    WEIGHT_HISTORY_COLUMNS,
    RollingICWeightConfig,
)
from src.factors.evaluation import IC_RESULT_COLUMNS, IC_SUMMARY_COLUMNS, FactorEvaluationConfig
from src.factors.examples import register_example_factors
from src.factors.financial_factors import register_financial_factors
from src.factors.forward_returns import ForwardReturnConfig
from src.factors.neutralization import NeutralizationConfig
from src.factors.preprocessing import PreprocessingConfig
from src.factors.quantile_evaluation import (
    LONG_SHORT_RESULT_COLUMNS,
    LONG_SHORT_SUMMARY_COLUMNS,
    QUANTILE_RESULT_COLUMNS,
    QUANTILE_SUMMARY_COLUMNS,
    QuantileEvaluationConfig,
)
from src.factors.registry import FactorRegistry
from src.factors.research_pipeline import (
    FactorResearchConfig,
    FactorResearchResult,
    FactorResearchRunner,
)
from src.factors.valuation import register_valuation_factors


NAMES = ("momentum_20d", "volatility_20d")


def _registry() -> FactorRegistry:
    registry = FactorRegistry()
    register_example_factors(registry)
    return registry


def _panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=32)
    codes = [f"S{i:02d}" for i in range(10)]
    factor_rows = []
    price_rows = []
    for date_index, date in enumerate(dates):
        for stock_index, code in enumerate(codes):
            growth = 0.0015 * (stock_index + 1)
            close = (
                100.0
                * (1.0 + growth) ** date_index
                * (1.0 + 0.002 * np.sin(date_index * 0.7 + stock_index))
            )
            price_rows.append(
                {"trade_date": date, "ts_code": code, "close": close}
            )
            if date_index < 30:
                factor_rows.append(
                    {"trade_date": date, "ts_code": code, "close": close}
                )
    score_dates = dates[22:26]
    score_panel = pd.DataFrame(
        [
            {"trade_date": date, "ts_code": code}
            for date in reversed(score_dates)
            for code in reversed(codes)
        ]
    )
    exposures = score_panel.copy()
    exposures["industry"] = [
        "I1" if int(code[1:]) < 5 else "I2" for code in exposures["ts_code"]
    ]
    exposures["log_total_mv"] = [
        8.0 + int(code[1:]) * 0.15 for code in exposures["ts_code"]
    ]
    return (
        pd.DataFrame(factor_rows),
        score_panel,
        pd.DataFrame(price_rows),
        exposures,
    )


def _runner(
    method: str = "equal",
    *,
    use_neutralization: bool = False,
    evaluate_components: bool = True,
    evaluate_composite: bool | None = None,
    composition_config: FactorCompositionConfig | None = None,
    rolling_config: RollingICWeightConfig | None = None,
    registry: FactorRegistry | None = None,
    factor_names: tuple[str, ...] = NAMES,
    return_col: str = "forward_return",
) -> FactorResearchRunner:
    if evaluate_composite is None:
        evaluate_composite = method != "none"
    return FactorResearchRunner(
        registry or _registry(),
        FactorResearchConfig(
            factor_names=factor_names,
            use_neutralization=use_neutralization,
            composition_method=method,
            evaluate_components=evaluate_components,
            evaluate_composite=evaluate_composite,
        ),
        preprocessing_config=PreprocessingConfig(
            missing_method="none",
            winsor_method="none",
            standardize_method="zscore",
            min_cross_section_size=5,
        ),
        neutralization_config=NeutralizationConfig(
            min_cross_section_size=5,
            min_industry_size=2,
            standardize_residuals=True,
        ),
        evaluation_config=FactorEvaluationConfig(
            return_col=return_col, min_cross_section_size=5
        ),
        quantile_config=QuantileEvaluationConfig(
            return_col=return_col,
            quantiles=5,
            min_cross_section_size=5,
            min_group_size=1,
        ),
        composition_config=composition_config,
        rolling_config=rolling_config,
        forward_return_config=ForwardReturnConfig(
            return_col=return_col,
            entry_lag_periods=1,
            holding_periods=1,
        ),
    )


def _run(runner: FactorResearchRunner | None = None) -> FactorResearchResult:
    factor_input, scores, prices, exposures = _panels()
    active = runner or _runner()
    return active.run(factor_input, scores, prices, exposures)


def test_basic_config_is_serializable_and_normalizes_tuple() -> None:
    config = FactorResearchConfig(factor_names=[" momentum_20d "])
    assert config.factor_names == ("momentum_20d",)
    json.dumps(config.to_dict())


@pytest.mark.parametrize("names", [(), [], "", ("",), ("a", "a")])
def test_invalid_factor_names_raise(names: object) -> None:
    with pytest.raises(ValueError):
        FactorResearchConfig(factor_names=names)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field", ["use_neutralization", "evaluate_components", "evaluate_composite"]
)
def test_boolean_config_fields_are_validated(field: str) -> None:
    with pytest.raises(ValueError):
        FactorResearchConfig(factor_names=NAMES, **{field: 1})


def test_invalid_composition_method_and_dependencies_raise() -> None:
    with pytest.raises(ValueError):
        FactorResearchConfig(factor_names=NAMES, composition_method="bad")
    with pytest.raises(ValueError):
        FactorResearchConfig(
            factor_names=NAMES,
            composition_method="none",
            evaluate_composite=True,
        )
    with pytest.raises(ValueError):
        FactorResearchConfig(
            factor_names=NAMES,
            composition_method="rolling_ic",
            evaluate_components=False,
        )


def test_runner_constructor_types_and_unregistered_factors_raise() -> None:
    config = FactorResearchConfig(factor_names=NAMES)
    with pytest.raises(TypeError):
        FactorResearchRunner(object(), config)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        FactorResearchRunner(_registry(), object())  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        FactorResearchRunner(
            _registry(), FactorResearchConfig(factor_names=("unknown",))
        )


def test_return_columns_must_match() -> None:
    with pytest.raises(ValueError):
        FactorResearchRunner(
            _registry(),
            FactorResearchConfig(factor_names=NAMES),
            evaluation_config=FactorEvaluationConfig(
                return_col="return_a", min_cross_section_size=5
            ),
            quantile_config=QuantileEvaluationConfig(
                return_col="return_b",
                quantiles=5,
                min_cross_section_size=5,
            ),
            forward_return_config=ForwardReturnConfig(return_col="return_a"),
        )


def test_static_composition_config_must_match_method() -> None:
    with pytest.raises(ValueError):
        _runner(
            method="equal",
            composition_config=FactorCompositionConfig(
                method="fixed", fixed_weights=((NAMES[0], 1.0), (NAMES[1], 1.0))
            ),
        )
    with pytest.raises(ValueError):
        _runner(
            method="fixed",
            composition_config=FactorCompositionConfig(method="equal"),
        )
    with pytest.raises(ValueError):
        _runner(method="fixed")


def test_rolling_config_metric_must_match_method() -> None:
    with pytest.raises(ValueError):
        _runner(
            method="rolling_ic",
            rolling_config=RollingICWeightConfig(metric="rank_ic"),
        )
    with pytest.raises(ValueError):
        _runner(
            method="rolling_rank_ic",
            rolling_config=RollingICWeightConfig(metric="ic"),
        )


def test_describe_config_is_serializable_and_complete() -> None:
    description = _runner().describe_config()
    assert set(description) == {
        "research_config",
        "preprocessing_config",
        "neutralization_config",
        "evaluation_config",
        "quantile_config",
        "composition_config",
        "rolling_config",
        "forward_return_config",
    }
    json.dumps(description)


def test_run_input_types_are_validated() -> None:
    factor_input, scores, prices, exposures = _panels()
    runner = _runner()
    with pytest.raises(TypeError):
        runner.run([], scores, prices, exposures)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        runner.run(factor_input, [], prices, exposures)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        runner.run(factor_input, scores, [], exposures)  # type: ignore[arg-type]


@pytest.mark.parametrize("column", ["trade_date", "ts_code"])
def test_score_panel_required_columns(column: str) -> None:
    factor_input, scores, prices, exposures = _panels()
    with pytest.raises(ValueError):
        _runner().run(
            factor_input, scores.drop(columns=column), prices, exposures
        )


def test_score_panel_duplicate_empty_code_and_missing_key_raise() -> None:
    factor_input, scores, prices, exposures = _panels()
    duplicate = pd.concat([scores, scores.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        _runner().run(factor_input, duplicate, prices, exposures)
    empty = scores.copy()
    empty.loc[0, "ts_code"] = " "
    with pytest.raises(ValueError):
        _runner().run(factor_input, empty, prices, exposures)
    missing = scores.copy()
    missing.loc[0, "ts_code"] = "NOT_REGISTERED_KEY"
    with pytest.raises(ValueError, match="computed factor panel"):
        _runner().run(factor_input, missing, prices, exposures)


def test_neutralization_requires_exposure_panel() -> None:
    factor_input, scores, prices, _ = _panels()
    with pytest.raises(ValueError):
        _runner(use_neutralization=True).run(factor_input, scores, prices)


def test_inputs_are_not_mutated() -> None:
    factor_input, scores, prices, exposures = _panels()
    originals = [
        frame.copy(deep=True) for frame in (factor_input, scores, prices, exposures)
    ]
    _runner(use_neutralization=True).run(
        factor_input, scores, prices, exposures
    )
    for actual, expected in zip(
        (factor_input, scores, prices, exposures), originals
    ):
        pdt.assert_frame_equal(actual, expected)


def test_full_history_is_used_before_score_selection_for_momentum() -> None:
    factor_input, scores, prices, exposures = _panels()
    result = _runner(
        method="none", evaluate_components=False, evaluate_composite=False
    ).run(factor_input, scores, prices, exposures)
    key = result.raw_factor_panel.iloc[0]
    stock = factor_input.loc[
        factor_input["ts_code"].eq(key["ts_code"])
    ].sort_values("trade_date")
    position = stock["trade_date"].tolist().index(key["trade_date"])
    expected = (
        stock["close"].iloc[position] / stock["close"].iloc[position - 20] - 1.0
    )
    assert key["momentum_20d"] == pytest.approx(expected)
    assert result.raw_factor_panel["momentum_20d"].notna().all()


def test_raw_panel_exactly_matches_sorted_score_keys() -> None:
    _, scores, _, _ = _panels()
    result = _run()
    expected = scores.copy()
    expected["ts_code"] = expected["ts_code"].astype("string").str.strip()
    expected["trade_date"] = pd.to_datetime(expected["trade_date"])
    expected = expected.sort_values(["trade_date", "ts_code"], ignore_index=True)
    pdt.assert_frame_equal(
        result.raw_factor_panel[["trade_date", "ts_code"]],
        expected[["trade_date", "ts_code"]],
    )
    assert len(result.raw_factor_panel) == len(scores)


def test_score_input_order_does_not_change_results() -> None:
    factor_input, scores, prices, exposures = _panels()
    baseline = _runner().run(factor_input, scores, prices, exposures)
    shuffled = _runner().run(
        factor_input,
        scores.sample(frac=1.0, random_state=4),
        prices,
        exposures.sample(frac=1.0, random_state=5),
    )
    pdt.assert_frame_equal(baseline.raw_factor_panel, shuffled.raw_factor_panel)
    pdt.assert_frame_equal(baseline.composite_scores, shuffled.composite_scores)


def test_d1_direction_adjustment_is_applied() -> None:
    result = _run()
    raw_date = result.raw_factor_panel["trade_date"].min()
    raw = result.raw_factor_panel.loc[
        result.raw_factor_panel["trade_date"].eq(raw_date)
    ].sort_values("volatility_20d")
    processed = result.processed_factor_panel.loc[
        result.processed_factor_panel["trade_date"].eq(raw_date)
    ].set_index("ts_code")
    low_vol_code = raw.iloc[0]["ts_code"]
    high_vol_code = raw.iloc[-1]["ts_code"]
    assert (
        processed.loc[low_vol_code, "volatility_20d"]
        > processed.loc[high_vol_code, "volatility_20d"]
    )


def test_without_d2_final_panel_is_equal_but_independent() -> None:
    result = _run()
    pdt.assert_frame_equal(result.final_factor_panel, result.processed_factor_panel)
    assert result.final_factor_panel is not result.processed_factor_panel


def test_neutralization_runs_and_missing_exposure_stays_missing() -> None:
    factor_input, scores, prices, exposures = _panels()
    missing_key = exposures.iloc[0][["trade_date", "ts_code"]]
    exposures = exposures.iloc[1:].copy()
    result = _runner(use_neutralization=True).run(
        factor_input, scores, prices, exposures
    )
    row = result.final_factor_panel.loc[
        result.final_factor_panel["trade_date"].eq(missing_key["trade_date"])
        & result.final_factor_panel["ts_code"].eq(missing_key["ts_code"])
    ]
    assert row[list(NAMES)].isna().all().all()
    assert result.used_neutralization


def test_future_exposure_change_does_not_change_past_neutralization() -> None:
    factor_input, scores, prices, exposures = _panels()
    baseline = _runner(use_neutralization=True).run(
        factor_input, scores, prices, exposures
    )
    future_date = exposures["trade_date"].max()
    changed = exposures.copy()
    changed.loc[changed["trade_date"].eq(future_date), "log_total_mv"] += 100.0
    result = _runner(use_neutralization=True).run(
        factor_input, scores, prices, changed
    )
    past = baseline.final_factor_panel["trade_date"].lt(future_date)
    pdt.assert_frame_equal(
        baseline.final_factor_panel.loc[past].reset_index(drop=True),
        result.final_factor_panel.loc[past].reset_index(drop=True),
    )


def test_forward_returns_are_audited_and_not_features() -> None:
    result = _run()
    assert len(result.forward_returns) == len(result.raw_factor_panel)
    assert {
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
        "forward_return",
    }.issubset(result.forward_returns.columns)
    for panel in (
        result.raw_factor_panel,
        result.processed_factor_panel,
        result.final_factor_panel,
        result.composite_scores,
    ):
        assert "forward_return" not in panel.columns
    row = result.forward_returns.iloc[0]
    assert row["forward_return"] == pytest.approx(
        row["exit_price"] / row["entry_price"] - 1.0
    )


def test_exit_price_change_changes_only_affected_labels() -> None:
    factor_input, scores, prices, exposures = _panels()
    baseline = _runner().run(factor_input, scores, prices, exposures)
    first = baseline.forward_returns.iloc[0]
    changed = prices.copy()
    changed.loc[
        changed["trade_date"].eq(first["exit_trade_date"])
        & changed["ts_code"].eq(first["ts_code"]),
        "close",
    ] *= 2.0
    result = _runner().run(factor_input, scores, changed, exposures)
    key = (
        baseline.forward_returns["trade_date"].eq(first["trade_date"])
        & baseline.forward_returns["ts_code"].eq(first["ts_code"])
    )
    assert (
        baseline.forward_returns.loc[key, "forward_return"].iloc[0]
        != result.forward_returns.loc[key, "forward_return"].iloc[0]
    )


def test_component_evaluation_produces_all_tables() -> None:
    result = _run()
    assert not result.factor_ic_results.empty
    assert set(result.factor_ic_results["factor_name"]) == set(NAMES)
    assert not result.factor_ic_summary.empty
    assert not result.factor_quantile_results.empty
    assert not result.factor_long_short_results.empty
    assert not result.factor_quantile_summary.empty
    assert not result.factor_long_short_summary.empty


def test_disabled_component_evaluation_has_stable_empty_schemas() -> None:
    result = _run(
        _runner(
            method="equal",
            evaluate_components=False,
            evaluate_composite=True,
        )
    )
    expected = [
        (result.factor_ic_results, IC_RESULT_COLUMNS),
        (result.factor_ic_summary, IC_SUMMARY_COLUMNS),
        (result.factor_quantile_results, QUANTILE_RESULT_COLUMNS),
        (result.factor_long_short_results, LONG_SHORT_RESULT_COLUMNS),
        (result.factor_quantile_summary, QUANTILE_SUMMARY_COLUMNS),
        (result.factor_long_short_summary, LONG_SHORT_SUMMARY_COLUMNS),
    ]
    for table, columns in expected:
        assert table.empty
        assert list(table.columns) == columns


def test_equal_composition_matches_processed_factor_mean() -> None:
    result = _run()
    expected = result.final_factor_panel[list(NAMES)].mean(axis=1)
    np.testing.assert_allclose(
        result.composite_scores["composite_score"], expected
    )
    assert result.weight_history.empty
    assert list(result.weight_history.columns) == WEIGHT_HISTORY_COLUMNS


def test_fixed_composition_matches_manual_formula() -> None:
    composition = FactorCompositionConfig(
        method="fixed",
        fixed_weights=((NAMES[0], 1.0), (NAMES[1], 3.0)),
    )
    result = _run(
        _runner(method="fixed", composition_config=composition)
    )
    expected = (
        result.final_factor_panel[NAMES[0]] * 0.25
        + result.final_factor_panel[NAMES[1]] * 0.75
    )
    np.testing.assert_allclose(
        result.composite_scores["composite_score"], expected
    )
    assert result.weight_history.empty


def test_none_composition_has_stable_empty_score_schema() -> None:
    result = _run(
        _runner(
            method="none",
            evaluate_components=False,
            evaluate_composite=False,
        )
    )
    assert result.composite_scores.empty
    assert list(result.composite_scores.columns) == [
        "trade_date",
        "ts_code",
        "composite_score",
        "valid_factor_count",
        "weight_coverage",
    ]
    assert result.weight_history.empty


@pytest.mark.parametrize(
    ("method", "metric"),
    [("rolling_ic", "ic"), ("rolling_rank_ic", "rank_ic")],
)
def test_rolling_composition_builds_strict_history(
    method: str, metric: str
) -> None:
    result = _run(
        _runner(
            method=method,
            rolling_config=RollingICWeightConfig(
                metric=metric,
                lookback_periods=3,
                min_periods=1,
                fallback_method="equal",
            ),
        )
    )
    assert len(result.composite_scores) == len(result.final_factor_panel)
    assert list(result.weight_history.columns) == WEIGHT_HISTORY_COLUMNS
    assert result.weight_history["metric"].eq(metric).all()
    histories = result.weight_history.dropna(subset=["history_end_date"])
    assert (histories["history_end_date"] < histories["trade_date"]).all()
    first_date = result.weight_history["trade_date"].min()
    assert result.weight_history.loc[
        result.weight_history["trade_date"].eq(first_date), "used_fallback"
    ].all()


def test_composite_evaluation_uses_actual_score_column() -> None:
    result = _run(
        _runner(
            composition_config=FactorCompositionConfig(
                method="equal", score_col="research_score"
            )
        )
    )
    assert "research_score" in result.composite_scores.columns
    assert set(result.composite_ic_results["factor_name"]) == {"research_score"}
    assert set(result.composite_quantile_results["factor_name"]) == {
        "research_score"
    }
    assert not result.composite_long_short_results.empty


def test_disabled_composite_evaluation_has_stable_schemas() -> None:
    result = _run(
        _runner(method="equal", evaluate_composite=False)
    )
    assert not result.composite_scores.empty
    for table, columns in (
        (result.composite_ic_results, IC_RESULT_COLUMNS),
        (result.composite_ic_summary, IC_SUMMARY_COLUMNS),
        (result.composite_quantile_results, QUANTILE_RESULT_COLUMNS),
        (result.composite_long_short_results, LONG_SHORT_RESULT_COLUMNS),
        (result.composite_quantile_summary, QUANTILE_SUMMARY_COLUMNS),
        (result.composite_long_short_summary, LONG_SHORT_SUMMARY_COLUMNS),
    ):
        assert table.empty
        assert list(table.columns) == columns


def test_result_summary_contains_shapes_not_table_contents() -> None:
    result = _run()
    assert isinstance(result, FactorResearchResult)
    shapes = result.table_shapes()
    assert shapes["raw_factor_panel"] == result.raw_factor_panel.shape
    summary = result.to_dict()
    assert summary["factor_names"] == list(NAMES)
    assert summary["requirements"] == result.requirements
    assert "raw_factor_panel" not in summary
    json.dumps(summary)


def test_requirements_match_engine_metadata() -> None:
    result = _run()
    assert result.requirements == {
        "factor_names": sorted(NAMES),
        "required_datasets": ["daily"],
        "source_fields": ["close"],
        "max_lookback_days": 20,
        "max_availability_lag_days": 0,
        "categories": ["momentum", "volatility"],
    }


def test_repeated_runs_are_equivalent_and_state_independent() -> None:
    factor_input, scores, prices, exposures = _panels()
    runner = _runner()
    first = runner.run(factor_input, scores, prices, exposures)
    second = runner.run(factor_input, scores, prices, exposures)
    assert first.to_dict() == second.to_dict()
    for table_name in FactorResearchResult.TABLE_FIELDS:
        pdt.assert_frame_equal(
            getattr(first, table_name), getattr(second, table_name)
        )


def test_future_factor_input_does_not_change_past_results() -> None:
    factor_input, scores, prices, exposures = _panels()
    baseline = _runner().run(factor_input, scores, prices, exposures)
    future = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2030-01-01"),
                "ts_code": code,
                "close": 9999.0,
            }
            for code in sorted(factor_input["ts_code"].unique())
        ]
    )
    result = _runner().run(
        pd.concat([factor_input, future], ignore_index=True),
        scores,
        prices,
        exposures,
    )
    pdt.assert_frame_equal(baseline.raw_factor_panel, result.raw_factor_panel)
    pdt.assert_frame_equal(
        baseline.processed_factor_panel, result.processed_factor_panel
    )


def test_valuation_and_pit_standard_financial_factors_complete_flow() -> None:
    dates = pd.bdate_range("2024-03-01", periods=3)
    codes = [f"S{i:02d}" for i in range(10)]
    rows = [
        {
            "trade_date": date,
            "ts_code": code,
            "pe_ttm": 10.0 + index,
            "fin_roe_ttm": 5.0 + index,
        }
        for date in dates
        for index, code in enumerate(codes)
    ]
    panel = pd.DataFrame(rows)
    scores = panel.loc[
        panel["trade_date"].eq(dates[0]), ["trade_date", "ts_code"]
    ]
    prices = pd.DataFrame(
        [
            {
                "trade_date": date,
                "ts_code": code,
                "close": 100.0 + date_index + stock_index,
            }
            for date_index, date in enumerate(dates)
            for stock_index, code in enumerate(codes)
        ]
    )
    registry = FactorRegistry()
    register_valuation_factors(registry)
    register_financial_factors(registry)
    for factor_name, expected_dataset in (
        ("ep_ttm", "daily_basic"),
        ("roe_ttm", "financial_pit"),
    ):
        result = _runner(
            method="none",
            evaluate_components=False,
            evaluate_composite=False,
            registry=registry,
            factor_names=(factor_name,),
        ).run(panel, scores, prices)
        assert result.raw_factor_panel[factor_name].notna().all()
        assert expected_dataset in result.requirements["required_datasets"]


def test_major_outputs_do_not_contain_infinity() -> None:
    result = _run()
    for table_name in FactorResearchResult.TABLE_FIELDS:
        table = getattr(result, table_name)
        numeric = table.select_dtypes(include=[np.number])
        assert not np.isinf(numeric).any().any()
