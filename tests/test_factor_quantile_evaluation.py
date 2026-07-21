"""Tests for rank-based quantile and long-short factor evaluation."""

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pandas as pd
import pytest

from src.factors.evaluation import FactorEvaluationConfig, FactorEvaluator
from src.factors.examples import register_example_factors
from src.factors.neutralization import FactorNeutralizer, NeutralizationConfig
from src.factors.preprocessing import FactorPreprocessor, PreprocessingConfig
from src.factors.quantile_evaluation import (
    LONG_SHORT_RESULT_COLUMNS,
    LONG_SHORT_SUMMARY_COLUMNS,
    QUANTILE_RESULT_COLUMNS,
    QUANTILE_SUMMARY_COLUMNS,
    FactorQuantileEvaluator,
    QuantileEvaluationConfig,
)
from src.factors.registry import FactorRegistry


def _panels(dates: int = 1, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_frames = []
    return_frames = []
    codes = [f"S{i:02d}" for i in range(n)]
    for date_index in range(dates):
        values = np.arange(1.0, n + 1.0) + date_index * n
        date = f"2024-01-{date_index + 2:02d}"
        factor_frames.append(
            pd.DataFrame(
                {
                    "trade_date": [date] * n,
                    "ts_code": codes,
                    "factor_a": values,
                    "factor_b": values[::-1] + np.sin(values),
                }
            )
        )
        return_frames.append(
            pd.DataFrame(
                {
                    "trade_date": [date] * n,
                    "ts_code": codes,
                    "forward_return": values * 0.01,
                }
            )
        )
    return pd.concat(factor_frames, ignore_index=True), pd.concat(return_frames, ignore_index=True)


def _config(**overrides: object) -> QuantileEvaluationConfig:
    defaults = {"quantiles": 5, "min_cross_section_size": 5}
    defaults.update(overrides)
    return QuantileEvaluationConfig(**defaults)


def _evaluate(
    factors: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    names: list[str] | None = None,
    **config: object,
) -> pd.DataFrame:
    default_factors, default_returns = _panels()
    return FactorQuantileEvaluator(_config(**config)).evaluate_quantiles(
        default_factors if factors is None else factors,
        default_returns if returns is None else returns,
        ["factor_a"] if names is None else names,
    )


def test_default_config_and_serializable_description() -> None:
    config = QuantileEvaluationConfig()
    assert config.to_dict() == {
        "return_col": "forward_return",
        "quantiles": 5,
        "min_cross_section_size": 20,
        "min_group_size": 1,
        "compute_monotonicity": True,
    }
    json.dumps(config.to_dict())
    assert FactorQuantileEvaluator(config).describe_config() == config.to_dict()


def test_config_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        QuantileEvaluationConfig().quantiles = 3  # type: ignore[misc]


@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_return_column_raises(value: object) -> None:
    with pytest.raises(ValueError):
        QuantileEvaluationConfig(return_col=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, 0, -1, 2.0, True, "5"])
def test_invalid_quantiles_raise(value: object) -> None:
    with pytest.raises(ValueError):
        QuantileEvaluationConfig(quantiles=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, 0, -1, 2.0, True, "20"])
def test_invalid_min_cross_section_size_raises(value: object) -> None:
    with pytest.raises(ValueError):
        QuantileEvaluationConfig(min_cross_section_size=value)  # type: ignore[arg-type]


def test_minimum_cross_section_must_cover_quantiles() -> None:
    with pytest.raises(ValueError):
        QuantileEvaluationConfig(quantiles=5, min_cross_section_size=4)


@pytest.mark.parametrize("value", [0, -1, 1.0, True, "1"])
def test_invalid_min_group_size_raises(value: object) -> None:
    with pytest.raises(ValueError):
        QuantileEvaluationConfig(min_group_size=value)  # type: ignore[arg-type]


def test_compute_monotonicity_must_be_bool() -> None:
    with pytest.raises(ValueError):
        QuantileEvaluationConfig(compute_monotonicity=1)  # type: ignore[arg-type]


def test_constructor_rejects_invalid_config() -> None:
    with pytest.raises(TypeError):
        FactorQuantileEvaluator(object())  # type: ignore[arg-type]


def test_factor_panel_must_be_dataframe() -> None:
    _, returns = _panels()
    with pytest.raises(TypeError):
        FactorQuantileEvaluator(_config()).evaluate_quantiles([], returns, ["factor_a"])  # type: ignore[arg-type]


def test_forward_returns_must_be_dataframe() -> None:
    factors, _ = _panels()
    with pytest.raises(TypeError):
        FactorQuantileEvaluator(_config()).evaluate_quantiles(factors, [], ["factor_a"])  # type: ignore[arg-type]


@pytest.mark.parametrize("names", [[], (), "factor_a"])
def test_factor_names_must_be_non_empty_sequence(names: object) -> None:
    factors, returns = _panels()
    with pytest.raises((TypeError, ValueError)):
        FactorQuantileEvaluator(_config()).evaluate_quantiles(factors, returns, names)  # type: ignore[arg-type]


def test_duplicate_factor_names_raise() -> None:
    with pytest.raises(ValueError):
        _evaluate(names=["factor_a", "factor_a"])


@pytest.mark.parametrize(
    ("panel", "column"),
    [
        ("factor", "trade_date"),
        ("factor", "ts_code"),
        ("factor", "factor_a"),
        ("return", "trade_date"),
        ("return", "ts_code"),
        ("return", "forward_return"),
    ],
)
def test_missing_required_columns_raise(panel: str, column: str) -> None:
    factors, returns = _panels()
    if panel == "factor":
        factors = factors.drop(columns=column)
    else:
        returns = returns.drop(columns=column)
    with pytest.raises(ValueError):
        _evaluate(factors, returns)


@pytest.mark.parametrize("panel", ["factor", "return"])
def test_duplicate_keys_raise_after_code_stripping(panel: str) -> None:
    factors, returns = _panels()
    target = factors if panel == "factor" else returns
    target.loc[1, "ts_code"] = " S00 "
    with pytest.raises(ValueError):
        _evaluate(factors, returns)


@pytest.mark.parametrize("panel", ["factor", "return"])
@pytest.mark.parametrize("empty", [None, "", "   "])
def test_empty_stock_codes_raise(panel: str, empty: object) -> None:
    factors, returns = _panels()
    target = factors if panel == "factor" else returns
    target.loc[0, "ts_code"] = empty
    with pytest.raises(ValueError):
        _evaluate(factors, returns)


def test_inputs_are_not_modified() -> None:
    factors, returns = _panels()
    expected_factors = factors.copy(deep=True)
    expected_returns = returns.copy(deep=True)
    _evaluate(factors, returns)
    pd.testing.assert_frame_equal(factors, expected_factors)
    pd.testing.assert_frame_equal(returns, expected_returns)


def test_universe_observation_and_coverage_contract() -> None:
    factors, returns = _panels()
    factors.loc[0, "factor_a"] = np.nan
    factors.loc[1, "factor_a"] = np.inf
    returns.loc[2, "forward_return"] = -np.inf
    returns = returns.iloc[:-1]
    returns = pd.concat(
        [returns, pd.DataFrame({"trade_date": ["2024-01-02"], "ts_code": ["EXTRA"], "forward_return": [9.0]})],
        ignore_index=True,
    )
    result = _evaluate(factors, returns, quantiles=3, min_cross_section_size=3)
    assert result["universe_count"].eq(10).all()
    assert result["n_obs"].eq(6).all()
    assert result["coverage"].eq(0.6).all()
    assert result["coverage"].between(0.0, 1.0).all()


def test_insufficient_sample_keeps_all_groups_with_nan_returns() -> None:
    factors, returns = _panels(n=5)
    factors.loc[0, "factor_a"] = np.nan
    result = _evaluate(factors, returns, quantiles=5, min_cross_section_size=5)
    assert result["quantile"].tolist() == [1, 2, 3, 4, 5]
    assert result["group_count"].eq(0).all()
    assert result["group_return"].isna().all()


def test_increasing_factor_has_correct_low_and_high_groups() -> None:
    result = _evaluate()
    assert result["quantile"].tolist() == [1, 2, 3, 4, 5]
    assert result.loc[0, "group_count"] == 2
    assert result.loc[4, "group_count"] == 2
    assert result.loc[0, "group_return"] < result.loc[4, "group_return"]


def test_all_dates_emit_complete_quantile_schema() -> None:
    factors, returns = _panels(2)
    result = _evaluate(factors, returns, names=["factor_b", "factor_a"])
    assert result.columns.tolist() == QUANTILE_RESULT_COLUMNS
    assert len(result) == 2 * 2 * 5
    assert result[["trade_date", "factor_name", "quantile"]].values.tolist() == sorted(
        result[["trade_date", "factor_name", "quantile"]].values.tolist()
    )


def test_ties_use_average_rank_and_are_not_split() -> None:
    factors, returns = _panels(n=10)
    factors["factor_a"] = [1, 1, 1, 1, 5, 6, 7, 8, 9, 10]
    result = _evaluate(factors, returns)
    assert result.loc[result["quantile"] == 1, "group_count"].item() == 4
    assert result.loc[result["quantile"] == 2, "group_count"].item() == 0
    assert np.isnan(result.loc[result["quantile"] == 2, "group_return"].item())


def test_input_order_does_not_change_groups() -> None:
    factors, returns = _panels()
    first = _evaluate(factors, returns)
    second = _evaluate(
        factors.sample(frac=1.0, random_state=1),
        returns.sample(frac=1.0, random_state=2),
    )
    pd.testing.assert_frame_equal(first, second)


def test_constant_factor_cannot_form_groups() -> None:
    factors, returns = _panels()
    factors["factor_a"] = 1.0
    result = _evaluate(factors, returns)
    assert result["group_count"].eq(0).all()
    assert result["group_return"].isna().all()


def test_grouping_is_date_isolated_and_future_safe() -> None:
    factors, returns = _panels(2)
    first = _evaluate(factors, returns)
    factors.loc[factors["trade_date"] == "2024-01-03", "factor_a"] *= -100
    returns.loc[returns["trade_date"] == "2024-01-03", "forward_return"] **= 2
    second = _evaluate(factors, returns)
    pd.testing.assert_frame_equal(first.iloc[:5], second.iloc[:5])


def test_group_return_is_equal_weight_mean_and_group_size_rule() -> None:
    result = _evaluate()
    assert result.loc[result["quantile"] == 1, "group_return"].item() == pytest.approx(0.015)
    strict = _evaluate(min_group_size=3)
    assert np.isnan(strict.loc[strict["quantile"] == 5, "group_return"].item())


def test_group_outputs_have_no_infinity() -> None:
    factors, returns = _panels()
    returns.loc[0, "forward_return"] = np.inf
    result = _evaluate(factors, returns)
    assert not np.isinf(result.select_dtypes(include="number").to_numpy()).any()


def test_multi_factor_grouping_is_independent() -> None:
    factors, returns = _panels()
    first = _evaluate(factors, returns, names=["factor_a", "factor_b"])
    factors["factor_a"] = factors["factor_a"] ** 3
    second = _evaluate(factors, returns, names=["factor_a", "factor_b"])
    pd.testing.assert_frame_equal(
        first[first["factor_name"] == "factor_b"].reset_index(drop=True),
        second[second["factor_name"] == "factor_b"].reset_index(drop=True),
    )


def test_long_short_values_nonempty_count_and_positive_direction() -> None:
    evaluator = FactorQuantileEvaluator(_config())
    long_short = evaluator.evaluate_long_short(_evaluate())
    assert long_short.columns.tolist() == LONG_SHORT_RESULT_COLUMNS
    assert long_short.loc[0, "low_group_return"] == pytest.approx(0.015)
    assert long_short.loc[0, "high_group_return"] == pytest.approx(0.095)
    assert long_short.loc[0, "long_short_return"] == pytest.approx(0.08)
    assert long_short.loc[0, "nonempty_quantiles"] == 5


def test_inverse_relation_produces_negative_long_short() -> None:
    factors, returns = _panels()
    factors["factor_a"] = factors["factor_a"][::-1].to_numpy()
    evaluator = FactorQuantileEvaluator(_config())
    long_short = evaluator.evaluate_long_short(_evaluate(factors, returns))
    assert long_short.loc[0, "long_short_return"] < 0


def test_invalid_extreme_group_makes_long_short_nan() -> None:
    evaluator = FactorQuantileEvaluator(_config(min_group_size=3))
    result = evaluator.evaluate_quantiles(*_panels(), ["factor_a"])
    long_short = evaluator.evaluate_long_short(result)
    assert np.isnan(long_short.loc[0, "high_group_return"])
    assert np.isnan(long_short.loc[0, "long_short_return"])


def test_monotonicity_increasing_decreasing_constant_and_switch() -> None:
    evaluator = FactorQuantileEvaluator(_config())
    increasing = evaluator.evaluate_long_short(_evaluate())
    assert increasing.loc[0, "monotonicity"] == pytest.approx(1.0)
    factors, returns = _panels()
    returns["forward_return"] *= -1
    decreasing = evaluator.evaluate_long_short(_evaluate(factors, returns))
    assert decreasing.loc[0, "monotonicity"] == pytest.approx(-1.0)
    returns["forward_return"] = 0.0
    constant = evaluator.evaluate_long_short(_evaluate(factors, returns))
    assert np.isnan(constant.loc[0, "monotonicity"])
    disabled = FactorQuantileEvaluator(_config(compute_monotonicity=False)).evaluate_long_short(_evaluate())
    assert disabled["monotonicity"].isna().all()


def test_monotonicity_with_fewer_than_two_valid_groups_is_nan() -> None:
    evaluator = FactorQuantileEvaluator(_config(min_group_size=3))
    long_short = evaluator.evaluate_long_short(evaluator.evaluate_quantiles(*_panels(), ["factor_a"]))
    assert long_short.loc[0, "nonempty_quantiles"] < 2
    assert np.isnan(long_short.loc[0, "monotonicity"])


def _quantile_summary_input() -> pd.DataFrame:
    rows = []
    for date, values, counts in [
        ("2024-01-02", [0.0, 0.02], [2, 3]),
        ("2024-01-03", [0.1, 0.04], [4, 1]),
        ("2024-01-04", [np.nan, -0.02], [0, 2]),
    ]:
        for quantile, value, count in zip([1, 2], values, counts):
            rows.append({"trade_date": date, "factor_name": "a", "quantile": quantile, "universe_count": 5, "n_obs": 5, "coverage": 1.0, "group_count": count, "group_return": value})
    return pd.DataFrame(rows)


def test_quantile_summary_metrics_and_schema() -> None:
    evaluator = FactorQuantileEvaluator(_config(quantiles=2, min_cross_section_size=2))
    summary = evaluator.summarize_quantiles(_quantile_summary_input())
    assert summary.columns.tolist() == QUANTILE_SUMMARY_COLUMNS
    first = summary[summary["quantile"] == 1].iloc[0]
    values = pd.Series([0.0, 0.1])
    assert first["total_periods"] == 3
    assert first["valid_periods"] == 2
    assert first["mean_group_return"] == pytest.approx(values.mean())
    assert first["std_group_return"] == pytest.approx(values.std(ddof=1))
    assert first["positive_return_ratio"] == pytest.approx(0.5)
    assert first["mean_group_count"] == pytest.approx(2.0)


def test_quantile_summary_single_valid_period_has_nan_std() -> None:
    data = _quantile_summary_input()
    data.loc[data["quantile"] == 1, "group_return"] = [np.nan, 0.1, np.nan]
    summary = FactorQuantileEvaluator(_config(quantiles=2, min_cross_section_size=2)).summarize_quantiles(data)
    assert np.isnan(summary.loc[summary["quantile"] == 1, "std_group_return"].item())


def _long_short_summary_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "factor_name": ["a"] * 3,
            "low_group_return": [0.0, 0.01, 0.02],
            "high_group_return": [0.1, 0.11, 0.12],
            "long_short_return": [0.1, 0.1, 0.1],
            "nonempty_quantiles": [5, 5, 4],
            "monotonicity": [1.0, 0.5, -0.5],
            "coverage": [1.0, 0.8, 0.6],
            "n_obs": [10, 8, 6],
        }
    )


def test_long_short_summary_metrics_schema_and_zero_std_ir() -> None:
    summary = FactorQuantileEvaluator(_config()).summarize_long_short(_long_short_summary_input())
    assert summary.columns.tolist() == LONG_SHORT_SUMMARY_COLUMNS
    row = summary.iloc[0]
    assert row["total_periods"] == 3
    assert row["valid_long_short_periods"] == 3
    assert row["mean_long_short_return"] == pytest.approx(0.1)
    assert row["std_long_short_return"] == pytest.approx(0.0)
    assert np.isnan(row["long_short_ir"])
    assert row["positive_long_short_ratio"] == 1.0
    assert row["valid_monotonicity_periods"] == 3
    assert row["mean_monotonicity"] == pytest.approx(1 / 3)
    assert row["positive_monotonicity_ratio"] == pytest.approx(2 / 3)
    assert row["mean_coverage"] == pytest.approx(0.8)
    assert row["mean_n_obs"] == pytest.approx(8.0)


def test_long_short_ir_uses_sample_std_without_annualization() -> None:
    data = _long_short_summary_input()
    data["long_short_return"] = [0.1, 0.2, -0.1]
    row = FactorQuantileEvaluator(_config()).summarize_long_short(data).iloc[0]
    values = pd.Series([0.1, 0.2, -0.1])
    assert row["std_long_short_return"] == pytest.approx(values.std(ddof=1))
    assert row["long_short_ir"] == pytest.approx(values.mean() / values.std(ddof=1))
    assert row["positive_long_short_ratio"] == pytest.approx(2 / 3)


def test_summary_validation_and_optional_coverage_columns() -> None:
    evaluator = FactorQuantileEvaluator(_config())
    core = _long_short_summary_input().drop(columns=["coverage", "n_obs"])
    summary = evaluator.summarize_long_short(core)
    assert np.isnan(summary.loc[0, "mean_coverage"])
    assert np.isnan(summary.loc[0, "mean_n_obs"])
    with pytest.raises(TypeError):
        evaluator.summarize_quantiles([])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evaluator.summarize_long_short(pd.DataFrame())


def _integration_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    codes = ["A", "B", "C", "D", "E", "F"]
    dates = ["2024-01-02"] * 6 + ["2024-01-03"] * 6
    raw = pd.DataFrame({"trade_date": dates, "ts_code": codes * 2, "momentum_20d": [1, 2, 4, 3, 6, 5, 11, 13, 12, 16, 14, 15], "volatility_20d": [6, 4, 5, 2, 3, 1, 16, 14, 15, 12, 13, 11]})
    exposures = pd.DataFrame({"trade_date": dates, "ts_code": codes * 2, "industry": ["X", "X", "X", "Y", "Y", "Y"] * 2, "log_total_mv": list(range(1, 7)) + list(range(11, 17))})
    returns = pd.DataFrame({"trade_date": dates, "ts_code": codes * 2, "forward_return": [0.01, 0.03, 0.02, -0.01, 0.05, 0.04] * 2})
    return raw, exposures, returns


def test_full_d1_d2_e1_e2_integration_and_direction_contract() -> None:
    raw, exposures, returns = _integration_inputs()
    registry = FactorRegistry()
    register_example_factors(registry)
    processed = FactorPreprocessor(registry, PreprocessingConfig(missing_method="none", winsor_method="none", standardize_method="zscore", min_cross_section_size=3)).transform(raw, ["momentum_20d", "volatility_20d"])
    assert processed.loc[0, "volatility_20d"] < processed.loc[5, "volatility_20d"]
    neutralized = FactorNeutralizer(NeutralizationConfig(neutralize_industry=True, neutralize_size=False, min_cross_section_size=3, min_industry_size=2, standardize_residuals=True)).transform(processed, exposures, ["momentum_20d", "volatility_20d"])
    ic = FactorEvaluator(FactorEvaluationConfig(min_cross_section_size=3)).evaluate_ic(neutralized, returns, ["momentum_20d", "volatility_20d"])
    evaluator = FactorQuantileEvaluator(QuantileEvaluationConfig(quantiles=3, min_cross_section_size=3))
    quantiles = evaluator.evaluate_quantiles(neutralized, returns, ["momentum_20d", "volatility_20d"])
    long_short = evaluator.evaluate_long_short(quantiles)
    assert ic[["ic", "rank_ic"]].notna().all().all()
    assert len(quantiles) == 12
    assert len(long_short) == 4
    assert not np.isinf(quantiles.select_dtypes(include="number").to_numpy()).any()
    assert not np.isinf(long_short.select_dtypes(include="number").to_numpy()).any()


def test_integrated_future_and_factor_isolation() -> None:
    raw, _, returns = _integration_inputs()
    evaluator = FactorQuantileEvaluator(QuantileEvaluationConfig(quantiles=3, min_cross_section_size=3))
    first = evaluator.evaluate_quantiles(raw, returns, ["momentum_20d", "volatility_20d"])
    changed = raw.copy()
    changed.loc[changed["trade_date"] == "2024-01-03", "momentum_20d"] *= -100
    second = evaluator.evaluate_quantiles(changed, returns, ["momentum_20d", "volatility_20d"])
    pd.testing.assert_frame_equal(first.iloc[:6], second.iloc[:6])
    pd.testing.assert_frame_equal(
        first[first["factor_name"] == "volatility_20d"].reset_index(drop=True),
        second[second["factor_name"] == "volatility_20d"].reset_index(drop=True),
    )
