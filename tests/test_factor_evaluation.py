"""Tests for same-date Pearson IC and Spearman RankIC evaluation."""

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pandas as pd
import pytest

from src.factors.evaluation import (
    IC_RESULT_COLUMNS,
    IC_SUMMARY_COLUMNS,
    FactorEvaluationConfig,
    FactorEvaluator,
)
from src.factors.examples import register_example_factors
from src.factors.neutralization import FactorNeutralizer, NeutralizationConfig
from src.factors.preprocessing import FactorPreprocessor, PreprocessingConfig
from src.factors.registry import FactorRegistry


def _panels(dates: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_frames = []
    return_frames = []
    codes = ["A", "B", "C", "D", "E"]
    for date_index in range(dates):
        base = np.arange(1.0, 6.0) + date_index * 10.0
        date = f"2024-01-{date_index + 2:02d}"
        factor_frames.append(
            pd.DataFrame(
                {
                    "trade_date": [date] * 5,
                    "ts_code": codes,
                    "factor_a": base,
                    "factor_b": base[::-1] + np.array([0.0, 0.2, -0.1, 0.1, 0.0]),
                }
            )
        )
        return_frames.append(
            pd.DataFrame(
                {
                    "trade_date": [date] * 5,
                    "ts_code": codes,
                    "forward_return": base * 0.02 - 0.03,
                }
            )
        )
    return pd.concat(factor_frames, ignore_index=True), pd.concat(return_frames, ignore_index=True)


def _config(**overrides: object) -> FactorEvaluationConfig:
    defaults = {"min_cross_section_size": 3}
    defaults.update(overrides)
    return FactorEvaluationConfig(**defaults)


def _evaluate(
    factors: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    names: list[str] | None = None,
    **config: object,
) -> pd.DataFrame:
    default_factors, default_returns = _panels()
    return FactorEvaluator(_config(**config)).evaluate_ic(
        default_factors if factors is None else factors,
        default_returns if returns is None else returns,
        ["factor_a"] if names is None else names,
    )


def test_default_config_and_serializable_description() -> None:
    config = FactorEvaluationConfig()
    assert config.to_dict() == {
        "return_col": "forward_return",
        "min_cross_section_size": 20,
        "compute_ic": True,
        "compute_rank_ic": True,
    }
    json.dumps(config.to_dict())
    assert FactorEvaluator(config).describe_config() == config.to_dict()


def test_config_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        FactorEvaluationConfig().return_col = "x"  # type: ignore[misc]


@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_return_col_raises(value: object) -> None:
    with pytest.raises(ValueError):
        FactorEvaluationConfig(return_col=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, 0, -1, 2.0, True, "20"])
def test_invalid_min_cross_section_size_raises(value: object) -> None:
    with pytest.raises(ValueError):
        FactorEvaluationConfig(min_cross_section_size=value)  # type: ignore[arg-type]


def test_both_metric_switches_disabled_raise() -> None:
    with pytest.raises(ValueError):
        FactorEvaluationConfig(compute_ic=False, compute_rank_ic=False)


@pytest.mark.parametrize("field", ["compute_ic", "compute_rank_ic"])
def test_metric_switches_must_be_bool(field: str) -> None:
    with pytest.raises(ValueError):
        FactorEvaluationConfig(**{field: 1})


def test_constructor_rejects_invalid_config() -> None:
    with pytest.raises(TypeError):
        FactorEvaluator(object())  # type: ignore[arg-type]


def test_factor_panel_must_be_dataframe() -> None:
    _, returns = _panels()
    with pytest.raises(TypeError):
        FactorEvaluator(_config()).evaluate_ic([], returns, ["factor_a"])  # type: ignore[arg-type]


def test_forward_returns_must_be_dataframe() -> None:
    factors, _ = _panels()
    with pytest.raises(TypeError):
        FactorEvaluator(_config()).evaluate_ic(factors, [], ["factor_a"])  # type: ignore[arg-type]


@pytest.mark.parametrize("names", [[], (), "factor_a"])
def test_factor_names_must_be_non_empty_sequence(names: object) -> None:
    factors, returns = _panels()
    with pytest.raises((TypeError, ValueError)):
        FactorEvaluator(_config()).evaluate_ic(factors, returns, names)  # type: ignore[arg-type]


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
def test_invalid_dates_raise(panel: str) -> None:
    factors, returns = _panels()
    target = factors if panel == "factor" else returns
    target.loc[0, "trade_date"] = "invalid"
    with pytest.raises(ValueError):
        _evaluate(factors, returns)


@pytest.mark.parametrize("panel", ["factor", "return"])
def test_duplicate_keys_raise_after_code_stripping(panel: str) -> None:
    factors, returns = _panels()
    target = factors if panel == "factor" else returns
    target.loc[1, "ts_code"] = " A "
    with pytest.raises(ValueError):
        _evaluate(factors, returns)


@pytest.mark.parametrize("panel", ["factor", "return"])
@pytest.mark.parametrize("empty", [None, "", "   "])
def test_empty_codes_raise(panel: str, empty: object) -> None:
    factors, returns = _panels()
    target = factors if panel == "factor" else returns
    target.loc[0, "ts_code"] = empty
    with pytest.raises(ValueError):
        _evaluate(factors, returns)


def test_stock_codes_are_stripped_before_matching() -> None:
    factors, returns = _panels()
    factors["ts_code"] = " " + factors["ts_code"] + " "
    result = _evaluate(factors, returns)
    assert result.loc[0, "n_obs"] == 5


def test_inputs_are_not_modified() -> None:
    factors, returns = _panels()
    expected_factors = factors.copy(deep=True)
    expected_returns = returns.copy(deep=True)
    _evaluate(factors, returns)
    pd.testing.assert_frame_equal(factors, expected_factors)
    pd.testing.assert_frame_equal(returns, expected_returns)


@pytest.mark.parametrize(("factor", "expected"), [([1, 2, 3, 4, 5], 1.0), ([5, 4, 3, 2, 1], -1.0)])
def test_perfect_linear_pearson_ic(factor: list[int], expected: float) -> None:
    factors, returns = _panels()
    factors["factor_a"] = factor
    assert _evaluate(factors, returns).loc[0, "ic"] == pytest.approx(expected)


@pytest.mark.parametrize("constant_target", ["factor", "return"])
def test_constant_cross_section_returns_nan_ic(constant_target: str) -> None:
    factors, returns = _panels()
    if constant_target == "factor":
        factors["factor_a"] = 1.0
    else:
        returns["forward_return"] = 1.0
    assert np.isnan(_evaluate(factors, returns).loc[0, "ic"])


def test_insufficient_sample_returns_nan_metrics() -> None:
    result = _evaluate(min_cross_section_size=6)
    assert np.isnan(result.loc[0, "ic"])
    assert np.isnan(result.loc[0, "rank_ic"])


def test_nan_and_infinity_are_excluded_from_pairs() -> None:
    factors, returns = _panels()
    factors.loc[0, "factor_a"] = np.nan
    factors.loc[1, "factor_a"] = np.inf
    returns.loc[2, "forward_return"] = -np.inf
    result = _evaluate(factors, returns, min_cross_section_size=2)
    assert result.loc[0, "n_obs"] == 2
    assert result.loc[0, "coverage"] == pytest.approx(0.4)


def test_ic_is_calculated_independently_by_date() -> None:
    factors, returns = _panels(2)
    returns.loc[returns["trade_date"] == "2024-01-03", "forward_return"] *= -1
    result = _evaluate(factors, returns)
    assert result["ic"].tolist() == pytest.approx([1.0, -1.0])


@pytest.mark.parametrize(("factor", "expected"), [([1, 2, 3, 4, 5], 1.0), ([5, 4, 3, 2, 1], -1.0)])
def test_perfect_rank_ic(factor: list[int], expected: float) -> None:
    factors, returns = _panels()
    factors["factor_a"] = factor
    assert _evaluate(factors, returns).loc[0, "rank_ic"] == pytest.approx(expected)


def test_rank_ic_uses_average_tie_ranks() -> None:
    factors, returns = _panels()
    factors["factor_a"] = [1, 1, 2, 3, 3]
    expected = pd.Series(factors["factor_a"]).rank(method="average").corr(
        pd.Series(returns["forward_return"]).rank(method="average")
    )
    assert _evaluate(factors, returns).loc[0, "rank_ic"] == pytest.approx(expected)


def test_constant_rank_returns_nan() -> None:
    factors, returns = _panels()
    factors["factor_a"] = 2.0
    assert np.isnan(_evaluate(factors, returns).loc[0, "rank_ic"])


def test_rank_ic_is_date_isolated() -> None:
    factors, returns = _panels(2)
    factors.loc[factors["trade_date"] == "2024-01-03", "factor_a"] = [5, 4, 3, 2, 1]
    result = _evaluate(factors, returns)
    assert result["rank_ic"].tolist() == pytest.approx([1.0, -1.0])


def test_metrics_are_bounded_or_nan() -> None:
    factors, returns = _panels(2)
    result = _evaluate(factors, returns, names=["factor_a", "factor_b"])
    for column in ["ic", "rank_ic"]:
        assert result[column].dropna().between(-1.0, 1.0).all()


def test_universe_observations_and_coverage_contract() -> None:
    factors, returns = _panels()
    factors.loc[0, "factor_a"] = np.nan
    returns = returns.iloc[1:].copy()
    returns = pd.concat(
        [returns, pd.DataFrame({"trade_date": ["2024-01-02"], "ts_code": ["EXTRA"], "forward_return": [9.0]})],
        ignore_index=True,
    )
    result = _evaluate(factors, returns)
    assert result.loc[0, "universe_count"] == 5
    assert result.loc[0, "n_obs"] == 4
    assert result.loc[0, "coverage"] == pytest.approx(0.8)
    assert result["coverage"].between(0.0, 1.0).all()


def test_evaluate_output_schema_sort_and_multi_factor_independence() -> None:
    factors, returns = _panels(2)
    first = _evaluate(factors, returns, names=["factor_b", "factor_a"])
    assert first.columns.tolist() == IC_RESULT_COLUMNS
    assert first[["trade_date", "factor_name"]].values.tolist() == sorted(
        first[["trade_date", "factor_name"]].values.tolist()
    )
    changed = factors.copy()
    changed["factor_a"] = changed["factor_a"] ** 3
    second = _evaluate(changed, returns, names=["factor_b", "factor_a"])
    np.testing.assert_allclose(
        first.loc[first["factor_name"] == "factor_b", "ic"],
        second.loc[second["factor_name"] == "factor_b", "ic"],
    )


def test_future_or_other_date_changes_do_not_change_past() -> None:
    factors, returns = _panels(2)
    first = _evaluate(factors, returns)
    factors.loc[factors["trade_date"] == "2024-01-03", "factor_a"] *= -100
    returns.loc[returns["trade_date"] == "2024-01-03", "forward_return"] **= 2
    second = _evaluate(factors, returns)
    pd.testing.assert_frame_equal(first.iloc[:1], second.iloc[:1])


def test_evaluate_output_contains_no_infinity() -> None:
    factors, returns = _panels()
    factors.loc[0, "factor_a"] = np.inf
    result = _evaluate(factors, returns)
    assert not np.isinf(result.select_dtypes(include="number").to_numpy()).any()


def _summary_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
            "factor_name": ["a"] * 3 + ["b"] * 3,
            "universe_count": [10] * 6,
            "n_obs": [8, 10, 6, 5, 5, 5],
            "coverage": [0.8, 1.0, 0.6, 0.5, 0.5, 0.5],
            "ic": [0.1, 0.3, -0.2, 0.0, 0.0, np.nan],
            "rank_ic": [0.2, 0.4, -0.1, 0.1, np.nan, np.nan],
        }
    )


def test_summary_metrics_match_manual_formulas() -> None:
    summary = FactorEvaluator(_config()).summarize_ic(_summary_input())
    row = summary[summary["factor_name"] == "a"].iloc[0]
    ic = pd.Series([0.1, 0.3, -0.2])
    rank_ic = pd.Series([0.2, 0.4, -0.1])
    assert row["total_periods"] == 3
    assert row["valid_ic_periods"] == 3
    assert row["mean_ic"] == pytest.approx(ic.mean())
    assert row["std_ic"] == pytest.approx(ic.std(ddof=1))
    assert row["icir"] == pytest.approx(ic.mean() / ic.std(ddof=1))
    assert row["positive_ic_ratio"] == pytest.approx(2 / 3)
    assert row["valid_rank_ic_periods"] == 3
    assert row["mean_rank_ic"] == pytest.approx(rank_ic.mean())
    assert row["std_rank_ic"] == pytest.approx(rank_ic.std(ddof=1))
    assert row["rank_icir"] == pytest.approx(rank_ic.mean() / rank_ic.std(ddof=1))
    assert row["positive_rank_ic_ratio"] == pytest.approx(2 / 3)
    assert row["mean_coverage"] == pytest.approx(0.8)
    assert row["mean_n_obs"] == pytest.approx(8.0)


def test_zero_ic_is_not_positive_and_zero_std_has_nan_icir() -> None:
    row = FactorEvaluator(_config()).summarize_ic(_summary_input()).query("factor_name == 'b'").iloc[0]
    assert row["valid_ic_periods"] == 2
    assert row["positive_ic_ratio"] == 0.0
    assert np.isnan(row["icir"])


def test_single_valid_period_has_nan_std_and_ratio() -> None:
    data = _summary_input().iloc[:1]
    row = FactorEvaluator(_config()).summarize_ic(data).iloc[0]
    assert np.isnan(row["std_ic"])
    assert np.isnan(row["icir"])


def test_summary_schema_sort_empty_and_validation() -> None:
    evaluator = FactorEvaluator(_config())
    summary = evaluator.summarize_ic(_summary_input().iloc[::-1])
    assert summary.columns.tolist() == IC_SUMMARY_COLUMNS
    assert summary["factor_name"].tolist() == ["a", "b"]
    assert evaluator.summarize_ic(pd.DataFrame(columns=IC_RESULT_COLUMNS)).empty
    with pytest.raises(TypeError):
        evaluator.summarize_ic([])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evaluator.summarize_ic(pd.DataFrame())


def test_compute_ic_disabled_keeps_stable_schema() -> None:
    result = _evaluate(compute_ic=False)
    assert result.columns.tolist() == IC_RESULT_COLUMNS
    assert result["ic"].isna().all()
    summary = FactorEvaluator(_config(compute_ic=False)).summarize_ic(result)
    assert summary.columns.tolist() == IC_SUMMARY_COLUMNS
    assert summary["valid_ic_periods"].eq(0).all()


def test_compute_rank_ic_disabled_keeps_stable_schema() -> None:
    result = _evaluate(compute_rank_ic=False)
    assert result.columns.tolist() == IC_RESULT_COLUMNS
    assert result["rank_ic"].isna().all()
    summary = FactorEvaluator(_config(compute_rank_ic=False)).summarize_ic(result)
    assert summary.columns.tolist() == IC_SUMMARY_COLUMNS
    assert summary["valid_rank_ic_periods"].eq(0).all()


def test_custom_return_column() -> None:
    factors, returns = _panels()
    returns = returns.rename(columns={"forward_return": "future_5d"})
    result = _evaluate(factors, returns, return_col="future_5d")
    assert result.loc[0, "ic"] == pytest.approx(1.0)


def _integration_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    codes = ["A", "B", "C", "D", "E", "F"]
    dates = ["2024-01-02"] * 6 + ["2024-01-03"] * 6
    raw = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": codes * 2,
            "momentum_20d": [1, 2, 4, 3, 6, 5, 11, 13, 12, 16, 14, 15],
            "volatility_20d": [6, 4, 5, 2, 3, 1, 16, 14, 15, 12, 13, 11],
        }
    )
    exposures = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": codes * 2,
            "industry": ["X", "X", "X", "Y", "Y", "Y"] * 2,
            "log_total_mv": list(range(1, 7)) + list(range(11, 17)),
        }
    )
    returns = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": codes * 2,
            "forward_return": [0.01, 0.03, 0.02, -0.01, 0.05, 0.04] * 2,
        }
    )
    return raw, exposures, returns


def test_d1_d2_e1_integration_and_no_second_direction_flip() -> None:
    raw, exposures, returns = _integration_inputs()
    registry = FactorRegistry()
    register_example_factors(registry)
    preprocessed = FactorPreprocessor(
        registry,
        PreprocessingConfig(
            missing_method="none",
            winsor_method="none",
            standardize_method="zscore",
            min_cross_section_size=3,
        ),
    ).transform(raw, ["momentum_20d", "volatility_20d"])
    assert preprocessed.loc[0, "volatility_20d"] < preprocessed.loc[5, "volatility_20d"]
    neutralized = FactorNeutralizer(
        NeutralizationConfig(
            neutralize_industry=True,
            neutralize_size=False,
            min_cross_section_size=3,
            min_industry_size=2,
            standardize_residuals=True,
        )
    ).transform(preprocessed, exposures, ["momentum_20d", "volatility_20d"])
    result = FactorEvaluator(_config(min_cross_section_size=3)).evaluate_ic(
        neutralized, returns, ["momentum_20d", "volatility_20d"]
    )
    assert len(result) == 4
    assert result[["ic", "rank_ic"]].notna().all().all()
    assert not np.isinf(result[["ic", "rank_ic"]].to_numpy()).any()


def test_integrated_future_changes_do_not_affect_past_or_other_factor() -> None:
    raw, exposures, returns = _integration_inputs()
    evaluator = FactorEvaluator(_config(min_cross_section_size=3))
    first = evaluator.evaluate_ic(raw, returns, ["momentum_20d", "volatility_20d"])
    changed = raw.copy()
    changed.loc[changed["trade_date"] == "2024-01-03", "momentum_20d"] *= -100
    second = evaluator.evaluate_ic(changed, returns, ["momentum_20d", "volatility_20d"])
    pd.testing.assert_frame_equal(first.iloc[:2], second.iloc[:2])
    np.testing.assert_allclose(
        first.loc[first["factor_name"] == "volatility_20d", "ic"],
        second.loc[second["factor_name"] == "volatility_20d", "ic"],
    )
