"""Tests for strictly historical rolling IC and RankIC factor weighting."""

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.factors.dynamic_composition import (
    WEIGHT_HISTORY_COLUMNS,
    RollingICFactorComposer,
    RollingICWeightConfig,
)
from src.factors.evaluation import FactorEvaluationConfig, FactorEvaluator
from src.factors.examples import register_example_factors
from src.factors.quantile_evaluation import (
    FactorQuantileEvaluator,
    QuantileEvaluationConfig,
)
from src.factors.registry import FactorRegistry


NAMES = ["momentum_20d", "volatility_20d"]


def _registry() -> FactorRegistry:
    registry = FactorRegistry()
    register_example_factors(registry)
    return registry


def _config(**overrides: object) -> RollingICWeightConfig:
    defaults = {"lookback_periods": 3, "min_periods": 2}
    defaults.update(overrides)
    return RollingICWeightConfig(**defaults)


def _composer(**config: object) -> RollingICFactorComposer:
    return RollingICFactorComposer(_registry(), _config(**config))


def _ic_results() -> pd.DataFrame:
    rows = []
    for index, date in enumerate(pd.date_range("2024-01-01", periods=6)):
        rows.extend(
            [
                {
                    "trade_date": date,
                    "factor_name": "momentum_20d",
                    "ic": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][index],
                    "rank_ic": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2][index],
                },
                {
                    "trade_date": date,
                    "factor_name": "volatility_20d",
                    "ic": [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4][index],
                    "rank_ic": [-0.4, -0.2, 0.2, 0.4, 0.6, 0.8][index],
                },
            ]
        )
    return pd.DataFrame(rows)


def _factor_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [
                "2024-01-04",
                "2024-01-04",
                "2024-01-05",
                "2024-01-05",
            ],
            "ts_code": [" B ", "A", "B", "A"],
            "momentum_20d": [2.0, 4.0, 6.0, 8.0],
            "volatility_20d": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_default_config_is_serializable_frozen_and_describable() -> None:
    default = RollingICWeightConfig()
    assert default.to_dict() == {
        "metric": "rank_ic",
        "lookback_periods": 12,
        "min_periods": 6,
        "negative_policy": "zero",
        "fallback_method": "equal",
        "missing_policy": "renormalize",
        "min_valid_factors": 1,
        "score_col": "composite_score",
    }
    json.dumps(default.to_dict())
    assert RollingICFactorComposer(_registry()).describe_config() == default.to_dict()
    with pytest.raises(FrozenInstanceError):
        default.metric = "ic"  # type: ignore[misc]


@pytest.mark.parametrize("value", ["bad", "", None])
def test_invalid_metric_raises(value: object) -> None:
    with pytest.raises(ValueError):
        RollingICWeightConfig(metric=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["lookback_periods", "min_periods", "min_valid_factors"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_positive_integer_config_fields_are_validated(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        RollingICWeightConfig(**{field: value})


def test_minimum_periods_cannot_exceed_lookback() -> None:
    with pytest.raises(ValueError):
        RollingICWeightConfig(lookback_periods=2, min_periods=3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("negative_policy", "signed"),
        ("fallback_method", "fixed"),
        ("missing_policy", "drop"),
    ],
)
def test_invalid_method_config_raises(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        RollingICWeightConfig(**{field: value})


@pytest.mark.parametrize("value", ["", " ", None])
def test_empty_score_column_raises(value: object) -> None:
    with pytest.raises(ValueError):
        RollingICWeightConfig(score_col=value)  # type: ignore[arg-type]


def test_constructor_types_are_validated() -> None:
    with pytest.raises(TypeError):
        RollingICFactorComposer(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RollingICFactorComposer(_registry(), object())  # type: ignore[arg-type]


@pytest.mark.parametrize("names", [[], (), "momentum_20d"])
def test_factor_names_must_be_non_empty_sequence(names: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _composer().build_weight_history(
            ["2024-01-04"], _ic_results(), names  # type: ignore[arg-type]
        )


def test_duplicate_empty_and_unregistered_factor_names_raise() -> None:
    for names in (
        ["momentum_20d", "momentum_20d"],
        ["momentum_20d", " "],
        ["momentum_20d", "unknown"],
    ):
        with pytest.raises(ValueError):
            _composer().build_weight_history(["2024-01-04"], _ic_results(), names)


def test_panel_and_ic_results_must_be_dataframes() -> None:
    with pytest.raises(TypeError):
        _composer().compose([], _ic_results(), NAMES)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _composer().compose(_factor_panel(), [], NAMES)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _composer().build_weight_history(["2024-01-04"], [], NAMES)  # type: ignore[arg-type]


@pytest.mark.parametrize("column", ["trade_date", "ts_code", "momentum_20d"])
def test_missing_factor_panel_columns_raise(column: str) -> None:
    with pytest.raises(ValueError):
        _composer().compose(_factor_panel().drop(columns=column), _ic_results(), NAMES)


@pytest.mark.parametrize("column", ["trade_date", "factor_name", "rank_ic"])
def test_missing_ic_result_columns_raise(column: str) -> None:
    with pytest.raises(ValueError):
        _composer().build_weight_history(
            ["2024-01-04"], _ic_results().drop(columns=column), NAMES
        )


def test_selected_ic_metric_contract_is_respected() -> None:
    ic_only = _ic_results().drop(columns="rank_ic")
    result = _composer(metric="ic").build_weight_history(
        ["2024-01-04"], ic_only, NAMES
    )
    assert result["metric"].eq("ic").all()


def test_duplicate_keys_and_empty_stock_codes_raise() -> None:
    panel = _factor_panel()
    duplicate_panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        _composer().compose(duplicate_panel, _ic_results(), NAMES)
    duplicate_ic = pd.concat(
        [_ic_results(), _ic_results().iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError):
        _composer().build_weight_history(["2024-01-04"], duplicate_ic, NAMES)
    empty_code = panel.copy()
    empty_code.loc[0, "ts_code"] = " "
    with pytest.raises(ValueError):
        _composer().compose(empty_code, _ic_results(), NAMES)


def test_invalid_dates_and_empty_ic_factor_names_raise() -> None:
    panel = _factor_panel()
    panel.loc[0, "trade_date"] = "bad"
    with pytest.raises(ValueError):
        _composer().compose(panel, _ic_results(), NAMES)
    evaluations = _ic_results()
    evaluations["trade_date"] = evaluations["trade_date"].astype(object)
    evaluations.loc[0, "trade_date"] = "bad"
    with pytest.raises(ValueError):
        _composer().build_weight_history(["2024-01-04"], evaluations, NAMES)
    evaluations = _ic_results()
    evaluations.loc[0, "factor_name"] = " "
    with pytest.raises(ValueError):
        _composer().build_weight_history(["2024-01-04"], evaluations, NAMES)
    with pytest.raises(ValueError):
        _composer().build_weight_history(["bad"], _ic_results(), NAMES)


def test_inputs_are_not_mutated() -> None:
    panel = _factor_panel()
    evaluations = _ic_results()
    panel_before = panel.copy(deep=True)
    evaluations_before = evaluations.copy(deep=True)
    _composer().compose(panel, evaluations, NAMES)
    pdt.assert_frame_equal(panel, panel_before)
    pdt.assert_frame_equal(evaluations, evaluations_before)


def test_empty_score_dates_and_empty_panel_have_stable_schemas() -> None:
    weights = _composer().build_weight_history([], _ic_results(), NAMES)
    assert weights.empty
    assert list(weights.columns) == WEIGHT_HISTORY_COLUMNS
    result = _composer().compose(_factor_panel().iloc[:0], _ic_results(), NAMES)
    assert result.empty
    assert list(result.columns) == [
        "trade_date",
        "ts_code",
        "composite_score",
        "valid_factor_count",
        "weight_coverage",
    ]


def test_strict_window_dates_counts_and_manual_mean() -> None:
    result = _composer(metric="ic").build_weight_history(
        ["2024-01-05"], _ic_results(), NAMES
    )
    momentum = result.loc[result["factor_name"] == "momentum_20d"].iloc[0]
    assert momentum["history_periods"] == 3
    assert momentum["history_start_date"] == pd.Timestamp("2024-01-02")
    assert momentum["history_end_date"] == pd.Timestamp("2024-01-04")
    assert momentum["historical_mean"] == pytest.approx((0.2 + 0.3 + 0.4) / 3)
    assert momentum["history_end_date"] < momentum["trade_date"]


def test_nonfinite_metrics_are_excluded_before_lookback() -> None:
    evaluations = _ic_results()
    mask = (
        (evaluations["factor_name"] == "momentum_20d")
        & evaluations["trade_date"].isin(
            [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        )
    )
    evaluations.loc[mask, "rank_ic"] = [np.nan, np.inf]
    result = _composer().build_weight_history(
        ["2024-01-05"], evaluations, NAMES
    )
    momentum = result.loc[result["factor_name"] == "momentum_20d"].iloc[0]
    assert momentum["history_periods"] == 2
    assert momentum["history_start_date"] == pd.Timestamp("2024-01-01")
    assert momentum["history_end_date"] == pd.Timestamp("2024-01-04")
    assert momentum["historical_mean"] == pytest.approx((0.2 + 0.8) / 2)


def test_no_history_uses_nat_and_equal_fallback() -> None:
    result = _composer().build_weight_history(
        ["2024-01-01"], _ic_results(), NAMES
    )
    assert result["history_periods"].eq(0).all()
    assert result["history_start_date"].isna().all()
    assert result["history_end_date"].isna().all()
    assert result["historical_mean"].isna().all()
    assert result["raw_weight"].isna().all()
    assert result["normalized_weight"].eq(0.5).all()
    assert result["used_fallback"].all()


def test_input_order_does_not_change_weight_history() -> None:
    baseline = _composer().build_weight_history(
        ["2024-01-06", "2024-01-04"], _ic_results(), NAMES
    )
    shuffled = _composer().build_weight_history(
        ["2024-01-04", "2024-01-06"],
        _ic_results().sample(frac=1.0, random_state=7),
        list(reversed(NAMES)),
    )
    pdt.assert_frame_equal(baseline, shuffled)
    assert list(baseline.columns) == WEIGHT_HISTORY_COLUMNS
    assert baseline["history_periods"].dtype.kind in "iu"
    assert baseline["used_fallback"].dtype == bool


def test_current_and_future_ic_do_not_change_current_weights() -> None:
    score_date = pd.Timestamp("2024-01-04")
    baseline = _composer().build_weight_history([score_date], _ic_results(), NAMES)
    changed = _ic_results()
    changed.loc[changed["trade_date"] >= score_date, "rank_ic"] = 999.0
    result = _composer().build_weight_history([score_date], changed, NAMES)
    pdt.assert_frame_equal(baseline, result)


def test_window_history_change_changes_current_weights() -> None:
    score_date = pd.Timestamp("2024-01-05")
    baseline = _composer().build_weight_history([score_date], _ic_results(), NAMES)
    changed = _ic_results()
    changed.loc[
        (changed["trade_date"] == pd.Timestamp("2024-01-04"))
        & (changed["factor_name"] == "volatility_20d"),
        "rank_ic",
    ] = 5.0
    result = _composer().build_weight_history([score_date], changed, NAMES)
    assert not np.allclose(
        baseline["normalized_weight"], result["normalized_weight"]
    )


def test_appending_future_records_does_not_change_past_history() -> None:
    score_dates = ["2024-01-04", "2024-01-05"]
    baseline = _composer().build_weight_history(score_dates, _ic_results(), NAMES)
    future = pd.DataFrame(
        [
            {
                "trade_date": "2030-01-01",
                "factor_name": name,
                "ic": -999.0,
                "rank_ic": 999.0,
            }
            for name in NAMES
        ]
    )
    result = _composer().build_weight_history(
        score_dates, pd.concat([_ic_results(), future], ignore_index=True), NAMES
    )
    pdt.assert_frame_equal(baseline, result)


def test_negative_zero_policy_and_normalization() -> None:
    result = _composer(negative_policy="zero").build_weight_history(
        ["2024-01-03"], _ic_results(), NAMES
    )
    momentum = result.loc[result["factor_name"] == "momentum_20d"].iloc[0]
    volatility = result.loc[result["factor_name"] == "volatility_20d"].iloc[0]
    assert momentum["raw_weight"] == pytest.approx(0.3)
    assert volatility["raw_weight"] == 0.0
    assert momentum["normalized_weight"] == 1.0
    assert volatility["normalized_weight"] == 0.0
    assert not result["used_fallback"].any()
    assert result["normalized_weight"].sum() == pytest.approx(1.0)


def test_absolute_policy_turns_negative_mean_into_positive_strength() -> None:
    result = _composer(negative_policy="absolute").build_weight_history(
        ["2024-01-03"], _ic_results(), NAMES
    )
    assert (result["raw_weight"] > 0.0).all()
    assert (result["normalized_weight"] >= 0.0).all()
    assert result["normalized_weight"].sum() == pytest.approx(1.0)
    expected = {"momentum_20d": 0.5, "volatility_20d": 0.5}
    assert result.set_index("factor_name")["normalized_weight"].to_dict() == expected


def test_insufficient_history_equal_and_none_fallbacks() -> None:
    equal = _composer(min_periods=3).build_weight_history(
        ["2024-01-03"], _ic_results(), NAMES
    )
    assert equal["history_periods"].eq(2).all()
    assert equal["raw_weight"].isna().all()
    assert equal["normalized_weight"].eq(0.5).all()
    assert equal["used_fallback"].all()
    none = _composer(min_periods=3, fallback_method="none").build_weight_history(
        ["2024-01-03"], _ic_results(), NAMES
    )
    assert none["normalized_weight"].isna().all()
    assert not none["used_fallback"].any()


def test_partial_history_does_not_trigger_equal_fallback() -> None:
    evaluations = _ic_results()
    evaluations.loc[
        (evaluations["factor_name"] == "volatility_20d")
        & (evaluations["trade_date"] < pd.Timestamp("2024-01-05")),
        "rank_ic",
    ] = np.nan
    result = _composer().build_weight_history(
        ["2024-01-05"], evaluations, NAMES
    )
    weights = result.set_index("factor_name")["normalized_weight"]
    assert weights["momentum_20d"] == 1.0
    assert weights["volatility_20d"] == 0.0
    assert not result["used_fallback"].any()


def test_fallback_none_makes_scores_and_coverage_unavailable() -> None:
    panel = _factor_panel().loc[lambda frame: frame["trade_date"] == "2024-01-04"]
    result = _composer(
        lookback_periods=6, min_periods=6, fallback_method="none"
    ).compose(panel, _ic_results(), NAMES)
    assert result["composite_score"].isna().all()
    assert result["weight_coverage"].isna().all()
    assert result["valid_factor_count"].eq(0).all()


def test_dynamic_score_matches_manual_weights() -> None:
    panel = _factor_panel().loc[lambda frame: frame["trade_date"] == "2024-01-04"]
    result = _composer(negative_policy="absolute").compose(panel, _ic_results(), NAMES)
    weights = _composer(negative_policy="absolute").build_weight_history(
        ["2024-01-04"], _ic_results(), NAMES
    ).set_index("factor_name")["normalized_weight"]
    normalized_panel = panel.assign(ts_code=panel["ts_code"].str.strip()).sort_values(
        ["trade_date", "ts_code"]
    )
    expected = (
        normalized_panel["momentum_20d"] * weights["momentum_20d"]
        + normalized_panel["volatility_20d"] * weights["volatility_20d"]
    )
    np.testing.assert_allclose(result["composite_score"], expected)
    assert result["valid_factor_count"].eq(2).all()
    assert result["weight_coverage"].eq(1.0).all()


def test_require_all_missing_positive_weight_is_nan() -> None:
    panel = _factor_panel().loc[lambda frame: frame["trade_date"] == "2024-01-04"].copy()
    panel.loc[panel["ts_code"].str.strip() == "A", "momentum_20d"] = np.nan
    result = _composer(missing_policy="require_all", negative_policy="absolute").compose(
        panel, _ic_results(), NAMES
    )
    stock_a = result.loc[result["ts_code"] == "A"].iloc[0]
    assert np.isnan(stock_a["composite_score"])
    assert stock_a["valid_factor_count"] == 1
    assert 0.0 < stock_a["weight_coverage"] < 1.0
    assert result.loc[result["ts_code"] == "B", "composite_score"].notna().all()


def test_zero_weight_missing_factor_does_not_block_require_all() -> None:
    panel = _factor_panel().loc[lambda frame: frame["trade_date"] == "2024-01-03"].copy()
    if panel.empty:
        panel = pd.DataFrame(
            {
                "trade_date": ["2024-01-03"],
                "ts_code": ["A"],
                "momentum_20d": [4.0],
                "volatility_20d": [np.nan],
            }
        )
    result = _composer(missing_policy="require_all").compose(
        panel, _ic_results(), NAMES
    )
    assert result.loc[0, "composite_score"] == 4.0
    assert result.loc[0, "valid_factor_count"] == 1
    assert result.loc[0, "weight_coverage"] == 1.0


def test_renormalize_uses_valid_dynamic_weights_and_minimum() -> None:
    panel = _factor_panel().loc[lambda frame: frame["trade_date"] == "2024-01-04"].copy()
    panel.loc[panel["ts_code"].str.strip() == "A", "volatility_20d"] = np.nan
    result = _composer(missing_policy="renormalize", negative_policy="absolute").compose(
        panel, _ic_results(), NAMES
    )
    stock_a = result.loc[result["ts_code"] == "A"].iloc[0]
    assert stock_a["composite_score"] == 4.0
    assert stock_a["valid_factor_count"] == 1
    assert 0.0 < stock_a["weight_coverage"] < 1.0
    minimum = _composer(
        missing_policy="renormalize", min_valid_factors=2, negative_policy="absolute"
    ).compose(panel, _ic_results(), NAMES)
    assert np.isnan(
        minimum.loc[minimum["ts_code"] == "A", "composite_score"].iloc[0]
    )


def test_invalid_string_factor_is_missing_not_zero() -> None:
    panel = _factor_panel().loc[lambda frame: frame["trade_date"] == "2024-01-04"].copy()
    panel["volatility_20d"] = panel["volatility_20d"].astype(object)
    panel.loc[panel["ts_code"].str.strip() == "A", "volatility_20d"] = "bad"
    result = _composer(negative_policy="absolute").compose(panel, _ic_results(), NAMES)
    stock_a = result.loc[result["ts_code"] == "A"].iloc[0]
    assert stock_a["composite_score"] == 4.0
    assert stock_a["weight_coverage"] < 1.0


def test_output_schema_sorting_custom_name_and_finite_contract() -> None:
    result = _composer(score_col="dynamic_score").compose(
        _factor_panel(), _ic_results(), NAMES
    )
    assert list(result.columns) == [
        "trade_date",
        "ts_code",
        "dynamic_score",
        "valid_factor_count",
        "weight_coverage",
    ]
    assert len(result) == len(_factor_panel())
    assert list(result["ts_code"]) == ["A", "B", "A", "B"]
    assert result["valid_factor_count"].dtype.kind in "iu"
    assert result["weight_coverage"].dropna().between(0.0, 1.0).all()
    assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()


def test_stock_and_date_missingness_are_isolated() -> None:
    baseline = _composer().compose(_factor_panel(), _ic_results(), NAMES)
    changed = _factor_panel()
    changed.loc[0, "momentum_20d"] = np.nan
    result = _composer().compose(changed, _ic_results(), NAMES)
    unaffected = baseline["ts_code"].eq("A") | baseline["trade_date"].eq(
        pd.Timestamp("2024-01-05")
    )
    pdt.assert_series_equal(
        result.loc[unaffected, "composite_score"].reset_index(drop=True),
        baseline.loc[unaffected, "composite_score"].reset_index(drop=True),
    )


def test_future_factor_and_ic_changes_do_not_change_past_scores() -> None:
    baseline = _composer().compose(_factor_panel(), _ic_results(), NAMES)
    future_panel = _factor_panel()
    future_panel.loc[
        future_panel["trade_date"] == "2024-01-05", "momentum_20d"
    ] = 999.0
    panel_result = _composer().compose(future_panel, _ic_results(), NAMES)
    past = baseline["trade_date"] == pd.Timestamp("2024-01-04")
    pdt.assert_series_equal(
        panel_result.loc[past, "composite_score"].reset_index(drop=True),
        baseline.loc[past, "composite_score"].reset_index(drop=True),
    )
    future_ic = _ic_results()
    future_ic.loc[future_ic["trade_date"] >= pd.Timestamp("2024-01-04"), "rank_ic"] = -999.0
    ic_result = _composer().compose(_factor_panel(), future_ic, NAMES)
    pdt.assert_series_equal(
        ic_result.loc[past, "composite_score"].reset_index(drop=True),
        baseline.loc[past, "composite_score"].reset_index(drop=True),
    )


def test_absolute_does_not_flip_negative_direction_factor_values() -> None:
    panel = pd.DataFrame(
        {
            "trade_date": ["2024-01-03"],
            "ts_code": ["A"],
            "momentum_20d": [10.0],
            # Already direction-adjusted by D1; F2 must use this value as given.
            "volatility_20d": [-20.0],
        }
    )
    result = _composer(negative_policy="absolute").compose(
        panel, _ic_results(), NAMES
    )
    assert result.loc[0, "composite_score"] == pytest.approx(-5.0)


def test_e1_f2_e1_and_quantile_integration() -> None:
    factor_rows = []
    return_rows = []
    dates = pd.date_range("2024-02-01", periods=5)
    codes = [f"S{i:02d}" for i in range(10)]
    for date_index, date in enumerate(dates):
        for index, code in enumerate(codes):
            base = float(index + 1)
            factor_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "momentum_20d": base + 0.1 * date_index,
                    "volatility_20d": -base + np.sin(index),
                }
            )
            return_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "forward_return": base * 0.01 + np.cos(index) * 0.001,
                }
            )
    factors = pd.DataFrame(factor_rows)
    returns = pd.DataFrame(return_rows)
    evaluator = FactorEvaluator(FactorEvaluationConfig(min_cross_section_size=5))
    evaluations = evaluator.evaluate_ic(factors, returns, NAMES)
    composer = RollingICFactorComposer(
        _registry(),
        RollingICWeightConfig(
            metric="rank_ic",
            lookback_periods=3,
            min_periods=1,
            fallback_method="equal",
        ),
    )
    weights = composer.build_weight_history(dates, evaluations, NAMES)
    scores = composer.compose(factors, evaluations, NAMES)
    score_ic = evaluator.evaluate_ic(scores, returns, ["composite_score"])
    quantiles = FactorQuantileEvaluator(
        QuantileEvaluationConfig(
            quantiles=5, min_cross_section_size=5, min_group_size=1
        )
    ).evaluate_quantiles(scores, returns, ["composite_score"])
    first_date_weights = weights.loc[weights["trade_date"] == dates[0]]
    assert first_date_weights["used_fallback"].all()
    for date in dates[1:]:
        assert (
            weights.loc[weights["trade_date"] == date, "history_end_date"] < date
        ).all()
    assert len(scores) == len(factors)
    assert set(score_ic["factor_name"]) == {"composite_score"}
    assert set(quantiles["factor_name"]) == {"composite_score"}
    assert not np.isinf(
        scores.select_dtypes(include=[np.number])
    ).any().any()
    assert "forward_return" not in scores.columns
    assert not any("top" in column.lower() for column in scores.columns)
