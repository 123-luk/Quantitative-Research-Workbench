"""Tests for equal- and fixed-weight composition of processed factors."""

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.factors.base import FactorMetadata, FunctionFactor
from src.factors.composition import FactorComposer, FactorCompositionConfig
from src.factors.evaluation import FactorEvaluationConfig, FactorEvaluator
from src.factors.examples import register_example_factors
from src.factors.neutralization import FactorNeutralizer, NeutralizationConfig
from src.factors.preprocessing import FactorPreprocessor, PreprocessingConfig
from src.factors.quantile_evaluation import (
    FactorQuantileEvaluator,
    QuantileEvaluationConfig,
)
from src.factors.registry import FactorRegistry


NAMES = ["momentum_20d", "volatility_20d"]


def _registry() -> FactorRegistry:
    registry = FactorRegistry()
    register_example_factors(registry)
    registry.register(
        FunctionFactor(
            FactorMetadata(
                name="quality",
                category="quality",
                direction=1,
                source_fields=("quality",),
            ),
            lambda frame: frame["quality"],
        )
    )
    return registry


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-03", "2024-01-02", "2024-01-02"],
            "ts_code": [" B ", "B", "A"],
            "momentum_20d": [3.0, 1.0, 2.0],
            "volatility_20d": [5.0, 3.0, 4.0],
            "quality": [7.0, 5.0, 6.0],
        }
    )


def _composer(**config: object) -> FactorComposer:
    return FactorComposer(_registry(), FactorCompositionConfig(**config))


def test_default_config_description_is_serializable_and_frozen() -> None:
    config = FactorCompositionConfig()
    assert config.to_dict() == {
        "method": "equal",
        "fixed_weights": (),
        "normalize_weights": True,
        "missing_policy": "renormalize",
        "min_valid_factors": 1,
        "score_col": "composite_score",
    }
    json.dumps(config.to_dict())
    assert _composer().describe_config() == config.to_dict()
    with pytest.raises(FrozenInstanceError):
        config.method = "fixed"  # type: ignore[misc]


@pytest.mark.parametrize("method", ["bad", "", None])
def test_invalid_method_raises(method: object) -> None:
    with pytest.raises(ValueError):
        FactorCompositionConfig(method=method)  # type: ignore[arg-type]


@pytest.mark.parametrize("policy", ["drop", "", None])
def test_invalid_missing_policy_raises(policy: object) -> None:
    with pytest.raises(ValueError):
        FactorCompositionConfig(missing_policy=policy)  # type: ignore[arg-type]


def test_normalize_weights_must_be_bool() -> None:
    with pytest.raises(ValueError):
        FactorCompositionConfig(normalize_weights=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "1"])
def test_min_valid_factors_must_be_positive_non_bool_integer(value: object) -> None:
    with pytest.raises(ValueError):
        FactorCompositionConfig(min_valid_factors=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["", " ", None])
def test_score_column_cannot_be_empty(value: object) -> None:
    with pytest.raises(ValueError):
        FactorCompositionConfig(score_col=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "weights",
    [
        (("", 1.0),),
        (("a", 1.0), ("a", 2.0)),
        (("a", np.nan),),
        (("a", np.inf),),
        (("a", -1.0),),
        (("a", 0.0), ("b", 0.0)),
        (("a", True),),
        (("a",),),
    ],
)
def test_invalid_fixed_weight_entries_raise(weights: object) -> None:
    with pytest.raises(ValueError):
        FactorCompositionConfig(fixed_weights=weights)  # type: ignore[arg-type]


def test_fixed_method_requires_weights_but_equal_ignores_valid_weights() -> None:
    with pytest.raises(ValueError):
        FactorCompositionConfig(method="fixed")
    config = FactorCompositionConfig(
        method="equal", fixed_weights=(("unused", 2.0),)
    )
    weights = FactorComposer(_registry(), config).resolve_weights(NAMES)
    assert weights == {"momentum_20d": 0.5, "volatility_20d": 0.5}


def test_constructor_types_are_validated() -> None:
    with pytest.raises(TypeError):
        FactorComposer(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        FactorComposer(_registry(), object())  # type: ignore[arg-type]


@pytest.mark.parametrize("names", [[], (), "momentum_20d"])
def test_factor_names_must_be_non_empty_sequence(names: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _composer().resolve_weights(names)  # type: ignore[arg-type]


def test_duplicate_empty_and_unregistered_names_raise() -> None:
    for names in (
        ["momentum_20d", "momentum_20d"],
        ["momentum_20d", " "],
        ["momentum_20d", "unknown"],
    ):
        with pytest.raises(ValueError):
            _composer().resolve_weights(names)


def test_equal_weights_are_stable_normalized_and_order_independent() -> None:
    first = _composer().resolve_weights(["quality", *NAMES])
    second = _composer().resolve_weights([*reversed(NAMES), "quality"])
    assert first == second
    assert list(first) == sorted(first)
    assert sum(first.values()) == pytest.approx(1.0)
    assert all(weight == pytest.approx(1.0 / 3.0) for weight in first.values())


def test_fixed_weights_normalize_and_match_by_name() -> None:
    composer = _composer(
        method="fixed",
        fixed_weights=(("volatility_20d", 3.0), ("momentum_20d", 1.0)),
    )
    assert composer.resolve_weights(NAMES) == {
        "momentum_20d": 0.25,
        "volatility_20d": 0.75,
    }
    assert composer.resolve_weights(list(reversed(NAMES))) == composer.resolve_weights(
        NAMES
    )


def test_fixed_raw_weights_are_preserved() -> None:
    weights = _composer(
        method="fixed",
        fixed_weights=(("momentum_20d", 2.0), ("volatility_20d", 3.0)),
        normalize_weights=False,
    ).resolve_weights(NAMES)
    assert weights == {"momentum_20d": 2.0, "volatility_20d": 3.0}


@pytest.mark.parametrize(
    "weights",
    [
        (("momentum_20d", 1.0),),
        (("momentum_20d", 1.0), ("volatility_20d", 1.0), ("quality", 1.0)),
    ],
)
def test_fixed_factor_set_must_match_selected_names(weights: object) -> None:
    composer = _composer(method="fixed", fixed_weights=weights)
    with pytest.raises(ValueError):
        composer.resolve_weights(NAMES)


def test_input_must_be_dataframe_and_contain_required_columns() -> None:
    with pytest.raises(TypeError):
        _composer().compose([], NAMES)  # type: ignore[arg-type]
    for column in ["trade_date", "ts_code", "momentum_20d"]:
        with pytest.raises(ValueError):
            _composer().compose(_panel().drop(columns=column), NAMES)


def test_invalid_date_empty_code_and_duplicate_keys_raise() -> None:
    invalid_date = _panel()
    invalid_date.loc[0, "trade_date"] = "not-a-date"
    with pytest.raises(ValueError):
        _composer().compose(invalid_date, NAMES)
    empty_code = _panel()
    empty_code.loc[0, "ts_code"] = " "
    with pytest.raises(ValueError):
        _composer().compose(empty_code, NAMES)
    duplicate = _panel()
    duplicate.loc[0, ["trade_date", "ts_code"]] = ["2024-01-02", " A "]
    with pytest.raises(ValueError):
        _composer().compose(duplicate, NAMES)


def test_input_is_not_mutated_and_output_is_sorted_with_fixed_schema() -> None:
    panel = _panel()
    before = panel.copy(deep=True)
    result = _composer().compose(panel, NAMES)
    pdt.assert_frame_equal(panel, before)
    assert list(result.columns) == [
        "trade_date",
        "ts_code",
        "composite_score",
        "valid_factor_count",
        "weight_coverage",
    ]
    assert len(result) == len(panel)
    assert list(result["ts_code"]) == ["A", "B", "B"]
    assert result["valid_factor_count"].dtype.kind in "iu"


def test_empty_input_returns_typed_fixed_schema() -> None:
    result = _composer().compose(_panel().iloc[:0], NAMES)
    assert result.empty
    assert list(result.columns) == [
        "trade_date",
        "ts_code",
        "composite_score",
        "valid_factor_count",
        "weight_coverage",
    ]
    assert result["valid_factor_count"].dtype.kind in "iu"


def test_equal_two_and_three_factor_scores_are_arithmetic_means() -> None:
    panel = _panel().sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    two = _composer().compose(panel, NAMES)
    three = _composer().compose(panel, [*NAMES, "quality"])
    np.testing.assert_allclose(
        two["composite_score"],
        panel[NAMES].mean(axis=1),
    )
    np.testing.assert_allclose(
        three["composite_score"],
        panel[[*NAMES, "quality"]].mean(axis=1),
    )


def test_identical_factor_values_keep_that_value() -> None:
    panel = _panel()
    panel["volatility_20d"] = panel["momentum_20d"]
    result = _composer().compose(panel, NAMES)
    expected = (
        panel.assign(ts_code=panel["ts_code"].str.strip())
        .sort_values(["trade_date", "ts_code"])["momentum_20d"]
        .to_numpy()
    )
    np.testing.assert_allclose(result["composite_score"], expected)


def test_fixed_normalized_and_raw_scores_follow_manual_formula() -> None:
    panel = _panel().sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    normalized = _composer(
        method="fixed",
        fixed_weights=(("momentum_20d", 1.0), ("volatility_20d", 3.0)),
    ).compose(panel, NAMES)
    raw = _composer(
        method="fixed",
        fixed_weights=(("momentum_20d", 1.0), ("volatility_20d", 3.0)),
        normalize_weights=False,
        missing_policy="require_all",
    ).compose(panel, NAMES)
    expected_raw = panel["momentum_20d"] + 3.0 * panel["volatility_20d"]
    np.testing.assert_allclose(normalized["composite_score"], expected_raw / 4.0)
    np.testing.assert_allclose(raw["composite_score"], expected_raw)


def test_factor_order_does_not_change_scores() -> None:
    first = _composer().compose(_panel(), NAMES)
    second = _composer().compose(_panel(), list(reversed(NAMES)))
    pdt.assert_frame_equal(first, second)


def test_zero_weight_factor_does_not_affect_score_but_counts_when_valid() -> None:
    result = _composer(
        method="fixed",
        fixed_weights=(("momentum_20d", 1.0), ("volatility_20d", 0.0)),
    ).compose(_panel(), NAMES)
    expected = (
        _panel()
        .assign(ts_code=_panel()["ts_code"].str.strip())
        .sort_values(["trade_date", "ts_code"])["momentum_20d"]
        .to_numpy()
    )
    np.testing.assert_allclose(result["composite_score"], expected)
    assert result["valid_factor_count"].eq(2).all()


def test_require_all_records_missing_audit_fields_and_rejects_inf() -> None:
    panel = _panel().sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    panel.loc[0, "momentum_20d"] = np.inf
    result = _composer(
        missing_policy="require_all", min_valid_factors=1
    ).compose(panel, NAMES)
    assert np.isnan(result.loc[0, "composite_score"])
    assert result.loc[0, "valid_factor_count"] == 1
    assert result.loc[0, "weight_coverage"] == pytest.approx(0.5)
    assert result.loc[1:, "composite_score"].notna().all()


def test_renormalize_uses_only_valid_values_and_base_weight_coverage() -> None:
    panel = _panel().sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    panel["volatility_20d"] = panel["volatility_20d"].astype(object)
    panel.loc[0, "volatility_20d"] = "bad"
    result = _composer(
        method="fixed",
        fixed_weights=(("momentum_20d", 1.0), ("volatility_20d", 3.0)),
    ).compose(panel, NAMES)
    assert result.loc[0, "composite_score"] == panel.loc[0, "momentum_20d"]
    assert result.loc[0, "valid_factor_count"] == 1
    assert result.loc[0, "weight_coverage"] == pytest.approx(0.25)


def test_renormalize_honors_minimum_and_positive_valid_weight() -> None:
    panel = _panel().sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    panel.loc[0, "volatility_20d"] = np.nan
    minimum = _composer(min_valid_factors=2).compose(panel, NAMES)
    assert np.isnan(minimum.loc[0, "composite_score"])
    zero_valid_weight = _composer(
        method="fixed",
        fixed_weights=(("momentum_20d", 0.0), ("volatility_20d", 1.0)),
    ).compose(panel, NAMES)
    assert np.isnan(zero_valid_weight.loc[0, "composite_score"])
    assert zero_valid_weight.loc[0, "weight_coverage"] == 0.0


def test_missingness_isolated_by_row_and_date() -> None:
    panel = _panel().sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    baseline = _composer().compose(panel, NAMES)
    changed = panel.copy()
    changed.loc[0, "volatility_20d"] = np.nan
    result = _composer().compose(changed, NAMES)
    assert result.loc[0, "composite_score"] == changed.loc[0, "momentum_20d"]
    pdt.assert_series_equal(
        result.loc[1:, "composite_score"].reset_index(drop=True),
        baseline.loc[1:, "composite_score"].reset_index(drop=True),
    )


def test_custom_score_column_and_finite_output_contract() -> None:
    panel = _panel()
    panel.loc[0, "momentum_20d"] = np.inf
    result = _composer(score_col=" score ").compose(panel, NAMES)
    assert list(result.columns) == [
        "trade_date",
        "ts_code",
        "score",
        "valid_factor_count",
        "weight_coverage",
    ]
    assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()
    assert result["weight_coverage"].between(0.0, 1.0).all()


def test_f1_does_not_reflip_negative_direction_or_restandardize() -> None:
    panel = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02"],
            "ts_code": ["A", "B"],
            "momentum_20d": [10.0, 20.0],
            # Already D1-adjusted values for direction=-1 volatility.
            "volatility_20d": [100.0, -100.0],
        }
    )
    result = _composer().compose(panel, NAMES)
    np.testing.assert_allclose(result["composite_score"], [55.0, -40.0])
    # A second standardization or neutralization would not preserve this formula.


def test_changing_one_stock_or_future_date_does_not_change_other_scores() -> None:
    panel = _panel().sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    baseline = _composer().compose(panel, NAMES)
    changed_stock = panel.copy()
    changed_stock.loc[0, "momentum_20d"] = 999.0
    stock_result = _composer().compose(changed_stock, NAMES)
    pdt.assert_series_equal(
        stock_result.loc[1:, "composite_score"].reset_index(drop=True),
        baseline.loc[1:, "composite_score"].reset_index(drop=True),
    )
    changed_future = panel.copy()
    changed_future.loc[2, "quality"] = -999.0
    future_result = _composer().compose(changed_future, NAMES)
    pdt.assert_series_equal(
        future_result["composite_score"], baseline["composite_score"]
    )


def test_full_d1_d2_f1_e1_e2_integration_has_finite_outputs() -> None:
    registry = _registry()
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    codes = [f"S{i:02d}" for i in range(10)]
    rows = []
    exposures = []
    returns = []
    for date_index, date in enumerate(dates):
        for index, code in enumerate(codes):
            signal = float(index + date_index)
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "momentum_20d": signal + 0.1 * (index % 3),
                    "volatility_20d": 20.0 - signal + 0.2 * (index % 2),
                    "quality": signal * 0.7 + np.sin(index),
                }
            )
            exposures.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "industry": "I1" if index < 5 else "I2",
                    "log_total_mv": 8.0 + index * 0.2,
                }
            )
            returns.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "forward_return": signal * 0.01 + np.cos(index) * 0.001,
                }
            )
    raw = pd.DataFrame(rows)
    raw_before = raw.copy(deep=True)
    processed = FactorPreprocessor(
        registry,
        PreprocessingConfig(
            missing_method="none",
            winsor_method="none",
            standardize_method="zscore",
            min_cross_section_size=3,
        ),
    ).transform(raw, [*NAMES, "quality"])
    neutralized = FactorNeutralizer(
        NeutralizationConfig(
            min_cross_section_size=6,
            min_industry_size=2,
            standardize_residuals=True,
        )
    ).transform(processed, pd.DataFrame(exposures), [*NAMES, "quality"])
    composed = FactorComposer(registry).compose(
        neutralized, [*NAMES, "quality"]
    )
    ic = FactorEvaluator(
        FactorEvaluationConfig(min_cross_section_size=5)
    ).evaluate_ic(composed, pd.DataFrame(returns), ["composite_score"])
    quantiles = FactorQuantileEvaluator(
        QuantileEvaluationConfig(
            quantiles=5, min_cross_section_size=5, min_group_size=1
        )
    ).evaluate_quantiles(
        composed, pd.DataFrame(returns), ["composite_score"]
    )
    pdt.assert_frame_equal(raw, raw_before)
    assert composed["composite_score"].notna().all()
    assert not np.isinf(composed.select_dtypes(include=[np.number])).any().any()
    assert set(ic["factor_name"]) == {"composite_score"}
    assert set(quantiles["factor_name"]) == {"composite_score"}
    assert np.isfinite(ic["ic"]).all()
    assert np.isfinite(quantiles["group_return"].dropna()).all()
