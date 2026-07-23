"""Tests for trade-date-isolated cross-sectional factor preprocessing."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from src.factors.base import FactorMetadata, FunctionFactor
from src.factors.examples import register_example_factors
from src.factors.financial_factors import register_financial_factors
from src.factors.preprocessing import FactorPreprocessor, PreprocessingConfig
from src.factors.price_volume import register_price_volume_factors
from src.factors.registry import FactorRegistry
from src.factors.valuation import register_valuation_factors


def _registry() -> FactorRegistry:
    registry = FactorRegistry()
    register_example_factors(registry)
    register_price_volume_factors(registry)
    register_valuation_factors(registry)
    register_financial_factors(registry)
    return registry


def _preprocessor(**config_overrides: object) -> FactorPreprocessor:
    defaults = {
        "missing_method": "none",
        "winsor_method": "none",
        "standardize_method": "none",
        "min_cross_section_size": 2,
    }
    defaults.update(config_overrides)
    return FactorPreprocessor(_registry(), PreprocessingConfig(**defaults))


def _panel(values: list[float], name: str = "momentum_20d") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * len(values),
            "ts_code": [f"{index:06d}.SZ" for index in range(len(values))],
            name: values,
        }
    )


def _two_date_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 3 + ["2024-01-03"] * 3,
            "ts_code": ["A", "B", "C"] * 2,
            "momentum_20d": [1.0, np.nan, 5.0, 100.0, 200.0, 300.0],
        }
    )


def test_default_config_and_serializable_summary() -> None:
    config = PreprocessingConfig()
    assert config.to_dict() == {
        "missing_method": "median",
        "winsor_method": "mad",
        "lower_quantile": 0.01,
        "upper_quantile": 0.99,
        "mad_limit": 3.0,
        "standardize_method": "zscore",
        "min_cross_section_size": 3,
    }
    assert FactorPreprocessor(_registry()).describe_config() == config.to_dict()


def test_config_is_frozen() -> None:
    config = PreprocessingConfig()
    with pytest.raises(FrozenInstanceError):
        config.missing_method = "none"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("missing_method", "mean"),
        ("winsor_method", "clip"),
        ("standardize_method", "minmax"),
    ],
)
def test_invalid_method_names_raise(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        PreprocessingConfig(**{field: value})


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(-0.1, 0.9), (0.5, 0.5), (0.9, 0.5), (0.1, 1.1), (np.nan, 0.9), (False, 0.9)],
)
def test_invalid_quantile_boundaries_raise(lower: object, upper: object) -> None:
    with pytest.raises(ValueError):
        PreprocessingConfig(lower_quantile=lower, upper_quantile=upper)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, np.inf, True])
def test_invalid_mad_limit_raises(value: object) -> None:
    with pytest.raises(ValueError):
        PreprocessingConfig(mad_limit=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, 1.0, True, "3"])
def test_invalid_min_cross_section_size_raises(value: object) -> None:
    with pytest.raises(ValueError):
        PreprocessingConfig(min_cross_section_size=value)  # type: ignore[arg-type]


def test_constructor_validates_registry_and_config() -> None:
    with pytest.raises(TypeError):
        FactorPreprocessor(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        FactorPreprocessor(_registry(), object())  # type: ignore[arg-type]


@pytest.mark.parametrize("factor_names", [[], (), "momentum_20d"])
def test_factor_names_must_be_non_empty_sequence(factor_names: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _preprocessor().transform(_panel([1.0, 2.0]), factor_names)  # type: ignore[arg-type]


def test_duplicate_factor_names_raise() -> None:
    with pytest.raises(ValueError):
        _preprocessor().transform(
            _panel([1.0, 2.0]), ["momentum_20d", "momentum_20d"]
        )


def test_unregistered_factor_raises() -> None:
    frame = _panel([1.0, 2.0], name="unknown")
    with pytest.raises(KeyError):
        _preprocessor().transform(frame, ["unknown"])


def test_input_must_be_dataframe() -> None:
    with pytest.raises(TypeError):
        _preprocessor().transform([], ["momentum_20d"])  # type: ignore[arg-type]


@pytest.mark.parametrize("column", ["trade_date", "ts_code", "momentum_20d"])
def test_missing_required_column_raises(column: str) -> None:
    with pytest.raises(ValueError):
        _preprocessor().transform(_panel([1.0, 2.0]).drop(columns=column), ["momentum_20d"])


def test_invalid_trade_date_raises() -> None:
    frame = _panel([1.0, 2.0])
    frame.loc[0, "trade_date"] = "invalid"
    with pytest.raises(ValueError):
        _preprocessor().transform(frame, ["momentum_20d"])


@pytest.mark.parametrize("empty_code", [None, "", "   "])
def test_empty_ts_code_raises(empty_code: object) -> None:
    frame = _panel([1.0, 2.0])
    frame.loc[0, "ts_code"] = empty_code
    with pytest.raises(ValueError):
        _preprocessor().transform(frame, ["momentum_20d"])


def test_duplicate_trade_date_and_code_raise() -> None:
    frame = _panel([1.0, 2.0])
    frame.loc[1, "ts_code"] = frame.loc[0, "ts_code"]
    with pytest.raises(ValueError):
        _preprocessor().transform(frame, ["momentum_20d"])


def test_transform_does_not_mutate_input() -> None:
    frame = _two_date_panel()
    expected = frame.copy(deep=True)
    _preprocessor(missing_method="median").transform(frame, ["momentum_20d"])
    pd.testing.assert_frame_equal(frame, expected)


def test_missing_none_preserves_nan_without_dropping_rows() -> None:
    result = _preprocessor().transform(_panel([1.0, np.nan, 3.0]), ["momentum_20d"])
    assert len(result) == 3
    assert result["momentum_20d"].isna().sum() == 1


def test_median_fill_uses_only_same_date() -> None:
    result = _preprocessor(missing_method="median").transform(
        _two_date_panel(), ["momentum_20d"]
    )
    first_date = result[result["trade_date"] == pd.Timestamp("2024-01-02")]
    assert first_date.loc[first_date["ts_code"] == "B", "momentum_20d"].item() == 3.0


def test_all_missing_date_remains_missing() -> None:
    frame = _two_date_panel()
    frame.loc[frame["trade_date"] == "2024-01-02", "momentum_20d"] = np.nan
    result = _preprocessor(missing_method="median").transform(frame, ["momentum_20d"])
    assert result.loc[result["trade_date"] == pd.Timestamp("2024-01-02"), "momentum_20d"].isna().all()


def test_other_date_cannot_change_median_fill() -> None:
    frame = _two_date_panel()
    changed = frame.copy()
    changed.loc[changed["trade_date"] == "2024-01-03", "momentum_20d"] *= 1000
    first = _preprocessor(missing_method="median").transform(frame, ["momentum_20d"])
    second = _preprocessor(missing_method="median").transform(changed, ["momentum_20d"])
    pd.testing.assert_frame_equal(first.iloc[:3], second.iloc[:3])


def test_quantile_winsorization_matches_manual_result() -> None:
    frame = _panel([0.0, 10.0, 20.0, 100.0])
    result = _preprocessor(
        winsor_method="quantile", lower_quantile=0.25, upper_quantile=0.75
    ).transform(frame, ["momentum_20d"])
    expected = pd.Series([0.0, 10.0, 20.0, 100.0]).clip(
        lower=pd.Series([0.0, 10.0, 20.0, 100.0]).quantile(0.25),
        upper=pd.Series([0.0, 10.0, 20.0, 100.0]).quantile(0.75),
    )
    np.testing.assert_allclose(result["momentum_20d"], expected)


def test_quantile_boundaries_are_date_isolated() -> None:
    frame = _two_date_panel().fillna({"momentum_20d": 3.0})
    first = _preprocessor(winsor_method="quantile", lower_quantile=0.25, upper_quantile=0.75).transform(frame, ["momentum_20d"])
    changed = frame.copy()
    changed.loc[changed["trade_date"] == "2024-01-03", "momentum_20d"] = [1e6, 2e6, 3e6]
    second = _preprocessor(winsor_method="quantile", lower_quantile=0.25, upper_quantile=0.75).transform(changed, ["momentum_20d"])
    pd.testing.assert_frame_equal(first.iloc[:3], second.iloc[:3])


def test_mad_winsorization_matches_manual_limits() -> None:
    result = _preprocessor(winsor_method="mad", mad_limit=1.0).transform(
        _panel([0.0, 1.0, 2.0, 100.0]), ["momentum_20d"]
    )
    median = 1.5
    mad = 1.0
    expected = pd.Series([0.0, 1.0, 2.0, 100.0]).clip(
        median - 1.4826 * mad, median + 1.4826 * mad
    )
    np.testing.assert_allclose(result["momentum_20d"], expected)


def test_mad_zero_keeps_finite_values_and_nan() -> None:
    result = _preprocessor(winsor_method="mad").transform(
        _panel([2.0, 2.0, np.nan, 100.0]), ["momentum_20d"]
    )
    assert result["momentum_20d"].tolist()[:2] == [2.0, 2.0]
    assert np.isnan(result.loc[2, "momentum_20d"])
    assert result.loc[3, "momentum_20d"] == 100.0


def test_direction_positive_is_unchanged() -> None:
    result = _preprocessor().transform(_panel([1.0, 2.0]), ["momentum_20d"])
    assert result["momentum_20d"].tolist() == [1.0, 2.0]


def test_direction_negative_is_multiplied_by_minus_one() -> None:
    result = _preprocessor().transform(
        _panel([1.0, 2.0], "volatility_20d"), ["volatility_20d"]
    )
    assert result["volatility_20d"].tolist() == [-1.0, -2.0]


def test_lower_raw_volatility_receives_higher_rank_score() -> None:
    result = _preprocessor(standardize_method="rank").transform(
        _panel([1.0, 2.0, 3.0], "volatility_20d"), ["volatility_20d"]
    )
    assert result.loc[0, "volatility_20d"] > result.loc[2, "volatility_20d"]


def test_daily_zscore_has_population_mean_zero_and_std_one() -> None:
    result = _preprocessor(standardize_method="zscore", min_cross_section_size=3).transform(
        _panel([1.0, 2.0, 4.0]), ["momentum_20d"]
    )
    assert result["momentum_20d"].mean() == pytest.approx(0.0)
    assert result["momentum_20d"].std(ddof=0) == pytest.approx(1.0)


def test_zscore_uses_each_dates_own_statistics() -> None:
    frame = _two_date_panel().fillna({"momentum_20d": 3.0})
    result = _preprocessor(standardize_method="zscore", min_cross_section_size=3).transform(frame, ["momentum_20d"])
    grouped = result.groupby("trade_date")["momentum_20d"]
    np.testing.assert_allclose(grouped.mean(), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(grouped.std(ddof=0), [1.0, 1.0], atol=1e-12)


def test_zscore_insufficient_count_returns_nan() -> None:
    result = _preprocessor(standardize_method="zscore", min_cross_section_size=3).transform(
        _panel([1.0, 2.0]), ["momentum_20d"]
    )
    assert result["momentum_20d"].isna().all()


def test_zscore_zero_standard_deviation_returns_nan() -> None:
    result = _preprocessor(standardize_method="zscore", min_cross_section_size=2).transform(
        _panel([5.0, 5.0, 5.0]), ["momentum_20d"]
    )
    assert result["momentum_20d"].isna().all()


def test_rank_range_endpoints_and_ties() -> None:
    result = _preprocessor(standardize_method="rank", min_cross_section_size=2).transform(
        _panel([1.0, 2.0, 2.0, 4.0]), ["momentum_20d"]
    )
    assert result["momentum_20d"].min() == pytest.approx(-1.0)
    assert result["momentum_20d"].max() == pytest.approx(1.0)
    assert result.loc[1, "momentum_20d"] == pytest.approx(result.loc[2, "momentum_20d"])
    assert result.loc[1, "momentum_20d"] == pytest.approx(0.0)


def test_rank_all_equal_returns_zero() -> None:
    result = _preprocessor(standardize_method="rank").transform(
        _panel([3.0, 3.0, 3.0]), ["momentum_20d"]
    )
    assert result["momentum_20d"].eq(0.0).all()


def test_rank_insufficient_count_returns_nan() -> None:
    result = _preprocessor(standardize_method="rank", min_cross_section_size=3).transform(
        _panel([1.0, 2.0]), ["momentum_20d"]
    )
    assert result["momentum_20d"].isna().all()


def test_output_shape_columns_sorting_and_numeric_dtype() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-03", "2024-01-02"],
            "ts_code": ["B", "A"],
            "momentum_20d": [2.0, 1.0],
            "unused": [9, 9],
        }
    )
    result = _preprocessor().transform(frame, ["momentum_20d"])
    assert len(result) == len(frame)
    assert result.columns.tolist() == ["trade_date", "ts_code", "momentum_20d"]
    assert result["trade_date"].is_monotonic_increasing
    assert pd.api.types.is_numeric_dtype(result["momentum_20d"])


def test_infinities_become_nan_and_never_reappear() -> None:
    result = _preprocessor().transform(
        _panel([1.0, np.inf, -np.inf]), ["momentum_20d"]
    )
    assert result["momentum_20d"].isna().sum() == 2
    assert not np.isinf(result["momentum_20d"].to_numpy()).any()


def test_future_values_do_not_change_past_results() -> None:
    frame = _two_date_panel().fillna({"momentum_20d": 3.0})
    processor = _preprocessor(
        winsor_method="quantile",
        lower_quantile=0.1,
        upper_quantile=0.9,
        standardize_method="zscore",
        min_cross_section_size=3,
    )
    first = processor.transform(frame, ["momentum_20d"])
    changed = frame.copy()
    changed.loc[changed["trade_date"] == "2024-01-03", "momentum_20d"] = [-1e9, np.nan, 1e9]
    second = processor.transform(changed, ["momentum_20d"])
    pd.testing.assert_frame_equal(first.iloc[:3], second.iloc[:3])


def test_missing_values_are_not_forward_or_backward_filled_across_dates() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03"],
            "ts_code": ["A", "A"],
            "momentum_20d": [1.0, np.nan],
        }
    )
    result = _preprocessor(missing_method="median").transform(frame, ["momentum_20d"])
    assert np.isnan(result.loc[1, "momentum_20d"])


def test_multiple_factors_use_independent_statistics_and_directions() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 3,
            "ts_code": ["A", "B", "C"],
            "momentum_20d": [1.0, 2.0, 3.0],
            "volatility_20d": [30.0, 20.0, 10.0],
        }
    )
    result = _preprocessor(standardize_method="zscore", min_cross_section_size=3).transform(
        frame, ["volatility_20d", "momentum_20d"]
    )
    assert result.columns.tolist() == ["trade_date", "ts_code", "volatility_20d", "momentum_20d"]
    np.testing.assert_allclose(result["volatility_20d"], result["momentum_20d"])


@pytest.mark.parametrize("name", ["momentum_20d", "volatility_20d", "ep_ttm", "debt_to_assets"])
def test_existing_registered_factor_can_be_preprocessed(name: str) -> None:
    result = _preprocessor().transform(_panel([1.0, 2.0], name), [name])
    assert result[name].notna().all()


def test_all_registration_entrypoints_work_together() -> None:
    registry = _registry()
    assert len(registry.list_names()) == 24
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 3,
            "ts_code": ["A", "B", "C"],
            "momentum_20d": [1.0, 2.0, 3.0],
            "volatility_20d": [3.0, 2.0, 1.0],
            "ep_ttm": [0.1, 0.2, 0.3],
            "debt_to_assets": [0.8, 0.5, 0.2],
        }
    )
    result = FactorPreprocessor(
        registry,
        PreprocessingConfig(
            missing_method="median",
            winsor_method="mad",
            standardize_method="rank",
            min_cross_section_size=3,
        ),
    ).transform(frame, ["momentum_20d", "volatility_20d", "ep_ttm", "debt_to_assets"])
    assert result.iloc[:, 2:].notna().all().all()


def test_custom_registered_factor_uses_metadata_without_mutating_it() -> None:
    metadata = FactorMetadata(name="custom", category="test", direction=-1)
    factor = FunctionFactor(metadata, lambda data: pd.to_numeric(data["custom"]))
    registry = FactorRegistry()
    registry.register(factor)
    result = FactorPreprocessor(
        registry,
        PreprocessingConfig(missing_method="none", winsor_method="none", standardize_method="none"),
    ).transform(_panel([1.0, 2.0], "custom"), ["custom"])
    assert result["custom"].tolist() == [-1.0, -2.0]
    assert factor.metadata is metadata
