"""Tests for same-date industry and size factor neutralization."""

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pandas as pd
import pytest

from src.factors.examples import register_example_factors
from src.factors.neutralization import FactorNeutralizer, NeutralizationConfig
from src.factors.preprocessing import FactorPreprocessor, PreprocessingConfig
from src.factors.registry import FactorRegistry


def _factor_panel(dates: int = 1) -> pd.DataFrame:
    codes = ["A", "B", "C", "D", "E", "F"]
    frames = []
    for date_index in range(dates):
        size = np.arange(1.0, 7.0) + date_index * 10.0
        industry_effect = np.array([2.0, 2.0, 2.0, -1.0, -1.0, -1.0])
        noise = np.array([-0.4, 0.2, 0.3, -0.3, 0.1, 0.5])
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": [f"2024-01-{date_index + 2:02d}"] * 6,
                    "ts_code": codes,
                    "factor_a": industry_effect + 0.7 * size + noise,
                    "factor_b": -industry_effect + 0.2 * size + noise[::-1],
                    "log_total_mv": size,
                    "log_circ_mv": size * 0.8 + noise,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _exposure_panel(dates: int = 1) -> pd.DataFrame:
    codes = ["A", "B", "C", "D", "E", "F"]
    frames = []
    for date_index in range(dates):
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": [f"2024-01-{date_index + 2:02d}"] * 6,
                    "ts_code": codes,
                    "industry": ["Tech", "Tech", "Tech", "Bank", "Bank", "Bank"],
                    "log_total_mv": np.arange(1.0, 7.0) + date_index * 10.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _config(**overrides: object) -> NeutralizationConfig:
    defaults = {
        "neutralize_industry": True,
        "neutralize_size": True,
        "min_cross_section_size": 3,
        "min_industry_size": 1,
        "standardize_residuals": False,
    }
    defaults.update(overrides)
    return NeutralizationConfig(**defaults)


def _neutralize(
    factors: pd.DataFrame | None = None,
    exposures: pd.DataFrame | None = None,
    names: list[str] | None = None,
    **config: object,
) -> pd.DataFrame:
    return FactorNeutralizer(_config(**config)).transform(
        _factor_panel() if factors is None else factors,
        _exposure_panel() if exposures is None else exposures,
        ["factor_a"] if names is None else names,
    )


def test_default_config_and_to_dict_are_serializable() -> None:
    config = NeutralizationConfig()
    assert config.neutralize_industry is True
    assert config.neutralize_size is True
    assert config.industry_col == "industry"
    assert config.size_col == "log_total_mv"
    assert config.min_cross_section_size == 10
    assert config.min_industry_size == 2
    assert config.standardize_residuals is True
    assert config.size_exempt_factors == ("log_total_mv", "log_circ_mv")
    json.dumps(config.to_dict())
    assert FactorNeutralizer(config).describe_config() == config.to_dict()


def test_config_is_frozen() -> None:
    config = NeutralizationConfig()
    with pytest.raises(FrozenInstanceError):
        config.neutralize_size = False  # type: ignore[misc]


def test_both_neutralizations_disabled_raises() -> None:
    with pytest.raises(ValueError):
        NeutralizationConfig(neutralize_industry=False, neutralize_size=False)


@pytest.mark.parametrize("field", ["industry_col", "size_col"])
@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_column_names_raise(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        NeutralizationConfig(**{field: value})


@pytest.mark.parametrize("value", [1, 0, -1, 2.0, True, "10"])
def test_invalid_min_cross_section_size_raises(value: object) -> None:
    with pytest.raises(ValueError):
        NeutralizationConfig(min_cross_section_size=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, 1.0, True, "2"])
def test_invalid_min_industry_size_raises(value: object) -> None:
    with pytest.raises(ValueError):
        NeutralizationConfig(min_industry_size=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [("ok", ""), ("ok", "   "), ("same", "same")])
def test_invalid_size_exempt_factors_raise(value: tuple[str, str]) -> None:
    with pytest.raises(ValueError):
        NeutralizationConfig(size_exempt_factors=value)


def test_exemption_list_is_normalized_to_tuple() -> None:
    config = NeutralizationConfig(size_exempt_factors=[" custom "])  # type: ignore[arg-type]
    assert config.size_exempt_factors == ("custom",)


def test_constructor_rejects_invalid_config() -> None:
    with pytest.raises(TypeError):
        FactorNeutralizer(object())  # type: ignore[arg-type]


def test_factor_panel_must_be_dataframe() -> None:
    with pytest.raises(TypeError):
        FactorNeutralizer(_config()).transform([], _exposure_panel(), ["factor_a"])  # type: ignore[arg-type]


def test_exposure_panel_must_be_dataframe() -> None:
    with pytest.raises(TypeError):
        FactorNeutralizer(_config()).transform(_factor_panel(), [], ["factor_a"])  # type: ignore[arg-type]


@pytest.mark.parametrize("names", [[], (), "factor_a"])
def test_factor_names_must_be_non_empty_sequence(names: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        FactorNeutralizer(_config()).transform(_factor_panel(), _exposure_panel(), names)  # type: ignore[arg-type]


def test_duplicate_factor_names_raise() -> None:
    with pytest.raises(ValueError):
        _neutralize(names=["factor_a", "factor_a"])


@pytest.mark.parametrize(
    ("panel_name", "column"),
    [
        ("factor", "trade_date"),
        ("factor", "ts_code"),
        ("factor", "factor_a"),
        ("exposure", "trade_date"),
        ("exposure", "ts_code"),
        ("exposure", "industry"),
        ("exposure", "log_total_mv"),
    ],
)
def test_missing_required_columns_raise(panel_name: str, column: str) -> None:
    factors = _factor_panel()
    exposures = _exposure_panel()
    if panel_name == "factor":
        factors = factors.drop(columns=column)
    else:
        exposures = exposures.drop(columns=column)
    with pytest.raises(ValueError):
        _neutralize(factors, exposures)


@pytest.mark.parametrize("panel_name", ["factor", "exposure"])
def test_invalid_dates_raise(panel_name: str) -> None:
    factors = _factor_panel()
    exposures = _exposure_panel()
    target = factors if panel_name == "factor" else exposures
    target.loc[0, "trade_date"] = "invalid"
    with pytest.raises(ValueError):
        _neutralize(factors, exposures)


@pytest.mark.parametrize("panel_name", ["factor", "exposure"])
def test_duplicate_keys_raise(panel_name: str) -> None:
    factors = _factor_panel()
    exposures = _exposure_panel()
    target = factors if panel_name == "factor" else exposures
    target.loc[1, ["trade_date", "ts_code"]] = target.loc[0, ["trade_date", "ts_code"]]
    with pytest.raises(ValueError):
        _neutralize(factors, exposures)


@pytest.mark.parametrize("panel_name", ["factor", "exposure"])
@pytest.mark.parametrize("empty", [None, "", "   "])
def test_empty_codes_raise(panel_name: str, empty: object) -> None:
    factors = _factor_panel()
    exposures = _exposure_panel()
    target = factors if panel_name == "factor" else exposures
    target.loc[0, "ts_code"] = empty
    with pytest.raises(ValueError):
        _neutralize(factors, exposures)


def test_inputs_are_not_modified() -> None:
    factors = _factor_panel()
    exposures = _exposure_panel()
    expected_factors = factors.copy(deep=True)
    expected_exposures = exposures.copy(deep=True)
    _neutralize(factors, exposures)
    pd.testing.assert_frame_equal(factors, expected_factors)
    pd.testing.assert_frame_equal(exposures, expected_exposures)


def test_industry_only_residual_mean_is_zero_within_each_industry() -> None:
    result = _neutralize(neutralize_size=False)
    exposures = _exposure_panel()
    exposures["trade_date"] = pd.to_datetime(exposures["trade_date"])
    merged = result.merge(exposures, on=["trade_date", "ts_code"])
    means = merged.groupby("industry")["factor_a"].mean()
    np.testing.assert_allclose(means, 0.0, atol=1e-12)


def test_industry_category_and_input_order_do_not_change_results() -> None:
    factors = _factor_panel().sample(frac=1.0, random_state=1).reset_index(drop=True)
    exposures = _exposure_panel().sample(frac=1.0, random_state=2).reset_index(drop=True)
    first = _neutralize(neutralize_size=False)
    second = _neutralize(factors, exposures, neutralize_size=False)
    pd.testing.assert_frame_equal(first, second)


def test_single_industry_reduces_to_demeaning() -> None:
    exposures = _exposure_panel()
    exposures["industry"] = "Only"
    factors = _factor_panel()
    result = _neutralize(factors, exposures, neutralize_size=False)
    expected = factors.sort_values(["trade_date", "ts_code"])["factor_a"]
    expected = expected - expected.mean()
    np.testing.assert_allclose(result["factor_a"], expected)


def test_small_industry_and_missing_industry_are_excluded() -> None:
    exposures = _exposure_panel()
    exposures.loc[0, "industry"] = "Tiny"
    exposures.loc[1, "industry"] = "   "
    result = _neutralize(
        exposures=exposures,
        neutralize_size=False,
        min_industry_size=2,
        min_cross_section_size=2,
    )
    assert result.loc[result["ts_code"].isin(["A", "B", "C"]), "factor_a"].isna().all()
    assert result.loc[result["ts_code"].isin(["D", "E", "F"]), "factor_a"].notna().all()


def test_industry_structures_are_date_specific() -> None:
    factors = _factor_panel(2)
    exposures = _exposure_panel(2)
    exposures.loc[exposures["trade_date"] == "2024-01-03", "industry"] = ["X", "Y", "X", "Y", "X", "Y"]
    first = _neutralize(factors, exposures, neutralize_size=False)
    changed = exposures.copy()
    changed.loc[changed["trade_date"] == "2024-01-03", "industry"] = "Future"
    second = _neutralize(factors, changed, neutralize_size=False)
    pd.testing.assert_frame_equal(first.iloc[:6], second.iloc[:6])


def test_size_only_residual_is_orthogonal_to_standardized_size() -> None:
    result = _neutralize(neutralize_industry=False)
    size = _exposure_panel()["log_total_mv"].to_numpy()
    size_z = (size - size.mean()) / size.std(ddof=0)
    assert float(np.dot(result["factor_a"], size_z)) == pytest.approx(0.0, abs=1e-12)
    assert result["factor_a"].mean() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(("scale", "offset"), [(1000.0, 0.0), (-3.0, 7.0)])
def test_linear_size_rescaling_does_not_change_residuals(scale: float, offset: float) -> None:
    first = _neutralize(neutralize_industry=False)
    exposures = _exposure_panel()
    exposures["log_total_mv"] = exposures["log_total_mv"] * scale + offset
    second = _neutralize(exposures=exposures, neutralize_industry=False)
    np.testing.assert_allclose(first["factor_a"], second["factor_a"], atol=1e-12)


def test_constant_size_returns_nan() -> None:
    exposures = _exposure_panel()
    exposures["log_total_mv"] = 5.0
    assert _neutralize(exposures=exposures, neutralize_industry=False)["factor_a"].isna().all()


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_missing_or_infinite_size_row_stays_nan(missing: float) -> None:
    exposures = _exposure_panel()
    exposures.loc[0, "log_total_mv"] = missing
    result = _neutralize(exposures=exposures, neutralize_industry=False)
    assert np.isnan(result.loc[result["ts_code"] == "A", "factor_a"].item())
    assert result.loc[result["ts_code"] != "A", "factor_a"].notna().all()


def test_each_date_uses_its_own_size_statistics() -> None:
    factors = _factor_panel(2)
    exposures = _exposure_panel(2)
    first = _neutralize(factors, exposures, neutralize_industry=False)
    changed = exposures.copy()
    changed.loc[changed["trade_date"] == "2024-01-03", "log_total_mv"] **= 3
    second = _neutralize(factors, changed, neutralize_industry=False)
    pd.testing.assert_frame_equal(first.iloc[:6], second.iloc[:6])


def test_joint_neutralization_residual_is_orthogonal_to_design() -> None:
    result = _neutralize()
    exposures = _exposure_panel()
    industry_dummy = (exposures["industry"] == "Tech").astype(float).to_numpy()
    size = exposures["log_total_mv"].to_numpy()
    size_z = (size - size.mean()) / size.std(ddof=0)
    design = np.column_stack([np.ones(6), industry_dummy, size_z])
    np.testing.assert_allclose(design.T @ result["factor_a"].to_numpy(), 0.0, atol=1e-12)


def test_joint_neutralization_removes_known_effects() -> None:
    result = _neutralize()
    assert result["factor_a"].std(ddof=0) < _factor_panel()["factor_a"].std(ddof=0)


def test_result_does_not_depend_on_original_row_order() -> None:
    first = _neutralize()
    second = _neutralize(
        _factor_panel().sample(frac=1.0, random_state=3),
        _exposure_panel().sample(frac=1.0, random_state=4),
    )
    pd.testing.assert_frame_equal(first, second)


def test_observations_not_greater_than_rank_return_nan() -> None:
    factors = _factor_panel().iloc[[0, 3]].copy()
    exposures = _exposure_panel().iloc[[0, 3]].copy()
    result = _neutralize(
        factors,
        exposures,
        neutralize_size=False,
        min_cross_section_size=2,
    )
    assert result["factor_a"].isna().all()


def test_fewer_than_minimum_observations_return_nan() -> None:
    result = _neutralize(min_cross_section_size=7)
    assert result["factor_a"].isna().all()


def test_standardized_residuals_have_mean_zero_and_std_one() -> None:
    result = _neutralize(standardize_residuals=True)
    assert result["factor_a"].mean() == pytest.approx(0.0, abs=1e-12)
    assert result["factor_a"].std(ddof=0) == pytest.approx(1.0, abs=1e-12)


def test_unstandardized_residuals_match_manual_lstsq() -> None:
    factors = _factor_panel()
    exposures = _exposure_panel()
    industry_dummy = (exposures["industry"] == "Tech").astype(float).to_numpy()
    size = exposures["log_total_mv"].to_numpy()
    size_z = (size - size.mean()) / size.std(ddof=0)
    design = np.column_stack([np.ones(6), industry_dummy, size_z])
    y = factors["factor_a"].to_numpy()
    expected = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    result = _neutralize(factors, exposures, standardize_residuals=False)
    np.testing.assert_allclose(result["factor_a"], expected, atol=1e-12)


def test_zero_residual_standard_deviation_returns_nan() -> None:
    factors = _factor_panel()
    exposures = _exposure_panel()
    factors["factor_a"] = exposures["log_total_mv"] * 2.0 + 1.0
    result = _neutralize(
        factors,
        exposures,
        neutralize_industry=False,
        standardize_residuals=True,
    )
    assert result["factor_a"].isna().all()


def test_output_never_contains_infinity() -> None:
    factors = _factor_panel()
    factors.loc[0, "factor_a"] = np.inf
    result = _neutralize(factors=factors, min_cross_section_size=3)
    assert not np.isinf(result["factor_a"].to_numpy()).any()


@pytest.mark.parametrize("factor_name", ["log_total_mv", "log_circ_mv"])
def test_default_size_exempt_factors_do_not_use_size_exposure(factor_name: str) -> None:
    factors = _factor_panel()
    exposures = _exposure_panel()
    exposures["log_total_mv"] = 1.0
    result = _neutralize(factors, exposures, [factor_name])
    assert result[factor_name].notna().all()


def test_exempt_size_factor_can_still_be_industry_neutralized() -> None:
    result = _neutralize(names=["log_total_mv"])
    exposures = _exposure_panel()
    exposures["trade_date"] = pd.to_datetime(exposures["trade_date"])
    merged = result.merge(exposures, on=["trade_date", "ts_code"])
    np.testing.assert_allclose(merged.groupby("industry")["log_total_mv_x"].mean(), 0.0, atol=1e-12)


def test_non_exempt_factor_uses_size_exposure() -> None:
    exposures = _exposure_panel()
    exposures["log_total_mv"] = 1.0
    assert _neutralize(exposures=exposures)["factor_a"].isna().all()


def test_custom_size_exemption_is_honored() -> None:
    exposures = _exposure_panel()
    exposures["log_total_mv"] = 1.0
    result = _neutralize(exposures=exposures, size_exempt_factors=("factor_a",))
    assert result["factor_a"].notna().all()


def test_output_contract_extra_exposures_and_missing_match() -> None:
    factors = _factor_panel()
    exposures = pd.concat(
        [
            _exposure_panel().iloc[1:],
            pd.DataFrame(
                {
                    "trade_date": ["2024-01-02"],
                    "ts_code": ["EXTRA"],
                    "industry": ["Other"],
                    "log_total_mv": [10.0],
                }
            ),
        ],
        ignore_index=True,
    )
    result = _neutralize(factors, exposures, min_cross_section_size=3)
    assert len(result) == len(factors)
    assert result.columns.tolist() == ["trade_date", "ts_code", "factor_a"]
    assert result["ts_code"].tolist() == sorted(factors["ts_code"])
    assert np.isnan(result.loc[result["ts_code"] == "A", "factor_a"].item())
    assert "EXTRA" not in result["ts_code"].tolist()


def test_future_factor_or_exposure_changes_do_not_change_past() -> None:
    factors = _factor_panel(2)
    exposures = _exposure_panel(2)
    first = _neutralize(factors, exposures)
    changed_factors = factors.copy()
    changed_exposures = exposures.copy()
    future = changed_factors["trade_date"] == "2024-01-03"
    changed_factors.loc[future, "factor_a"] *= -1000
    changed_exposures.loc[future, "log_total_mv"] **= 2
    second = _neutralize(changed_factors, changed_exposures)
    pd.testing.assert_frame_equal(first.iloc[:6], second.iloc[:6])


def test_changing_one_date_does_not_change_another_date() -> None:
    factors = _factor_panel(2)
    exposures = _exposure_panel(2)
    first = _neutralize(factors, exposures)
    factors.loc[factors["trade_date"] == "2024-01-02", "factor_a"] += 999
    second = _neutralize(factors, exposures)
    pd.testing.assert_frame_equal(first.iloc[6:], second.iloc[6:])


def test_changing_factor_a_does_not_change_factor_b() -> None:
    factors = _factor_panel()
    first = _neutralize(factors, names=["factor_a", "factor_b"])
    factors["factor_a"] = factors["factor_a"] ** 3
    second = _neutralize(factors, names=["factor_a", "factor_b"])
    np.testing.assert_allclose(first["factor_b"], second["factor_b"])


def test_same_date_stocks_have_expected_cross_sectional_influence() -> None:
    factors = _factor_panel()
    first = _neutralize(factors)
    factors.loc[0, "factor_a"] += 100.0
    second = _neutralize(factors)
    assert not np.allclose(first.loc[1:, "factor_a"], second.loc[1:, "factor_a"])


def test_no_forward_or_backward_fill_of_exposure() -> None:
    factors = _factor_panel(2)
    exposures = _exposure_panel(2)
    exposures = exposures[
        ~((exposures["trade_date"] == "2024-01-03") & (exposures["ts_code"] == "A"))
    ]
    result = _neutralize(factors, exposures)
    current = result[
        (result["trade_date"] == pd.Timestamp("2024-01-03"))
        & (result["ts_code"] == "A")
    ]
    assert current["factor_a"].isna().all()


def _d1_registry() -> FactorRegistry:
    registry = FactorRegistry()
    register_example_factors(registry)
    return registry


def test_d1_then_d2_integration_and_direction_handling() -> None:
    dates = ["2024-01-02"] * 6 + ["2024-01-03"] * 6
    codes = ["A", "B", "C", "D", "E", "F"] * 2
    raw = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": codes,
            "momentum_20d": [1, 2, 4, 3, 6, 5, 11, 13, 12, 16, 14, 15],
            "volatility_20d": [6, 4, 5, 2, 3, 1, 16, 14, 15, 12, 13, 11],
        }
    )
    exposures = _exposure_panel(2)
    preprocessed = FactorPreprocessor(
        _d1_registry(),
        PreprocessingConfig(
            missing_method="none",
            winsor_method="none",
            standardize_method="zscore",
            min_cross_section_size=3,
        ),
    ).transform(raw, ["momentum_20d", "volatility_20d"])
    assert preprocessed.loc[0, "volatility_20d"] < preprocessed.loc[5, "volatility_20d"]
    result = FactorNeutralizer(
        _config(standardize_residuals=True)
    ).transform(preprocessed, exposures, ["momentum_20d", "volatility_20d"])
    assert len(result) == len(raw)
    assert not np.isinf(result.iloc[:, 2:].to_numpy()).any()
    assert result.iloc[:, 2:].notna().all().all()


def test_d1_d2_future_isolation() -> None:
    factors = _factor_panel(2)
    exposures = _exposure_panel(2)
    first = _neutralize(factors, exposures, names=["factor_a", "factor_b"])
    factors.loc[factors["trade_date"] == "2024-01-03", ["factor_a", "factor_b"]] *= 100
    second = _neutralize(factors, exposures, names=["factor_a", "factor_b"])
    pd.testing.assert_frame_equal(first.iloc[:6], second.iloc[:6])
