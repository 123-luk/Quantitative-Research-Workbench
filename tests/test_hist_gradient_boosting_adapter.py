"""Tests for the V3-D1 time-safe HistGradientBoosting adapter."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml import (
    HistGradientBoostingModelAdapter,
    HistGradientBoostingModelConfig,
    ModelConfigError,
    ModelDataError,
    ModelFeatureImportanceUnavailableError,
    ModelFeatureMismatchError,
    ModelFitAudit,
    ModelNotFittedError,
)
import src.ml.models.tree as tree_module


def _data(
    *,
    start: int = 0,
    rows: int = 48,
) -> tuple[pd.DataFrame, pd.Series]:
    positions = np.arange(rows, dtype=np.float64)
    index = pd.Index(range(start, start + rows), name="row")
    X = pd.DataFrame(
        {
            "factor_a": positions,
            "factor_b": np.sin(positions / 3.0),
            "factor_nan": np.where(positions % 7 == 0, np.nan, positions / 10.0),
            "constant": np.where(positions % 9 == 0, np.nan, 3.0),
        },
        index=index,
    )
    y = pd.Series(
        0.4 * positions + np.sin(positions),
        index=index,
        name="forward_return",
        dtype=np.float64,
    )
    return X, y


def _fast_config(**overrides: object) -> HistGradientBoostingModelConfig:
    return HistGradientBoostingModelConfig.from_dict(
        {
            "max_iter": 12,
            "min_samples_leaf": 3,
            "random_state": 17,
            **overrides,
        }
    )


def test_default_config_is_complete_frozen_and_json_safe() -> None:
    config = HistGradientBoostingModelConfig.from_dict(None)
    expected = {
        "loss": "squared_error",
        "quantile": None,
        "learning_rate": 0.1,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "max_depth": None,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_features": 1.0,
        "max_bins": 255,
        "early_stopping": False,
        "n_iter_no_change": 10,
        "tol": 1e-7,
        "warm_start": False,
        "random_state": 42,
        "verbose": 0,
    }
    assert config.as_dict() == expected
    estimator_params = config.to_estimator_params()
    assert {name: estimator_params[name] for name in expected} == expected
    assert estimator_params["validation_fraction"] is None
    json.dumps(config.as_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        config.max_iter = 2


def test_every_public_parameter_is_overridable_and_normalized() -> None:
    values = {
        "loss": " QUANTILE ",
        "quantile": np.float64(0.7),
        "learning_rate": np.float64(0.03),
        "max_iter": np.int64(300),
        "max_leaf_nodes": None,
        "max_depth": np.int64(6),
        "min_samples_leaf": np.int64(5),
        "l2_regularization": np.float64(0.2),
        "max_features": np.float64(0.8),
        "max_bins": np.int64(128),
        "early_stopping": True,
        "n_iter_no_change": np.int64(7),
        "tol": np.float64(1e-6),
        "warm_start": True,
        "random_state": np.int64(123),
        "verbose": np.int64(1),
    }
    config = HistGradientBoostingModelConfig.from_dict(values)
    assert config.as_dict() == {
        **values,
        "loss": "quantile",
        "quantile": 0.7,
        "learning_rate": 0.03,
        "max_iter": 300,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "l2_regularization": 0.2,
        "max_features": 0.8,
        "max_bins": 128,
        "n_iter_no_change": 7,
        "tol": 1e-6,
        "random_state": 123,
        "verbose": 1,
    }


def test_schema_is_stable_complete_and_json_safe() -> None:
    first = HistGradientBoostingModelConfig.parameter_schema()
    second = HistGradientBoostingModelConfig.parameter_schema()
    expected_names = tuple(HistGradientBoostingModelConfig().as_dict())
    assert tuple(spec.name for spec in first) == expected_names
    assert first == second
    payload = [spec.as_dict() for spec in first]
    json.dumps(payload, allow_nan=False)
    assert {item["ui_control"] for item in payload} <= {
        "number",
        "checkbox",
        "select",
    }
    quantile = next(spec for spec in first if spec.name == "quantile")
    assert quantile.value_type == "optional_float"
    max_bins = next(spec for spec in first if spec.name == "max_bins")
    assert max_bins.minimum == 2
    assert max_bins.maximum == 255


@pytest.mark.parametrize(
    "field",
    [
        "unknown",
        "categorical_features",
        "monotonic_cst",
        "interaction_cst",
        "validation_fraction",
        "scoring",
    ],
)
def test_unknown_and_complex_parameters_are_rejected(field: str) -> None:
    with pytest.raises(ModelConfigError, match="unknown parameter"):
        HistGradientBoostingModelConfig.from_dict({field: object()})


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"loss": "invalid"}, "loss"),
        ({"loss": object()}, "loss"),
        ({"loss": "quantile"}, "quantile"),
        ({"loss": "quantile", "quantile": 0.0}, "quantile"),
        ({"loss": "quantile", "quantile": 1.0}, "quantile"),
        ({"loss": "quantile", "quantile": True}, "quantile"),
        ({"quantile": 0.5}, "unless"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": True}, "learning_rate"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": True}, "max_iter"),
        ({"max_leaf_nodes": 1}, "max_leaf_nodes"),
        ({"max_leaf_nodes": True}, "max_leaf_nodes"),
        ({"max_depth": 0}, "max_depth"),
        ({"max_depth": False}, "max_depth"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"l2_regularization": -0.1}, "l2_regularization"),
        ({"max_features": 0.0}, "max_features"),
        ({"max_features": 1.1}, "max_features"),
        ({"max_bins": 1}, "max_bins"),
        ({"max_bins": 256}, "max_bins"),
        ({"max_bins": True}, "max_bins"),
        ({"early_stopping": "auto"}, "early_stopping"),
        ({"early_stopping": 1}, "early_stopping"),
        ({"n_iter_no_change": 0}, "n_iter_no_change"),
        ({"tol": 0.0}, "tol"),
        ({"warm_start": 1}, "warm_start"),
        ({"random_state": True}, "random_state"),
        ({"verbose": -1}, "verbose"),
        ({"verbose": False}, "verbose"),
    ],
)
def test_invalid_configuration_is_rejected(
    params: dict[str, object], match: str
) -> None:
    with pytest.raises(ModelConfigError, match=match):
        HistGradientBoostingModelConfig.from_dict(params)


@pytest.mark.parametrize(
    "loss",
    ["squared_error", "absolute_error", "poisson", "gamma"],
)
def test_actual_non_quantile_losses_are_open_and_normalized(loss: str) -> None:
    config = HistGradientBoostingModelConfig(loss=f" {loss.upper()} ")
    assert config.loss == loss


def test_normal_fit_uses_native_missing_without_preprocessors() -> None:
    X, y = _data()
    X_before = X.copy(deep=True)
    y_before = y.copy(deep=True)
    adapter = HistGradientBoostingModelAdapter(_fast_config())

    audit = adapter.fit(X, y)

    assert adapter.model_name == "hist_gradient_boosting"
    assert adapter.is_fitted
    assert isinstance(audit, ModelFitAudit)
    assert isinstance(adapter._pipeline, HistGradientBoostingRegressor)
    assert not isinstance(adapter._pipeline, Pipeline)
    assert not isinstance(adapter._pipeline, (SimpleImputer, StandardScaler))
    assert audit.native_missing_support
    assert not audit.imputer_enabled
    assert not audit.scaler_enabled
    assert audit.estimator_intercept is None
    assert audit.imputation_values == ()
    assert audit.scaler_means == ()
    assert audit.scaler_scales == ()
    assert audit.constant_features == ("constant",)
    assert audit.n_iterations is not None
    assert audit.n_iterations <= adapter.config.max_iter
    assert audit.best_iteration is None
    pdt.assert_frame_equal(X, X_before)
    pdt.assert_series_equal(y, y_before)


def test_custom_parameters_reach_estimator_and_audit() -> None:
    X, y = _data()
    config = _fast_config(
        learning_rate=0.03,
        max_depth=4,
        max_leaf_nodes=15,
        l2_regularization=0.4,
        max_features=0.8,
        max_bins=64,
        warm_start=True,
        random_state=123,
    )
    adapter = HistGradientBoostingModelAdapter(config)
    audit = adapter.fit(X, y)
    assert dict(audit.resolved_parameters) == config.as_dict()
    assert adapter._pipeline.get_params()["learning_rate"] == 0.03
    assert adapter._pipeline.get_params()["random_state"] == 123
    assert adapter._pipeline.get_params()["validation_fraction"] is None


@pytest.mark.parametrize(
    ("x_change", "y_change", "error", "match"),
    [
        (
            lambda X: X.assign(factor_a=np.inf),
            lambda y: y,
            ModelDataError,
            "infinity",
        ),
        (
            lambda X: X.assign(factor_a=np.nan),
            lambda y: y,
            ModelDataError,
            "entirely missing",
        ),
        (
            lambda X: X,
            lambda y: y.mask(y.index == y.index[0]),
            ModelDataError,
            "missing",
        ),
        (
            lambda X: X,
            lambda y: y.mask(y.index == y.index[0], np.inf),
            ModelDataError,
            "infinity",
        ),
        (
            lambda X: X,
            lambda y: y.set_axis(range(1000, 1000 + len(y))),
            ModelDataError,
            "index",
        ),
        (
            lambda X: X.rename(columns={"factor_a": "prediction"}),
            lambda y: y,
            ModelDataError,
            "reserved",
        ),
    ],
)
def test_training_contract_rejects_invalid_data(
    x_change: object,
    y_change: object,
    error: type[Exception],
    match: str,
) -> None:
    X, y = _data()
    with pytest.raises(error, match=match):
        HistGradientBoostingModelAdapter(_fast_config()).fit(
            x_change(X),
            y_change(y),
        )


def test_refit_rebuilds_estimator_even_with_warm_start() -> None:
    X, y = _data()
    adapter = HistGradientBoostingModelAdapter(_fast_config(warm_start=True))
    first_audit = adapter.fit(X, y)
    first_estimator = adapter._pipeline
    shifted = X.copy()
    shifted["factor_a"] += 1000.0
    second_audit = adapter.fit(shifted, y)
    assert adapter._pipeline is not first_estimator
    assert first_audit.train_missing_counts == second_audit.train_missing_counts


def test_failed_initial_fit_leaves_no_fitted_state() -> None:
    X, y = _data()
    adapter = HistGradientBoostingModelAdapter(_fast_config())
    with pytest.raises(ModelDataError):
        adapter.fit(X.assign(factor_a=np.inf), y)
    assert not adapter.is_fitted
    assert adapter.feature_names == ()


def test_same_configuration_and_data_are_reproducible() -> None:
    X, y = _data()
    first = HistGradientBoostingModelAdapter(_fast_config(random_state=7))
    second = HistGradientBoostingModelAdapter(_fast_config(random_state=7))
    first.fit(X, y)
    second.fit(X, y)
    pdt.assert_series_equal(first.predict(X), second.predict(X))


def test_early_stopping_false_never_passes_external_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _data()
    valid_x, valid_y = _data(start=1000, rows=16)
    calls: list[dict[str, object]] = []
    original = HistGradientBoostingRegressor.fit

    def spy(
        estimator: HistGradientBoostingRegressor,
        fit_x: object,
        fit_y: object,
        *args: object,
        **kwargs: object,
    ) -> HistGradientBoostingRegressor:
        calls.append(
            {
                "kwargs": dict(kwargs),
                "validation_fraction": estimator.validation_fraction,
            }
        )
        return original(estimator, fit_x, fit_y, *args, **kwargs)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", spy)
    audit = HistGradientBoostingModelAdapter(_fast_config()).fit(
        X, y, valid_x, valid_y
    )
    assert calls == [{"kwargs": {}, "validation_fraction": None}]
    assert audit.validation_provided
    assert not audit.validation_used_for_fit


def test_early_stopping_true_passes_only_external_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _data()
    valid_x, valid_y = _data(start=1000, rows=20)
    valid_x.iloc[0, 0] = np.nan
    before = valid_x.copy(deep=True)
    observed: dict[str, object] = {}
    original = HistGradientBoostingRegressor.fit

    def spy(
        estimator: HistGradientBoostingRegressor,
        fit_x: object,
        fit_y: object,
        *args: object,
        **kwargs: object,
    ) -> HistGradientBoostingRegressor:
        observed.update(kwargs)
        observed["validation_fraction"] = estimator.validation_fraction
        return original(estimator, fit_x, fit_y, *args, **kwargs)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", spy)
    adapter = HistGradientBoostingModelAdapter(
        _fast_config(early_stopping=True, n_iter_no_change=3)
    )
    audit = adapter.fit(X, y, valid_x, valid_y)
    assert observed["X_val"] is not X
    pdt.assert_frame_equal(observed["X_val"], valid_x)
    pdt.assert_series_equal(observed["y_val"], valid_y)
    assert observed["validation_fraction"] is None
    assert audit.validation_used_for_fit
    assert audit.validation_missing_counts[0] == ("factor_a", 1)
    pdt.assert_frame_equal(valid_x, before)


@pytest.mark.parametrize(
    ("provide_x", "provide_y"),
    [(False, False), (True, False), (False, True)],
)
def test_early_stopping_requires_both_validation_parts(
    provide_x: bool, provide_y: bool
) -> None:
    X, y = _data()
    valid_x, valid_y = _data(start=1000, rows=16)
    with pytest.raises((ModelConfigError, ModelDataError)):
        HistGradientBoostingModelAdapter(
            _fast_config(early_stopping=True)
        ).fit(
            X,
            y,
            valid_x if provide_x else None,
            valid_y if provide_y else None,
        )


def test_early_stopping_rejects_overlapping_indices() -> None:
    X, y = _data()
    valid_x, valid_y = _data(start=0, rows=12)
    with pytest.raises(ModelDataError, match="must not overlap"):
        HistGradientBoostingModelAdapter(
            _fast_config(early_stopping=True)
        ).fit(X, y, valid_x, valid_y)


@pytest.mark.parametrize(
    ("x_change", "y_change", "error"),
    [
        (
            lambda X: X[["factor_b", "factor_a", "factor_nan", "constant"]],
            lambda y: y,
            ModelFeatureMismatchError,
        ),
        (
            lambda X: X.assign(extra=1.0),
            lambda y: y,
            ModelFeatureMismatchError,
        ),
        (
            lambda X: X.assign(factor_a=np.inf),
            lambda y: y,
            ModelDataError,
        ),
        (
            lambda X: X,
            lambda y: y.mask(y.index == y.index[0]),
            ModelDataError,
        ),
        (
            lambda X: X,
            lambda y: y.mask(y.index == y.index[0], np.inf),
            ModelDataError,
        ),
    ],
)
def test_early_stopping_validation_contract(
    x_change: object,
    y_change: object,
    error: type[Exception],
) -> None:
    X, y = _data()
    valid_x, valid_y = _data(start=1000, rows=16)
    with pytest.raises(error):
        HistGradientBoostingModelAdapter(
            _fast_config(early_stopping=True)
        ).fit(X, y, x_change(valid_x), y_change(valid_y))


def test_unsupported_external_validation_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _data()
    valid_x, valid_y = _data(start=1000, rows=16)
    monkeypatch.setattr(tree_module, "_SUPPORTS_EXTERNAL_VALIDATION", False)
    with pytest.raises(ModelConfigError, match="internal validation is forbidden"):
        HistGradientBoostingModelAdapter(
            _fast_config(early_stopping=True)
        ).fit(X, y, valid_x, valid_y)


def test_validation_changes_do_not_change_training_structure_audit() -> None:
    X, y = _data()
    first_x, first_y = _data(start=1000, rows=16)
    second_x = first_x * 100000.0
    second_y = first_y * -100.0
    first = HistGradientBoostingModelAdapter(_fast_config())
    second = HistGradientBoostingModelAdapter(_fast_config())
    first_audit = first.fit(X, y, first_x, first_y)
    second_audit = second.fit(X, y, second_x, second_y)
    assert first_audit.train_missing_counts == second_audit.train_missing_counts
    assert first_audit.constant_features == second_audit.constant_features
    pdt.assert_series_equal(first.predict(X), second.predict(X))


def test_unfitted_prediction_is_rejected() -> None:
    with pytest.raises(ModelNotFittedError):
        HistGradientBoostingModelAdapter().predict(_data()[0])


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (
            lambda X: X.drop(columns="factor_a"),
            ModelFeatureMismatchError,
        ),
        (
            lambda X: X.assign(extra=1.0),
            ModelFeatureMismatchError,
        ),
        (
            lambda X: X[["factor_b", "factor_a", "factor_nan", "constant"]],
            ModelFeatureMismatchError,
        ),
        (
            lambda X: X.assign(factor_a=np.inf),
            ModelDataError,
        ),
    ],
)
def test_prediction_contract(
    change: object, error: type[Exception]
) -> None:
    X, y = _data()
    adapter = HistGradientBoostingModelAdapter(_fast_config())
    adapter.fit(X, y)
    with pytest.raises(error):
        adapter.predict(change(X))


def test_prediction_accepts_nan_preserves_index_and_input() -> None:
    X, y = _data()
    adapter = HistGradientBoostingModelAdapter(_fast_config())
    adapter.fit(X, y)
    predict_x = X.iloc[:7].copy()
    predict_x.iloc[0, :] = np.nan
    before = predict_x.copy(deep=True)
    output = adapter.predict(predict_x)
    assert output.name == "prediction"
    assert output.dtype == np.dtype("float64")
    assert output.index.equals(predict_x.index)
    assert len(output) == len(predict_x)
    assert np.isfinite(output).all()
    pdt.assert_frame_equal(predict_x, before)


def test_audit_and_metadata_are_complete_and_json_safe() -> None:
    X, y = _data()
    valid_x, valid_y = _data(start=1000, rows=16)
    config = _fast_config(learning_rate=0.04)
    adapter = HistGradientBoostingModelAdapter(config)
    audit = adapter.fit(X, y, valid_x, valid_y)
    assert audit.train_missing_counts == (
        ("factor_a", 0),
        ("factor_b", 0),
        ("factor_nan", 7),
        ("constant", 6),
    )
    assert dict(audit.resolved_parameters) == config.as_dict()
    assert dict(audit.preprocessing_parameters) == {
        "native_missing_support": True,
        "imputer": None,
        "scaler": None,
    }
    assert audit.python_version
    assert audit.numpy_version
    assert audit.sklearn_version
    json.dumps(audit.as_dict(), allow_nan=False)
    json.dumps(adapter.get_metadata(), allow_nan=False)


def test_feature_importance_errors_distinguish_state_and_capability() -> None:
    X, y = _data()
    adapter = HistGradientBoostingModelAdapter(_fast_config())
    with pytest.raises(ModelNotFittedError):
        adapter.get_feature_importance()
    adapter.fit(X, y)
    with pytest.raises(
        ModelFeatureImportanceUnavailableError,
        match="permutation importance.*V3-F",
    ):
        adapter.get_feature_importance()
