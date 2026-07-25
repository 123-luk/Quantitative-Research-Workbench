"""Contract tests for configurable V3-C linear regression adapters."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.ml import (
    ElasticNetModelAdapter,
    ElasticNetModelConfig,
    ModelConfigError,
    ModelDataError,
    ModelFeatureMismatchError,
    ModelFitError,
    ModelNotFittedError,
    RidgeModelAdapter,
    RidgeModelConfig,
)


def _data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "factor_a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "factor_b": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "constant": [2.0] * 6,
        },
        index=pd.Index([10, 11, 12, 13, 14, 15], name="row"),
    )
    y = pd.Series(
        [0.5, 1.2, 1.8, 2.9, 3.7, 4.4],
        index=X.index,
        name="forward_return",
    )
    return X, y


@pytest.mark.parametrize(
    ("config_class", "expected"),
    [
        (
            RidgeModelConfig,
            {
                "alpha": 1.0,
                "fit_intercept": True,
                "solver": "auto",
                "tol": 1e-4,
                "max_iter": None,
                "positive": False,
                "random_state": None,
            },
        ),
        (
            ElasticNetModelConfig,
            {
                "alpha": 1.0,
                "l1_ratio": 0.5,
                "fit_intercept": True,
                "max_iter": 5000,
                "tol": 1e-4,
                "selection": "cyclic",
                "random_state": 42,
                "positive": False,
                "warm_start": False,
            },
        ),
    ],
)
def test_default_configs_are_complete_frozen_and_json_safe(
    config_class: type, expected: dict[str, object]
) -> None:
    config = config_class.from_dict(None)
    assert config.as_dict() == expected
    assert config.to_estimator_params() == expected
    json.dumps(config.as_dict(), allow_nan=False)
    json.dumps(
        [spec.as_dict() for spec in config.parameter_schema()],
        allow_nan=False,
    )
    with pytest.raises(FrozenInstanceError):
        config.alpha = 9.0


def test_ridge_all_parameters_are_overridable_and_normalized() -> None:
    config = RidgeModelConfig.from_dict(
        {
            "alpha": np.float64(2.5),
            "fit_intercept": False,
            "solver": "  L-BFGS  ".replace("-", ""),
            "tol": 1e-6,
            "max_iter": np.int64(200),
            "positive": True,
            "random_state": np.int64(7),
        }
    )
    assert config.as_dict() == {
        "alpha": 2.5,
        "fit_intercept": False,
        "solver": "lbfgs",
        "tol": 1e-6,
        "max_iter": 200,
        "positive": True,
        "random_state": 7,
    }


def test_elastic_net_all_parameters_are_overridable_and_normalized() -> None:
    config = ElasticNetModelConfig.from_dict(
        {
            "alpha": 0.2,
            "l1_ratio": 0.8,
            "fit_intercept": False,
            "max_iter": 8000,
            "tol": 1e-6,
            "selection": " RANDOM ",
            "random_state": 9,
            "positive": True,
            "warm_start": True,
        }
    )
    assert config.as_dict() == {
        "alpha": 0.2,
        "l1_ratio": 0.8,
        "fit_intercept": False,
        "max_iter": 8000,
        "tol": 1e-6,
        "selection": "random",
        "random_state": 9,
        "positive": True,
        "warm_start": True,
    }


@pytest.mark.parametrize("config_class", [RidgeModelConfig, ElasticNetModelConfig])
def test_unknown_config_parameter_is_rejected(config_class: type) -> None:
    with pytest.raises(ModelConfigError, match="unknown parameter.*alpah"):
        config_class.from_dict({"alpah": 1.0})


@pytest.mark.parametrize(
    ("factory", "params", "match"),
    [
        (RidgeModelConfig.from_dict, {"alpha": "1"}, "alpha"),
        (RidgeModelConfig.from_dict, {"alpha": np.inf}, "finite"),
        (RidgeModelConfig.from_dict, {"alpha": -0.1}, "alpha"),
        (RidgeModelConfig.from_dict, {"fit_intercept": 1}, "fit_intercept"),
        (RidgeModelConfig.from_dict, {"tol": 0.0}, "tol"),
        (RidgeModelConfig.from_dict, {"max_iter": 0}, "max_iter"),
        (RidgeModelConfig.from_dict, {"max_iter": True}, "max_iter"),
        (RidgeModelConfig.from_dict, {"random_state": False}, "random_state"),
        (RidgeModelConfig.from_dict, {"solver": "bad"}, "solver"),
        (
            RidgeModelConfig.from_dict,
            {"solver": "lbfgs", "positive": False},
            "requires positive",
        ),
        (
            RidgeModelConfig.from_dict,
            {"solver": "svd", "positive": True},
            "requires solver",
        ),
        (ElasticNetModelConfig.from_dict, {"alpha": 0.0}, "alpha"),
        (ElasticNetModelConfig.from_dict, {"l1_ratio": 0.0}, "l1_ratio"),
        (ElasticNetModelConfig.from_dict, {"l1_ratio": 1.1}, "l1_ratio"),
        (ElasticNetModelConfig.from_dict, {"max_iter": 0}, "max_iter"),
        (ElasticNetModelConfig.from_dict, {"max_iter": True}, "max_iter"),
        (ElasticNetModelConfig.from_dict, {"tol": 0.0}, "tol"),
        (ElasticNetModelConfig.from_dict, {"selection": "bad"}, "selection"),
        (ElasticNetModelConfig.from_dict, {"random_state": True}, "random_state"),
        (ElasticNetModelConfig.from_dict, {"positive": 1}, "positive"),
        (ElasticNetModelConfig.from_dict, {"warm_start": 0}, "warm_start"),
    ],
)
def test_invalid_parameters_are_rejected(
    factory: object, params: dict[str, object], match: str
) -> None:
    with pytest.raises(ModelConfigError, match=match):
        factory(params)


@pytest.mark.parametrize(
    "adapter", [RidgeModelAdapter(), ElasticNetModelAdapter()]
)
def test_normal_fit_predict_audit_metadata_and_importance(adapter: object) -> None:
    X, y = _data()
    X_before = X.copy(deep=True)
    y_before = y.copy(deep=True)

    audit = adapter.fit(X, y)
    prediction = adapter.predict(X)
    importance = adapter.get_feature_importance()
    metadata = adapter.get_metadata()

    assert adapter.is_fitted
    assert adapter.feature_names == tuple(X.columns)
    assert audit.model_name == adapter.model_name
    assert audit.n_train_rows == 6
    assert audit.n_validation_rows == 0
    assert audit.train_missing_counts == (
        ("factor_a", 1),
        ("factor_b", 0),
        ("constant", 0),
    )
    assert audit.imputation_values[0] == ("factor_a", 4.0)
    assert audit.constant_features == ("constant",)
    assert isinstance(audit.estimator_intercept, float)
    assert not audit.native_missing_support
    assert audit.imputer_enabled
    assert audit.scaler_enabled
    assert audit.best_iteration is None
    assert audit.n_iterations is None
    assert not audit.validation_provided
    assert not audit.validation_used_for_fit
    assert dict(audit.resolved_parameters) == adapter.config.as_dict()
    assert prediction.name == "prediction"
    assert prediction.dtype == np.dtype("float64")
    assert prediction.index.equals(X.index)
    assert np.isfinite(prediction).all()
    assert list(importance.columns) == [
        "feature_name",
        "feature_position",
        "coefficient",
        "abs_coefficient",
        "importance_rank",
        "direction",
    ]
    assert importance["feature_name"].tolist() == list(X.columns)
    assert importance["feature_position"].tolist() == [0, 1, 2]
    assert importance["importance_rank"].sort_values().tolist() == [1, 2, 3]
    assert set(importance["direction"]) <= {"positive", "negative", "zero"}
    assert np.isfinite(importance["coefficient"]).all()
    json.dumps(audit.as_dict(), allow_nan=False)
    json.dumps(metadata, allow_nan=False)
    pdt.assert_frame_equal(X, X_before)
    pdt.assert_series_equal(y, y_before)


@pytest.mark.parametrize(
    "method_name", ["predict", "get_feature_importance", "get_metadata"]
)
def test_fitted_methods_reject_unfitted_adapter(method_name: str) -> None:
    adapter = RidgeModelAdapter()
    assert adapter.feature_names == ()
    method = getattr(adapter, method_name)
    args = (_data()[0],) if method_name == "predict" else ()
    with pytest.raises(ModelNotFittedError, match="ridge"):
        method(*args)


@pytest.mark.parametrize(
    ("x_transform", "y_transform", "error", "match"),
    [
        (lambda X: X.to_numpy(), lambda y: y, ModelDataError, "DataFrame"),
        (lambda X: X, lambda y: y.to_frame(), ModelDataError, "Series"),
        (lambda X: X.iloc[:1], lambda y: y.iloc[:1], ModelDataError, "at least 2"),
        (lambda X: X.iloc[:, :0], lambda y: y, ModelDataError, "feature"),
        (
            lambda X: X.set_axis(["a", "a", "c"], axis=1),
            lambda y: y,
            ModelDataError,
            "columns must be unique",
        ),
        (
            lambda X: X.rename(columns={"factor_a": " "}),
            lambda y: y,
            ModelDataError,
            "non-empty strings",
        ),
        (
            lambda X: X.rename(columns={"factor_a": "prediction"}),
            lambda y: y,
            ModelDataError,
            "reserved",
        ),
        (
            lambda X: X.assign(factor_a=["bad"] * len(X)),
            lambda y: y,
            ModelDataError,
            "non-numeric",
        ),
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
            lambda y: y.iloc[:-1],
            ModelDataError,
            "length",
        ),
        (
            lambda X: X,
            lambda y: y.set_axis(range(len(y))),
            ModelDataError,
            "index",
        ),
        (
            lambda X: X.set_axis([10, 10, 12, 13, 14, 15]),
            lambda y: y,
            ModelDataError,
            "index must be unique",
        ),
        (
            lambda X: X,
            lambda y: y.set_axis([10, 10, 12, 13, 14, 15]),
            ModelDataError,
            "index must be unique",
        ),
    ],
)
def test_training_input_contract(
    x_transform: object,
    y_transform: object,
    error: type[Exception],
    match: str,
) -> None:
    X, y = _data()
    changed_x = x_transform(X)
    changed_y = y_transform(y)
    with pytest.raises(error, match=match):
        RidgeModelAdapter().fit(changed_x, changed_y)


def test_validation_is_audited_but_never_used_for_fit() -> None:
    X, y = _data()
    valid_x = pd.DataFrame(
        {
            "factor_a": [np.nan, 1000000.0],
            "factor_b": [-1000000.0, np.nan],
            "constant": [2.0, 2.0],
        },
        index=[20, 21],
    )
    valid_y = pd.Series([1.0, 2.0], index=valid_x.index)
    valid_before = valid_x.copy(deep=True)
    first = RidgeModelAdapter()
    second = RidgeModelAdapter()

    audit = first.fit(X, y, valid_x, valid_y)
    changed_valid_y = pd.Series([999.0, -999.0], index=valid_x.index)
    second_audit = second.fit(X, y, valid_x * 100.0, changed_valid_y)

    assert audit.validation_missing_counts == (
        ("factor_a", 1),
        ("factor_b", 1),
        ("constant", 0),
    )
    assert audit.imputation_values == second_audit.imputation_values
    assert audit.scaler_means == second_audit.scaler_means
    assert audit.scaler_scales == second_audit.scaler_scales
    pdt.assert_frame_equal(
        first.get_feature_importance(),
        second.get_feature_importance(),
    )
    assert not audit.validation_used_for_fit
    assert np.isfinite(first.predict(valid_x)).all()
    pdt.assert_frame_equal(valid_x, valid_before)


@pytest.mark.parametrize(
    ("valid_x_change", "valid_y_change", "error", "match"),
    [
        (lambda X: X, lambda y: None, ModelDataError, "provided together"),
        (lambda X: None, lambda y: y, ModelDataError, "provided together"),
        (
            lambda X: X.drop(columns="factor_a"),
            lambda y: y,
            ModelFeatureMismatchError,
            "exactly match",
        ),
        (
            lambda X: X.assign(extra=1.0),
            lambda y: y,
            ModelFeatureMismatchError,
            "exactly match",
        ),
        (
            lambda X: X[["factor_b", "factor_a", "constant"]],
            lambda y: y,
            ModelFeatureMismatchError,
            "exactly match",
        ),
        (
            lambda X: X,
            lambda y: y.set_axis(range(len(y))),
            ModelDataError,
            "index",
        ),
        (
            lambda X: X,
            lambda y: y.mask(y.index == y.index[0]),
            ModelDataError,
            "missing",
        ),
    ],
)
def test_validation_contract(
    valid_x_change: object,
    valid_y_change: object,
    error: type[Exception],
    match: str,
) -> None:
    X, y = _data()
    valid_x = X.iloc[:2].copy()
    valid_x.index = [20, 21]
    valid_y = y.iloc[:2].copy()
    valid_y.index = valid_x.index
    with pytest.raises(error, match=match):
        RidgeModelAdapter().fit(
            X,
            y,
            valid_x_change(valid_x),
            valid_y_change(valid_y),
        )


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda X: X.drop(columns="factor_a"), ModelFeatureMismatchError),
        (lambda X: X.assign(extra=1.0), ModelFeatureMismatchError),
        (
            lambda X: X[["factor_b", "factor_a", "constant"]],
            ModelFeatureMismatchError,
        ),
        (lambda X: X.assign(factor_a=np.inf), ModelDataError),
    ],
)
def test_prediction_contract(change: object, error: type[Exception]) -> None:
    X, y = _data()
    adapter = RidgeModelAdapter()
    adapter.fit(X, y)
    with pytest.raises(error):
        adapter.predict(change(X))


def test_prediction_accepts_missing_values_and_does_not_modify_input() -> None:
    X, y = _data()
    adapter = ElasticNetModelAdapter()
    adapter.fit(X, y)
    prediction_x = X.iloc[:2].copy()
    prediction_x.iloc[:, :] = np.nan
    before = prediction_x.copy(deep=True)
    output = adapter.predict(prediction_x)
    assert np.isfinite(output).all()
    pdt.assert_frame_equal(prediction_x, before)


def test_refit_rebuilds_statistics_and_failed_initial_fit_stays_unfitted() -> None:
    X, y = _data()
    adapter = RidgeModelAdapter()
    first = adapter.fit(X, y)
    shifted = X.copy()
    shifted["factor_a"] += 100.0
    second = adapter.fit(shifted, y)
    assert first.imputation_values != second.imputation_values

    fresh = RidgeModelAdapter()
    with pytest.raises(ModelDataError):
        fresh.fit(X.assign(factor_a=np.inf), y)
    assert not fresh.is_fitted
    assert fresh.feature_names == ()


def test_returned_importance_is_defensive_and_ties_are_stable() -> None:
    X = pd.DataFrame({"first": [1.0, 2.0, 3.0], "second": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    adapter = RidgeModelAdapter()
    adapter.fit(X, y)
    importance = adapter.get_feature_importance()
    assert importance["importance_rank"].tolist() == [1, 2]
    importance.loc[0, "coefficient"] = 999.0
    assert adapter.get_feature_importance().loc[0, "coefficient"] != 999.0


def test_parameters_change_behavior_and_are_not_overwritten() -> None:
    X, y = _data()
    weak = RidgeModelAdapter(RidgeModelConfig(alpha=0.0))
    strong = RidgeModelAdapter(RidgeModelConfig(alpha=1000.0))
    weak_audit = weak.fit(X, y)
    strong_audit = strong.fit(X, y)
    assert dict(strong_audit.resolved_parameters)["alpha"] == 1000.0
    assert not np.allclose(weak.predict(X), strong.predict(X))


def test_elastic_net_convergence_warning_is_a_fit_error() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(80, 20)), columns=[f"factor_{i}" for i in range(20)]
    )
    y = pd.Series(rng.normal(size=80))
    adapter = ElasticNetModelAdapter(
        ElasticNetModelConfig(alpha=1e-8, max_iter=1, tol=1e-15)
    )
    with pytest.raises(ModelFitError, match="max_iter.*tol.*alpha"):
        adapter.fit(X, y)
    assert not adapter.is_fitted
