"""Tests for the explicit V3-C model registry and UI-facing parameter schema."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.ml import (
    ElasticNetModelAdapter,
    ElasticNetModelConfig,
    HistGradientBoostingModelAdapter,
    HistGradientBoostingModelConfig,
    ModelConfigError,
    ModelRegistry,
    ModelRegistryError,
    RegressionModelAdapter,
    RidgeModelAdapter,
    RidgeModelConfig,
    create_default_model_registry,
)


def test_default_registry_models_are_stable_and_complete() -> None:
    registry = ModelRegistry()
    assert registry.list_models() == (
        "elastic_net", "hist_gradient_boosting", "ridge"
    )


@pytest.mark.parametrize(
    ("name", "adapter_type", "config_type"),
    [
        ("ridge", RidgeModelAdapter, RidgeModelConfig),
        ("elastic_net", ElasticNetModelAdapter, ElasticNetModelConfig),
        (
            "hist_gradient_boosting",
            HistGradientBoostingModelAdapter,
            HistGradientBoostingModelConfig,
        ),
    ],
)
def test_create_defaults_returns_expected_adapter(
    name: str, adapter_type: type, config_type: type
) -> None:
    adapter = create_default_model_registry().create(name)
    assert isinstance(adapter, adapter_type)
    assert isinstance(adapter, RegressionModelAdapter)
    assert isinstance(adapter.config, config_type)


def test_create_normalizes_name_and_preserves_partial_overrides() -> None:
    registry = ModelRegistry()
    ridge = registry.create(" RIDGE ", {"alpha": 3.5, "tol": 1e-6})
    elastic = registry.create(
        " Elastic_Net ",
        {"alpha": 0.2, "l1_ratio": 0.8, "max_iter": 8000},
    )
    assert ridge.config.alpha == 3.5
    assert ridge.config.tol == 1e-6
    assert ridge.config.solver == "auto"
    assert elastic.config.alpha == 0.2
    assert elastic.config.l1_ratio == 0.8
    assert elastic.config.max_iter == 8000
    assert elastic.config.random_state == 42


@pytest.mark.parametrize("name", ["", " ", "unknown"])
def test_unknown_or_empty_model_is_rejected(name: str) -> None:
    with pytest.raises(ModelRegistryError):
        ModelRegistry().create(name)


def test_registry_does_not_ignore_bad_parameters() -> None:
    registry = ModelRegistry()
    with pytest.raises(ModelConfigError, match="unknown parameter"):
        registry.create("ridge", {"alpah": 2.0})
    with pytest.raises(ModelConfigError, match="alpha"):
        registry.create("elastic_net", {"alpha": "bad"})


def test_duplicate_registration_is_rejected() -> None:
    with pytest.raises(ModelRegistryError, match="already registered"):
        ModelRegistry().register("ridge", RidgeModelConfig, RidgeModelAdapter)


def test_empty_registry_is_an_explicit_extension_point() -> None:
    registry = ModelRegistry(include_builtin_models=False)
    assert registry.list_models() == ()
    registry.register("custom_ridge", RidgeModelConfig, RidgeModelAdapter)
    assert registry.list_models() == ("custom_ridge",)
    assert isinstance(registry.create("custom_ridge"), RidgeModelAdapter)


def test_arbitrary_estimator_callable_is_rejected() -> None:
    registry = ModelRegistry(include_builtin_models=False)
    with pytest.raises(ModelRegistryError, match="RegressionModelAdapter"):
        registry.register("unsafe", RidgeModelConfig, lambda: object())


@pytest.mark.parametrize(
    ("name", "expected_names"),
    [
        (
            "ridge",
            [
                "alpha",
                "fit_intercept",
                "solver",
                "tol",
                "max_iter",
                "positive",
                "random_state",
            ],
        ),
        (
            "elastic_net",
            [
                "alpha",
                "l1_ratio",
                "fit_intercept",
                "max_iter",
                "tol",
                "selection",
                "random_state",
                "positive",
                "warm_start",
            ],
        ),
        (
            "hist_gradient_boosting",
            [
                "loss",
                "quantile",
                "learning_rate",
                "max_iter",
                "max_leaf_nodes",
                "max_depth",
                "min_samples_leaf",
                "l2_regularization",
                "max_features",
                "max_bins",
                "early_stopping",
                "n_iter_no_change",
                "tol",
                "warm_start",
                "random_state",
                "verbose",
            ],
        ),
    ],
)
def test_parameter_schema_is_stable_complete_and_json_safe(
    name: str, expected_names: list[str]
) -> None:
    registry = ModelRegistry()
    first = registry.get_parameter_schema(name)
    second = registry.get_parameter_schema(name)
    assert tuple(spec.name for spec in first) == tuple(expected_names)
    assert first == second
    payload = [spec.as_dict() for spec in first]
    json.dumps(payload, allow_nan=False)
    assert all(
        {
            "name",
            "display_name",
            "value_type",
            "default",
            "description",
            "advanced",
            "ui_control",
        }
        <= set(item)
        for item in payload
    )


@pytest.mark.parametrize("name", ["ridge", "elastic_net"])
def test_default_parameters_are_complete_and_defensive(name: str) -> None:
    registry = ModelRegistry()
    defaults = registry.get_default_parameters(name)
    schema_defaults = {
        spec.name: spec.default for spec in registry.get_parameter_schema(name)
    }
    assert defaults == schema_defaults
    defaults["alpha"] = 999.0
    assert registry.get_default_parameters(name)["alpha"] == 1.0


def test_all_public_parameters_can_flow_through_registry() -> None:
    registry = ModelRegistry()
    ridge_params = {
        "alpha": 2.0,
        "fit_intercept": False,
        "solver": "lbfgs",
        "tol": 1e-6,
        "max_iter": 500,
        "positive": True,
        "random_state": 5,
    }
    elastic_params = {
        "alpha": 0.2,
        "l1_ratio": 0.9,
        "fit_intercept": False,
        "max_iter": 7000,
        "tol": 1e-6,
        "selection": "random",
        "random_state": 8,
        "positive": True,
        "warm_start": True,
    }
    assert registry.create("ridge", ridge_params).config.as_dict() == ridge_params
    assert (
        registry.create("elastic_net", elastic_params).config.as_dict()
        == elastic_params
    )


    tree_params = {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_depth": 6,
        "early_stopping": False,
        "random_state": 123,
    }
    tree = registry.create("hist_gradient_boosting", tree_params)
    assert isinstance(tree, HistGradientBoostingModelAdapter)
    for name, value in tree_params.items():
        assert tree.config.as_dict()[name] == value


def test_tree_defaults_are_complete_and_defensive() -> None:
    registry = ModelRegistry()
    defaults = registry.get_default_parameters("hist_gradient_boosting")
    schema_defaults = {
        spec.name: spec.default
        for spec in registry.get_parameter_schema("hist_gradient_boosting")
    }
    assert defaults == schema_defaults
    defaults["max_iter"] = 999

def test_fit_audit_keeps_every_registry_resolved_parameter() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    y = pd.Series([1.0, 1.5, 2.5])
    adapter = ModelRegistry().create("ridge", {"alpha": 4.0})
    audit = adapter.fit(X, y)
    assert dict(audit.resolved_parameters) == adapter.config.as_dict()
    assert len(audit.resolved_parameters) == len(RidgeModelConfig().as_dict())


def test_new_default_registry_instances_are_independent() -> None:
    first = create_default_model_registry()
    second = create_default_model_registry()
    first.register("another_ridge", RidgeModelConfig, RidgeModelAdapter)
    assert "another_ridge" in first.list_models()
    assert "another_ridge" not in second.list_models()
