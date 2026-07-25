"""Tests for the V3 in-memory ML experiment orchestration layer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pandas as pd
import pandas.testing as pdt
import pytest

from src.ml import (
    MLExperimentAudit,
    MLExperimentConfig,
    MLExperimentConfigError,
    MLExperimentDataError,
    MLExperimentResult,
    MLExperimentRunner,
    MLExperimentStageError,
    MLDatasetConfig,
    ModelEvaluationConfig,
    PermutationImportanceOptionsConfig,
    WalkForwardConfig,
    WalkForwardTrainingConfig,
)


def _frame(periods: int = 16, stocks: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, date in enumerate(
        pd.date_range("2024-01-01", periods=periods, freq="D")
    ):
        for stock_number in range(stocks):
            factor_a = float(date_number + stock_number)
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": f"S{stock_number:02d}",
                    "factor_a": factor_a,
                    "factor_b": float(stock_number - date_number / 10),
                    "entry_trade_date": date + pd.Timedelta(days=1),
                    "exit_trade_date": date + pd.Timedelta(days=2),
                    "forward_return": factor_a / 100.0
                    + (stock_number % 2) / 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _config(
    model_name: str = "ridge",
    params: dict[str, object] | None = None,
    *,
    importance: bool = False,
) -> MLExperimentConfig:
    return MLExperimentConfig(
        dataset_config=MLDatasetConfig(),
        walk_forward_config=WalkForwardConfig(
            train_window_periods=2,
            validation_periods=2,
            window_type="rolling",
            retrain_frequency=3,
            embargo_periods=1,
        ),
        training_config=WalkForwardTrainingConfig(model_name, params),
        evaluation_config=ModelEvaluationConfig(
            minimum_cross_section_size=3
        ),
        permutation_importance=(
            PermutationImportanceOptionsConfig(n_repeats=2, random_state=7)
            if importance
            else None
        ),
    )


def test_importance_options_defaults_custom_frozen_and_json() -> None:
    default = PermutationImportanceOptionsConfig()
    assert default.as_dict() == {
        "scoring": "rmse",
        "n_repeats": 5,
        "random_state": 42,
        "permutation_scope": "within_trade_date",
    }
    custom = PermutationImportanceOptionsConfig.from_dict(
        {
            "scoring": " MAE ",
            "n_repeats": 1,
            "random_state": 0,
            "permutation_scope": " within_trade_date ",
        }
    )
    assert custom.scoring == "mae"
    json.dumps(custom.as_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        custom.scoring = "rmse"  # type: ignore[misc]


@pytest.mark.parametrize(
    "values",
    [
        {"scoring": "r2"},
        {"n_repeats": 0},
        {"n_repeats": True},
        {"random_state": -1},
        {"random_state": False},
        {"permutation_scope": "global"},
        {"extra": 1},
    ],
)
def test_importance_options_reject_invalid(values: dict[str, object]) -> None:
    with pytest.raises(MLExperimentConfigError):
        PermutationImportanceOptionsConfig.from_dict(values)


def test_experiment_config_direct_and_minimal_from_dict() -> None:
    direct = _config()
    assert direct.permutation_importance is None
    config = MLExperimentConfig.from_dict(
        {
            "walk_forward": {
                "train_window_periods": 2,
                "validation_periods": 2,
                "retrain_frequency": 3,
                "embargo_periods": 1,
            },
            "training": {
                "model_name": "ridge",
                "model_params": {"alpha": 2.0},
            },
        }
    )
    assert config.dataset_config == MLDatasetConfig()
    assert config.evaluation_config == ModelEvaluationConfig()
    assert config.permutation_importance is None
    assert config.training_config.as_dict()["model_params"] == {"alpha": 2.0}
    json.dumps(config.as_dict(), allow_nan=False)


def test_experiment_config_full_nested_and_defensive() -> None:
    source = {
        "dataset": {"label_col": "forward_return"},
        "walk_forward": {
            "train_window_periods": 2,
            "validation_periods": 2,
            "window_type": "expanding",
            "retrain_frequency": 3,
            "embargo_periods": 1,
        },
        "training": {
            "model_name": "hist_gradient_boosting",
            "model_params": {
                "learning_rate": 0.03,
                "max_iter": 20,
                "max_depth": 7,
                "min_samples_leaf": 2,
                "early_stopping": True,
                "random_state": 123,
            },
        },
        "evaluation": {"minimum_cross_section_size": 4},
        "permutation_importance": {
            "scoring": "mae",
            "n_repeats": 2,
            "random_state": 9,
            "permutation_scope": "within_trade_date",
        },
    }
    config = MLExperimentConfig.from_dict(source)
    source["training"]["model_params"]["max_depth"] = 99  # type: ignore[index]
    returned = config.as_dict()
    returned["training"]["model_params"]["max_depth"] = 88  # type: ignore[index]
    assert config.training_config.as_dict()["model_params"]["max_depth"] == 7  # type: ignore[index]
    assert config.permutation_importance is not None
    assert config.permutation_importance.scoring == "mae"


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"walk_forward": {}, "training": {"model_name": "ridge"}},
        {
            "walk_forward": {
                "train_window_periods": 2,
                "validation_periods": 2,
            }
        },
        {
            "walk_forward": {
                "train_window_periods": 2,
                "validation_periods": 2,
            },
            "training": {"model_name": "ridge"},
            "unknown": 1,
        },
        {
            "walk_forward": "path",
            "training": {"model_name": "ridge"},
        },
        {
            "walk_forward": {
                "train_window_periods": 2,
                "validation_periods": 2,
            },
            "training": [],
        },
        {
            "walk_forward": {
                "train_window_periods": 2,
                "validation_periods": 2,
            },
            "training": {"model_name": "ridge"},
            "dataset": [],
        },
    ],
)
def test_experiment_config_rejects_missing_unknown_or_wrong_nested(
    values: dict[str, object],
) -> None:
    with pytest.raises(MLExperimentConfigError):
        MLExperimentConfig.from_dict(values)


@pytest.mark.parametrize(
    ("model", "params"),
    [
        ("ridge", {"alpha": 1.0}),
        (
            "elastic_net",
            {"alpha": 0.05, "l1_ratio": 0.2, "max_iter": 1000},
        ),
        (
            "hist_gradient_boosting",
            {
                "max_iter": 10,
                "min_samples_leaf": 2,
                "early_stopping": False,
                "random_state": 3,
            },
        ),
    ],
)
def test_real_experiment_without_importance(
    model: str, params: dict[str, object]
) -> None:
    result = MLExperimentRunner().run(_frame(), _config(model, params))
    assert isinstance(result, MLExperimentResult)
    assert result.permutation_importance_result is None
    assert result.audit.evaluation_completed
    assert not result.audit.permutation_importance_enabled
    assert result.audit.stage_sequence == (
        "dataset_build",
        "walk_forward_split",
        "training",
        "evaluation",
        "integrity_validation",
        "result_build",
    )


@pytest.mark.parametrize(
    ("model", "params"),
    [
        ("ridge", {"alpha": 2.0}),
        ("elastic_net", {"alpha": 0.05, "l1_ratio": 0.2}),
        (
            "hist_gradient_boosting",
            {
                "max_iter": 10,
                "min_samples_leaf": 2,
                "early_stopping": False,
                "random_state": 3,
            },
        ),
    ],
)
def test_real_experiment_with_importance_uses_same_model_config(
    model: str, params: dict[str, object]
) -> None:
    result = MLExperimentRunner().run(
        _frame(), _config(model, params, importance=True)
    )
    importance = result.permutation_importance_result
    assert importance is not None
    training_audit = result.training_result.audit
    assert importance.audit.model_name == training_audit.model_name
    assert (
        importance.audit.resolved_model_parameters
        == training_audit.resolved_model_parameters
    )
    assert result.audit.permutation_importance_enabled
    assert result.audit.permutation_importance_completed
    assert result.audit.permutation_importance_scoring == "rmse"
    assert result.audit.permutation_importance_n_repeats == 2
    assert result.audit.stage_sequence[-3:] == (
        "permutation_importance",
        "integrity_validation",
        "result_build",
    )


def test_experiment_audit_exact_counts_config_and_json() -> None:
    frame = _frame()
    config = _config(importance=True)
    result = MLExperimentRunner().run(frame, config)
    audit = result.audit
    training = result.training_result.audit
    assert isinstance(audit, MLExperimentAudit)
    assert audit.model_name == training.model_name
    assert audit.resolved_model_parameters == training.resolved_model_parameters
    assert audit.input_rows == len(frame)
    assert audit.dataset_rows == result.dataset_audit.output_rows
    assert audit.n_features == 2
    assert audit.feature_names == ("factor_a", "factor_b")
    assert audit.n_folds == training.n_folds
    assert audit.n_prediction_rows == training.n_prediction_rows
    assert audit.n_prediction_dates == training.n_prediction_dates
    assert audit.first_prediction_date == training.first_prediction_date
    assert audit.last_prediction_date == training.last_prediction_date
    assert audit.config == config
    json.dumps(audit.as_dict(), allow_nan=False)


def test_result_memory_boundaries_and_nested_defensive_behavior() -> None:
    result = MLExperimentRunner().run(_frame(), _config(importance=True))
    assert not hasattr(result, "frame")
    assert not hasattr(result, "dataset")
    assert not hasattr(result, "registry")
    assert not hasattr(result, "model")
    first_plan = result.walk_forward_plan
    second_plan = result.walk_forward_plan
    assert first_plan is not second_plan
    predictions = result.training_result.predictions
    predictions.iloc[0, predictions.columns.get_loc("prediction")] = 999.0
    assert result.training_result.predictions.iloc[0]["prediction"] != 999.0
    dates = result.evaluation_result.date_metrics
    dates.loc[0, "n_obs"] = 999
    assert result.evaluation_result.date_metrics.loc[0, "n_obs"] != 999
    importance = result.permutation_importance_result
    assert importance is not None
    table = importance.feature_importance
    table.loc[0, "importance_rank"] = 999
    assert importance.feature_importance.loc[0, "importance_rank"] != 999


def test_repeat_runs_are_independent_deterministic_and_input_unchanged() -> None:
    frame = _frame()
    before = frame.copy(deep=True)
    runner = MLExperimentRunner()
    first = runner.run(frame, _config(importance=True))
    second = runner.run(frame, _config(importance=True))
    pdt.assert_frame_equal(frame, before)
    pdt.assert_frame_equal(
        first.training_result.predictions,
        second.training_result.predictions,
    )
    assert first.audit.as_dict() == second.audit.as_dict()
    assert not hasattr(runner, "result")
    assert not hasattr(runner, "dataset")


@pytest.mark.parametrize("bad", [None, [], "frame", 1])
def test_run_rejects_non_frame_input(bad: object) -> None:
    with pytest.raises(MLExperimentDataError):
        MLExperimentRunner().run(bad, _config())  # type: ignore[arg-type]


def test_run_rejects_empty_frame_config_and_registry_types() -> None:
    with pytest.raises(MLExperimentDataError):
        MLExperimentRunner().run(pd.DataFrame(), _config())
    with pytest.raises(MLExperimentConfigError):
        MLExperimentRunner().run(_frame(), object())  # type: ignore[arg-type]
    with pytest.raises(MLExperimentConfigError):
        MLExperimentRunner(object())  # type: ignore[arg-type]


def test_dataset_stage_failure_is_wrapped_and_chained() -> None:
    frame = _frame().drop(columns="forward_return")
    with pytest.raises(MLExperimentStageError) as caught:
        MLExperimentRunner().run(frame, _config())
    assert caught.value.stage == "dataset_build"
    assert "dataset_build" in str(caught.value)
    assert caught.value.__cause__ is not None


def test_split_stage_failure_is_wrapped_and_chained() -> None:
    config = MLExperimentConfig(
        MLDatasetConfig(),
        WalkForwardConfig(100, 20),
        WalkForwardTrainingConfig("ridge"),
        ModelEvaluationConfig(),
    )
    with pytest.raises(MLExperimentStageError) as caught:
        MLExperimentRunner().run(_frame(), config)
    assert caught.value.stage == "walk_forward_split"
    assert caught.value.__cause__ is not None


def test_training_stage_failure_is_wrapped_and_chained() -> None:
    with pytest.raises(MLExperimentStageError) as caught:
        MLExperimentRunner().run(
            _frame(), _config("ridge", {"not_a_parameter": 1})
        )
    assert caught.value.stage == "training"
    assert caught.value.__cause__ is not None


def test_importance_options_cannot_override_model() -> None:
    with pytest.raises(MLExperimentConfigError, match="unknown"):
        MLExperimentConfig.from_dict(
            {
                "walk_forward": {
                    "train_window_periods": 2,
                    "validation_periods": 2,
                },
                "training": {"model_name": "ridge"},
                "permutation_importance": {"model_name": "elastic_net"},
            }
        )
