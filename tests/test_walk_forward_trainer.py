"""Tests for strict walk-forward model training and OOS prediction assembly."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import json
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.ml import (
    MLDataset,
    MLDatasetBuilder,
    ModelRegistry,
    WalkForwardConfig,
    WalkForwardFoldAudit,
    WalkForwardFoldError,
    WalkForwardPlan,
    WalkForwardSplit,
    WalkForwardSplitter,
    WalkForwardTrainer,
    WalkForwardTrainingAudit,
    WalkForwardTrainingConfig,
    WalkForwardTrainingConfigError,
    WalkForwardTrainingDataError,
    WalkForwardTrainingIntegrityError,
    WalkForwardTrainingResult,
)


OUTPUT_COLUMNS = [
    "trade_date",
    "ts_code",
    "entry_trade_date",
    "exit_trade_date",
    "target",
    "prediction",
    "fold_id",
]


def _dataset(periods: int = 16, stocks: int = 2) -> MLDataset:
    factor_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for date_number, trade_date in enumerate(
        pd.date_range("2024-01-01", periods=periods, freq="D")
    ):
        for stock_number in range(stocks):
            code = f"S{stocks - stock_number:02d}"
            factor_a = float(date_number + stock_number / 10)
            factor_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "factor_a": factor_a,
                    "factor_b": float((date_number % 4) - stock_number),
                }
            )
            label_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "entry_trade_date": trade_date + pd.Timedelta(days=1),
                    "exit_trade_date": trade_date + pd.Timedelta(days=2),
                    "forward_return": 0.01 * factor_a
                    + 0.002 * float(stock_number),
                }
            )
    return MLDatasetBuilder().build(
        pd.DataFrame(factor_rows),
        pd.DataFrame(label_rows),
        ("factor_a", "factor_b"),
    )


def _plan(dataset: MLDataset, **overrides: object) -> WalkForwardPlan:
    values: dict[str, object] = {
        "train_window_periods": 2,
        "validation_periods": 2,
        "window_type": "rolling",
        "retrain_frequency": 3,
        "embargo_periods": 1,
    }
    values.update(overrides)
    return WalkForwardSplitter(
        WalkForwardConfig(**values)  # type: ignore[arg-type]
    ).build(dataset)


def _run(
    model_name: str = "ridge",
    model_params: dict[str, object] | None = None,
    *,
    dataset: MLDataset | None = None,
    plan: WalkForwardPlan | None = None,
    trainer: WalkForwardTrainer | None = None,
) -> WalkForwardTrainingResult:
    data = dataset or _dataset()
    split_plan = plan or _plan(data)
    return (trainer or WalkForwardTrainer()).run(
        data,
        split_plan,
        WalkForwardTrainingConfig(model_name, model_params),
    )


def _forge(value: Any, **overrides: object) -> Any:
    forged = object.__new__(type(value))
    for field in fields(value):
        object.__setattr__(
            forged,
            field.name,
            overrides.get(field.name, getattr(value, field.name)),
        )
    return forged


class _AdapterProxy:
    def __init__(
        self,
        adapter: object,
        owner: "_TrackingRegistry",
        ordinal: int,
    ) -> None:
        self._adapter = adapter
        self._owner = owner
        self._ordinal = ordinal

    def fit(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        valid_x: pd.DataFrame | None = None,
        valid_y: pd.Series | None = None,
    ) -> object:
        self._owner.fit_calls.append(
            (
                self._ordinal,
                tuple(train_x.index),
                tuple(train_y.index),
                None if valid_x is None else tuple(valid_x.index),
                None if valid_y is None else tuple(valid_y.index),
            )
        )
        if self._owner.fail_fit_at == self._ordinal:
            raise ValueError("injected fit failure")
        return self._adapter.fit(train_x, train_y, valid_x, valid_y)  # type: ignore[attr-defined]

    def predict(self, prediction_x: pd.DataFrame) -> pd.Series:
        self._owner.predict_calls.append(
            (self._ordinal, tuple(prediction_x.index))
        )
        if self._owner.fail_predict_at == self._ordinal:
            raise RuntimeError("injected prediction failure")
        return self._adapter.predict(prediction_x)  # type: ignore[attr-defined]


class _TrackingRegistry(ModelRegistry):
    def __init__(
        self,
        *,
        fail_create_at: int | None = None,
        fail_fit_at: int | None = None,
        fail_predict_at: int | None = None,
    ) -> None:
        super().__init__()
        self.fail_create_at = fail_create_at
        self.fail_fit_at = fail_fit_at
        self.fail_predict_at = fail_predict_at
        self.create_calls: list[tuple[str, dict[str, object]]] = []
        self.adapters: list[_AdapterProxy] = []
        self.fit_calls: list[tuple[object, ...]] = []
        self.predict_calls: list[tuple[object, ...]] = []

    def create(  # type: ignore[override]
        self,
        model_name: str,
        params: dict[str, object] | None = None,
    ) -> object:
        ordinal = len(self.create_calls)
        self.create_calls.append((model_name, dict(params or {})))
        if self.fail_create_at == ordinal:
            raise LookupError("injected create failure")
        proxy = _AdapterProxy(
            super().create(model_name, params),
            self,
            ordinal,
        )
        self.adapters.append(proxy)
        return proxy


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("ridge", {"alpha": 2.0}),
        ("elastic_net", {"alpha": 0.05, "l1_ratio": 0.3}),
        (
            "hist_gradient_boosting",
            {
                "max_iter": 12,
                "min_samples_leaf": 2,
                "early_stopping": False,
                "random_state": 7,
            },
        ),
    ],
)
def test_config_normalizes_supported_model_names_and_is_json_safe(
    model_name: str, params: dict[str, object]
) -> None:
    config = WalkForwardTrainingConfig(f"  {model_name.upper()}  ", params)
    assert config.model_name == model_name
    assert config.as_dict()["model_params"] == params
    json.dumps(config.as_dict(), allow_nan=False)


def test_config_none_is_empty_and_copies_all_mappings() -> None:
    source = {"alpha": 2.0}
    configured = WalkForwardTrainingConfig("ridge", source)
    source["alpha"] = 9.0
    returned = configured.as_dict()
    returned["model_params"]["alpha"] = 7.0  # type: ignore[index]
    assert configured.as_dict() == {
        "model_name": "ridge",
        "model_params": {"alpha": 2.0},
    }
    assert WalkForwardTrainingConfig("ridge").as_dict()["model_params"] == {}
    with pytest.raises(TypeError):
        configured.model_params["alpha"] = 3.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        configured.model_name = "elastic_net"  # type: ignore[misc]


@pytest.mark.parametrize("name", ["", "   ", None, 1])
def test_config_rejects_invalid_model_name(name: object) -> None:
    with pytest.raises(WalkForwardTrainingConfigError, match="model_name"):
        WalkForwardTrainingConfig(name)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "params",
    [
        [],
        lambda: None,
        {"bad": lambda: None},
        {"bad": pd.DataFrame({"x": [1]})},
        {"bad": pd.Series([1])},
        {"bad": np.array([1])},
        {"bad": [1]},
        {"bad": (1,)},
        {"bad": {"nested": 1}},
        {"bad": np.inf},
        {"bad": np.nan},
    ],
)
def test_config_rejects_non_mapping_or_non_scalar_values(params: object) -> None:
    with pytest.raises(WalkForwardTrainingConfigError):
        WalkForwardTrainingConfig("ridge", params)  # type: ignore[arg-type]


def test_config_from_dict_rejects_unknown_and_missing_fields() -> None:
    with pytest.raises(WalkForwardTrainingConfigError, match="unknown"):
        WalkForwardTrainingConfig.from_dict(
            {"model_name": "ridge", "extra": 1}
        )
    with pytest.raises(WalkForwardTrainingConfigError, match="requires"):
        WalkForwardTrainingConfig.from_dict({})
    with pytest.raises(WalkForwardTrainingConfigError, match="Mapping"):
        WalkForwardTrainingConfig.from_dict([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("ridge", {"alpha": 1.5}),
        (
            "elastic_net",
            {
                "alpha": 0.05,
                "l1_ratio": 0.2,
                "max_iter": 1000,
                "random_state": 3,
            },
        ),
        (
            "hist_gradient_boosting",
            {
                "max_iter": 12,
                "min_samples_leaf": 2,
                "early_stopping": False,
                "random_state": 3,
            },
        ),
    ],
)
def test_real_models_run_multiple_folds_deterministically(
    model_name: str, params: dict[str, object]
) -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    first = _run(model_name, params, dataset=dataset, plan=plan)
    second = _run(model_name, params, dataset=dataset, plan=plan)
    pdt.assert_frame_equal(first.predictions, second.predictions)
    assert isinstance(first, WalkForwardTrainingResult)
    assert isinstance(first.audit, WalkForwardTrainingAudit)
    assert first.audit.n_folds == len(plan.splits) > 1


def test_single_fold_ridge_run() -> None:
    dataset = _dataset(periods=10)
    plan = _plan(dataset, retrain_frequency=5)
    result = _run(dataset=dataset, plan=plan)
    assert result.audit.n_folds == 1


def test_prediction_frame_has_exact_schema_types_keys_and_alignment() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    result = _run(dataset=dataset, plan=plan)
    predictions = result.predictions
    expected_indices = sorted(
        index for split in plan.splits for index in split.prediction_indices
    )
    assert list(predictions.columns) == OUTPUT_COLUMNS
    assert predictions.index.tolist() == expected_indices
    assert predictions.index.name == "dataset_index"
    assert predictions.index.is_unique
    assert predictions.index.is_monotonic_increasing
    assert predictions["prediction"].dtype == np.dtype("float64")
    assert predictions["target"].dtype == np.dtype("float64")
    assert predictions["fold_id"].dtype == np.dtype("int64")
    assert np.isfinite(predictions[["target", "prediction"]]).all().all()
    assert not predictions.duplicated(["trade_date", "ts_code"]).any()
    pdt.assert_frame_equal(
        predictions.loc[:, OUTPUT_COLUMNS[:4]],
        dataset.metadata.iloc[expected_indices].rename_axis("dataset_index"),
    )
    pdt.assert_series_equal(
        predictions["target"],
        dataset.labels.iloc[expected_indices]
        .rename("target")
        .rename_axis("dataset_index"),
    )
    assert not set(dataset.feature_names) & set(predictions.columns)
    assert predictions["ts_code"].iloc[:2].tolist() == ["S01", "S02"]


def test_prediction_frame_and_audit_are_defensive_and_sample_free() -> None:
    result = _run()
    original = result.predictions
    changed = result.predictions
    changed.loc[changed.index[0], "prediction"] = 999.0
    pdt.assert_frame_equal(result.predictions, original)
    audit_dict = result.audit.as_dict()
    json.dumps(audit_dict, allow_nan=False)
    assert "predictions" not in audit_dict
    assert "indices" not in json.dumps(audit_dict)
    assert not hasattr(result, "model")
    assert not hasattr(result, "dataset")
    assert not hasattr(result, "plan")


def test_fold_and_overall_audits_match_plan_and_fit_audits() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    config = WalkForwardTrainingConfig("ridge", {"alpha": 2.5})
    result = WalkForwardTrainer().run(dataset, plan, config)
    audit = result.audit
    assert audit.model_name == "ridge"
    assert dict(audit.resolved_model_parameters)["alpha"] == 2.5
    assert audit.n_prediction_rows == len(result.predictions)
    assert audit.n_prediction_dates == result.predictions["trade_date"].nunique()
    assert audit.first_prediction_date == plan.first_prediction_date
    assert audit.last_prediction_date == plan.last_prediction_date
    assert audit.source_label_name == "forward_return"
    assert [fold.fold_id for fold in audit.fold_audits] == list(
        range(len(plan.splits))
    )
    for split, fold in zip(plan.splits, audit.fold_audits, strict=True):
        assert isinstance(fold, WalkForwardFoldAudit)
        assert fold.train_rows == split.n_train_rows
        assert fold.validation_rows == split.n_validation_rows
        assert fold.prediction_rows == split.n_prediction_rows
        assert fold.train_start_date == split.train_start_date
        assert fold.validation_start_date == split.validation_start_date
        assert fold.prediction_end_date == split.prediction_end_date
        assert fold.validation_provided
        assert (
            fold.validation_used_for_fit
            == fold.model_fit_audit.validation_used_for_fit
        )
        assert not hasattr(fold, "indices")
        assert not hasattr(fold, "estimator")
        json.dumps(fold.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"alpha": 2.0}, False),
        (
            {
                "max_iter": 10,
                "min_samples_leaf": 2,
                "early_stopping": False,
            },
            False,
        ),
        (
            {
                "max_iter": 10,
                "min_samples_leaf": 2,
                "early_stopping": True,
                "n_iter_no_change": 3,
            },
            True,
        ),
    ],
)
def test_validation_usage_comes_from_real_fit_audit(
    params: dict[str, object], expected: bool
) -> None:
    model_name = (
        "ridge" if set(params) == {"alpha"} else "hist_gradient_boosting"
    )
    result = _run(model_name, params)
    assert all(fold.validation_provided for fold in result.audit.fold_audits)
    assert all(
        fold.validation_used_for_fit is expected
        for fold in result.audit.fold_audits
    )


def test_each_fold_and_repeated_run_create_fresh_adapters_once() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    registry = _TrackingRegistry()
    trainer = WalkForwardTrainer(registry)
    trainer.run(
        dataset,
        plan,
        WalkForwardTrainingConfig("ridge", {"alpha": 2.0}),
    )
    trainer.run(
        dataset,
        plan,
        WalkForwardTrainingConfig("ridge", {"alpha": 2.0}),
    )
    expected = 2 * len(plan.splits)
    assert len(registry.create_calls) == expected
    assert len(registry.adapters) == expected
    assert len({id(adapter) for adapter in registry.adapters}) == expected
    assert len(registry.fit_calls) == expected
    assert len(registry.predict_calls) == expected
    assert all(call == ("ridge", {"alpha": 2.0}) for call in registry.create_calls)


def test_adapter_receives_only_each_split_partitions() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    registry = _TrackingRegistry()
    _run(dataset=dataset, plan=plan, trainer=WalkForwardTrainer(registry))
    for split, fit_call, predict_call in zip(
        plan.splits,
        registry.fit_calls,
        registry.predict_calls,
        strict=True,
    ):
        assert fit_call[1] == split.train_indices
        assert fit_call[2] == split.train_indices
        assert fit_call[3] == split.validation_indices
        assert fit_call[4] == split.validation_indices
        assert predict_call[1] == split.prediction_indices


def test_prediction_targets_do_not_affect_predictions() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    prediction_indices = [
        index for split in plan.splits for index in split.prediction_indices
    ]
    labels = dataset.labels
    labels.iloc[prediction_indices] += 1000.0
    changed = MLDataset(
        dataset.features,
        labels,
        dataset.metadata,
        dataset.feature_names,
        dataset.label_name,
        dataset.audit,
    )
    baseline = _run(dataset=dataset, plan=plan).predictions
    altered = _run(dataset=changed, plan=plan).predictions
    pdt.assert_series_equal(baseline["prediction"], altered["prediction"])
    assert not baseline["target"].equals(altered["target"])


def test_later_fold_data_do_not_affect_first_fold_prediction() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    first = plan.splits[0]
    used = set(first.train_indices + first.validation_indices + first.prediction_indices)
    later_only = [index for index in range(dataset.n_samples) if index not in used]
    features = dataset.features
    features.loc[later_only, :] += 10000.0
    changed = MLDataset(
        features,
        dataset.labels,
        dataset.metadata,
        dataset.feature_names,
        dataset.label_name,
        dataset.audit,
    )
    baseline = _run(dataset=dataset, plan=plan).predictions
    altered = _run(dataset=changed, plan=plan).predictions
    first_indices = list(first.prediction_indices)
    pdt.assert_series_equal(
        baseline.loc[first_indices, "prediction"],
        altered.loc[first_indices, "prediction"],
    )


def test_run_does_not_modify_dataset_plan_or_config() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    config = WalkForwardTrainingConfig("ridge", {"alpha": 2.0})
    features = dataset.features
    labels = dataset.labels
    metadata = dataset.metadata
    plan_dict = plan.summary()
    config_dict = config.as_dict()
    WalkForwardTrainer().run(dataset, plan, config)
    pdt.assert_frame_equal(dataset.features, features)
    pdt.assert_series_equal(dataset.labels, labels)
    pdt.assert_frame_equal(dataset.metadata, metadata)
    assert plan.summary() == plan_dict
    assert config.as_dict() == config_dict


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("dataset", WalkForwardTrainingDataError),
        ("plan", WalkForwardTrainingDataError),
        ("config", WalkForwardTrainingConfigError),
        ("registry", WalkForwardTrainingConfigError),
    ],
)
def test_public_inputs_require_contract_types(
    target: str, error: type[Exception]
) -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    config = WalkForwardTrainingConfig("ridge")
    if target == "registry":
        with pytest.raises(error):
            WalkForwardTrainer(object())  # type: ignore[arg-type]
        return
    args: list[object] = [dataset, plan, config]
    args[{"dataset": 0, "plan": 1, "config": 2}[target]] = object()
    with pytest.raises(error):
        WalkForwardTrainer().run(*args)  # type: ignore[arg-type]


def test_empty_plan_is_rejected_before_model_creation() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    forged = _forge(plan, splits=())
    registry = _TrackingRegistry()
    with pytest.raises(WalkForwardTrainingDataError, match="at least one"):
        _run(dataset=dataset, plan=forged, trainer=WalkForwardTrainer(registry))
    assert registry.create_calls == []


@pytest.mark.parametrize(
    ("partition", "indices", "error"),
    [
        ("train_indices", (), WalkForwardTrainingDataError),
        ("prediction_indices", (), WalkForwardTrainingDataError),
        ("train_indices", (-1,), WalkForwardTrainingDataError),
        ("train_indices", (999,), WalkForwardTrainingDataError),
        ("train_indices", (0, 0), WalkForwardTrainingIntegrityError),
    ],
)
def test_invalid_partition_indices_are_rejected(
    partition: str,
    indices: tuple[int, ...],
    error: type[Exception],
) -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    split = _forge(plan.splits[0], **{partition: indices})
    forged = _forge(plan, splits=(split,) + plan.splits[1:])
    with pytest.raises(error):
        _run(dataset=dataset, plan=forged)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("train_indices", "validation_indices"),
        ("train_indices", "prediction_indices"),
        ("validation_indices", "prediction_indices"),
    ],
)
def test_partition_overlap_is_rejected(left: str, right: str) -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    split = plan.splits[0]
    overrides = {
        right: (getattr(split, left)[0],) + tuple(getattr(split, right)[1:])
    }
    forged = _forge(
        plan, splits=(_forge(split, **overrides),) + plan.splits[1:]
    )
    with pytest.raises(WalkForwardTrainingIntegrityError, match="overlap"):
        _run(dataset=dataset, plan=forged)


@pytest.mark.parametrize(
    ("partition", "date_field"),
    [
        ("train_indices", "train_dates"),
        ("validation_indices", "validation_dates"),
        ("prediction_indices", "prediction_dates"),
    ],
)
def test_incomplete_date_cross_section_is_rejected(
    partition: str, date_field: str
) -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    split = plan.splits[0]
    shortened = tuple(getattr(split, partition))[1:]
    forged_split = _forge(split, **{partition: shortened})
    forged = _forge(plan, splits=(forged_split,) + plan.splits[1:])
    with pytest.raises(
        WalkForwardTrainingIntegrityError, match="complete date cross-sections"
    ):
        _run(dataset=dataset, plan=forged)


def test_cross_fold_prediction_index_and_date_overlap_are_rejected() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    second = plan.splits[1]
    first = plan.splits[0]
    repeated_index = _forge(
        second,
        prediction_indices=first.prediction_indices,
    )
    with pytest.raises(WalkForwardTrainingIntegrityError, match="overlap"):
        _run(
            dataset=dataset,
            plan=_forge(
                plan,
                splits=(first, repeated_index) + plan.splits[2:],
            ),
        )
    repeated_date = _forge(
        second,
        prediction_dates=first.prediction_dates,
    )
    with pytest.raises(WalkForwardTrainingIntegrityError):
        _run(
            dataset=dataset,
            plan=_forge(
                plan,
                splits=(first, repeated_date) + plan.splits[2:],
            ),
        )


def test_plan_date_universe_boundary_and_gap_are_rejected() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    with pytest.raises(WalkForwardTrainingIntegrityError, match="all_score_dates"):
        _run(
            dataset=dataset,
            plan=_forge(plan, all_score_dates=plan.all_score_dates[:-1]),
        )
    with pytest.raises(WalkForwardTrainingIntegrityError):
        _run(
            dataset=dataset,
            plan=_forge(
                plan,
                first_prediction_date=pd.Timestamp("1999-01-01"),
            ),
        )
    with pytest.raises(WalkForwardTrainingIntegrityError, match="boundaries"):
        _run(
            dataset=dataset,
            plan=_forge(
                plan,
                last_prediction_date=plan.first_prediction_date,
            ),
        )
    with pytest.raises(WalkForwardTrainingIntegrityError, match="skipped"):
        _run(
            dataset=dataset,
            plan=_forge(plan, skipped_initial_prediction_dates=()),
        )


def test_split_date_boundaries_are_rechecked_against_metadata() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    first = _forge(
        plan.splits[0], train_start_date=pd.Timestamp("2020-01-01")
    )
    with pytest.raises(
        WalkForwardTrainingIntegrityError, match="train_start_date"
    ):
        _run(
            dataset=dataset,
            plan=_forge(plan, splits=(first,) + plan.splits[1:]),
        )


def test_temporal_order_and_exit_cutoffs_are_rechecked() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    first = plan.splits[0]
    bad_order = _forge(
        first,
        train_indices=first.validation_indices,
        train_dates=first.validation_dates,
        validation_indices=first.train_indices,
        validation_dates=first.train_dates,
    )
    with pytest.raises(WalkForwardTrainingIntegrityError):
        _run(
            dataset=dataset,
            plan=_forge(plan, splits=(bad_order,) + plan.splits[1:]),
        )
    metadata = dataset.metadata
    metadata.loc[
        list(first.train_indices), "exit_trade_date"
    ] = first.validation_start_date
    forged_dataset = MLDataset(
        dataset.features,
        dataset.labels,
        metadata,
        dataset.feature_names,
        dataset.label_name,
        dataset.audit,
    )
    with pytest.raises(WalkForwardTrainingIntegrityError, match="cutoff"):
        _run(dataset=forged_dataset, plan=plan)


@pytest.mark.parametrize(
    ("failure", "cause_type"),
    [
        ("create", LookupError),
        ("fit", ValueError),
        ("predict", RuntimeError),
    ],
)
def test_fold_failures_are_atomic_contextual_and_chained(
    failure: str, cause_type: type[Exception]
) -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    registry = _TrackingRegistry(
        fail_create_at=1 if failure == "create" else None,
        fail_fit_at=1 if failure == "fit" else None,
        fail_predict_at=1 if failure == "predict" else None,
    )
    with pytest.raises(WalkForwardFoldError) as caught:
        _run(dataset=dataset, plan=plan, trainer=WalkForwardTrainer(registry))
    message = str(caught.value)
    assert "fold 1 failed for model ridge" in message
    assert "train=" in message
    assert "validation=" in message
    assert "prediction=" in message
    assert f"cause={cause_type.__name__}" in message
    assert isinstance(caught.value.__cause__, cause_type)
    assert len(registry.create_calls) == 2


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("unknown", {}),
        ("ridge", {"unknown_parameter": 1}),
    ],
)
def test_registry_rejections_are_fold_errors(
    model_name: str, params: dict[str, object]
) -> None:
    with pytest.raises(WalkForwardFoldError) as caught:
        _run(model_name, params)
    assert "fold 0" in str(caught.value)
    assert caught.value.__cause__ is not None

