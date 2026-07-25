"""Tests for strict walk-forward OOS permutation importance."""

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
    PermutationImportanceFoldAudit,
    WalkForwardConfig,
    WalkForwardPermutationImportanceAudit,
    WalkForwardPermutationImportanceConfig,
    WalkForwardPermutationImportanceConfigError,
    WalkForwardPermutationImportanceDataError,
    WalkForwardPermutationImportanceFoldError,
    WalkForwardPermutationImportanceIntegrityError,
    WalkForwardPermutationImportanceResult,
    WalkForwardPermutationImportanceRunner,
    WalkForwardPlan,
    WalkForwardSplitter,
)
from src.ml.importance import _permute_within_trade_date, _score


REPEAT_COLUMNS = [
    "fold_id",
    "feature_name",
    "feature_position",
    "repeat_id",
    "baseline_score",
    "permuted_score",
    "importance",
]
FOLD_COLUMNS = [
    "fold_id",
    "feature_name",
    "feature_position",
    "baseline_score",
    "importance_mean",
    "importance_std",
    "importance_min",
    "importance_max",
    "positive_fraction",
    "n_repeats",
]
FEATURE_COLUMNS = [
    "feature_name",
    "feature_position",
    "importance_mean",
    "importance_std",
    "importance_median",
    "importance_min",
    "importance_max",
    "positive_fraction",
    "n_folds",
    "n_observations",
    "importance_rank",
]


def _dataset(periods: int = 16, stocks: int = 4) -> MLDataset:
    factor_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for date_number, date in enumerate(
        pd.date_range("2024-01-01", periods=periods, freq="D")
    ):
        for stock_number in range(stocks):
            code = f"S{stock_number:02d}"
            factor_a = float(date_number + stock_number)
            factor_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "factor_a": factor_a,
                    "factor_b": float(stock_number - date_number / 5),
                }
            )
            label_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "entry_trade_date": date + pd.Timedelta(days=1),
                    "exit_trade_date": date + pd.Timedelta(days=2),
                    "forward_return": factor_a / 100.0
                    + float(stock_number % 2) / 1000.0,
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
    params: dict[str, object] | None = None,
    *,
    scoring: str = "rmse",
    n_repeats: int = 2,
    random_state: int = 42,
    dataset: MLDataset | None = None,
    plan: WalkForwardPlan | None = None,
    runner: WalkForwardPermutationImportanceRunner | None = None,
) -> WalkForwardPermutationImportanceResult:
    data = dataset or _dataset()
    split_plan = plan or _plan(data)
    config = WalkForwardPermutationImportanceConfig(
        model_name,
        params,
        scoring,
        n_repeats,
        random_state,
    )
    return (runner or WalkForwardPermutationImportanceRunner()).run(
        data, split_plan, config
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


class _Proxy:
    def __init__(self, adapter: object, registry: "_TrackingRegistry", ordinal: int):
        self.adapter = adapter
        self.registry = registry
        self.ordinal = ordinal

    def fit(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        valid_x: pd.DataFrame | None = None,
        valid_y: pd.Series | None = None,
    ) -> object:
        self.registry.fit_calls.append(
            (
                self.ordinal,
                tuple(train_x.index),
                tuple(train_y.index),
                None if valid_x is None else tuple(valid_x.index),
                None if valid_y is None else tuple(valid_y.index),
            )
        )
        if self.registry.fail_fit == self.ordinal:
            raise ValueError("injected fit failure")
        return self.adapter.fit(train_x, train_y, valid_x, valid_y)  # type: ignore[attr-defined]

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        call_number = len(self.registry.predict_calls)
        self.registry.predict_calls.append(
            (self.ordinal, tuple(frame.index), frame.copy(deep=True))
        )
        if self.registry.fail_predict == call_number:
            raise RuntimeError("injected prediction failure")
        return self.adapter.predict(frame)  # type: ignore[attr-defined]


class _TrackingRegistry(ModelRegistry):
    def __init__(
        self,
        *,
        fail_create: int | None = None,
        fail_fit: int | None = None,
        fail_predict: int | None = None,
    ) -> None:
        super().__init__()
        self.fail_create = fail_create
        self.fail_fit = fail_fit
        self.fail_predict = fail_predict
        self.create_calls: list[tuple[str, dict[str, object]]] = []
        self.fit_calls: list[tuple[object, ...]] = []
        self.predict_calls: list[tuple[int, tuple[int, ...], pd.DataFrame]] = []
        self.proxies: list[_Proxy] = []

    def create(  # type: ignore[override]
        self, model_name: str, params: dict[str, object] | None = None
    ) -> object:
        ordinal = len(self.create_calls)
        self.create_calls.append((model_name, dict(params or {})))
        if self.fail_create == ordinal:
            raise LookupError("injected create failure")
        proxy = _Proxy(super().create(model_name, params), self, ordinal)
        self.proxies.append(proxy)
        return proxy


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("ridge", {"alpha": 2.0}),
        ("elastic_net", {"alpha": 0.05, "l1_ratio": 0.3}),
        (
            "hist_gradient_boosting",
            {
                "max_iter": 10,
                "min_samples_leaf": 2,
                "early_stopping": False,
                "random_state": 7,
            },
        ),
    ],
)
def test_config_supported_models_normalization_and_json(
    model_name: str, params: dict[str, object]
) -> None:
    config = WalkForwardPermutationImportanceConfig(
        f" {model_name.upper()} ", params
    )
    assert config.model_name == model_name
    assert config.scoring == "rmse"
    assert config.permutation_scope == "within_trade_date"
    json.dumps(config.as_dict(), allow_nan=False)


def test_config_defensive_copy_from_dict_and_frozen() -> None:
    params = {"alpha": 2.0}
    config = WalkForwardPermutationImportanceConfig.from_dict(
        {
            "model_name": "ridge",
            "model_params": params,
            "scoring": " MAE ",
            "n_repeats": 1,
            "random_state": 0,
            "permutation_scope": " within_trade_date ",
        }
    )
    params["alpha"] = 9.0
    returned = config.as_dict()
    returned["model_params"]["alpha"] = 8.0  # type: ignore[index]
    assert config.as_dict()["model_params"] == {"alpha": 2.0}
    assert config.scoring == "mae"
    with pytest.raises(FrozenInstanceError):
        config.scoring = "rmse"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": ""},
        {"model_name": "ridge", "model_params": []},
        {"model_name": "ridge", "model_params": {"bad": {"x": 1}}},
        {"model_name": "ridge", "model_params": {"bad": lambda: None}},
        {"model_name": "ridge", "model_params": {"bad": np.inf}},
        {"model_name": "ridge", "scoring": "r2"},
        {"model_name": "ridge", "n_repeats": 0},
        {"model_name": "ridge", "n_repeats": True},
        {"model_name": "ridge", "random_state": -1},
        {"model_name": "ridge", "random_state": False},
        {"model_name": "ridge", "permutation_scope": "global"},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(WalkForwardPermutationImportanceConfigError):
        WalkForwardPermutationImportanceConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_unknown_or_missing_fields() -> None:
    with pytest.raises(WalkForwardPermutationImportanceConfigError, match="unknown"):
        WalkForwardPermutationImportanceConfig.from_dict(
            {"model_name": "ridge", "extra": 1}
        )
    with pytest.raises(WalkForwardPermutationImportanceConfigError, match="requires"):
        WalkForwardPermutationImportanceConfig.from_dict({})


def test_score_formulas_and_nonfinite_rejection() -> None:
    target = pd.Series([1.0, 2.0, 4.0])
    prediction = pd.Series([2.0, 2.0, 1.0])
    residual = target - prediction
    assert _score(target, prediction, "rmse") == pytest.approx(
        np.sqrt(np.mean(residual**2))
    )
    assert _score(target, prediction, "mae") == pytest.approx(
        np.mean(np.abs(residual))
    )
    with pytest.raises(WalkForwardPermutationImportanceIntegrityError):
        _score(target, pd.Series([np.inf, 1.0, 1.0]), "rmse")


def test_within_date_permutation_contract_and_rng_isolation() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 10.0, 11.0, 99.0],
            "b": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        },
        index=[10, 11, 12, 20, 21, 30],
    )
    dates = pd.Series(
        pd.to_datetime(
            ["2024-01-01"] * 3 + ["2024-01-02"] * 2 + ["2024-01-03"]
        ),
        index=frame.index,
    )
    before = frame.copy(deep=True)
    state_before = np.random.get_state()
    first = _permute_within_trade_date(frame, dates, "a", 0, 0, 0, 42)
    second = _permute_within_trade_date(frame, dates, "a", 0, 0, 0, 42)
    different = _permute_within_trade_date(frame, dates, "a", 0, 0, 1, 42)
    state_after = np.random.get_state()
    pdt.assert_frame_equal(frame, before)
    pdt.assert_frame_equal(first, second)
    pdt.assert_series_equal(first["b"], frame["b"])
    assert first.index.equals(frame.index)
    assert list(first.columns) == list(frame.columns)
    for date in dates.unique():
        positions = dates.eq(date)
        left = first.loc[positions, "a"]
        right = frame.loc[positions, "a"]
        assert sorted(left.dropna()) == sorted(right.dropna())
        assert left.isna().sum() == right.isna().sum()
    assert first.loc[30, "a"] == 99.0
    assert not first.equals(different)
    assert all(np.array_equal(a, b) for a, b in zip(state_before, state_after))


@pytest.mark.parametrize(
    ("model_name", "params"),
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
def test_real_models_multifold_are_deterministic(
    model_name: str, params: dict[str, object]
) -> None:
    first = _run(model_name, params)
    second = _run(model_name, params)
    assert isinstance(first, WalkForwardPermutationImportanceResult)
    pdt.assert_frame_equal(first.repeat_importance, second.repeat_importance)
    pdt.assert_frame_equal(first.feature_importance, second.feature_importance)


def test_single_fold_ridge_and_mae() -> None:
    dataset = _dataset(periods=10)
    result = _run(
        scoring="mae",
        dataset=dataset,
        plan=_plan(dataset, retrain_frequency=5),
    )
    assert result.audit.n_folds == 1
    assert result.audit.scoring == "mae"


def test_repeat_table_exact_contract_formula_and_counts() -> None:
    result = _run(n_repeats=3)
    frame = result.repeat_importance
    audit = result.audit
    assert list(frame.columns) == REPEAT_COLUMNS
    assert isinstance(frame.index, pd.RangeIndex)
    assert len(frame) == audit.n_folds * audit.n_features * audit.n_repeats
    assert frame[["fold_id", "feature_position", "repeat_id"]].equals(
        frame[["fold_id", "feature_position", "repeat_id"]].sort_values(
            ["fold_id", "feature_position", "repeat_id"]
        ).reset_index(drop=True)
    )
    assert np.allclose(
        frame["importance"],
        frame["permuted_score"] - frame["baseline_score"],
    )
    assert np.isfinite(
        frame[["baseline_score", "permuted_score", "importance"]]
    ).all().all()
    assert frame.groupby("fold_id")["baseline_score"].nunique().eq(1).all()
    assert frame["feature_name"].unique().tolist() == ["factor_a", "factor_b"]


def test_fold_summary_exact_aggregation_and_single_repeat_std() -> None:
    result = _run(n_repeats=3)
    repeat = result.repeat_importance
    fold = result.fold_importance
    assert list(fold.columns) == FOLD_COLUMNS
    assert isinstance(fold.index, pd.RangeIndex)
    for row in fold.itertuples(index=False):
        values = repeat.loc[
            repeat["fold_id"].eq(row.fold_id)
            & repeat["feature_position"].eq(row.feature_position),
            "importance",
        ].to_numpy()
        assert row.importance_mean == pytest.approx(values.mean())
        assert row.importance_std == pytest.approx(values.std(ddof=1))
        assert row.importance_min == pytest.approx(values.min())
        assert row.importance_max == pytest.approx(values.max())
        assert row.positive_fraction == pytest.approx(np.mean(values > 0))
    one = _run(n_repeats=1).fold_importance
    assert one["importance_std"].isna().all()


def test_feature_summary_equal_weight_rank_and_counts() -> None:
    result = _run(n_repeats=3)
    repeat = result.repeat_importance
    feature = result.feature_importance
    assert list(feature.columns) == FEATURE_COLUMNS
    assert feature["feature_position"].tolist() == [0, 1]
    for row in feature.itertuples(index=False):
        values = repeat.loc[
            repeat["feature_position"].eq(row.feature_position), "importance"
        ].to_numpy()
        assert row.importance_mean == pytest.approx(values.mean())
        assert row.importance_std == pytest.approx(values.std(ddof=1))
        assert row.importance_median == pytest.approx(np.median(values))
        assert row.n_folds == result.audit.n_folds
        assert row.n_observations == result.audit.n_folds * 3
    ranked = feature.sort_values(
        ["importance_mean", "feature_position"], ascending=[False, True]
    )
    assert ranked["importance_rank"].tolist() == [1, 2]


def test_audits_fields_parameters_validation_and_json() -> None:
    result = _run("ridge", {"alpha": 2.5}, n_repeats=3)
    audit = result.audit
    assert isinstance(audit, WalkForwardPermutationImportanceAudit)
    assert audit.score_direction == "lower_is_better"
    assert audit.permutation_scope == "within_trade_date"
    assert audit.feature_names == ("factor_a", "factor_b")
    assert dict(audit.resolved_model_parameters)["alpha"] == 2.5
    assert audit.n_repeat_evaluations == len(result.repeat_importance)
    for fold in audit.fold_audits:
        assert isinstance(fold, PermutationImportanceFoldAudit)
        assert fold.n_permutations == fold.n_features * fold.n_repeats
        assert (
            fold.validation_used_for_fit
            == fold.model_fit_audit.validation_used_for_fit
        )
        assert not hasattr(fold, "indices")
        assert not hasattr(fold, "estimator")
    json.dumps(audit.as_dict(), allow_nan=False)
    json.dumps(result.as_dict(), allow_nan=False)


def test_result_tables_are_defensive_and_nan_serializes_to_none() -> None:
    result = _run(n_repeats=1)
    before = result.fold_importance
    changed = result.fold_importance
    changed.loc[0, "importance_mean"] = 999.0
    pdt.assert_frame_equal(result.fold_importance, before)
    changed = result.feature_importance
    changed.loc[0, "importance_rank"] = 99
    assert result.feature_importance.loc[0, "importance_rank"] != 99
    changed = result.repeat_importance
    changed.loc[0, "importance"] = 999.0
    assert result.repeat_importance.loc[0, "importance"] != 999.0
    assert result.as_dict()["fold_importance"][0]["importance_std"] is None  # type: ignore[index]
    assert not hasattr(result, "model")
    assert not hasattr(result, "dataset")


def test_fresh_adapter_call_counts_and_exact_partitions() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    registry = _TrackingRegistry()
    _run(
        n_repeats=2,
        dataset=dataset,
        plan=plan,
        runner=WalkForwardPermutationImportanceRunner(registry),
    )
    assert len(registry.create_calls) == len(plan.splits)
    assert len(registry.fit_calls) == len(plan.splits)
    assert len({id(proxy) for proxy in registry.proxies}) == len(plan.splits)
    expected_predictions = len(plan.splits) * (1 + dataset.n_features * 2)
    assert len(registry.predict_calls) == expected_predictions
    for split, call in zip(plan.splits, registry.fit_calls, strict=True):
        assert call[1] == split.train_indices
        assert call[2] == split.train_indices
        assert call[3] == split.validation_indices
        assert call[4] == split.validation_indices


def test_only_prediction_features_are_permuted_and_inputs_unchanged() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    features_before = dataset.features
    labels_before = dataset.labels
    metadata_before = dataset.metadata
    registry = _TrackingRegistry()
    _run(
        dataset=dataset,
        plan=plan,
        n_repeats=1,
        runner=WalkForwardPermutationImportanceRunner(registry),
    )
    pdt.assert_frame_equal(dataset.features, features_before)
    pdt.assert_series_equal(dataset.labels, labels_before)
    pdt.assert_frame_equal(dataset.metadata, metadata_before)
    first_fold_calls = [
        call for call in registry.predict_calls if call[0] == 0
    ]
    assert first_fold_calls[0][1] == plan.splits[0].prediction_indices
    assert all(call[1] == plan.splits[0].prediction_indices for call in first_fold_calls)


def test_constant_features_zero_importance_and_negative_values_are_not_clipped() -> None:
    dataset = _dataset()
    features = dataset.features
    features["factor_b"] = 1.0
    constant = MLDataset(
        features,
        dataset.labels,
        dataset.metadata,
        dataset.feature_names,
        dataset.label_name,
        dataset.audit,
    )
    result = _run(dataset=constant, plan=_plan(constant), n_repeats=2)
    values = result.repeat_importance.loc[
        result.repeat_importance["feature_name"].eq("factor_b"), "importance"
    ]
    assert np.allclose(values, 0.0)
    assert (
        result.fold_importance.loc[
            result.fold_importance["feature_name"].eq("factor_b"),
            "positive_fraction",
        ]
        == 0.0
    ).all()
    assert not (result.repeat_importance["importance"] < 0).any() or (
        result.repeat_importance["importance"].min() < 0
    )


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("dataset", WalkForwardPermutationImportanceDataError),
        ("plan", WalkForwardPermutationImportanceDataError),
        ("config", WalkForwardPermutationImportanceConfigError),
        ("registry", WalkForwardPermutationImportanceConfigError),
    ],
)
def test_public_inputs_require_contract_types(
    target: str, error: type[Exception]
) -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    config = WalkForwardPermutationImportanceConfig("ridge")
    if target == "registry":
        with pytest.raises(error):
            WalkForwardPermutationImportanceRunner(object())  # type: ignore[arg-type]
        return
    args: list[object] = [dataset, plan, config]
    args[{"dataset": 0, "plan": 1, "config": 2}[target]] = object()
    with pytest.raises(error):
        WalkForwardPermutationImportanceRunner().run(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "values", "error"),
    [
        ("train_indices", (), WalkForwardPermutationImportanceDataError),
        ("prediction_indices", (), WalkForwardPermutationImportanceDataError),
        ("train_indices", (-1,), WalkForwardPermutationImportanceDataError),
        ("train_indices", (999,), WalkForwardPermutationImportanceDataError),
        ("train_indices", (0, 0), WalkForwardPermutationImportanceIntegrityError),
    ],
)
def test_invalid_plan_indices_are_rejected(
    field: str, values: tuple[int, ...], error: type[Exception]
) -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    split = _forge(plan.splits[0], **{field: values})
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
def test_plan_partition_overlap_is_rejected(left: str, right: str) -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    split = plan.splits[0]
    replacement = (getattr(split, left)[0],) + getattr(split, right)[1:]
    forged_split = _forge(split, **{right: replacement})
    with pytest.raises(WalkForwardPermutationImportanceIntegrityError, match="overlap"):
        _run(
            dataset=dataset,
            plan=_forge(plan, splits=(forged_split,) + plan.splits[1:]),
        )


def test_empty_plan_cross_section_and_cutoff_are_rejected() -> None:
    dataset = _dataset()
    plan = _plan(dataset)
    with pytest.raises(WalkForwardPermutationImportanceDataError):
        _run(dataset=dataset, plan=_forge(plan, splits=()))
    first = plan.splits[0]
    incomplete = _forge(
        first, prediction_indices=first.prediction_indices[1:]
    )
    with pytest.raises(WalkForwardPermutationImportanceIntegrityError):
        _run(
            dataset=dataset,
            plan=_forge(plan, splits=(incomplete,) + plan.splits[1:]),
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
    with pytest.raises(WalkForwardPermutationImportanceIntegrityError, match="cutoff"):
        _run(dataset=forged_dataset, plan=plan)


@pytest.mark.parametrize(
    ("failure", "cause"),
    [
        ("create", LookupError),
        ("fit", ValueError),
        ("baseline", RuntimeError),
        ("permutation", RuntimeError),
    ],
)
def test_fold_failures_are_contextual_atomic_and_chained(
    failure: str, cause: type[Exception]
) -> None:
    registry = _TrackingRegistry(
        fail_create=0 if failure == "create" else None,
        fail_fit=0 if failure == "fit" else None,
        fail_predict=0
        if failure == "baseline"
        else (1 if failure == "permutation" else None),
    )
    with pytest.raises(WalkForwardPermutationImportanceFoldError) as caught:
        _run(runner=WalkForwardPermutationImportanceRunner(registry))
    message = str(caught.value)
    assert "fold 0 failed for model ridge" in message
    assert "train=" in message and "validation=" in message and "prediction=" in message
    assert f"cause={cause.__name__}" in message
    assert isinstance(caught.value.__cause__, cause)
    if failure == "permutation":
        assert "feature_name=factor_a" in message
        assert "repeat_id=0" in message


@pytest.mark.parametrize(
    ("model", "params"),
    [("unknown", {}), ("ridge", {"not_a_parameter": 1})],
)
def test_registry_rejections_are_fold_errors(
    model: str, params: dict[str, object]
) -> None:
    with pytest.raises(WalkForwardPermutationImportanceFoldError) as caught:
        _run(model, params)
    assert "fold 0" in str(caught.value)
    assert caught.value.__cause__ is not None
