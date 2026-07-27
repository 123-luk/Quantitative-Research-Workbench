"""Tests for strict evaluation of frozen out-of-sample predictions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import json
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.ml import (
    CrossSectionalMetricSummary,
    MLDatasetBuilder,
    ModelEvaluationAudit,
    ModelEvaluationConfig,
    ModelEvaluationConfigError,
    ModelEvaluationDataError,
    ModelEvaluationIntegrityError,
    ModelEvaluationResult,
    OOSModelEvaluator,
    RegressionMetrics,
    WalkForwardConfig,
    WalkForwardSplitter,
    WalkForwardTrainer,
    WalkForwardTrainingConfig,
    WalkForwardTrainingResult,
)


DATE_COLUMNS = [
    "trade_date",
    "fold_id",
    "n_obs",
    "pearson_ic",
    "pearson_valid",
    "pearson_invalid_reason",
    "spearman_rank_ic",
    "rank_ic_valid",
    "rank_ic_invalid_reason",
]

FOLD_COLUMNS = [
    "fold_id",
    "start_date",
    "end_date",
    "n_rows",
    "n_dates",
    "mae",
    "rmse",
    "r2",
    "r2_valid",
    "r2_invalid_reason",
    "pearson_ic_mean",
    "pearson_ic_std",
    "pearson_icir",
    "pearson_valid_dates",
    "pearson_invalid_dates",
    "rank_ic_mean",
    "rank_ic_std",
    "rank_icir",
    "rank_ic_valid_dates",
    "rank_ic_invalid_dates",
]


def _forge(value: Any, **overrides: object) -> Any:
    forged = object.__new__(type(value))
    for field in fields(value):
        object.__setattr__(
            forged,
            field.name,
            overrides.get(field.name, getattr(value, field.name)),
        )
    return forged


def _training_result(periods: int = 16, stocks: int = 4) -> WalkForwardTrainingResult:
    factors: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for date_number, date in enumerate(
        pd.date_range("2024-01-01", periods=periods, freq="D")
    ):
        for stock_number in range(stocks):
            code = f"S{stock_number:02d}"
            value = float(date_number + stock_number)
            factors.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "factor_a": value,
                    "factor_b": float(stock_number - date_number / 10),
                }
            )
            labels.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "entry_trade_date": date + pd.Timedelta(days=1),
                    "exit_trade_date": date + pd.Timedelta(days=2),
                    "forward_return": value / 100.0,
                }
            )
    dataset = MLDatasetBuilder().build(
        pd.DataFrame(factors),
        pd.DataFrame(labels),
        ("factor_a", "factor_b"),
    )
    plan = WalkForwardSplitter(
        WalkForwardConfig(
            train_window_periods=2,
            validation_periods=2,
            window_type="rolling",
            retrain_frequency=3,
            embargo_periods=1,
        )
    ).build(dataset)
    return WalkForwardTrainer().run(
        dataset,
        plan,
        WalkForwardTrainingConfig("ridge", {"alpha": 1.0}),
    )


@pytest.fixture(scope="module")
def training_result() -> WalkForwardTrainingResult:
    return _training_result()


def _with_predictions(
    result: WalkForwardTrainingResult,
    predictions: pd.DataFrame,
    *,
    audit: object | None = None,
) -> WalkForwardTrainingResult:
    return WalkForwardTrainingResult(
        predictions,
        result.audit if audit is None else audit,  # type: ignore[arg-type]
    )


def _single_row_result(result: WalkForwardTrainingResult) -> WalkForwardTrainingResult:
    frame = result.predictions.iloc[[0]].copy()
    frame.index.name = "dataset_index"
    date = pd.Timestamp(frame["trade_date"].iloc[0])
    old_fold = result.audit.fold_audits[0]
    fold = _forge(
        old_fold,
        prediction_rows=1,
        prediction_start_date=date,
        prediction_end_date=date,
    )
    audit = _forge(
        result.audit,
        n_folds=1,
        n_prediction_rows=1,
        n_prediction_dates=1,
        first_prediction_date=date,
        last_prediction_date=date,
        fold_audits=(fold,),
    )
    frame["fold_id"] = np.int64(0)
    return _with_predictions(result, frame, audit=audit)


def test_config_defaults_custom_value_frozen_and_json_safe() -> None:
    default = ModelEvaluationConfig()
    assert default.minimum_cross_section_size == 3
    assert ModelEvaluationConfig.from_dict(None) == default
    custom = ModelEvaluationConfig.from_dict(
        {"minimum_cross_section_size": 5}
    )
    assert custom.as_dict() == {"minimum_cross_section_size": 5}
    json.dumps(custom.as_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        custom.minimum_cross_section_size = 3  # type: ignore[misc]


@pytest.mark.parametrize("value", [True, False, 1, 0, -1, 2.5, "3", None])
def test_config_rejects_invalid_cross_section_size(value: object) -> None:
    with pytest.raises(ModelEvaluationConfigError):
        ModelEvaluationConfig(value)  # type: ignore[arg-type]


def test_config_rejects_unknown_field_and_non_mapping() -> None:
    with pytest.raises(ModelEvaluationConfigError, match="unknown"):
        ModelEvaluationConfig.from_dict({"extra": 1})
    with pytest.raises(ModelEvaluationConfigError, match="Mapping"):
        ModelEvaluationConfig.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(ModelEvaluationConfigError):
        OOSModelEvaluator(object())  # type: ignore[arg-type]


def test_normal_evaluation_returns_all_contract_types(
    training_result: WalkForwardTrainingResult,
) -> None:
    result = OOSModelEvaluator().evaluate(training_result)
    assert isinstance(result, ModelEvaluationResult)
    assert isinstance(result.regression_metrics, RegressionMetrics)
    assert isinstance(
        result.pearson_ic_summary, CrossSectionalMetricSummary
    )
    assert isinstance(
        result.rank_ic_summary, CrossSectionalMetricSummary
    )
    assert isinstance(result.audit, ModelEvaluationAudit)
    assert list(result.date_metrics.columns) == DATE_COLUMNS
    assert list(result.fold_metrics.columns) == FOLD_COLUMNS


def test_overall_regression_formulas_and_perfect_prediction(
    training_result: WalkForwardTrainingResult,
) -> None:
    frame = training_result.predictions
    target = np.arange(len(frame), dtype=np.float64)
    frame["target"] = target
    frame["prediction"] = target
    metrics = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, frame)
    ).regression_metrics
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.r2_valid


def test_overall_regression_general_and_negative_r2(
    training_result: WalkForwardTrainingResult,
) -> None:
    frame = training_result.predictions
    target = np.arange(len(frame), dtype=np.float64)
    prediction = target + np.where(target % 2 == 0, 2.0, -1.0)
    frame["target"] = target
    frame["prediction"] = prediction
    metrics = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, frame)
    ).regression_metrics
    residual = target - prediction
    assert metrics.mae == pytest.approx(np.mean(np.abs(residual)))
    assert metrics.rmse == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert metrics.r2 == pytest.approx(
        1.0
        - np.sum(residual**2)
        / np.sum((target - np.mean(target)) ** 2)
    )
    frame["prediction"] = target + 1000.0
    negative = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, frame)
    ).regression_metrics
    assert negative.r2 is not None and negative.r2 < 0.0


def test_single_observation_and_constant_target_keep_mae_rmse(
    training_result: WalkForwardTrainingResult,
) -> None:
    single = _single_row_result(training_result)
    single_frame = single.predictions
    single_frame["target"] = 1.0
    single_frame["prediction"] = 3.0
    single_result = OOSModelEvaluator().evaluate(
        _with_predictions(single, single_frame)
    ).regression_metrics
    assert single_result.mae == pytest.approx(2.0)
    assert single_result.rmse == pytest.approx(2.0)
    assert not single_result.r2_valid
    assert single_result.r2 is None
    assert single_result.r2_invalid_reason == "insufficient_observations"

    frame = training_result.predictions
    frame["target"] = 2.0
    frame["prediction"] = np.arange(len(frame), dtype=np.float64)
    constant = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, frame)
    ).regression_metrics
    assert constant.mae >= 0.0 and constant.rmse >= 0.0
    assert not constant.r2_valid
    assert constant.r2_invalid_reason == "constant_target"


def test_daily_positive_negative_pearson_and_rank_ic(
    training_result: WalkForwardTrainingResult,
) -> None:
    frame = training_result.predictions
    for _, indices in frame.groupby("trade_date").groups.items():
        values = np.arange(len(indices), dtype=np.float64)
        frame.loc[indices, "target"] = values
        frame.loc[indices, "prediction"] = values
    positive = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, frame)
    ).date_metrics
    assert np.allclose(positive["pearson_ic"], 1.0)
    assert np.allclose(positive["spearman_rank_ic"], 1.0)

    for _, indices in frame.groupby("trade_date").groups.items():
        frame.loc[indices, "prediction"] = -np.arange(
            len(indices), dtype=np.float64
        )
    negative = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, frame)
    ).date_metrics
    assert np.allclose(negative["pearson_ic"], -1.0)
    assert np.allclose(negative["spearman_rank_ic"], -1.0)


def test_rank_ic_uses_average_ties_and_is_order_independent(
    training_result: WalkForwardTrainingResult,
) -> None:
    frame = training_result.predictions
    first_date = frame["trade_date"].min()
    indices = frame.index[frame["trade_date"].eq(first_date)]
    frame.loc[indices, "target"] = [1.0, 1.0, 3.0, 4.0]
    frame.loc[indices, "prediction"] = [1.0, 2.0, 2.0, 4.0]
    evaluated = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, frame)
    ).date_metrics.iloc[0]
    expected = np.corrcoef([1.5, 1.5, 3.0, 4.0], [1.0, 2.5, 2.5, 4.0])[0, 1]
    assert evaluated["spearman_rank_ic"] == pytest.approx(expected)

    reordered = frame.copy()
    reordered.loc[indices, "ts_code"] = list(
        reversed(reordered.loc[indices, "ts_code"].tolist())
    )
    changed_codes = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, reordered)
    ).date_metrics.iloc[0]
    assert changed_codes["spearman_rank_ic"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("target_constant", "prediction_constant", "reason"),
    [
        (True, True, "constant_target_and_prediction"),
        (True, False, "constant_target"),
        (False, True, "constant_prediction"),
    ],
)
def test_invalid_daily_correlations_are_retained_with_priority_reason(
    training_result: WalkForwardTrainingResult,
    target_constant: bool,
    prediction_constant: bool,
    reason: str,
) -> None:
    frame = training_result.predictions
    first_date = frame["trade_date"].min()
    indices = frame.index[frame["trade_date"].eq(first_date)]
    sequence = np.arange(len(indices), dtype=np.float64)
    frame.loc[indices, "target"] = 1.0 if target_constant else sequence
    frame.loc[indices, "prediction"] = (
        2.0 if prediction_constant else sequence
    )
    row = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, frame)
    ).date_metrics.iloc[0]
    assert np.isnan(row["pearson_ic"])
    assert np.isnan(row["spearman_rank_ic"])
    assert not row["pearson_valid"]
    assert not row["rank_ic_valid"]
    assert row["pearson_invalid_reason"] == reason
    assert row["rank_ic_invalid_reason"] == reason


def test_minimum_cross_section_reason_has_highest_priority(
    training_result: WalkForwardTrainingResult,
) -> None:
    evaluated = OOSModelEvaluator(
        ModelEvaluationConfig(minimum_cross_section_size=5)
    ).evaluate(training_result)
    assert not evaluated.date_metrics["pearson_valid"].any()
    assert not evaluated.date_metrics["rank_ic_valid"].any()
    assert set(evaluated.date_metrics["pearson_invalid_reason"]) == {
        "insufficient_cross_section"
    }
    assert evaluated.pearson_ic_summary.valid_dates == 0
    assert evaluated.pearson_ic_summary.mean is None
    assert evaluated.pearson_ic_summary.std is None
    assert evaluated.pearson_ic_summary.information_ratio is None


def test_cross_sectional_summary_uses_sample_std_and_unannualized_ratio(
    training_result: WalkForwardTrainingResult,
) -> None:
    frame = training_result.predictions
    signs = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    for sign, (_, indices) in zip(
        signs, frame.groupby("trade_date").groups.items(), strict=True
    ):
        values = np.arange(len(indices), dtype=np.float64)
        frame.loc[indices, "target"] = values
        frame.loc[indices, "prediction"] = sign * values
    summary = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, frame)
    ).pearson_ic_summary
    expected = np.asarray(signs)
    assert summary.mean == pytest.approx(expected.mean())
    assert summary.std == pytest.approx(expected.std(ddof=1))
    assert summary.information_ratio == pytest.approx(
        expected.mean() / expected.std(ddof=1)
    )
    assert summary.valid_dates == len(signs)
    assert summary.invalid_dates == 0


def test_one_valid_date_and_zero_std_have_no_information_ratio(
    training_result: WalkForwardTrainingResult,
) -> None:
    single = OOSModelEvaluator(
        ModelEvaluationConfig(minimum_cross_section_size=2)
    ).evaluate(_single_row_result(training_result))
    assert single.pearson_ic_summary.valid_dates == 0
    frame = training_result.predictions
    for _, indices in frame.groupby("trade_date").groups.items():
        values = np.arange(len(indices), dtype=np.float64)
        frame.loc[indices, "target"] = values
        frame.loc[indices, "prediction"] = values
    summary = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, frame)
    ).pearson_ic_summary
    assert summary.std == pytest.approx(0.0)
    assert summary.information_ratio is None


def test_date_metrics_contract_sorting_types_and_counts(
    training_result: WalkForwardTrainingResult,
) -> None:
    result = OOSModelEvaluator().evaluate(training_result)
    frame = result.date_metrics
    source = training_result.predictions
    assert list(frame.columns) == DATE_COLUMNS
    assert isinstance(frame.index, pd.RangeIndex)
    assert frame["trade_date"].is_monotonic_increasing
    assert frame["trade_date"].is_unique
    assert frame["fold_id"].dtype == np.dtype("int64")
    assert frame["n_obs"].dtype == np.dtype("int64")
    assert frame["pearson_valid"].dtype == bool
    assert frame["rank_ic_valid"].dtype == bool
    assert frame["n_obs"].sum() == len(source)
    expected_folds = source.groupby("trade_date")["fold_id"].first()
    assert frame.set_index("trade_date")["fold_id"].to_dict() == expected_folds.to_dict()


def test_fold_metrics_contract_and_training_audit_alignment(
    training_result: WalkForwardTrainingResult,
) -> None:
    result = OOSModelEvaluator().evaluate(training_result)
    frame = result.fold_metrics
    assert list(frame.columns) == FOLD_COLUMNS
    assert isinstance(frame.index, pd.RangeIndex)
    assert frame["fold_id"].tolist() == list(range(result.audit.n_folds))
    for fold, audit in zip(
        frame.itertuples(index=False),
        training_result.audit.fold_audits,
        strict=True,
    ):
        assert fold.n_rows == audit.prediction_rows
        assert fold.start_date == audit.prediction_start_date
        assert fold.end_date == audit.prediction_end_date
        assert fold.pearson_valid_dates + fold.pearson_invalid_dates == fold.n_dates
        assert fold.rank_ic_valid_dates + fold.rank_ic_invalid_dates == fold.n_dates
        assert math_is_finite_nonnegative(fold.mae)
        assert math_is_finite_nonnegative(fold.rmse)


def math_is_finite_nonnegative(value: object) -> bool:
    return bool(np.isfinite(float(value)) and float(value) >= 0.0)


def test_evaluation_audit_exact_coverage_and_source(
    training_result: WalkForwardTrainingResult,
) -> None:
    result = OOSModelEvaluator().evaluate(training_result)
    audit = result.audit
    source = training_result.audit
    assert audit.model_name == source.model_name
    assert audit.source_label_name == source.source_label_name
    assert audit.n_rows == audit.expected_rows == source.n_prediction_rows
    assert audit.n_dates == audit.expected_dates == source.n_prediction_dates
    assert audit.n_folds == source.n_folds
    assert audit.row_coverage == 1.0
    assert audit.date_coverage == 1.0
    assert audit.first_date == source.first_prediction_date
    assert audit.last_date == source.last_prediction_date
    json.dumps(audit.as_dict(), allow_nan=False)


def test_result_tables_are_defensive_and_as_dict_is_json_safe(
    training_result: WalkForwardTrainingResult,
) -> None:
    result = OOSModelEvaluator(
        ModelEvaluationConfig(minimum_cross_section_size=5)
    ).evaluate(training_result)
    date_before = result.date_metrics
    fold_before = result.fold_metrics
    date_changed = result.date_metrics
    fold_changed = result.fold_metrics
    date_changed.loc[0, "n_obs"] = 999
    fold_changed.loc[0, "n_rows"] = 999
    pdt.assert_frame_equal(result.date_metrics, date_before)
    pdt.assert_frame_equal(result.fold_metrics, fold_before)
    serialized = result.as_dict()
    json.dumps(serialized, allow_nan=False)
    assert serialized["date_metrics"][0]["pearson_ic"] is None  # type: ignore[index]
    assert serialized["date_metrics"][0]["trade_date"] == "2024-01-10"  # type: ignore[index]
    assert "predictions" not in serialized
    assert not hasattr(result, "training_result")
    assert not hasattr(result, "model")


def test_evaluate_is_repeatable_and_keeps_no_previous_result(
    training_result: WalkForwardTrainingResult,
) -> None:
    evaluator = OOSModelEvaluator()
    first = evaluator.evaluate(training_result)
    second = evaluator.evaluate(training_result)
    assert first.as_dict() == second.as_dict()
    assert not hasattr(evaluator, "result")
    assert not hasattr(evaluator, "last_result")


def test_evaluate_requires_training_result() -> None:
    with pytest.raises(ModelEvaluationDataError, match="WalkForwardTrainingResult"):
        OOSModelEvaluator().evaluate(pd.DataFrame())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("empty", ModelEvaluationDataError),
        ("missing_column", ModelEvaluationDataError),
        ("extra_column", ModelEvaluationDataError),
        ("reordered_columns", ModelEvaluationDataError),
        ("duplicate_index", ModelEvaluationIntegrityError),
        ("wrong_index_name", ModelEvaluationDataError),
        ("descending_index", ModelEvaluationIntegrityError),
        ("duplicate_key", ModelEvaluationIntegrityError),
        ("empty_code", ModelEvaluationDataError),
        ("missing_date", ModelEvaluationDataError),
        ("target_nan", ModelEvaluationDataError),
        ("target_inf", ModelEvaluationDataError),
        ("prediction_nan", ModelEvaluationDataError),
        ("prediction_inf", ModelEvaluationDataError),
        ("float_fold", ModelEvaluationDataError),
        ("negative_fold", ModelEvaluationDataError),
        ("missing_fold", ModelEvaluationDataError),
        ("noncontinuous_fold", ModelEvaluationIntegrityError),
    ],
)
def test_prediction_contract_rejects_invalid_frames(
    training_result: WalkForwardTrainingResult,
    mutation: str,
    error: type[Exception],
) -> None:
    frame = training_result.predictions
    if mutation == "empty":
        frame = frame.iloc[0:0]
    elif mutation == "missing_column":
        frame = frame.drop(columns="target")
    elif mutation == "extra_column":
        frame["extra"] = 1
    elif mutation == "reordered_columns":
        frame = frame[list(reversed(frame.columns))]
    elif mutation == "duplicate_index":
        frame.index = pd.Index([0] * len(frame), name="dataset_index")
    elif mutation == "wrong_index_name":
        frame.index.name = "wrong"
    elif mutation == "descending_index":
        frame.index = pd.Index(
            list(reversed(range(len(frame)))), name="dataset_index"
        )
    elif mutation == "duplicate_key":
        frame.iloc[1, frame.columns.get_loc("trade_date")] = frame.iloc[0]["trade_date"]
        frame.iloc[1, frame.columns.get_loc("ts_code")] = frame.iloc[0]["ts_code"]
    elif mutation == "empty_code":
        frame.iloc[0, frame.columns.get_loc("ts_code")] = " "
    elif mutation == "missing_date":
        frame.iloc[0, frame.columns.get_loc("trade_date")] = pd.NaT
    elif mutation == "target_nan":
        frame.iloc[0, frame.columns.get_loc("target")] = np.nan
    elif mutation == "target_inf":
        frame.iloc[0, frame.columns.get_loc("target")] = np.inf
    elif mutation == "prediction_nan":
        frame.iloc[0, frame.columns.get_loc("prediction")] = np.nan
    elif mutation == "prediction_inf":
        frame.iloc[0, frame.columns.get_loc("prediction")] = -np.inf
    elif mutation == "float_fold":
        frame["fold_id"] = frame["fold_id"].astype(float)
    elif mutation == "negative_fold":
        frame.loc[frame.index[0], "fold_id"] = -1
    elif mutation == "missing_fold":
        frame["fold_id"] = frame["fold_id"].astype("Int64")
        frame.loc[frame.index[0], "fold_id"] = pd.NA
    elif mutation == "noncontinuous_fold":
        frame.loc[frame["fold_id"].eq(1), "fold_id"] = 2
    with pytest.raises(error):
        OOSModelEvaluator().evaluate(_with_predictions(training_result, frame))


def test_prediction_input_is_not_modified_on_success_or_failure(
    training_result: WalkForwardTrainingResult,
) -> None:
    frame = training_result.predictions
    before = frame.copy(deep=True)
    wrapped = _with_predictions(training_result, frame)
    OOSModelEvaluator().evaluate(wrapped)
    pdt.assert_frame_equal(frame, before)
    bad = frame.copy(deep=True)
    bad.loc[bad.index[0], "prediction"] = np.nan
    bad_before = bad.copy(deep=True)
    with pytest.raises(ModelEvaluationDataError):
        OOSModelEvaluator().evaluate(_with_predictions(training_result, bad))
    pdt.assert_frame_equal(bad, bad_before)


@pytest.mark.parametrize(
    "field",
    [
        "n_prediction_rows",
        "n_prediction_dates",
        "n_folds",
        "first_prediction_date",
        "last_prediction_date",
    ],
)
def test_training_audit_overall_mismatches_are_rejected(
    training_result: WalkForwardTrainingResult,
    field: str,
) -> None:
    audit = training_result.audit
    replacements: dict[str, object] = {
        "n_prediction_rows": audit.n_prediction_rows + 1,
        "n_prediction_dates": audit.n_prediction_dates + 1,
        "n_folds": audit.n_folds + 1,
        "first_prediction_date": audit.first_prediction_date
        - pd.Timedelta(days=1),
        "last_prediction_date": audit.last_prediction_date
        + pd.Timedelta(days=1),
    }
    forged = _forge(audit, **{field: replacements[field]})
    with pytest.raises(ModelEvaluationIntegrityError):
        OOSModelEvaluator().evaluate(
            _with_predictions(training_result, training_result.predictions, audit=forged)
        )


@pytest.mark.parametrize(
    "field",
    [
        "prediction_rows",
        "prediction_start_date",
        "prediction_end_date",
        "model_name",
        "fold_id",
    ],
)
def test_fold_audit_mismatches_are_rejected(
    training_result: WalkForwardTrainingResult,
    field: str,
) -> None:
    audit = training_result.audit
    fold = audit.fold_audits[0]
    replacements: dict[str, object] = {
        "prediction_rows": fold.prediction_rows + 1,
        "prediction_start_date": fold.prediction_start_date
        - pd.Timedelta(days=1),
        "prediction_end_date": fold.prediction_end_date
        + pd.Timedelta(days=1),
        "model_name": "other",
        "fold_id": 2,
    }
    forged_fold = _forge(fold, **{field: replacements[field]})
    forged_audit = _forge(
        audit, fold_audits=(forged_fold,) + audit.fold_audits[1:]
    )
    with pytest.raises(ModelEvaluationIntegrityError):
        OOSModelEvaluator().evaluate(
            _with_predictions(
                training_result,
                training_result.predictions,
                audit=forged_audit,
            )
        )


def test_one_date_cannot_belong_to_multiple_folds(
    training_result: WalkForwardTrainingResult,
) -> None:
    frame = training_result.predictions
    first_date = frame["trade_date"].min()
    indices = frame.index[frame["trade_date"].eq(first_date)]
    frame.loc[indices[0], "fold_id"] = 1
    with pytest.raises(ModelEvaluationIntegrityError, match="trade_date"):
        OOSModelEvaluator().evaluate(_with_predictions(training_result, frame))


def test_overflow_in_metrics_fails_atomically(
    training_result: WalkForwardTrainingResult,
) -> None:
    frame = training_result.predictions
    frame["target"] = np.finfo(np.float64).max
    frame["prediction"] = -np.finfo(np.float64).max
    with pytest.raises(ModelEvaluationIntegrityError, match="non-finite"):
        OOSModelEvaluator().evaluate(_with_predictions(training_result, frame))


def test_adjusting_values_changes_only_evaluation_output(
    training_result: WalkForwardTrainingResult,
) -> None:
    original = OOSModelEvaluator().evaluate(training_result)
    frame = training_result.predictions
    frame["target"] = frame["target"] + np.arange(len(frame))
    changed_target = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, frame)
    )
    frame = training_result.predictions
    frame["prediction"] = frame["prediction"] - np.arange(len(frame))
    changed_prediction = OOSModelEvaluator().evaluate(
        _with_predictions(training_result, frame)
    )
    assert original.regression_metrics != changed_target.regression_metrics
    assert original.regression_metrics != changed_prediction.regression_metrics
    assert training_result.predictions.equals(training_result.predictions)
