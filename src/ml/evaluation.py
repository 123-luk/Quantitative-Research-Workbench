"""Strict evaluation contracts for frozen out-of-sample model predictions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.ml.training import (
    WalkForwardFoldAudit,
    WalkForwardTrainingAudit,
    WalkForwardTrainingResult,
)


class ModelEvaluationError(Exception):
    """Base error for out-of-sample model evaluation."""


class ModelEvaluationConfigError(ModelEvaluationError):
    """Raised when evaluation configuration is invalid."""


class ModelEvaluationDataError(ModelEvaluationError):
    """Raised when prediction data cannot be evaluated safely."""


class ModelEvaluationIntegrityError(ModelEvaluationError):
    """Raised when predictions, audits, or computed outputs disagree."""


_PREDICTION_COLUMNS = [
    "trade_date",
    "ts_code",
    "entry_trade_date",
    "exit_trade_date",
    "target",
    "prediction",
    "fold_id",
]

_DATE_METRIC_COLUMNS = [
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

_FOLD_METRIC_COLUMNS = [
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

_METRIC_NAMES = {"pearson_ic", "spearman_rank_ic"}


def _timestamp(field_name: str, value: object) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ModelEvaluationIntegrityError(
            f"{field_name} must be a valid timestamp"
        ) from exc
    if pd.isna(result) or result.tz is not None:
        raise ModelEvaluationIntegrityError(
            f"{field_name} must be timezone-naive and valid"
        )
    return result


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None else pd.Timestamp(value).strftime("%Y-%m-%d")


def _finite_float(field_name: str, value: object, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ModelEvaluationIntegrityError(f"{field_name} must be a finite float")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelEvaluationIntegrityError(
            f"{field_name} must be a finite float"
        ) from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite non-negative" if nonnegative else "finite"
        raise ModelEvaluationIntegrityError(
            f"{field_name} must be a {qualifier} float"
        )
    return result


def _nonnegative_int(field_name: str, value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < 0
    ):
        raise ModelEvaluationIntegrityError(
            f"{field_name} must be a non-negative integer"
        )
    return int(value)


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Immutable configuration for cross-sectional validity."""

    minimum_cross_section_size: int = 3

    def __post_init__(self) -> None:
        value = self.minimum_cross_section_size
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 2
        ):
            raise ModelEvaluationConfigError(
                "minimum_cross_section_size must be an integer >= 2"
            )
        object.__setattr__(self, "minimum_cross_section_size", int(value))

    @classmethod
    def from_dict(
        cls, values: Mapping[str, object] | None
    ) -> "ModelEvaluationConfig":
        """Build from the single supported configuration field."""
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ModelEvaluationConfigError(
                "evaluation config must be a Mapping or None"
            )
        unknown = [key for key in values if key != "minimum_cross_section_size"]
        if unknown:
            raise ModelEvaluationConfigError(
                f"unknown evaluation config field(s): {unknown!r}"
            )
        return cls(
            minimum_cross_section_size=values.get(
                "minimum_cross_section_size", 3
            )  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        """Return a fresh JSON-safe configuration dictionary."""
        return {
            "minimum_cross_section_size": self.minimum_cross_section_size,
        }


@dataclass(frozen=True)
class RegressionMetrics:
    """Immutable aggregate regression metrics without sample details."""

    n_obs: int
    mae: float
    rmse: float
    r2: float | None
    r2_valid: bool
    r2_invalid_reason: str | None

    def __post_init__(self) -> None:
        n_obs = _nonnegative_int("n_obs", self.n_obs)
        if n_obs == 0:
            raise ModelEvaluationIntegrityError("n_obs must be positive")
        object.__setattr__(self, "n_obs", n_obs)
        object.__setattr__(
            self, "mae", _finite_float("mae", self.mae, nonnegative=True)
        )
        object.__setattr__(
            self, "rmse", _finite_float("rmse", self.rmse, nonnegative=True)
        )
        if not isinstance(self.r2_valid, bool):
            raise ModelEvaluationIntegrityError("r2_valid must be bool")
        if self.r2_valid:
            if self.r2 is None or self.r2_invalid_reason is not None:
                raise ModelEvaluationIntegrityError(
                    "valid r2 requires a value and no invalid reason"
                )
            object.__setattr__(self, "r2", _finite_float("r2", self.r2))
        elif (
            self.r2 is not None
            or not isinstance(self.r2_invalid_reason, str)
            or not self.r2_invalid_reason
        ):
            raise ModelEvaluationIntegrityError(
                "invalid r2 requires no value and a non-empty reason"
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe regression summary."""
        return {
            "n_obs": self.n_obs,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "r2_valid": self.r2_valid,
            "r2_invalid_reason": self.r2_invalid_reason,
        }


@dataclass(frozen=True)
class CrossSectionalMetricSummary:
    """Immutable summary of valid daily correlation observations."""

    metric_name: str
    total_dates: int
    valid_dates: int
    invalid_dates: int
    mean: float | None
    std: float | None
    information_ratio: float | None

    def __post_init__(self) -> None:
        if self.metric_name not in _METRIC_NAMES:
            raise ModelEvaluationIntegrityError(
                "metric_name must be pearson_ic or spearman_rank_ic"
            )
        total = _nonnegative_int("total_dates", self.total_dates)
        valid = _nonnegative_int("valid_dates", self.valid_dates)
        invalid = _nonnegative_int("invalid_dates", self.invalid_dates)
        if total != valid + invalid:
            raise ModelEvaluationIntegrityError(
                "total_dates must equal valid_dates plus invalid_dates"
            )
        object.__setattr__(self, "total_dates", total)
        object.__setattr__(self, "valid_dates", valid)
        object.__setattr__(self, "invalid_dates", invalid)
        if valid == 0:
            if any(
                value is not None
                for value in (self.mean, self.std, self.information_ratio)
            ):
                raise ModelEvaluationIntegrityError(
                    "empty metric summary values must all be None"
                )
            return
        if self.mean is None:
            raise ModelEvaluationIntegrityError(
                "a metric summary with valid dates requires mean"
            )
        object.__setattr__(self, "mean", _finite_float("mean", self.mean))
        if valid == 1:
            if self.std is not None or self.information_ratio is not None:
                raise ModelEvaluationIntegrityError(
                    "a one-date summary cannot have std or information_ratio"
                )
            return
        if self.std is None:
            raise ModelEvaluationIntegrityError(
                "a multi-date summary requires std"
            )
        std = _finite_float("std", self.std, nonnegative=True)
        object.__setattr__(self, "std", std)
        if std == 0.0:
            if self.information_ratio is not None:
                raise ModelEvaluationIntegrityError(
                    "zero std requires information_ratio=None"
                )
        elif self.information_ratio is None:
            raise ModelEvaluationIntegrityError(
                "nonzero std requires information_ratio"
            )
        else:
            object.__setattr__(
                self,
                "information_ratio",
                _finite_float("information_ratio", self.information_ratio),
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe cross-sectional summary."""
        return {
            "metric_name": self.metric_name,
            "total_dates": self.total_dates,
            "valid_dates": self.valid_dates,
            "invalid_dates": self.invalid_dates,
            "mean": self.mean,
            "std": self.std,
            "information_ratio": self.information_ratio,
        }


@dataclass(frozen=True)
class ModelEvaluationAudit:
    """Immutable prediction coverage and source training audit."""

    model_name: str
    source_label_name: str | None
    n_rows: int
    n_dates: int
    n_folds: int
    expected_rows: int
    expected_dates: int
    row_coverage: float
    date_coverage: float
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    config: ModelEvaluationConfig

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ModelEvaluationIntegrityError(
                "evaluation audit model_name must be non-empty"
            )
        if self.source_label_name is not None and (
            not isinstance(self.source_label_name, str)
            or not self.source_label_name
        ):
            raise ModelEvaluationIntegrityError(
                "source_label_name must be non-empty or None"
            )
        for field_name in (
            "n_rows",
            "n_dates",
            "n_folds",
            "expected_rows",
            "expected_dates",
        ):
            value = _nonnegative_int(field_name, getattr(self, field_name))
            if value == 0:
                raise ModelEvaluationIntegrityError(
                    f"{field_name} must be positive"
                )
            object.__setattr__(self, field_name, value)
        if self.n_rows != self.expected_rows:
            raise ModelEvaluationIntegrityError(
                "n_rows must equal expected_rows"
            )
        if self.n_dates != self.expected_dates:
            raise ModelEvaluationIntegrityError(
                "n_dates must equal expected_dates"
            )
        if self.n_folds > self.n_dates:
            raise ModelEvaluationIntegrityError(
                "n_folds cannot exceed n_dates"
            )
        row_coverage = _finite_float(
            "row_coverage", self.row_coverage, nonnegative=True
        )
        date_coverage = _finite_float(
            "date_coverage", self.date_coverage, nonnegative=True
        )
        if row_coverage != 1.0 or date_coverage != 1.0:
            raise ModelEvaluationIntegrityError(
                "successful evaluation coverage must equal 1.0"
            )
        object.__setattr__(self, "row_coverage", row_coverage)
        object.__setattr__(self, "date_coverage", date_coverage)
        first = _timestamp("first_date", self.first_date)
        last = _timestamp("last_date", self.last_date)
        if first > last:
            raise ModelEvaluationIntegrityError(
                "first_date cannot exceed last_date"
            )
        object.__setattr__(self, "first_date", first)
        object.__setattr__(self, "last_date", last)
        if not isinstance(self.config, ModelEvaluationConfig):
            raise ModelEvaluationIntegrityError(
                "evaluation audit config must be ModelEvaluationConfig"
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe coverage audit."""
        return {
            "model_name": self.model_name,
            "source_label_name": self.source_label_name,
            "n_rows": self.n_rows,
            "n_dates": self.n_dates,
            "n_folds": self.n_folds,
            "expected_rows": self.expected_rows,
            "expected_dates": self.expected_dates,
            "row_coverage": self.row_coverage,
            "date_coverage": self.date_coverage,
            "first_date": _iso(self.first_date),
            "last_date": _iso(self.last_date),
            "config": self.config.as_dict(),
        }


def _json_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return _iso(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    raise ModelEvaluationIntegrityError(
        f"evaluation output contains unsupported {type(value).__name__}"
    )


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {column: _json_value(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


class ModelEvaluationResult:
    """Defensively expose immutable summaries and evaluation tables."""

    def __init__(
        self,
        regression_metrics: RegressionMetrics,
        pearson_ic_summary: CrossSectionalMetricSummary,
        rank_ic_summary: CrossSectionalMetricSummary,
        date_metrics: pd.DataFrame,
        fold_metrics: pd.DataFrame,
        audit: ModelEvaluationAudit,
    ) -> None:
        if not isinstance(regression_metrics, RegressionMetrics):
            raise ModelEvaluationIntegrityError(
                "regression_metrics must be RegressionMetrics"
            )
        if not isinstance(pearson_ic_summary, CrossSectionalMetricSummary):
            raise ModelEvaluationIntegrityError(
                "pearson_ic_summary must be CrossSectionalMetricSummary"
            )
        if pearson_ic_summary.metric_name != "pearson_ic":
            raise ModelEvaluationIntegrityError(
                "pearson_ic_summary metric_name is invalid"
            )
        if not isinstance(rank_ic_summary, CrossSectionalMetricSummary):
            raise ModelEvaluationIntegrityError(
                "rank_ic_summary must be CrossSectionalMetricSummary"
            )
        if rank_ic_summary.metric_name != "spearman_rank_ic":
            raise ModelEvaluationIntegrityError(
                "rank_ic_summary metric_name is invalid"
            )
        if not isinstance(date_metrics, pd.DataFrame) or list(
            date_metrics.columns
        ) != _DATE_METRIC_COLUMNS:
            raise ModelEvaluationIntegrityError(
                "date_metrics columns are invalid"
            )
        if not isinstance(fold_metrics, pd.DataFrame) or list(
            fold_metrics.columns
        ) != _FOLD_METRIC_COLUMNS:
            raise ModelEvaluationIntegrityError(
                "fold_metrics columns are invalid"
            )
        if not isinstance(audit, ModelEvaluationAudit):
            raise ModelEvaluationIntegrityError(
                "audit must be ModelEvaluationAudit"
            )
        self._regression_metrics = regression_metrics
        self._pearson_ic_summary = pearson_ic_summary
        self._rank_ic_summary = rank_ic_summary
        self._date_metrics = date_metrics.copy(deep=True)
        self._fold_metrics = fold_metrics.copy(deep=True)
        self._audit = audit

    @property
    def regression_metrics(self) -> RegressionMetrics:
        return self._regression_metrics

    @property
    def pearson_ic_summary(self) -> CrossSectionalMetricSummary:
        return self._pearson_ic_summary

    @property
    def rank_ic_summary(self) -> CrossSectionalMetricSummary:
        return self._rank_ic_summary

    @property
    def date_metrics(self) -> pd.DataFrame:
        return self._date_metrics.copy(deep=True)

    @property
    def fold_metrics(self) -> pd.DataFrame:
        return self._fold_metrics.copy(deep=True)

    @property
    def audit(self) -> ModelEvaluationAudit:
        return self._audit

    def as_dict(self) -> dict[str, object]:
        """Return all evaluation outputs in deterministic JSON-safe form."""
        result = {
            "regression_metrics": self.regression_metrics.as_dict(),
            "pearson_ic_summary": self.pearson_ic_summary.as_dict(),
            "rank_ic_summary": self.rank_ic_summary.as_dict(),
            "date_metrics": _frame_records(self._date_metrics),
            "fold_metrics": _frame_records(self._fold_metrics),
            "audit": self.audit.as_dict(),
        }
        json.dumps(result, allow_nan=False)
        return result


@dataclass(frozen=True)
class _CorrelationResult:
    value: float | None
    valid: bool
    invalid_reason: str | None


def _regression_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> RegressionMetrics:
    n_obs = len(target)
    if n_obs == 0 or len(prediction) != n_obs:
        raise ModelEvaluationIntegrityError(
            "regression inputs must be aligned and non-empty"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        residual = target - prediction
        absolute_error = np.abs(residual)
        squared_error = residual * residual
        mae = float(np.mean(absolute_error, dtype=np.float64))
        rmse = float(np.sqrt(np.mean(squared_error, dtype=np.float64)))
    if not math.isfinite(mae) or not math.isfinite(rmse):
        raise ModelEvaluationIntegrityError(
            "regression metric calculation produced a non-finite value"
        )
    if n_obs < 2:
        return RegressionMetrics(
            n_obs, mae, rmse, None, False, "insufficient_observations"
        )
    if np.all(target == target[0]):
        return RegressionMetrics(
            n_obs, mae, rmse, None, False, "constant_target"
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        centered = target - np.mean(target, dtype=np.float64)
        sse = float(np.sum(squared_error, dtype=np.float64))
        sst = float(np.dot(centered, centered))
        r2 = float(1.0 - sse / sst)
    if not all(math.isfinite(value) for value in (sse, sst, r2)):
        raise ModelEvaluationIntegrityError(
            "r2 calculation produced a non-finite value"
        )
    return RegressionMetrics(n_obs, mae, rmse, r2, True, None)


def _pearson(
    target: np.ndarray,
    prediction: np.ndarray,
    minimum_size: int,
) -> _CorrelationResult:
    if len(target) < minimum_size:
        return _CorrelationResult(
            None, False, "insufficient_cross_section"
        )
    target_constant = bool(np.all(target == target[0]))
    prediction_constant = bool(np.all(prediction == prediction[0]))
    if target_constant and prediction_constant:
        return _CorrelationResult(
            None, False, "constant_target_and_prediction"
        )
    if target_constant:
        return _CorrelationResult(None, False, "constant_target")
    if prediction_constant:
        return _CorrelationResult(None, False, "constant_prediction")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        target_centered = target - np.mean(target, dtype=np.float64)
        prediction_centered = prediction - np.mean(
            prediction, dtype=np.float64
        )
        numerator = float(np.dot(target_centered, prediction_centered))
        target_ss = float(np.dot(target_centered, target_centered))
        prediction_ss = float(
            np.dot(prediction_centered, prediction_centered)
        )
        denominator = float(np.sqrt(target_ss * prediction_ss))
        correlation = float(numerator / denominator)
    if not all(
        math.isfinite(value)
        for value in (numerator, target_ss, prediction_ss, denominator, correlation)
    ):
        raise ModelEvaluationIntegrityError(
            "correlation calculation produced a non-finite value"
        )
    return _CorrelationResult(
        float(np.clip(correlation, -1.0, 1.0)), True, None
    )


def _summary(
    metric_name: str,
    values: pd.Series,
    valid: pd.Series,
) -> CrossSectionalMetricSummary:
    valid_values = values.loc[valid].to_numpy(dtype=np.float64)
    total_dates = len(values)
    valid_dates = len(valid_values)
    invalid_dates = total_dates - valid_dates
    if valid_dates == 0:
        return CrossSectionalMetricSummary(
            metric_name,
            total_dates,
            valid_dates,
            invalid_dates,
            None,
            None,
            None,
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = float(np.mean(valid_values, dtype=np.float64))
    if not math.isfinite(mean):
        raise ModelEvaluationIntegrityError(
            f"{metric_name} mean is non-finite"
        )
    if valid_dates == 1:
        return CrossSectionalMetricSummary(
            metric_name,
            total_dates,
            valid_dates,
            invalid_dates,
            mean,
            None,
            None,
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        std = float(np.std(valid_values, ddof=1, dtype=np.float64))
    if not math.isfinite(std):
        raise ModelEvaluationIntegrityError(
            f"{metric_name} sample standard deviation is non-finite"
        )
    information_ratio = None
    if std != 0.0:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            information_ratio = float(mean / std)
        if not math.isfinite(information_ratio):
            raise ModelEvaluationIntegrityError(
                f"{metric_name} information ratio is non-finite"
            )
    return CrossSectionalMetricSummary(
        metric_name,
        total_dates,
        valid_dates,
        invalid_dates,
        mean,
        std,
        information_ratio,
    )


class OOSModelEvaluator:
    """Evaluate only frozen OOS predictions and their training audit."""

    def __init__(self, config: ModelEvaluationConfig | None = None) -> None:
        if config is not None and not isinstance(config, ModelEvaluationConfig):
            raise ModelEvaluationConfigError(
                "config must be ModelEvaluationConfig or None"
            )
        self._config = ModelEvaluationConfig() if config is None else config

    def evaluate(
        self, result: WalkForwardTrainingResult
    ) -> ModelEvaluationResult:
        """Validate and evaluate a complete OOS prediction result."""
        if not isinstance(result, WalkForwardTrainingResult):
            raise ModelEvaluationDataError(
                "result must be WalkForwardTrainingResult"
            )
        predictions = result.predictions
        training_audit = result.audit
        normalized = self._validate_predictions(predictions)
        self._validate_training_audit(normalized, training_audit)

        regression = _regression_metrics(
            normalized["target"].to_numpy(dtype=np.float64),
            normalized["prediction"].to_numpy(dtype=np.float64),
        )
        date_metrics = self._date_metrics(normalized)
        pearson_summary = _summary(
            "pearson_ic",
            date_metrics["pearson_ic"],
            date_metrics["pearson_valid"],
        )
        rank_summary = _summary(
            "spearman_rank_ic",
            date_metrics["spearman_rank_ic"],
            date_metrics["rank_ic_valid"],
        )
        fold_metrics = self._fold_metrics(normalized, date_metrics)
        first_date = pd.Timestamp(normalized["trade_date"].min())
        last_date = pd.Timestamp(normalized["trade_date"].max())
        audit = ModelEvaluationAudit(
            model_name=training_audit.model_name,
            source_label_name=training_audit.source_label_name,
            n_rows=len(normalized),
            n_dates=int(normalized["trade_date"].nunique()),
            n_folds=int(normalized["fold_id"].nunique()),
            expected_rows=training_audit.n_prediction_rows,
            expected_dates=training_audit.n_prediction_dates,
            row_coverage=len(normalized) / training_audit.n_prediction_rows,
            date_coverage=(
                normalized["trade_date"].nunique()
                / training_audit.n_prediction_dates
            ),
            first_date=first_date,
            last_date=last_date,
            config=self._config,
        )
        return ModelEvaluationResult(
            regression,
            pearson_summary,
            rank_summary,
            date_metrics,
            fold_metrics,
            audit,
        )

    @staticmethod
    def _validate_predictions(predictions: object) -> pd.DataFrame:
        if not isinstance(predictions, pd.DataFrame):
            raise ModelEvaluationDataError(
                "predictions must be a pandas DataFrame"
            )
        if predictions.empty:
            raise ModelEvaluationDataError("predictions must not be empty")
        if list(predictions.columns) != _PREDICTION_COLUMNS:
            raise ModelEvaluationDataError(
                "prediction columns and order are invalid"
            )
        if predictions.index.name != "dataset_index":
            raise ModelEvaluationDataError(
                "prediction index name must be dataset_index"
            )
        if not predictions.index.is_unique:
            raise ModelEvaluationIntegrityError(
                "prediction dataset index must be unique"
            )
        if not predictions.index.is_monotonic_increasing:
            raise ModelEvaluationIntegrityError(
                "prediction dataset index must be increasing"
            )
        if predictions.duplicated(["trade_date", "ts_code"]).any():
            raise ModelEvaluationIntegrityError(
                "prediction trade_date and ts_code keys must be unique"
            )
        normalized = predictions.copy(deep=True)
        for column in (
            "trade_date",
            "entry_trade_date",
            "exit_trade_date",
        ):
            if normalized[column].isna().any() or not isinstance(
                normalized[column].dtype, pd.DatetimeTZDtype
            ) and not pd.api.types.is_datetime64_ns_dtype(
                normalized[column].dtype
            ):
                raise ModelEvaluationDataError(
                    f"{column} must be non-missing datetime64[ns]"
                )
            if getattr(normalized[column].dt, "tz", None) is not None:
                raise ModelEvaluationDataError(
                    f"{column} must be timezone-naive"
                )
        codes = normalized["ts_code"]
        if any(
            not isinstance(value, str) or not value.strip()
            for value in codes.tolist()
        ):
            raise ModelEvaluationDataError(
                "ts_code must contain non-empty strings"
            )
        for column in ("target", "prediction"):
            try:
                values = normalized[column].to_numpy(dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ModelEvaluationDataError(
                    f"{column} must be safely convertible to float64"
                ) from exc
            if not np.isfinite(values).all():
                raise ModelEvaluationDataError(
                    f"{column} must contain only finite values"
                )
            normalized[column] = values
        if (
            pd.api.types.is_bool_dtype(normalized["fold_id"].dtype)
            or not pd.api.types.is_integer_dtype(normalized["fold_id"].dtype)
            or normalized["fold_id"].isna().any()
        ):
            raise ModelEvaluationDataError(
                "fold_id must be a non-missing integer dtype"
            )
        fold_ids = normalized["fold_id"].to_numpy(dtype=np.int64)
        if (fold_ids < 0).any():
            raise ModelEvaluationDataError("fold_id must be non-negative")
        normalized["fold_id"] = fold_ids
        unique_folds = tuple(int(value) for value in np.unique(fold_ids))
        if unique_folds != tuple(range(len(unique_folds))):
            raise ModelEvaluationIntegrityError(
                "fold_id must start at zero and be continuous"
            )
        date_fold_counts = normalized.groupby(
            "trade_date", sort=False
        )["fold_id"].nunique()
        if (date_fold_counts != 1).any():
            raise ModelEvaluationIntegrityError(
                "each trade_date must belong to exactly one fold"
            )
        if normalized.groupby("fold_id", sort=False).size().eq(0).any():
            raise ModelEvaluationIntegrityError(
                "every fold must contain prediction rows"
            )
        return normalized

    @staticmethod
    def _validate_training_audit(
        predictions: pd.DataFrame,
        audit: object,
    ) -> None:
        if not isinstance(audit, WalkForwardTrainingAudit):
            raise ModelEvaluationIntegrityError(
                "result audit must be WalkForwardTrainingAudit"
            )
        n_rows = len(predictions)
        n_dates = int(predictions["trade_date"].nunique())
        fold_ids = tuple(
            int(value) for value in sorted(predictions["fold_id"].unique())
        )
        if n_rows != audit.n_prediction_rows:
            raise ModelEvaluationIntegrityError(
                "prediction row count differs from training audit"
            )
        if n_dates != audit.n_prediction_dates:
            raise ModelEvaluationIntegrityError(
                "prediction date count differs from training audit"
            )
        if len(fold_ids) != audit.n_folds:
            raise ModelEvaluationIntegrityError(
                "prediction fold count differs from training audit"
            )
        first_date = pd.Timestamp(predictions["trade_date"].min())
        last_date = pd.Timestamp(predictions["trade_date"].max())
        if first_date != audit.first_prediction_date:
            raise ModelEvaluationIntegrityError(
                "first prediction date differs from training audit"
            )
        if last_date != audit.last_prediction_date:
            raise ModelEvaluationIntegrityError(
                "last prediction date differs from training audit"
            )
        folds = tuple(audit.fold_audits)
        expected_ids = tuple(range(audit.n_folds))
        audit_ids = tuple(fold.fold_id for fold in folds)
        if fold_ids != expected_ids or audit_ids != expected_ids:
            raise ModelEvaluationIntegrityError(
                "fold ids or fold audit order are inconsistent"
            )
        for fold in folds:
            if not isinstance(fold, WalkForwardFoldAudit):
                raise ModelEvaluationIntegrityError(
                    "fold audit type is invalid"
                )
            if fold.model_name != audit.model_name:
                raise ModelEvaluationIntegrityError(
                    "fold model_name differs from training audit"
                )
            block = predictions.loc[predictions["fold_id"].eq(fold.fold_id)]
            if len(block) != fold.prediction_rows:
                raise ModelEvaluationIntegrityError(
                    f"fold_id={fold.fold_id} row count differs from audit"
                )
            if pd.Timestamp(block["trade_date"].min()) != fold.prediction_start_date:
                raise ModelEvaluationIntegrityError(
                    f"fold_id={fold.fold_id} start date differs from audit"
                )
            if pd.Timestamp(block["trade_date"].max()) != fold.prediction_end_date:
                raise ModelEvaluationIntegrityError(
                    f"fold_id={fold.fold_id} end date differs from audit"
                )

    def _date_metrics(self, predictions: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for trade_date, block in predictions.groupby("trade_date", sort=True):
            target = block["target"].to_numpy(dtype=np.float64)
            prediction = block["prediction"].to_numpy(dtype=np.float64)
            pearson = _pearson(
                target,
                prediction,
                self._config.minimum_cross_section_size,
            )
            target_rank = pd.Series(target).rank(
                method="average"
            ).to_numpy(dtype=np.float64)
            prediction_rank = pd.Series(prediction).rank(
                method="average"
            ).to_numpy(dtype=np.float64)
            rank_ic = _pearson(
                target_rank,
                prediction_rank,
                self._config.minimum_cross_section_size,
            )
            rows.append(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "fold_id": int(block["fold_id"].iloc[0]),
                    "n_obs": len(block),
                    "pearson_ic": (
                        np.nan if pearson.value is None else pearson.value
                    ),
                    "pearson_valid": pearson.valid,
                    "pearson_invalid_reason": pearson.invalid_reason,
                    "spearman_rank_ic": (
                        np.nan if rank_ic.value is None else rank_ic.value
                    ),
                    "rank_ic_valid": rank_ic.valid,
                    "rank_ic_invalid_reason": rank_ic.invalid_reason,
                }
            )
        frame = pd.DataFrame(rows, columns=_DATE_METRIC_COLUMNS)
        frame["fold_id"] = frame["fold_id"].astype(np.int64)
        frame["n_obs"] = frame["n_obs"].astype(np.int64)
        frame["pearson_valid"] = frame["pearson_valid"].astype(bool)
        frame["rank_ic_valid"] = frame["rank_ic_valid"].astype(bool)
        frame.index = pd.RangeIndex(len(frame))
        return frame

    def _fold_metrics(
        self,
        predictions: pd.DataFrame,
        date_metrics: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for fold_id, block in predictions.groupby("fold_id", sort=True):
            fold_dates = date_metrics.loc[date_metrics["fold_id"].eq(fold_id)]
            regression = _regression_metrics(
                block["target"].to_numpy(dtype=np.float64),
                block["prediction"].to_numpy(dtype=np.float64),
            )
            pearson = _summary(
                "pearson_ic",
                fold_dates["pearson_ic"],
                fold_dates["pearson_valid"],
            )
            rank_ic = _summary(
                "spearman_rank_ic",
                fold_dates["spearman_rank_ic"],
                fold_dates["rank_ic_valid"],
            )
            rows.append(
                {
                    "fold_id": int(fold_id),
                    "start_date": pd.Timestamp(block["trade_date"].min()),
                    "end_date": pd.Timestamp(block["trade_date"].max()),
                    "n_rows": len(block),
                    "n_dates": int(block["trade_date"].nunique()),
                    "mae": regression.mae,
                    "rmse": regression.rmse,
                    "r2": np.nan if regression.r2 is None else regression.r2,
                    "r2_valid": regression.r2_valid,
                    "r2_invalid_reason": regression.r2_invalid_reason,
                    "pearson_ic_mean": (
                        np.nan if pearson.mean is None else pearson.mean
                    ),
                    "pearson_ic_std": (
                        np.nan if pearson.std is None else pearson.std
                    ),
                    "pearson_icir": (
                        np.nan
                        if pearson.information_ratio is None
                        else pearson.information_ratio
                    ),
                    "pearson_valid_dates": pearson.valid_dates,
                    "pearson_invalid_dates": pearson.invalid_dates,
                    "rank_ic_mean": (
                        np.nan if rank_ic.mean is None else rank_ic.mean
                    ),
                    "rank_ic_std": (
                        np.nan if rank_ic.std is None else rank_ic.std
                    ),
                    "rank_icir": (
                        np.nan
                        if rank_ic.information_ratio is None
                        else rank_ic.information_ratio
                    ),
                    "rank_ic_valid_dates": rank_ic.valid_dates,
                    "rank_ic_invalid_dates": rank_ic.invalid_dates,
                }
            )
        frame = pd.DataFrame(rows, columns=_FOLD_METRIC_COLUMNS)
        integer_columns = [
            "fold_id",
            "n_rows",
            "n_dates",
            "pearson_valid_dates",
            "pearson_invalid_dates",
            "rank_ic_valid_dates",
            "rank_ic_invalid_dates",
        ]
        for column in integer_columns:
            frame[column] = frame[column].astype(np.int64)
        frame["r2_valid"] = frame["r2_valid"].astype(bool)
        frame.index = pd.RangeIndex(len(frame))
        return frame
