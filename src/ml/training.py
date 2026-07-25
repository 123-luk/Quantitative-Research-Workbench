"""Strict walk-forward model training and out-of-sample prediction contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.ml.contracts import METADATA_COLUMNS, MLDataset
from src.ml.models import (
    ModelFitAudit,
    ModelRegistry,
    create_default_model_registry,
)
from src.ml.splitting import WalkForwardPlan, WalkForwardSplit


class WalkForwardTrainingError(Exception):
    """Base error for walk-forward training configuration and execution."""


class WalkForwardTrainingConfigError(WalkForwardTrainingError):
    """Raised when the trainer or its model configuration is invalid."""


class WalkForwardTrainingDataError(WalkForwardTrainingError):
    """Raised when the dataset or plan cannot be consumed safely."""


class WalkForwardTrainingIntegrityError(WalkForwardTrainingError):
    """Raised when temporal, index, or output integrity is violated."""


class WalkForwardFoldError(WalkForwardTrainingError):
    """Raised when model creation, fitting, or prediction fails for one fold."""


def _iso_date(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _timestamp(
    field_name: str, value: object, *, optional: bool = False
) -> pd.Timestamp | None:
    if value is None and optional:
        return None
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise WalkForwardTrainingDataError(
            f"{field_name} must be a valid timestamp"
        ) from exc
    if pd.isna(result) or result.tz is not None:
        raise WalkForwardTrainingDataError(
            f"{field_name} must be a timezone-naive valid timestamp"
        )
    return result


def _json_scalar(field_name: str, value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WalkForwardTrainingConfigError(
                f"model_params[{field_name!r}] must be finite"
            )
        return value
    raise WalkForwardTrainingConfigError(
        f"model_params[{field_name!r}] must be a JSON-safe scalar; "
        f"received {type(value).__name__}"
    )


@dataclass(frozen=True)
class WalkForwardTrainingConfig:
    """Select one registry model and immutable scalar parameter overrides."""

    model_name: str
    model_params: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise WalkForwardTrainingConfigError(
                "model_name must be a non-empty string"
            )
        name = self.model_name.strip().lower()
        raw = {} if self.model_params is None else self.model_params
        if not isinstance(raw, Mapping):
            raise WalkForwardTrainingConfigError(
                "model_params must be a Mapping or None"
            )
        normalized: dict[str, object] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key:
                raise WalkForwardTrainingConfigError(
                    "model_params keys must be non-empty strings"
                )
            normalized[key] = _json_scalar(key, value)
        try:
            json.dumps(normalized, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise WalkForwardTrainingConfigError(
                "model_params must be directly JSON serializable"
            ) from exc
        object.__setattr__(self, "model_name", name)
        object.__setattr__(
            self, "model_params", MappingProxyType(dict(normalized))
        )

    @classmethod
    def from_dict(
        cls, values: Mapping[str, object]
    ) -> "WalkForwardTrainingConfig":
        """Build a config from exactly the supported top-level fields."""
        if not isinstance(values, Mapping):
            raise WalkForwardTrainingConfigError(
                "training config must be a Mapping"
            )
        allowed = {"model_name", "model_params"}
        unknown = [key for key in values if key not in allowed]
        if unknown:
            raise WalkForwardTrainingConfigError(
                f"unknown training config field(s): {unknown!r}"
            )
        if "model_name" not in values:
            raise WalkForwardTrainingConfigError(
                "training config requires model_name"
            )
        return cls(
            model_name=values["model_name"],  # type: ignore[arg-type]
            model_params=values.get("model_params"),  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        """Return a defensive JSON-safe configuration dictionary."""
        return {
            "model_name": self.model_name,
            "model_params": dict(self.model_params or {}),
        }


@dataclass(frozen=True)
class WalkForwardFoldAudit:
    """Immutable audit for one model fit and prediction block."""

    fold_id: int
    model_name: str
    train_rows: int
    validation_rows: int
    prediction_rows: int
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    validation_start_date: pd.Timestamp | None
    validation_end_date: pd.Timestamp | None
    prediction_start_date: pd.Timestamp
    prediction_end_date: pd.Timestamp
    validation_provided: bool
    validation_used_for_fit: bool
    model_fit_audit: ModelFitAudit

    def __post_init__(self) -> None:
        if (
            isinstance(self.fold_id, bool)
            or not isinstance(self.fold_id, Integral)
            or self.fold_id < 0
        ):
            raise WalkForwardTrainingIntegrityError(
                "fold_id must be a non-negative integer"
            )
        object.__setattr__(self, "fold_id", int(self.fold_id))
        if not isinstance(self.model_name, str) or not self.model_name:
            raise WalkForwardTrainingIntegrityError(
                "fold audit model_name must be non-empty"
            )
        for field_name in ("train_rows", "validation_rows", "prediction_rows"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0
            ):
                raise WalkForwardTrainingIntegrityError(
                    f"{field_name} must be a non-negative integer"
                )
            object.__setattr__(self, field_name, int(value))
        if self.prediction_rows == 0:
            raise WalkForwardTrainingIntegrityError(
                "prediction_rows must be positive"
            )
        for field_name in (
            "train_start_date",
            "train_end_date",
            "prediction_start_date",
            "prediction_end_date",
        ):
            object.__setattr__(
                self, field_name, _timestamp(field_name, getattr(self, field_name))
            )
        for field_name in ("validation_start_date", "validation_end_date"):
            object.__setattr__(
                self,
                field_name,
                _timestamp(
                    field_name, getattr(self, field_name), optional=True
                ),
            )
        if not isinstance(self.model_fit_audit, ModelFitAudit):
            raise WalkForwardTrainingIntegrityError(
                "model_fit_audit must be a ModelFitAudit"
            )
        if self.model_fit_audit.model_name != self.model_name:
            raise WalkForwardTrainingIntegrityError(
                "fold model_name must match model_fit_audit"
            )
        if (
            self.validation_used_for_fit
            != self.model_fit_audit.validation_used_for_fit
        ):
            raise WalkForwardTrainingIntegrityError(
                "validation_used_for_fit must come from model_fit_audit"
            )
        if self.validation_provided != self.model_fit_audit.validation_provided:
            raise WalkForwardTrainingIntegrityError(
                "validation_provided must match model_fit_audit"
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe audit without indices, samples, or estimators."""
        return {
            "fold_id": self.fold_id,
            "model_name": self.model_name,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "prediction_rows": self.prediction_rows,
            "train_start_date": _iso_date(self.train_start_date),
            "train_end_date": _iso_date(self.train_end_date),
            "validation_start_date": _iso_date(self.validation_start_date),
            "validation_end_date": _iso_date(self.validation_end_date),
            "prediction_start_date": _iso_date(self.prediction_start_date),
            "prediction_end_date": _iso_date(self.prediction_end_date),
            "validation_provided": self.validation_provided,
            "validation_used_for_fit": self.validation_used_for_fit,
            "model_fit_audit": self.model_fit_audit.as_dict(),
        }


@dataclass(frozen=True)
class WalkForwardTrainingAudit:
    """Immutable summary across every successful walk-forward fold."""

    model_name: str
    resolved_model_parameters: tuple[tuple[str, object], ...]
    n_folds: int
    n_prediction_rows: int
    n_prediction_dates: int
    first_prediction_date: pd.Timestamp
    last_prediction_date: pd.Timestamp
    source_label_name: str | None
    fold_audits: tuple[WalkForwardFoldAudit, ...]

    def __post_init__(self) -> None:
        folds = tuple(self.fold_audits)
        if not folds or any(
            not isinstance(fold, WalkForwardFoldAudit) for fold in folds
        ):
            raise WalkForwardTrainingIntegrityError(
                "fold_audits must contain at least one fold audit"
            )
        object.__setattr__(self, "fold_audits", folds)
        if self.n_folds != len(folds):
            raise WalkForwardTrainingIntegrityError(
                "n_folds must equal fold_audits length"
            )
        if self.n_prediction_rows != sum(
            fold.prediction_rows for fold in folds
        ):
            raise WalkForwardTrainingIntegrityError(
                "n_prediction_rows must equal fold prediction row counts"
            )
        if self.n_prediction_dates <= 0:
            raise WalkForwardTrainingIntegrityError(
                "n_prediction_dates must be positive"
            )
        parameters = tuple(self.resolved_model_parameters)
        object.__setattr__(self, "resolved_model_parameters", parameters)
        for fold in folds:
            if fold.model_name != self.model_name:
                raise WalkForwardTrainingIntegrityError(
                    "all fold model names must match the training audit"
                )
            if fold.model_fit_audit.resolved_parameters != parameters:
                raise WalkForwardTrainingIntegrityError(
                    "resolved model parameters differ across folds"
                )
        object.__setattr__(
            self,
            "first_prediction_date",
            _timestamp("first_prediction_date", self.first_prediction_date),
        )
        object.__setattr__(
            self,
            "last_prediction_date",
            _timestamp("last_prediction_date", self.last_prediction_date),
        )
        if self.first_prediction_date > self.last_prediction_date:
            raise WalkForwardTrainingIntegrityError(
                "first_prediction_date cannot exceed last_prediction_date"
            )
        if self.source_label_name is not None and (
            not isinstance(self.source_label_name, str)
            or not self.source_label_name
        ):
            raise WalkForwardTrainingIntegrityError(
                "source_label_name must be non-empty or None"
            )
        json.dumps(self.as_dict(), allow_nan=False)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe summary without prediction details or models."""
        return {
            "model_name": self.model_name,
            "resolved_model_parameters": dict(
                self.resolved_model_parameters
            ),
            "n_folds": self.n_folds,
            "n_prediction_rows": self.n_prediction_rows,
            "n_prediction_dates": self.n_prediction_dates,
            "first_prediction_date": _iso_date(self.first_prediction_date),
            "last_prediction_date": _iso_date(self.last_prediction_date),
            "source_label_name": self.source_label_name,
            "fold_audits": [fold.as_dict() for fold in self.fold_audits],
        }


class WalkForwardTrainingResult:
    """Defensively expose OOS predictions and immutable training audit."""

    def __init__(
        self,
        predictions: pd.DataFrame,
        audit: WalkForwardTrainingAudit,
    ) -> None:
        if not isinstance(predictions, pd.DataFrame):
            raise WalkForwardTrainingDataError(
                "predictions must be a pandas DataFrame"
            )
        if not isinstance(audit, WalkForwardTrainingAudit):
            raise WalkForwardTrainingDataError(
                "audit must be a WalkForwardTrainingAudit"
            )
        self._predictions = predictions.copy(deep=True)
        self._audit = audit

    @property
    def predictions(self) -> pd.DataFrame:
        """Return a deep defensive copy of OOS prediction rows."""
        return self._predictions.copy(deep=True)

    @property
    def audit(self) -> WalkForwardTrainingAudit:
        """Return the immutable aggregate audit."""
        return self._audit


_PREDICTION_COLUMNS = [
    "trade_date",
    "ts_code",
    "entry_trade_date",
    "exit_trade_date",
    "target",
    "prediction",
    "fold_id",
]


class WalkForwardTrainer:
    """Train a fresh registry model per split and combine strict OOS predictions."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        if registry is not None and not isinstance(registry, ModelRegistry):
            raise WalkForwardTrainingConfigError(
                "registry must be a ModelRegistry or None"
            )
        self._registry = (
            create_default_model_registry() if registry is None else registry
        )

    def run(
        self,
        dataset: MLDataset,
        plan: WalkForwardPlan,
        config: WalkForwardTrainingConfig,
    ) -> WalkForwardTrainingResult:
        """Execute every split atomically and return only OOS rows and audits."""
        if not isinstance(dataset, MLDataset):
            raise WalkForwardTrainingDataError(
                "dataset must be an MLDataset"
            )
        if not isinstance(plan, WalkForwardPlan):
            raise WalkForwardTrainingDataError(
                "plan must be a WalkForwardPlan"
            )
        if not isinstance(config, WalkForwardTrainingConfig):
            raise WalkForwardTrainingConfigError(
                "config must be a WalkForwardTrainingConfig"
            )

        features = dataset.features
        labels = dataset.labels
        metadata = dataset.metadata
        self._validate_dataset(dataset, features, labels, metadata)
        self._validate_plan(plan, metadata, len(features))

        prediction_frames: list[pd.DataFrame] = []
        fold_audits: list[WalkForwardFoldAudit] = []
        resolved_parameters: tuple[tuple[str, object], ...] | None = None

        for fold_id, split in enumerate(plan.splits):
            train_x = self._take_positions(features, split.train_indices)
            train_y = self._take_positions(labels, split.train_indices)
            valid_x = self._take_positions(
                features, split.validation_indices
            )
            valid_y = self._take_positions(labels, split.validation_indices)
            prediction_x = self._take_positions(
                features, split.prediction_indices
            )

            try:
                adapter = self._registry.create(
                    config.model_name, dict(config.model_params or {})
                )
                fit_audit = adapter.fit(
                    train_x,
                    train_y,
                    valid_x,
                    valid_y,
                )
                prediction = adapter.predict(prediction_x)
            except Exception as exc:
                raise self._fold_error(
                    fold_id, config.model_name, split, exc
                ) from exc

            self._validate_fit_audit(
                fit_audit,
                config.model_name,
                tuple(features.columns),
                len(train_x),
                len(valid_x),
            )
            self._validate_prediction(
                prediction, prediction_x.index, fold_id
            )
            if resolved_parameters is None:
                resolved_parameters = fit_audit.resolved_parameters
            elif fit_audit.resolved_parameters != resolved_parameters:
                raise WalkForwardTrainingIntegrityError(
                    f"fold_id={fold_id} resolved model parameters differ"
                )

            prediction_meta = self._take_positions(
                metadata, split.prediction_indices
            )
            prediction_target = self._take_positions(
                labels, split.prediction_indices
            )
            fold_frame = prediction_meta.copy(deep=True)
            fold_frame["target"] = prediction_target.astype(np.float64)
            fold_frame["prediction"] = prediction.astype(np.float64)
            fold_frame["fold_id"] = np.int64(fold_id)
            fold_frame = fold_frame.loc[:, _PREDICTION_COLUMNS]
            prediction_frames.append(fold_frame)
            fold_audits.append(
                WalkForwardFoldAudit(
                    fold_id=fold_id,
                    model_name=config.model_name,
                    train_rows=len(train_x),
                    validation_rows=len(valid_x),
                    prediction_rows=len(prediction_x),
                    train_start_date=split.train_start_date,
                    train_end_date=split.train_end_date,
                    validation_start_date=split.validation_start_date,
                    validation_end_date=split.validation_end_date,
                    prediction_start_date=split.prediction_start_date,
                    prediction_end_date=split.prediction_end_date,
                    validation_provided=True,
                    validation_used_for_fit=fit_audit.validation_used_for_fit,
                    model_fit_audit=fit_audit,
                )
            )

        if resolved_parameters is None:
            raise WalkForwardTrainingDataError(
                "plan must contain at least one split"
            )
        predictions = pd.concat(prediction_frames, axis=0).sort_index()
        predictions.index.name = "dataset_index"
        self._validate_output(predictions)
        prediction_dates = predictions["trade_date"].drop_duplicates()
        audit = WalkForwardTrainingAudit(
            model_name=config.model_name,
            resolved_model_parameters=resolved_parameters,
            n_folds=len(fold_audits),
            n_prediction_rows=len(predictions),
            n_prediction_dates=int(predictions["trade_date"].nunique()),
            first_prediction_date=pd.Timestamp(prediction_dates.iloc[0]),
            last_prediction_date=pd.Timestamp(prediction_dates.iloc[-1]),
            source_label_name=dataset.label_name,
            fold_audits=tuple(fold_audits),
        )
        return WalkForwardTrainingResult(predictions, audit)

    @staticmethod
    def _take_positions(
        value: pd.DataFrame | pd.Series, positions: tuple[int, ...]
    ) -> pd.DataFrame | pd.Series:
        """Extract row positions with iloc while preserving dataset labels."""
        return value.iloc[list(positions)].copy(deep=True)

    @staticmethod
    def _validate_dataset(
        dataset: MLDataset,
        features: pd.DataFrame,
        labels: pd.Series,
        metadata: pd.DataFrame,
    ) -> None:
        expected_index = pd.RangeIndex(len(features))
        if (
            not features.index.equals(expected_index)
            or not labels.index.equals(expected_index)
            or not metadata.index.equals(expected_index)
        ):
            raise WalkForwardTrainingDataError(
                "dataset objects must share RangeIndex(0, n)"
            )
        if tuple(features.columns) != dataset.feature_names:
            raise WalkForwardTrainingDataError(
                "dataset feature columns do not match feature_names"
            )
        if labels.name != dataset.label_name:
            raise WalkForwardTrainingDataError(
                "dataset labels.name does not match label_name"
            )
        if tuple(metadata.columns) != tuple(METADATA_COLUMNS):
            raise WalkForwardTrainingDataError(
                "dataset metadata columns are invalid"
            )
        if len({len(features), len(labels), len(metadata)}) != 1:
            raise WalkForwardTrainingDataError(
                "dataset object row counts must match"
            )
        label_values = labels.to_numpy(dtype=np.float64)
        if not np.isfinite(label_values).all():
            raise WalkForwardTrainingDataError(
                "dataset labels must all be finite"
            )
        if metadata.isna().any().any():
            raise WalkForwardTrainingDataError(
                "dataset metadata must not contain missing values"
            )
        for column in ("trade_date", "entry_trade_date", "exit_trade_date"):
            if not pd.api.types.is_datetime64_ns_dtype(metadata[column]):
                raise WalkForwardTrainingDataError(
                    f"metadata {column} must use datetime64[ns]"
                )
            if getattr(metadata[column].dt, "tz", None) is not None:
                raise WalkForwardTrainingDataError(
                    f"metadata {column} must be timezone-naive"
                )
        if not metadata["trade_date"].is_monotonic_increasing:
            raise WalkForwardTrainingDataError(
                "metadata trade_date must be sorted"
            )
        if (
            metadata.duplicated(["trade_date", "ts_code"], keep=False).any()
        ):
            raise WalkForwardTrainingDataError(
                "metadata prediction keys must be unique"
            )
        codes = metadata["ts_code"].astype("string")
        if codes.isna().any() or codes.str.strip().eq("").any():
            raise WalkForwardTrainingDataError(
                "metadata ts_code must contain non-empty strings"
            )

    def _validate_plan(
        self,
        plan: WalkForwardPlan,
        metadata: pd.DataFrame,
        n_rows: int,
    ) -> None:
        try:
            splits = tuple(plan.splits)
        except (AttributeError, TypeError) as exc:
            raise WalkForwardTrainingDataError(
                "plan splits must be a non-empty tuple"
            ) from exc
        if not splits:
            raise WalkForwardTrainingDataError(
                "plan must contain at least one split"
            )
        if any(not isinstance(split, WalkForwardSplit) for split in splits):
            raise WalkForwardTrainingDataError(
                "plan splits must contain WalkForwardSplit values"
            )
        dataset_dates = tuple(
            pd.Timestamp(value)
            for value in metadata["trade_date"].drop_duplicates()
        )
        if tuple(plan.all_score_dates) != dataset_dates:
            raise WalkForwardTrainingIntegrityError(
                "plan all_score_dates do not match the dataset"
            )

        seen_prediction_indices: set[int] = set()
        seen_prediction_dates: set[pd.Timestamp] = set()
        combined_prediction_dates: list[pd.Timestamp] = []
        for fold_id, split in enumerate(splits):
            if split.retrain_id != fold_id + 1:
                raise WalkForwardTrainingIntegrityError(
                    "split retrain_id sequence does not match plan order"
                )
            train = self._position_tuple(
                "train_indices", split.train_indices, n_rows
            )
            valid = self._position_tuple(
                "validation_indices", split.validation_indices, n_rows
            )
            prediction = self._position_tuple(
                "prediction_indices", split.prediction_indices, n_rows
            )
            if (
                set(train) & set(valid)
                or set(train) & set(prediction)
                or set(valid) & set(prediction)
            ):
                raise WalkForwardTrainingIntegrityError(
                    f"fold_id={fold_id} partition indices overlap"
                )
            duplicate_prediction = seen_prediction_indices & set(prediction)
            if duplicate_prediction:
                raise WalkForwardTrainingIntegrityError(
                    "prediction indices overlap across folds"
                )
            seen_prediction_indices.update(prediction)

            self._validate_partition(
                fold_id,
                "train",
                train,
                split.train_dates,
                metadata,
            )
            self._validate_partition(
                fold_id,
                "validation",
                valid,
                split.validation_dates,
                metadata,
            )
            self._validate_partition(
                fold_id,
                "prediction",
                prediction,
                split.prediction_dates,
                metadata,
            )
            prediction_dates = tuple(
                pd.Timestamp(value) for value in split.prediction_dates
            )
            if seen_prediction_dates & set(prediction_dates):
                raise WalkForwardTrainingIntegrityError(
                    "prediction dates overlap across folds"
                )
            seen_prediction_dates.update(prediction_dates)
            combined_prediction_dates.extend(prediction_dates)
            self._validate_temporal_cutoffs(fold_id, split, metadata)
            if (
                split.n_train_rows != len(train)
                or split.n_validation_rows != len(valid)
                or split.n_prediction_rows != len(prediction)
            ):
                raise WalkForwardTrainingIntegrityError(
                    f"fold_id={fold_id} split row counts are inconsistent"
                )

        first_prediction_date = pd.Timestamp(plan.first_prediction_date)
        if first_prediction_date not in dataset_dates:
            raise WalkForwardTrainingIntegrityError(
                "plan first_prediction_date is not a dataset score date"
            )
        first_position = dataset_dates.index(first_prediction_date)
        skipped_dates = tuple(
            pd.Timestamp(value)
            for value in plan.skipped_initial_prediction_dates
        )
        if skipped_dates != dataset_dates[:first_position]:
            raise WalkForwardTrainingIntegrityError(
                "plan skipped prediction dates do not match the dataset prefix"
            )
        if tuple(combined_prediction_dates) != tuple(
            dataset_dates[first_position:]
        ):
            raise WalkForwardTrainingIntegrityError(
                "prediction dates must cover the dataset suffix without gaps"
            )
        if (
            plan.first_prediction_date != combined_prediction_dates[0]
            or plan.last_prediction_date != combined_prediction_dates[-1]
        ):
            raise WalkForwardTrainingIntegrityError(
                "plan prediction boundaries are inconsistent"
            )

    @staticmethod
    def _position_tuple(
        field_name: str, values: object, n_rows: int
    ) -> tuple[int, ...]:
        try:
            raw = tuple(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise WalkForwardTrainingDataError(
                f"{field_name} must contain integer row positions"
            ) from exc
        if not raw:
            raise WalkForwardTrainingDataError(
                f"{field_name} must not be empty"
            )
        normalized: list[int] = []
        for value in raw:
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0
                or value >= n_rows
            ):
                raise WalkForwardTrainingDataError(
                    f"{field_name} contains invalid row position {value!r}"
                )
            normalized.append(int(value))
        result = tuple(normalized)
        if len(set(result)) != len(result):
            raise WalkForwardTrainingIntegrityError(
                f"{field_name} contains duplicate row positions"
            )
        return result

    @staticmethod
    def _validate_partition(
        fold_id: int,
        partition_name: str,
        positions: tuple[int, ...],
        declared_dates: tuple[pd.Timestamp, ...],
        metadata: pd.DataFrame,
    ) -> None:
        dates = tuple(pd.Timestamp(value) for value in declared_dates)
        if not dates or dates != tuple(sorted(set(dates))):
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} {partition_name} dates are invalid"
            )
        expected_positions = tuple(
            int(value)
            for value in metadata.index[
                metadata["trade_date"].isin(dates)
            ]
        )
        if positions != expected_positions:
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} {partition_name} must contain complete "
                "date cross-sections in dataset order"
            )
        actual_dates = tuple(
            pd.Timestamp(value)
            for value in metadata.iloc[list(positions)]["trade_date"]
            .drop_duplicates()
        )
        if actual_dates != dates:
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} {partition_name} dates do not match indices"
            )

    @staticmethod
    def _validate_temporal_cutoffs(
        fold_id: int,
        split: WalkForwardSplit,
        metadata: pd.DataFrame,
    ) -> None:
        train_meta = metadata.iloc[list(split.train_indices)]
        valid_meta = metadata.iloc[list(split.validation_indices)]
        prediction_meta = metadata.iloc[list(split.prediction_indices)]
        train_end = pd.Timestamp(train_meta["trade_date"].max())
        valid_start = pd.Timestamp(valid_meta["trade_date"].min())
        valid_end = pd.Timestamp(valid_meta["trade_date"].max())
        prediction_start = pd.Timestamp(
            prediction_meta["trade_date"].min()
        )
        declared_boundaries = {
            "train_start_date": split.train_start_date,
            "train_end_date": split.train_end_date,
            "validation_start_date": split.validation_start_date,
            "validation_end_date": split.validation_end_date,
            "prediction_start_date": split.prediction_start_date,
            "prediction_end_date": split.prediction_end_date,
        }
        actual_boundaries = {
            "train_start_date": pd.Timestamp(train_meta["trade_date"].min()),
            "train_end_date": train_end,
            "validation_start_date": valid_start,
            "validation_end_date": valid_end,
            "prediction_start_date": prediction_start,
            "prediction_end_date": pd.Timestamp(
                prediction_meta["trade_date"].max()
            ),
        }
        for field_name, actual in actual_boundaries.items():
            declared = _timestamp(
                f"split.{field_name}", declared_boundaries[field_name]
            )
            if declared != actual:
                raise WalkForwardTrainingIntegrityError(
                    f"fold_id={fold_id} split {field_name} does not match "
                    "metadata"
                )
        if not train_end < valid_start < prediction_start:
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} train/validation/prediction order is invalid"
            )
        if not valid_end < prediction_start:
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} validation must precede prediction"
            )
        max_train_exit = pd.Timestamp(train_meta["exit_trade_date"].max())
        max_valid_exit = pd.Timestamp(valid_meta["exit_trade_date"].max())
        if not max_train_exit < valid_start:
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} training max exit_trade_date violates "
                "the strict validation cutoff"
            )
        if not max_valid_exit < prediction_start:
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} validation max exit_trade_date violates "
                "the strict prediction cutoff"
            )
        if (
            max_train_exit != pd.Timestamp(split.max_train_exit_date)
            or max_valid_exit != pd.Timestamp(split.max_validation_exit_date)
        ):
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} split exit-date audit is inconsistent"
            )

    @staticmethod
    def _validate_fit_audit(
        audit: object,
        model_name: str,
        feature_names: tuple[str, ...],
        train_rows: int,
        validation_rows: int,
    ) -> None:
        if not isinstance(audit, ModelFitAudit):
            raise WalkForwardTrainingIntegrityError(
                "adapter.fit must return ModelFitAudit"
            )
        if audit.model_name != model_name:
            raise WalkForwardTrainingIntegrityError(
                "fit audit model_name differs from training config"
            )
        if audit.feature_names != feature_names:
            raise WalkForwardTrainingIntegrityError(
                "fit audit feature_names differ from the dataset"
            )
        if (
            audit.n_train_rows != train_rows
            or audit.n_validation_rows != validation_rows
        ):
            raise WalkForwardTrainingIntegrityError(
                "fit audit partition row counts are inconsistent"
            )

    @staticmethod
    def _validate_prediction(
        prediction: object, expected_index: pd.Index, fold_id: int
    ) -> None:
        if not isinstance(prediction, pd.Series):
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} adapter.predict must return a Series"
            )
        if (
            len(prediction) != len(expected_index)
            or not prediction.index.equals(expected_index)
        ):
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} prediction index or length is invalid"
            )
        if prediction.name != "prediction":
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} prediction name must be 'prediction'"
            )
        try:
            values = prediction.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} predictions must be numeric"
            ) from exc
        if not np.isfinite(values).all():
            raise WalkForwardTrainingIntegrityError(
                f"fold_id={fold_id} predictions must all be finite"
            )

    @staticmethod
    def _validate_output(predictions: pd.DataFrame) -> None:
        if list(predictions.columns) != _PREDICTION_COLUMNS:
            raise WalkForwardTrainingIntegrityError(
                "prediction output columns are invalid"
            )
        if not predictions.index.is_unique or not predictions.index.is_monotonic_increasing:
            raise WalkForwardTrainingIntegrityError(
                "prediction dataset index must be unique and increasing"
            )
        if predictions.index.name != "dataset_index":
            raise WalkForwardTrainingIntegrityError(
                "prediction index name must be dataset_index"
            )
        if predictions.duplicated(["trade_date", "ts_code"]).any():
            raise WalkForwardTrainingIntegrityError(
                "prediction keys must be unique"
            )
        if not np.isfinite(
            predictions["prediction"].to_numpy(dtype=np.float64)
        ).all() or not np.isfinite(
            predictions["target"].to_numpy(dtype=np.float64)
        ).all():
            raise WalkForwardTrainingIntegrityError(
                "prediction and target values must be finite"
            )
        predictions["prediction"] = predictions["prediction"].astype(
            np.float64
        )
        predictions["target"] = predictions["target"].astype(np.float64)
        predictions["fold_id"] = predictions["fold_id"].astype(np.int64)
        if predictions.loc[:, list(METADATA_COLUMNS)].isna().any().any():
            raise WalkForwardTrainingIntegrityError(
                "prediction metadata must not contain missing values"
            )

    @staticmethod
    def _fold_error(
        fold_id: int,
        model_name: str,
        split: WalkForwardSplit,
        cause: Exception,
    ) -> WalkForwardFoldError:
        return WalkForwardFoldError(
            f"fold {fold_id} failed for model {model_name}; "
            f"train={_iso_date(split.train_start_date)}.."
            f"{_iso_date(split.train_end_date)}; "
            f"validation={_iso_date(split.validation_start_date)}.."
            f"{_iso_date(split.validation_end_date)}; "
            f"prediction={_iso_date(split.prediction_start_date)}.."
            f"{_iso_date(split.prediction_end_date)}; "
            f"cause={type(cause).__name__}"
        )
