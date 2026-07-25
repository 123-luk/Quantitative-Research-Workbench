"""Strict walk-forward out-of-sample permutation importance contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from src.ml.contracts import METADATA_COLUMNS, MLDataset
from src.ml.models import ModelFitAudit, ModelRegistry, create_default_model_registry
from src.ml.splitting import WalkForwardPlan, WalkForwardSplit


class WalkForwardPermutationImportanceError(Exception):
    """Base error for strict OOS permutation importance."""


class WalkForwardPermutationImportanceConfigError(
    WalkForwardPermutationImportanceError
):
    """Raised when importance configuration is invalid."""


class WalkForwardPermutationImportanceDataError(
    WalkForwardPermutationImportanceError
):
    """Raised when dataset or plan data cannot be consumed."""


class WalkForwardPermutationImportanceIntegrityError(
    WalkForwardPermutationImportanceError
):
    """Raised when plan, model output, or aggregation integrity fails."""


class WalkForwardPermutationImportanceFoldError(
    WalkForwardPermutationImportanceError
):
    """Raised when model creation, fitting, prediction, or scoring fails."""


_REPEAT_COLUMNS = [
    "fold_id",
    "feature_name",
    "feature_position",
    "repeat_id",
    "baseline_score",
    "permuted_score",
    "importance",
]

_FOLD_COLUMNS = [
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

_FEATURE_COLUMNS = [
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


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None else pd.Timestamp(value).strftime("%Y-%m-%d")


def _timestamp(
    field_name: str, value: object, *, optional: bool = False
) -> pd.Timestamp | None:
    if value is None and optional:
        return None
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise WalkForwardPermutationImportanceDataError(
            f"{field_name} must be a valid timestamp"
        ) from exc
    if pd.isna(result) or result.tz is not None:
        raise WalkForwardPermutationImportanceDataError(
            f"{field_name} must be timezone-naive and valid"
        )
    return result


def _config_scalar(name: str, value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WalkForwardPermutationImportanceConfigError(
                f"model_params[{name!r}] must be finite"
            )
        return value
    raise WalkForwardPermutationImportanceConfigError(
        f"model_params[{name!r}] must be a JSON-safe scalar"
    )


@dataclass(frozen=True)
class WalkForwardPermutationImportanceConfig:
    """Immutable model, scoring, and deterministic permutation configuration."""

    model_name: str
    model_params: Mapping[str, object] | None = None
    scoring: str = "rmse"
    n_repeats: int = 5
    random_state: int = 42
    permutation_scope: str = "within_trade_date"

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise WalkForwardPermutationImportanceConfigError(
                "model_name must be a non-empty string"
            )
        model_name = self.model_name.strip().lower()
        raw_params = {} if self.model_params is None else self.model_params
        if not isinstance(raw_params, Mapping):
            raise WalkForwardPermutationImportanceConfigError(
                "model_params must be a Mapping or None"
            )
        params: dict[str, object] = {}
        for key, value in raw_params.items():
            if not isinstance(key, str) or not key:
                raise WalkForwardPermutationImportanceConfigError(
                    "model_params keys must be non-empty strings"
                )
            params[key] = _config_scalar(key, value)
        if not isinstance(self.scoring, str):
            raise WalkForwardPermutationImportanceConfigError(
                "scoring must be rmse or mae"
            )
        scoring = self.scoring.strip().lower()
        if scoring not in {"rmse", "mae"}:
            raise WalkForwardPermutationImportanceConfigError(
                "scoring must be rmse or mae"
            )
        if (
            isinstance(self.n_repeats, bool)
            or not isinstance(self.n_repeats, Integral)
            or int(self.n_repeats) < 1
        ):
            raise WalkForwardPermutationImportanceConfigError(
                "n_repeats must be an integer >= 1"
            )
        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, Integral)
            or int(self.random_state) < 0
        ):
            raise WalkForwardPermutationImportanceConfigError(
                "random_state must be a non-negative integer"
            )
        if not isinstance(self.permutation_scope, str):
            raise WalkForwardPermutationImportanceConfigError(
                "permutation_scope must be within_trade_date"
            )
        scope = self.permutation_scope.strip().lower()
        if scope != "within_trade_date":
            raise WalkForwardPermutationImportanceConfigError(
                "permutation_scope must be within_trade_date"
            )
        try:
            json.dumps(params, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise WalkForwardPermutationImportanceConfigError(
                "model_params must be directly JSON serializable"
            ) from exc
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "model_params", MappingProxyType(dict(params)))
        object.__setattr__(self, "scoring", scoring)
        object.__setattr__(self, "n_repeats", int(self.n_repeats))
        object.__setattr__(self, "random_state", int(self.random_state))
        object.__setattr__(self, "permutation_scope", scope)

    @classmethod
    def from_dict(
        cls, values: Mapping[str, object]
    ) -> "WalkForwardPermutationImportanceConfig":
        """Build from exactly the six supported fields."""
        if not isinstance(values, Mapping):
            raise WalkForwardPermutationImportanceConfigError(
                "importance config must be a Mapping"
            )
        allowed = {
            "model_name",
            "model_params",
            "scoring",
            "n_repeats",
            "random_state",
            "permutation_scope",
        }
        unknown = [key for key in values if key not in allowed]
        if unknown:
            raise WalkForwardPermutationImportanceConfigError(
                f"unknown importance config field(s): {unknown!r}"
            )
        if "model_name" not in values:
            raise WalkForwardPermutationImportanceConfigError(
                "importance config requires model_name"
            )
        return cls(
            model_name=values["model_name"],  # type: ignore[arg-type]
            model_params=values.get("model_params"),  # type: ignore[arg-type]
            scoring=values.get("scoring", "rmse"),  # type: ignore[arg-type]
            n_repeats=values.get("n_repeats", 5),  # type: ignore[arg-type]
            random_state=values.get("random_state", 42),  # type: ignore[arg-type]
            permutation_scope=values.get(
                "permutation_scope", "within_trade_date"
            ),  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        """Return a fresh JSON-safe configuration dictionary."""
        return {
            "model_name": self.model_name,
            "model_params": dict(self.model_params or {}),
            "scoring": self.scoring,
            "n_repeats": self.n_repeats,
            "random_state": self.random_state,
            "permutation_scope": self.permutation_scope,
        }


def _positive_int(field_name: str, value: object, *, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < lower
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise WalkForwardPermutationImportanceIntegrityError(
            f"{field_name} must be a {qualifier} integer"
        )
    return int(value)


def _finite(field_name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise WalkForwardPermutationImportanceIntegrityError(
            f"{field_name} must be finite"
        ) from exc
    if not math.isfinite(result):
        raise WalkForwardPermutationImportanceIntegrityError(
            f"{field_name} must be finite"
        )
    return result


@dataclass(frozen=True)
class PermutationImportanceFoldAudit:
    """Immutable sample-free audit for one fitted fold."""

    fold_id: int
    model_name: str
    train_rows: int
    validation_rows: int
    prediction_rows: int
    prediction_dates: int
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    validation_start_date: pd.Timestamp | None
    validation_end_date: pd.Timestamp | None
    prediction_start_date: pd.Timestamp
    prediction_end_date: pd.Timestamp
    baseline_score: float
    scoring: str
    validation_used_for_fit: bool
    n_features: int
    n_repeats: int
    n_permutations: int
    model_fit_audit: ModelFitAudit

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fold_id", _positive_int("fold_id", self.fold_id, allow_zero=True)
        )
        if not isinstance(self.model_name, str) or not self.model_name:
            raise WalkForwardPermutationImportanceIntegrityError(
                "fold audit model_name must be non-empty"
            )
        for field_name in (
            "train_rows",
            "prediction_rows",
            "prediction_dates",
            "n_features",
            "n_repeats",
            "n_permutations",
        ):
            object.__setattr__(
                self, field_name, _positive_int(field_name, getattr(self, field_name))
            )
        object.__setattr__(
            self,
            "validation_rows",
            _positive_int(
                "validation_rows", self.validation_rows, allow_zero=True
            ),
        )
        if self.n_permutations != self.n_features * self.n_repeats:
            raise WalkForwardPermutationImportanceIntegrityError(
                "n_permutations must equal n_features times n_repeats"
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
        object.__setattr__(
            self,
            "baseline_score",
            _finite("baseline_score", self.baseline_score),
        )
        if self.scoring not in {"rmse", "mae"}:
            raise WalkForwardPermutationImportanceIntegrityError(
                "fold audit scoring is invalid"
            )
        if not isinstance(self.validation_used_for_fit, bool):
            raise WalkForwardPermutationImportanceIntegrityError(
                "validation_used_for_fit must be bool"
            )
        if not isinstance(self.model_fit_audit, ModelFitAudit):
            raise WalkForwardPermutationImportanceIntegrityError(
                "model_fit_audit must be ModelFitAudit"
            )
        if self.model_fit_audit.model_name != self.model_name:
            raise WalkForwardPermutationImportanceIntegrityError(
                "fold model_name must match model_fit_audit"
            )
        if (
            self.validation_used_for_fit
            != self.model_fit_audit.validation_used_for_fit
        ):
            raise WalkForwardPermutationImportanceIntegrityError(
                "validation_used_for_fit must come from model_fit_audit"
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe fold audit."""
        return {
            "fold_id": self.fold_id,
            "model_name": self.model_name,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "prediction_rows": self.prediction_rows,
            "prediction_dates": self.prediction_dates,
            "train_start_date": _iso(self.train_start_date),
            "train_end_date": _iso(self.train_end_date),
            "validation_start_date": _iso(self.validation_start_date),
            "validation_end_date": _iso(self.validation_end_date),
            "prediction_start_date": _iso(self.prediction_start_date),
            "prediction_end_date": _iso(self.prediction_end_date),
            "baseline_score": self.baseline_score,
            "scoring": self.scoring,
            "validation_used_for_fit": self.validation_used_for_fit,
            "n_features": self.n_features,
            "n_repeats": self.n_repeats,
            "n_permutations": self.n_permutations,
            "model_fit_audit": self.model_fit_audit.as_dict(),
        }


@dataclass(frozen=True)
class WalkForwardPermutationImportanceAudit:
    """Immutable aggregate importance audit."""

    model_name: str
    resolved_model_parameters: tuple[tuple[str, object], ...]
    scoring: str
    score_direction: str
    permutation_scope: str
    n_repeats: int
    random_state: int
    n_folds: int
    n_features: int
    n_repeat_evaluations: int
    feature_names: tuple[str, ...]
    first_prediction_date: pd.Timestamp
    last_prediction_date: pd.Timestamp
    fold_audits: tuple[PermutationImportanceFoldAudit, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise WalkForwardPermutationImportanceIntegrityError(
                "importance audit model_name must be non-empty"
            )
        if self.scoring not in {"rmse", "mae"}:
            raise WalkForwardPermutationImportanceIntegrityError(
                "importance audit scoring is invalid"
            )
        if self.score_direction != "lower_is_better":
            raise WalkForwardPermutationImportanceIntegrityError(
                "score_direction must be lower_is_better"
            )
        if self.permutation_scope != "within_trade_date":
            raise WalkForwardPermutationImportanceIntegrityError(
                "importance audit permutation_scope is invalid"
            )
        for field_name in ("n_repeats", "n_folds", "n_features"):
            object.__setattr__(
                self, field_name, _positive_int(field_name, getattr(self, field_name))
            )
        object.__setattr__(
            self,
            "random_state",
            _positive_int("random_state", self.random_state, allow_zero=True),
        )
        object.__setattr__(
            self,
            "n_repeat_evaluations",
            _positive_int(
                "n_repeat_evaluations", self.n_repeat_evaluations
            ),
        )
        if self.n_repeat_evaluations != (
            self.n_folds * self.n_features * self.n_repeats
        ):
            raise WalkForwardPermutationImportanceIntegrityError(
                "n_repeat_evaluations count is inconsistent"
            )
        names = tuple(self.feature_names)
        if (
            len(names) != self.n_features
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise WalkForwardPermutationImportanceIntegrityError(
                "feature_names are invalid"
            )
        object.__setattr__(self, "feature_names", names)
        folds = tuple(self.fold_audits)
        if len(folds) != self.n_folds or any(
            not isinstance(fold, PermutationImportanceFoldAudit)
            for fold in folds
        ):
            raise WalkForwardPermutationImportanceIntegrityError(
                "fold_audits must match n_folds"
            )
        object.__setattr__(self, "fold_audits", folds)
        parameters = tuple(self.resolved_model_parameters)
        object.__setattr__(self, "resolved_model_parameters", parameters)
        for fold_id, fold in enumerate(folds):
            if fold.fold_id != fold_id or fold.model_name != self.model_name:
                raise WalkForwardPermutationImportanceIntegrityError(
                    "fold audit order or model_name is inconsistent"
                )
            if fold.model_fit_audit.resolved_parameters != parameters:
                raise WalkForwardPermutationImportanceIntegrityError(
                    "resolved model parameters differ across folds"
                )
            if fold.model_fit_audit.feature_names != names:
                raise WalkForwardPermutationImportanceIntegrityError(
                    "model feature names differ across folds"
                )
        first = _timestamp("first_prediction_date", self.first_prediction_date)
        last = _timestamp("last_prediction_date", self.last_prediction_date)
        if first is None or last is None or first > last:
            raise WalkForwardPermutationImportanceIntegrityError(
                "prediction date range is invalid"
            )
        object.__setattr__(self, "first_prediction_date", first)
        object.__setattr__(self, "last_prediction_date", last)
        json.dumps(self.as_dict(), allow_nan=False)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe aggregate audit."""
        return {
            "model_name": self.model_name,
            "resolved_model_parameters": dict(self.resolved_model_parameters),
            "scoring": self.scoring,
            "score_direction": self.score_direction,
            "permutation_scope": self.permutation_scope,
            "n_repeats": self.n_repeats,
            "random_state": self.random_state,
            "n_folds": self.n_folds,
            "n_features": self.n_features,
            "n_repeat_evaluations": self.n_repeat_evaluations,
            "feature_names": list(self.feature_names),
            "first_prediction_date": _iso(self.first_prediction_date),
            "last_prediction_date": _iso(self.last_prediction_date),
            "fold_audits": [fold.as_dict() for fold in self.fold_audits],
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
    raise WalkForwardPermutationImportanceIntegrityError(
        f"result contains unsupported {type(value).__name__}"
    )


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {column: _json_value(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


class WalkForwardPermutationImportanceResult:
    """Defensively expose repeat, fold, and feature importance tables."""

    def __init__(
        self,
        feature_importance: pd.DataFrame,
        fold_importance: pd.DataFrame,
        repeat_importance: pd.DataFrame,
        audit: WalkForwardPermutationImportanceAudit,
    ) -> None:
        table_contracts = (
            ("feature_importance", feature_importance, _FEATURE_COLUMNS),
            ("fold_importance", fold_importance, _FOLD_COLUMNS),
            ("repeat_importance", repeat_importance, _REPEAT_COLUMNS),
        )
        for name, frame, columns in table_contracts:
            if not isinstance(frame, pd.DataFrame) or list(frame.columns) != columns:
                raise WalkForwardPermutationImportanceIntegrityError(
                    f"{name} columns are invalid"
                )
        if not isinstance(audit, WalkForwardPermutationImportanceAudit):
            raise WalkForwardPermutationImportanceIntegrityError(
                "audit must be WalkForwardPermutationImportanceAudit"
            )
        if len(repeat_importance) != audit.n_repeat_evaluations:
            raise WalkForwardPermutationImportanceIntegrityError(
                "repeat_importance row count differs from audit"
            )
        self._feature_importance = feature_importance.copy(deep=True)
        self._fold_importance = fold_importance.copy(deep=True)
        self._repeat_importance = repeat_importance.copy(deep=True)
        self._audit = audit

    @property
    def feature_importance(self) -> pd.DataFrame:
        return self._feature_importance.copy(deep=True)

    @property
    def fold_importance(self) -> pd.DataFrame:
        return self._fold_importance.copy(deep=True)

    @property
    def repeat_importance(self) -> pd.DataFrame:
        return self._repeat_importance.copy(deep=True)

    @property
    def audit(self) -> WalkForwardPermutationImportanceAudit:
        return self._audit

    def as_dict(self) -> dict[str, object]:
        """Return deterministic JSON-safe importance outputs."""
        result = {
            "feature_importance": _records(self._feature_importance),
            "fold_importance": _records(self._fold_importance),
            "repeat_importance": _records(self._repeat_importance),
            "audit": self.audit.as_dict(),
        }
        json.dumps(result, allow_nan=False)
        return result


def _score(
    target: pd.Series,
    prediction: pd.Series,
    scoring: str,
) -> float:
    target_values = target.to_numpy(dtype=np.float64)
    prediction_values = prediction.to_numpy(dtype=np.float64)
    if len(target_values) == 0 or len(target_values) != len(prediction_values):
        raise WalkForwardPermutationImportanceIntegrityError(
            "score inputs must be aligned and non-empty"
        )
    if not np.isfinite(target_values).all() or not np.isfinite(
        prediction_values
    ).all():
        raise WalkForwardPermutationImportanceIntegrityError(
            "score inputs must be finite"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        residual = target_values - prediction_values
        if scoring == "rmse":
            value = float(
                np.sqrt(np.mean(residual * residual, dtype=np.float64))
            )
        elif scoring == "mae":
            value = float(np.mean(np.abs(residual), dtype=np.float64))
        else:
            raise WalkForwardPermutationImportanceIntegrityError(
                "unsupported scoring value"
            )
    if not math.isfinite(value):
        raise WalkForwardPermutationImportanceIntegrityError(
            "score calculation produced a non-finite value"
        )
    return value


def _permute_within_trade_date(
    features: pd.DataFrame,
    trade_dates: pd.Series,
    feature_name: str,
    fold_id: int,
    feature_position: int,
    repeat_id: int,
    random_state: int,
) -> pd.DataFrame:
    """Return a copy with one feature independently permuted within each date."""
    permuted = features.copy(deep=True)
    seed = np.random.SeedSequence(
        [random_state, fold_id, feature_position, repeat_id]
    )
    rng = np.random.default_rng(seed)
    date_values = trade_dates.to_numpy()
    for date in pd.unique(date_values):
        positions = np.flatnonzero(date_values == date)
        if len(positions) <= 1:
            continue
        original = permuted.iloc[positions][feature_name].to_numpy(copy=True)
        shuffled = original[rng.permutation(len(positions))]
        column_position = permuted.columns.get_loc(feature_name)
        permuted.iloc[positions, column_position] = shuffled
    return permuted


class WalkForwardPermutationImportanceRunner:
    """Fit one fresh model per fold and score OOS feature permutations."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        if registry is not None and not isinstance(registry, ModelRegistry):
            raise WalkForwardPermutationImportanceConfigError(
                "registry must be ModelRegistry or None"
            )
        self._registry = (
            create_default_model_registry() if registry is None else registry
        )

    def run(
        self,
        dataset: MLDataset,
        plan: WalkForwardPlan,
        config: WalkForwardPermutationImportanceConfig,
    ) -> WalkForwardPermutationImportanceResult:
        """Execute deterministic prediction-block permutations for every fold."""
        if not isinstance(dataset, MLDataset):
            raise WalkForwardPermutationImportanceDataError(
                "dataset must be MLDataset"
            )
        if not isinstance(plan, WalkForwardPlan):
            raise WalkForwardPermutationImportanceDataError(
                "plan must be WalkForwardPlan"
            )
        if not isinstance(config, WalkForwardPermutationImportanceConfig):
            raise WalkForwardPermutationImportanceConfigError(
                "config must be WalkForwardPermutationImportanceConfig"
            )
        features = dataset.features
        labels = dataset.labels
        metadata = dataset.metadata
        self._validate_dataset(dataset, features, labels, metadata)
        self._validate_plan(plan, metadata, len(features))

        repeat_rows: list[dict[str, object]] = []
        fold_audits: list[PermutationImportanceFoldAudit] = []
        resolved_parameters: tuple[tuple[str, object], ...] | None = None

        for fold_id, split in enumerate(plan.splits):
            train_x = self._take(features, split.train_indices)
            train_y = self._take(labels, split.train_indices)
            prediction_x = self._take(features, split.prediction_indices)
            prediction_y = self._take(labels, split.prediction_indices)
            prediction_meta = self._take(metadata, split.prediction_indices)
            valid_x: pd.DataFrame | None = None
            valid_y: pd.Series | None = None
            if split.validation_indices:
                valid_x = self._take(features, split.validation_indices)
                valid_y = self._take(labels, split.validation_indices)
            try:
                adapter = self._registry.create(
                    config.model_name, dict(config.model_params or {})
                )
                fit_audit = adapter.fit(train_x, train_y, valid_x, valid_y)
                baseline_prediction = adapter.predict(prediction_x)
            except Exception as exc:
                raise self._fold_error(
                    fold_id, config.model_name, split, exc
                ) from exc
            self._validate_fit_audit(
                fit_audit,
                config.model_name,
                tuple(features.columns),
                len(train_x),
                0 if valid_x is None else len(valid_x),
            )
            self._validate_prediction(
                baseline_prediction, prediction_x.index, fold_id
            )
            try:
                baseline_score = _score(
                    prediction_y, baseline_prediction, config.scoring
                )
            except Exception as exc:
                raise self._fold_error(
                    fold_id, config.model_name, split, exc
                ) from exc
            if resolved_parameters is None:
                resolved_parameters = fit_audit.resolved_parameters
            elif resolved_parameters != fit_audit.resolved_parameters:
                raise WalkForwardPermutationImportanceIntegrityError(
                    f"fold_id={fold_id} resolved model parameters differ"
                )
            for feature_position, feature_name in enumerate(features.columns):
                for repeat_id in range(config.n_repeats):
                    try:
                        permuted_x = _permute_within_trade_date(
                            prediction_x,
                            prediction_meta["trade_date"],
                            feature_name,
                            fold_id,
                            feature_position,
                            repeat_id,
                            config.random_state,
                        )
                        permuted_prediction = adapter.predict(permuted_x)
                    except Exception as exc:
                        raise self._fold_error(
                            fold_id,
                            config.model_name,
                            split,
                            exc,
                            feature_name=feature_name,
                            repeat_id=repeat_id,
                        ) from exc
                    self._validate_prediction(
                        permuted_prediction, prediction_x.index, fold_id
                    )
                    try:
                        permuted_score = _score(
                            prediction_y,
                            permuted_prediction,
                            config.scoring,
                        )
                        importance = float(
                            permuted_score - baseline_score
                        )
                    except Exception as exc:
                        raise self._fold_error(
                            fold_id,
                            config.model_name,
                            split,
                            exc,
                            feature_name=feature_name,
                            repeat_id=repeat_id,
                        ) from exc
                    if not math.isfinite(importance):
                        raise WalkForwardPermutationImportanceIntegrityError(
                            "importance calculation produced a non-finite value"
                        )
                    repeat_rows.append(
                        {
                            "fold_id": fold_id,
                            "feature_name": feature_name,
                            "feature_position": feature_position,
                            "repeat_id": repeat_id,
                            "baseline_score": baseline_score,
                            "permuted_score": permuted_score,
                            "importance": importance,
                        }
                    )
            fold_audits.append(
                PermutationImportanceFoldAudit(
                    fold_id=fold_id,
                    model_name=config.model_name,
                    train_rows=len(train_x),
                    validation_rows=0 if valid_x is None else len(valid_x),
                    prediction_rows=len(prediction_x),
                    prediction_dates=int(
                        prediction_meta["trade_date"].nunique()
                    ),
                    train_start_date=split.train_start_date,
                    train_end_date=split.train_end_date,
                    validation_start_date=split.validation_start_date,
                    validation_end_date=split.validation_end_date,
                    prediction_start_date=split.prediction_start_date,
                    prediction_end_date=split.prediction_end_date,
                    baseline_score=baseline_score,
                    scoring=config.scoring,
                    validation_used_for_fit=fit_audit.validation_used_for_fit,
                    n_features=len(features.columns),
                    n_repeats=config.n_repeats,
                    n_permutations=len(features.columns) * config.n_repeats,
                    model_fit_audit=fit_audit,
                )
            )
        if resolved_parameters is None:
            raise WalkForwardPermutationImportanceDataError(
                "plan must contain at least one split"
            )
        repeat_importance = pd.DataFrame(
            repeat_rows, columns=_REPEAT_COLUMNS
        )
        self._normalize_repeat_types(repeat_importance)
        fold_importance = self._aggregate_folds(
            repeat_importance, config.n_repeats
        )
        feature_importance = self._aggregate_features(
            repeat_importance, tuple(features.columns)
        )
        audit = WalkForwardPermutationImportanceAudit(
            model_name=config.model_name,
            resolved_model_parameters=resolved_parameters,
            scoring=config.scoring,
            score_direction="lower_is_better",
            permutation_scope=config.permutation_scope,
            n_repeats=config.n_repeats,
            random_state=config.random_state,
            n_folds=len(plan.splits),
            n_features=len(features.columns),
            n_repeat_evaluations=len(repeat_importance),
            feature_names=tuple(features.columns),
            first_prediction_date=plan.first_prediction_date,
            last_prediction_date=plan.last_prediction_date,
            fold_audits=tuple(fold_audits),
        )
        self._validate_outputs(
            repeat_importance,
            fold_importance,
            feature_importance,
            audit,
        )
        return WalkForwardPermutationImportanceResult(
            feature_importance,
            fold_importance,
            repeat_importance,
            audit,
        )

    @staticmethod
    def _take(
        value: pd.DataFrame | pd.Series, positions: tuple[int, ...]
    ) -> pd.DataFrame | pd.Series:
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
            features.empty
            or len(features.columns) == 0
            or tuple(features.columns) != dataset.feature_names
        ):
            raise WalkForwardPermutationImportanceDataError(
                "dataset must contain its declared features"
            )
        if (
            not features.index.equals(expected_index)
            or not labels.index.equals(expected_index)
            or not metadata.index.equals(expected_index)
            or len({len(features), len(labels), len(metadata)}) != 1
        ):
            raise WalkForwardPermutationImportanceDataError(
                "dataset objects must share RangeIndex(0, n)"
            )
        if labels.name != dataset.label_name:
            raise WalkForwardPermutationImportanceDataError(
                "labels.name differs from dataset label_name"
            )
        if tuple(metadata.columns) != tuple(METADATA_COLUMNS):
            raise WalkForwardPermutationImportanceDataError(
                "dataset metadata columns are invalid"
            )
        try:
            label_values = labels.to_numpy(dtype=np.float64)
            feature_values = features.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise WalkForwardPermutationImportanceDataError(
                "features and labels must be numeric"
            ) from exc
        if not np.isfinite(label_values).all():
            raise WalkForwardPermutationImportanceDataError(
                "labels must be finite"
            )
        if np.isinf(feature_values).any():
            raise WalkForwardPermutationImportanceDataError(
                "features must not contain infinity"
            )
        if metadata.isna().any().any():
            raise WalkForwardPermutationImportanceDataError(
                "metadata must not contain missing values"
            )
        for column in ("trade_date", "entry_trade_date", "exit_trade_date"):
            if not pd.api.types.is_datetime64_ns_dtype(metadata[column]):
                raise WalkForwardPermutationImportanceDataError(
                    f"metadata {column} must use datetime64[ns]"
                )
            if getattr(metadata[column].dt, "tz", None) is not None:
                raise WalkForwardPermutationImportanceDataError(
                    f"metadata {column} must be timezone-naive"
                )
        if not metadata["trade_date"].is_monotonic_increasing:
            raise WalkForwardPermutationImportanceDataError(
                "metadata trade_date must be sorted"
            )
        if metadata.duplicated(["trade_date", "ts_code"]).any():
            raise WalkForwardPermutationImportanceDataError(
                "metadata keys must be unique"
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
            raise WalkForwardPermutationImportanceDataError(
                "plan splits must be a tuple"
            ) from exc
        if not splits:
            raise WalkForwardPermutationImportanceDataError(
                "plan must contain at least one split"
            )
        if any(not isinstance(split, WalkForwardSplit) for split in splits):
            raise WalkForwardPermutationImportanceDataError(
                "plan splits must contain WalkForwardSplit values"
            )
        dataset_dates = tuple(
            pd.Timestamp(value)
            for value in metadata["trade_date"].drop_duplicates()
        )
        if tuple(plan.all_score_dates) != dataset_dates:
            raise WalkForwardPermutationImportanceIntegrityError(
                "plan all_score_dates differ from dataset"
            )
        seen_indices: set[int] = set()
        seen_dates: set[pd.Timestamp] = set()
        combined_dates: list[pd.Timestamp] = []
        for fold_id, split in enumerate(splits):
            if split.retrain_id != fold_id + 1:
                raise WalkForwardPermutationImportanceIntegrityError(
                    "split retrain_id order is invalid"
                )
            train = self._positions(
                "train_indices", split.train_indices, n_rows
            )
            valid = self._positions(
                "validation_indices", split.validation_indices, n_rows
            )
            prediction = self._positions(
                "prediction_indices", split.prediction_indices, n_rows
            )
            if (
                set(train) & set(valid)
                or set(train) & set(prediction)
                or set(valid) & set(prediction)
            ):
                raise WalkForwardPermutationImportanceIntegrityError(
                    f"fold_id={fold_id} partition indices overlap"
                )
            if seen_indices & set(prediction):
                raise WalkForwardPermutationImportanceIntegrityError(
                    "prediction indices overlap across folds"
                )
            seen_indices.update(prediction)
            self._validate_partition(
                fold_id, "train", train, split.train_dates, metadata
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
            if seen_dates & set(prediction_dates):
                raise WalkForwardPermutationImportanceIntegrityError(
                    "prediction dates overlap across folds"
                )
            seen_dates.update(prediction_dates)
            combined_dates.extend(prediction_dates)
            self._validate_cutoffs(fold_id, split, metadata)
            if (
                split.n_train_rows != len(train)
                or split.n_validation_rows != len(valid)
                or split.n_prediction_rows != len(prediction)
            ):
                raise WalkForwardPermutationImportanceIntegrityError(
                    f"fold_id={fold_id} row counts differ from split"
                )
        first = pd.Timestamp(plan.first_prediction_date)
        if first not in dataset_dates:
            raise WalkForwardPermutationImportanceIntegrityError(
                "first_prediction_date is not a dataset date"
            )
        first_position = dataset_dates.index(first)
        skipped = tuple(
            pd.Timestamp(value)
            for value in plan.skipped_initial_prediction_dates
        )
        if skipped != dataset_dates[:first_position]:
            raise WalkForwardPermutationImportanceIntegrityError(
                "skipped prediction dates differ from dataset prefix"
            )
        if tuple(combined_dates) != dataset_dates[first_position:]:
            raise WalkForwardPermutationImportanceIntegrityError(
                "prediction dates must cover the dataset suffix"
            )
        if (
            plan.first_prediction_date != combined_dates[0]
            or plan.last_prediction_date != combined_dates[-1]
        ):
            raise WalkForwardPermutationImportanceIntegrityError(
                "plan prediction boundaries are inconsistent"
            )

    @staticmethod
    def _positions(
        field_name: str, values: object, n_rows: int
    ) -> tuple[int, ...]:
        try:
            raw = tuple(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise WalkForwardPermutationImportanceDataError(
                f"{field_name} must contain integer positions"
            ) from exc
        if not raw:
            raise WalkForwardPermutationImportanceDataError(
                f"{field_name} must not be empty"
            )
        normalized: list[int] = []
        for value in raw:
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 0
                or int(value) >= n_rows
            ):
                raise WalkForwardPermutationImportanceDataError(
                    f"{field_name} contains invalid position {value!r}"
                )
            normalized.append(int(value))
        result = tuple(normalized)
        if len(set(result)) != len(result):
            raise WalkForwardPermutationImportanceIntegrityError(
                f"{field_name} contains duplicate positions"
            )
        return result

    @staticmethod
    def _validate_partition(
        fold_id: int,
        name: str,
        positions: tuple[int, ...],
        declared_dates: tuple[pd.Timestamp, ...],
        metadata: pd.DataFrame,
    ) -> None:
        dates = tuple(pd.Timestamp(value) for value in declared_dates)
        if not dates or dates != tuple(sorted(set(dates))):
            raise WalkForwardPermutationImportanceIntegrityError(
                f"fold_id={fold_id} {name} dates are invalid"
            )
        expected = tuple(
            int(value)
            for value in metadata.index[metadata["trade_date"].isin(dates)]
        )
        if positions != expected:
            raise WalkForwardPermutationImportanceIntegrityError(
                f"fold_id={fold_id} {name} must contain complete date cross-sections"
            )

    @staticmethod
    def _validate_cutoffs(
        fold_id: int,
        split: WalkForwardSplit,
        metadata: pd.DataFrame,
    ) -> None:
        train = metadata.iloc[list(split.train_indices)]
        valid = metadata.iloc[list(split.validation_indices)]
        prediction = metadata.iloc[list(split.prediction_indices)]
        boundaries = {
            "train_start_date": pd.Timestamp(train["trade_date"].min()),
            "train_end_date": pd.Timestamp(train["trade_date"].max()),
            "validation_start_date": pd.Timestamp(valid["trade_date"].min()),
            "validation_end_date": pd.Timestamp(valid["trade_date"].max()),
            "prediction_start_date": pd.Timestamp(
                prediction["trade_date"].min()
            ),
            "prediction_end_date": pd.Timestamp(
                prediction["trade_date"].max()
            ),
        }
        for field_name, actual in boundaries.items():
            declared = _timestamp(
                f"split.{field_name}", getattr(split, field_name)
            )
            if declared != actual:
                raise WalkForwardPermutationImportanceIntegrityError(
                    f"fold_id={fold_id} {field_name} differs from metadata"
                )
        if not (
            boundaries["train_end_date"]
            < boundaries["validation_start_date"]
            < boundaries["prediction_start_date"]
        ):
            raise WalkForwardPermutationImportanceIntegrityError(
                f"fold_id={fold_id} temporal order is invalid"
            )
        max_train_exit = pd.Timestamp(train["exit_trade_date"].max())
        max_valid_exit = pd.Timestamp(valid["exit_trade_date"].max())
        if max_train_exit >= boundaries["validation_start_date"]:
            raise WalkForwardPermutationImportanceIntegrityError(
                f"fold_id={fold_id} training exit cutoff is invalid"
            )
        if max_valid_exit >= boundaries["prediction_start_date"]:
            raise WalkForwardPermutationImportanceIntegrityError(
                f"fold_id={fold_id} validation exit cutoff is invalid"
            )
        if (
            max_train_exit != pd.Timestamp(split.max_train_exit_date)
            or max_valid_exit != pd.Timestamp(split.max_validation_exit_date)
        ):
            raise WalkForwardPermutationImportanceIntegrityError(
                f"fold_id={fold_id} exit date audit is inconsistent"
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
            raise WalkForwardPermutationImportanceIntegrityError(
                "adapter.fit must return ModelFitAudit"
            )
        if audit.model_name != model_name:
            raise WalkForwardPermutationImportanceIntegrityError(
                "fit audit model_name differs from config"
            )
        if audit.feature_names != feature_names:
            raise WalkForwardPermutationImportanceIntegrityError(
                "fit audit feature_names differ from dataset"
            )
        if (
            audit.n_train_rows != train_rows
            or audit.n_validation_rows != validation_rows
        ):
            raise WalkForwardPermutationImportanceIntegrityError(
                "fit audit row counts are inconsistent"
            )

    @staticmethod
    def _validate_prediction(
        prediction: object, expected_index: pd.Index, fold_id: int
    ) -> None:
        if not isinstance(prediction, pd.Series):
            raise WalkForwardPermutationImportanceIntegrityError(
                f"fold_id={fold_id} prediction must be a Series"
            )
        if (
            len(prediction) != len(expected_index)
            or not prediction.index.equals(expected_index)
            or prediction.name != "prediction"
        ):
            raise WalkForwardPermutationImportanceIntegrityError(
                f"fold_id={fold_id} prediction contract is invalid"
            )
        try:
            values = prediction.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise WalkForwardPermutationImportanceIntegrityError(
                f"fold_id={fold_id} prediction must be numeric"
            ) from exc
        if not np.isfinite(values).all():
            raise WalkForwardPermutationImportanceIntegrityError(
                f"fold_id={fold_id} prediction must be finite"
            )

    @staticmethod
    def _normalize_repeat_types(frame: pd.DataFrame) -> None:
        for column in ("fold_id", "feature_position", "repeat_id"):
            frame[column] = frame[column].astype(np.int64)
        for column in ("baseline_score", "permuted_score", "importance"):
            frame[column] = frame[column].astype(np.float64)
        frame.index = pd.RangeIndex(len(frame))

    @staticmethod
    def _aggregate_folds(
        repeat: pd.DataFrame, n_repeats: int
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for (fold_id, feature_position), block in repeat.groupby(
            ["fold_id", "feature_position"], sort=True
        ):
            values = block["importance"].to_numpy(dtype=np.float64)
            std = (
                np.nan
                if len(values) == 1
                else float(np.std(values, ddof=1, dtype=np.float64))
            )
            rows.append(
                {
                    "fold_id": int(fold_id),
                    "feature_name": str(block["feature_name"].iloc[0]),
                    "feature_position": int(feature_position),
                    "baseline_score": float(block["baseline_score"].iloc[0]),
                    "importance_mean": float(np.mean(values)),
                    "importance_std": std,
                    "importance_min": float(np.min(values)),
                    "importance_max": float(np.max(values)),
                    "positive_fraction": float(np.mean(values > 0.0)),
                    "n_repeats": n_repeats,
                }
            )
        frame = pd.DataFrame(rows, columns=_FOLD_COLUMNS)
        for column in ("fold_id", "feature_position", "n_repeats"):
            frame[column] = frame[column].astype(np.int64)
        frame.index = pd.RangeIndex(len(frame))
        return frame

    @staticmethod
    def _aggregate_features(
        repeat: pd.DataFrame, feature_names: tuple[str, ...]
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for feature_position, feature_name in enumerate(feature_names):
            block = repeat.loc[
                repeat["feature_position"].eq(feature_position)
            ]
            values = block["importance"].to_numpy(dtype=np.float64)
            std = (
                np.nan
                if len(values) == 1
                else float(np.std(values, ddof=1, dtype=np.float64))
            )
            rows.append(
                {
                    "feature_name": feature_name,
                    "feature_position": feature_position,
                    "importance_mean": float(np.mean(values)),
                    "importance_std": std,
                    "importance_median": float(np.median(values)),
                    "importance_min": float(np.min(values)),
                    "importance_max": float(np.max(values)),
                    "positive_fraction": float(np.mean(values > 0.0)),
                    "n_folds": int(block["fold_id"].nunique()),
                    "n_observations": len(values),
                    "importance_rank": 0,
                }
            )
        ranked = sorted(
            rows,
            key=lambda row: (
                -float(row["importance_mean"]),
                int(row["feature_position"]),
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            rows[int(row["feature_position"])]["importance_rank"] = rank
        frame = pd.DataFrame(rows, columns=_FEATURE_COLUMNS)
        for column in (
            "feature_position",
            "n_folds",
            "n_observations",
            "importance_rank",
        ):
            frame[column] = frame[column].astype(np.int64)
        frame.index = pd.RangeIndex(len(frame))
        return frame

    @staticmethod
    def _validate_outputs(
        repeat: pd.DataFrame,
        fold: pd.DataFrame,
        feature: pd.DataFrame,
        audit: WalkForwardPermutationImportanceAudit,
    ) -> None:
        if len(repeat) != audit.n_repeat_evaluations:
            raise WalkForwardPermutationImportanceIntegrityError(
                "repeat output row count is inconsistent"
            )
        if len(fold) != audit.n_folds * audit.n_features:
            raise WalkForwardPermutationImportanceIntegrityError(
                "fold output row count is inconsistent"
            )
        if len(feature) != audit.n_features:
            raise WalkForwardPermutationImportanceIntegrityError(
                "feature output row count is inconsistent"
            )
        for frame in (repeat, fold, feature):
            numeric = frame.select_dtypes(include=[np.number])
            for column in numeric:
                values = numeric[column].to_numpy(dtype=np.float64)
                if column == "importance_std":
                    values = values[~np.isnan(values)]
                if not np.isfinite(values).all():
                    raise WalkForwardPermutationImportanceIntegrityError(
                        f"output column {column} contains non-finite values"
                    )

    @staticmethod
    def _fold_error(
        fold_id: int,
        model_name: str,
        split: WalkForwardSplit,
        cause: Exception,
        *,
        feature_name: str | None = None,
        repeat_id: int | None = None,
    ) -> WalkForwardPermutationImportanceFoldError:
        detail = ""
        if feature_name is not None:
            detail += f"; feature_name={feature_name}"
        if repeat_id is not None:
            detail += f"; repeat_id={repeat_id}"
        return WalkForwardPermutationImportanceFoldError(
            f"fold {fold_id} failed for model {model_name}{detail}; "
            f"train={_iso(split.train_start_date)}..{_iso(split.train_end_date)}; "
            f"validation={_iso(split.validation_start_date)}.."
            f"{_iso(split.validation_end_date)}; "
            f"prediction={_iso(split.prediction_start_date)}.."
            f"{_iso(split.prediction_end_date)}; "
            f"cause={type(cause).__name__}"
        )
