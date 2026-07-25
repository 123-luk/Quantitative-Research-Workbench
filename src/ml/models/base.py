"""Shared contracts and validation for regression model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import math
import platform
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ModelError(Exception):
    """Base exception for model configuration, data, fit, and prediction errors."""


class ModelConfigError(ModelError):
    """Raised when a public model parameter is invalid."""


class ModelDataError(ModelError):
    """Raised when model input data violates the public contract."""


class ModelNotFittedError(ModelError):
    """Raised when fitted state is required but unavailable."""


class ModelFeatureMismatchError(ModelDataError):
    """Raised when validation or prediction features do not exactly match training."""



class ModelFeatureImportanceUnavailableError(ModelError):
    """Raised when a fitted model has no supported native feature importance."""

class ModelFitError(ModelError):
    """Raised when the estimator cannot be fitted safely."""


class ModelPredictionError(ModelError):
    """Raised when prediction fails or produces invalid output."""


class ModelRegistryError(ModelError):
    """Raised for invalid model registry operations."""


_VALUE_TYPES = {
    "int",
    "optional_int",
    "float",
    "optional_float",
    "bool",
    "str",
    "choice",
}
_UI_CONTROLS = {"number", "checkbox", "select"}
_RESERVED_FEATURES = {
    "trade_date",
    "ts_code",
    "entry_trade_date",
    "exit_trade_date",
    "entry_price",
    "exit_price",
    "forward_return",
    "prediction",
}


def _json_safe(value: Any) -> Any:
    """Return a recursively JSON-safe copy or raise a configuration error."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelConfigError("model metadata contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise ModelConfigError(
        f"model metadata contains unsupported value type {type(value).__name__}"
    )


def _validate_schema_default(spec: "ModelParameterSpec") -> None:
    value = spec.default
    if isinstance(value, np.generic):
        value = value.item()
    if spec.value_type == "int":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif spec.value_type == "optional_int":
        valid = value is None or (
            isinstance(value, int) and not isinstance(value, bool)
        )
    elif spec.value_type == "float":
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    elif spec.value_type == "optional_float":
        valid = value is None or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    elif spec.value_type == "bool":
        valid = isinstance(value, bool)
    elif spec.value_type in {"str", "choice"}:
        valid = isinstance(value, str)
    else:
        valid = False
    if not valid:
        raise ModelConfigError(
            f"parameter schema {spec.name!r} has invalid default for "
            f"{spec.value_type}"
        )
    if spec.minimum is not None and value is not None and float(value) < spec.minimum:
        raise ModelConfigError(
            f"parameter schema {spec.name!r} default is below minimum"
        )
    if spec.maximum is not None and value is not None and float(value) > spec.maximum:
        raise ModelConfigError(
            f"parameter schema {spec.name!r} default is above maximum"
        )
    if spec.choices is not None and value not in spec.choices:
        raise ModelConfigError(
            f"parameter schema {spec.name!r} default is not in choices"
        )


@dataclass(frozen=True)
class ModelParameterSpec:
    """Declarative, JSON-safe model parameter description for future UIs."""

    name: str
    display_name: str
    value_type: str
    default: object
    description: str
    advanced: bool = False
    required: bool = False
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[object, ...] | None = None
    ui_control: str = "number"
    step: int | float | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "display_name", "description"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ModelConfigError(
                    f"parameter schema field {field_name} must be a non-empty string"
                )
        if self.value_type not in _VALUE_TYPES:
            raise ModelConfigError(
                f"parameter schema {self.name!r} has invalid value_type "
                f"{self.value_type!r}"
            )
        if self.ui_control not in _UI_CONTROLS:
            raise ModelConfigError(
                f"parameter schema {self.name!r} has invalid ui_control "
                f"{self.ui_control!r}"
            )
        for field_name in ("advanced", "required"):
            if not isinstance(getattr(self, field_name), bool):
                raise ModelConfigError(
                    f"parameter schema {self.name!r} field {field_name} must be bool"
                )
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ModelConfigError(
                    f"parameter schema {self.name!r} minimum exceeds maximum"
                )
        if self.step is not None:
            if (
                isinstance(self.step, bool)
                or not isinstance(self.step, (int, float))
                or not math.isfinite(float(self.step))
                or self.step <= 0
            ):
                raise ModelConfigError(
                    f"parameter schema {self.name!r} step must be positive"
                )
        if self.choices is not None:
            if not isinstance(self.choices, tuple) or not self.choices:
                raise ModelConfigError(
                    f"parameter schema {self.name!r} choices must be a non-empty tuple"
                )
            if len(set(self.choices)) != len(self.choices):
                raise ModelConfigError(
                    f"parameter schema {self.name!r} choices must be unique"
                )
        if self.value_type == "choice" and self.choices is None:
            raise ModelConfigError(
                f"parameter schema {self.name!r} choice requires choices"
            )
        _validate_schema_default(self)
        json.dumps(self.as_dict(), allow_nan=False)

    def as_dict(self) -> dict[str, object]:
        """Return a defensive JSON-safe representation."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "value_type": self.value_type,
            "default": _json_safe(self.default),
            "description": self.description,
            "advanced": self.advanced,
            "required": self.required,
            "minimum": _json_safe(self.minimum),
            "maximum": _json_safe(self.maximum),
            "choices": (
                None if self.choices is None else _json_safe(self.choices)
            ),
            "ui_control": self.ui_control,
            "step": _json_safe(self.step),
        }


_PAIR_FIELDS = (
    "train_missing_counts",
    "validation_missing_counts",
    "imputation_values",
    "scaler_means",
    "scaler_scales",
)


@dataclass(frozen=True)
class ModelFitAudit:
    """Immutable, sample-free audit of one successful model fit."""

    model_name: str
    estimator_class: str
    feature_names: tuple[str, ...]
    n_train_rows: int
    n_validation_rows: int
    n_features: int
    train_missing_counts: tuple[tuple[str, int], ...]
    validation_missing_counts: tuple[tuple[str, int], ...]
    imputation_values: tuple[tuple[str, float], ...]
    scaler_means: tuple[tuple[str, float], ...]
    scaler_scales: tuple[tuple[str, float], ...]
    constant_features: tuple[str, ...]
    validation_provided: bool
    validation_used_for_fit: bool
    resolved_parameters: tuple[tuple[str, object], ...]
    preprocessing_parameters: tuple[tuple[str, object], ...]
    estimator_intercept: float | None
    python_version: str
    numpy_version: str
    sklearn_version: str
    native_missing_support: bool = False
    imputer_enabled: bool = True
    scaler_enabled: bool = True
    best_iteration: int | None = None
    n_iterations: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ModelConfigError("fit audit model_name must be non-empty")
        if not isinstance(self.estimator_class, str) or not self.estimator_class:
            raise ModelConfigError("fit audit estimator_class must be non-empty")
        if (
            self.n_train_rows < 0
            or self.n_validation_rows < 0
            or self.n_features < 0
        ):
            raise ModelConfigError("fit audit row and feature counts cannot be negative")
        if self.n_features != len(self.feature_names):
            raise ModelConfigError(
                "fit audit n_features must equal feature_names length"
            )
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ModelConfigError("fit audit feature_names must be unique")
        for field_name in _PAIR_FIELDS:
            pairs = getattr(self, field_name)
            if field_name == "validation_missing_counts":
                expected = self.feature_names if self.validation_provided else ()
            elif field_name == "imputation_values":
                expected = self.feature_names if self.imputer_enabled else ()
            elif field_name in {"scaler_means", "scaler_scales"}:
                expected = self.feature_names if self.scaler_enabled else ()
            else:
                expected = self.feature_names
            if tuple(name for name, _ in pairs) != tuple(expected):
                raise ModelConfigError(
                    f"fit audit {field_name} order must match feature_names"
                )
        for field_name in ("train_missing_counts", "validation_missing_counts"):
            if any(value < 0 for _, value in getattr(self, field_name)):
                raise ModelConfigError(f"fit audit {field_name} cannot be negative")
        if self.estimator_intercept is not None and not math.isfinite(
            float(self.estimator_intercept)
        ):
            raise ModelConfigError("fit audit estimator_intercept must be finite")
        for field_name in ("best_iteration", "n_iterations"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ModelConfigError(
                    f"fit audit {field_name} must be None or a non-negative integer"
                )
        if not set(self.constant_features).issubset(self.feature_names):
            raise ModelConfigError(
                "fit audit constant_features must be training features"
            )
        json.dumps(self.as_dict(), allow_nan=False)

    def as_dict(self) -> dict[str, object]:
        """Return a defensive JSON-safe dictionary without samples or indices."""
        return {
            "model_name": self.model_name,
            "estimator_class": self.estimator_class,
            "feature_names": list(self.feature_names),
            "n_train_rows": self.n_train_rows,
            "n_validation_rows": self.n_validation_rows,
            "n_features": self.n_features,
            "train_missing_counts": dict(self.train_missing_counts),
            "validation_missing_counts": dict(self.validation_missing_counts),
            "imputation_values": dict(self.imputation_values),
            "scaler_means": dict(self.scaler_means),
            "scaler_scales": dict(self.scaler_scales),
            "constant_features": list(self.constant_features),
            "validation_provided": self.validation_provided,
            "validation_used_for_fit": self.validation_used_for_fit,
            "resolved_parameters": _json_safe(dict(self.resolved_parameters)),
            "preprocessing_parameters": _json_safe(
                dict(self.preprocessing_parameters)
            ),
            "estimator_intercept": (
                None
                if self.estimator_intercept is None
                else float(self.estimator_intercept)
            ),
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "sklearn_version": self.sklearn_version,
            "native_missing_support": self.native_missing_support,
            "imputer_enabled": self.imputer_enabled,
            "scaler_enabled": self.scaler_enabled,
            "best_iteration": self.best_iteration,
            "n_iterations": self.n_iterations,
        }


def _numeric_frame(
    frame: object,
    *,
    model_name: str,
    context: str,
    minimum_rows: int,
    expected_features: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ModelDataError(f"{model_name}: {context} must be a pandas DataFrame")
    if len(frame) < minimum_rows:
        raise ModelDataError(
            f"{model_name}: {context} must contain at least {minimum_rows} row(s)"
        )
    if frame.shape[1] == 0:
        raise ModelDataError(f"{model_name}: {context} must contain a feature")
    if not frame.index.is_unique:
        raise ModelDataError(f"{model_name}: {context} index must be unique")
    if not frame.columns.is_unique:
        raise ModelDataError(f"{model_name}: {context} columns must be unique")
    invalid_names = [
        name for name in frame.columns
        if not isinstance(name, str) or not name.strip()
    ]
    if invalid_names:
        raise ModelDataError(
            f"{model_name}: {context} feature names must be non-empty strings"
        )
    features = tuple(frame.columns)
    if expected_features is not None and features != expected_features:
        raise ModelFeatureMismatchError(
            f"{model_name}: {context} feature names and order must exactly match "
            f"training features {expected_features!r}; received {features!r}"
        )
    reserved = [name for name in features if name in _RESERVED_FEATURES]
    if reserved:
        raise ModelDataError(
            f"{model_name}: {context} contains reserved feature(s) {reserved!r}"
        )
    converted: dict[str, pd.Series] = {}
    for name in features:
        source = frame[name]
        numeric = pd.to_numeric(source, errors="coerce")
        invalid = source.notna() & numeric.isna()
        if bool(invalid.any()):
            raise ModelDataError(
                f"{model_name}: {context} feature {name!r} contains "
                f"{int(invalid.sum())} non-numeric value(s)"
            )
        values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        if bool(np.isinf(values).any()):
            raise ModelDataError(
                f"{model_name}: {context} feature {name!r} contains infinity"
            )
        converted[name] = pd.Series(values, index=frame.index, name=name)
    result = pd.DataFrame(converted, index=frame.index).astype(np.float64)
    if context == "X_train":
        all_missing = [name for name in features if result[name].isna().all()]
        if all_missing:
            raise ModelDataError(
                f"{model_name}: X_train feature(s) are entirely missing "
                f"{all_missing!r}"
            )
    return result


def _numeric_target(
    target: object,
    *,
    model_name: str,
    context: str,
    expected_index: pd.Index,
) -> pd.Series:
    if not isinstance(target, pd.Series):
        raise ModelDataError(f"{model_name}: {context} must be a pandas Series")
    if not target.index.is_unique:
        raise ModelDataError(f"{model_name}: {context} index must be unique")
    if len(target) != len(expected_index):
        raise ModelDataError(
            f"{model_name}: {context} length must match its feature frame"
        )
    if not target.index.equals(expected_index):
        raise ModelDataError(
            f"{model_name}: {context} index must exactly match its feature frame"
        )
    numeric = pd.to_numeric(target, errors="coerce")
    invalid = target.notna() & numeric.isna()
    if bool(invalid.any()):
        raise ModelDataError(
            f"{model_name}: {context} contains non-numeric value(s)"
        )
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    if bool(np.isnan(values).any()):
        raise ModelDataError(f"{model_name}: {context} contains missing values")
    if bool(np.isinf(values).any()):
        raise ModelDataError(f"{model_name}: {context} contains infinity")
    return pd.Series(values, index=target.index, name=target.name, dtype=np.float64)


class RegressionModelAdapter(ABC):
    """Uniform regression adapter with leak-resistant preprocessing and auditing."""

    _MODEL_NAME: str

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._audit: ModelFitAudit | None = None
        self._importance: pd.DataFrame | None = None

    @property
    def model_name(self) -> str:
        """Return the stable registry name."""
        return self._MODEL_NAME

    @property
    def is_fitted(self) -> bool:
        """Return whether a fit completed successfully."""
        return self._pipeline is not None

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return fitted feature names, or an empty tuple before fit."""
        return () if self._audit is None else self._audit.feature_names

    @property
    @abstractmethod
    def config(self) -> object:
        """Return the immutable project configuration object."""

    @abstractmethod
    def _build_estimator(self) -> RegressorMixin:
        """Create a fresh estimator using only validated configuration."""

    @abstractmethod
    def get_parameter_schema(self) -> tuple[ModelParameterSpec, ...]:
        """Return the stable immutable parameter schema."""

    def _config_as_dict(self) -> dict[str, object]:
        method = getattr(self.config, "as_dict")
        return method()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> ModelFitAudit:
        """Fit a new pipeline on training data only and atomically publish it."""
        train_x = _numeric_frame(
            X_train,
            model_name=self.model_name,
            context="X_train",
            minimum_rows=2,
        )
        train_y = _numeric_target(
            y_train,
            model_name=self.model_name,
            context="y_train",
            expected_index=train_x.index,
        )
        if (X_valid is None) != (y_valid is None):
            raise ModelDataError(
                f"{self.model_name}: X_valid and y_valid must be provided together"
            )
        valid_x: pd.DataFrame | None = None
        if X_valid is not None and y_valid is not None:
            valid_x = _numeric_frame(
                X_valid,
                model_name=self.model_name,
                context="X_valid",
                minimum_rows=1,
                expected_features=tuple(train_x.columns),
            )
            _numeric_target(
                y_valid,
                model_name=self.model_name,
                context="y_valid",
                expected_index=valid_x.index,
            )

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("estimator", self._build_estimator()),
            ]
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(train_x, train_y)
            convergence = [
                warning
                for warning in caught
                if issubclass(warning.category, ConvergenceWarning)
            ]
            if convergence:
                raise ModelFitError(
                    f"{self.model_name}: estimator did not converge; consider "
                    "increasing max_iter or adjusting tol or alpha"
                )
            imputer = pipeline.named_steps["imputer"]
            scaler = pipeline.named_steps["scaler"]
            estimator = pipeline.named_steps["estimator"]
            imputed = imputer.transform(train_x)
            coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
            intercept = float(np.asarray(estimator.intercept_).reshape(-1)[0])
            if coefficients.shape != (train_x.shape[1],) or not np.isfinite(
                coefficients
            ).all():
                raise ModelFitError(
                    f"{self.model_name}: estimator returned invalid coefficients"
                )
            if not math.isfinite(intercept):
                raise ModelFitError(
                    f"{self.model_name}: estimator returned an invalid intercept"
                )
            feature_names = tuple(train_x.columns)
            constant_features = tuple(
                name
                for position, name in enumerate(feature_names)
                if float(np.var(imputed[:, position])) == 0.0
            )
            audit = ModelFitAudit(
                model_name=self.model_name,
                estimator_class=type(estimator).__name__,
                feature_names=feature_names,
                n_train_rows=len(train_x),
                n_validation_rows=0 if valid_x is None else len(valid_x),
                n_features=train_x.shape[1],
                train_missing_counts=tuple(
                    (name, int(train_x[name].isna().sum())) for name in feature_names
                ),
                validation_missing_counts=(
                    ()
                    if valid_x is None
                    else tuple(
                        (name, int(valid_x[name].isna().sum()))
                        for name in feature_names
                    )
                ),
                imputation_values=tuple(
                    (name, float(value))
                    for name, value in zip(feature_names, imputer.statistics_)
                ),
                scaler_means=tuple(
                    (name, float(value))
                    for name, value in zip(feature_names, scaler.mean_)
                ),
                scaler_scales=tuple(
                    (name, float(value))
                    for name, value in zip(feature_names, scaler.scale_)
                ),
                constant_features=constant_features,
                validation_provided=valid_x is not None,
                validation_used_for_fit=False,
                resolved_parameters=tuple(self._config_as_dict().items()),
                preprocessing_parameters=(
                    ("imputer_strategy", "median"),
                    ("scaler_type", "StandardScaler"),
                    ("scaler_enabled", True),
                ),
                estimator_intercept=intercept,
                python_version=platform.python_version(),
                numpy_version=np.__version__,
                sklearn_version=sklearn.__version__,
                native_missing_support=False,
                imputer_enabled=True,
                scaler_enabled=True,
                best_iteration=None,
                n_iterations=None,
            )
            importance = self._make_importance(feature_names, coefficients)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"{self.model_name}: model fit failed: {type(exc).__name__}: {exc}"
            ) from exc

        self._pipeline = pipeline
        self._audit = audit
        self._importance = importance
        return audit

    @staticmethod
    def _make_importance(
        feature_names: tuple[str, ...], coefficients: np.ndarray
    ) -> pd.DataFrame:
        order = sorted(
            range(len(feature_names)),
            key=lambda position: (-abs(float(coefficients[position])), position),
        )
        ranks = {position: rank for rank, position in enumerate(order, start=1)}
        return pd.DataFrame(
            {
                "feature_name": list(feature_names),
                "feature_position": np.arange(len(feature_names), dtype=np.int64),
                "coefficient": coefficients.astype(np.float64),
                "abs_coefficient": np.abs(coefficients).astype(np.float64),
                "importance_rank": [ranks[position] for position in range(len(feature_names))],
                "direction": [
                    "positive" if value > 0 else "negative" if value < 0 else "zero"
                    for value in coefficients
                ],
            }
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict with exact fitted feature order while preserving the input index."""
        if self._pipeline is None or self._audit is None:
            raise ModelNotFittedError(
                f"{self.model_name}: predict requires a successfully fitted model"
            )
        frame = _numeric_frame(
            X,
            model_name=self.model_name,
            context="X",
            minimum_rows=1,
            expected_features=self._audit.feature_names,
        )
        try:
            values = np.asarray(self._pipeline.predict(frame), dtype=np.float64)
        except Exception as exc:
            raise ModelPredictionError(
                f"{self.model_name}: prediction failed: {type(exc).__name__}: {exc}"
            ) from exc
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ModelPredictionError(
                f"{self.model_name}: prediction returned invalid shape or values"
            )
        return pd.Series(
            values,
            index=X.index.copy(),
            name="prediction",
            dtype=np.float64,
        )

    def get_feature_importance(self) -> pd.DataFrame:
        """Return a defensive copy of standardized-input model coefficients."""
        if self._importance is None:
            raise ModelNotFittedError(
                f"{self.model_name}: feature importance requires a fitted model"
            )
        return self._importance.copy(deep=True)

    def get_metadata(self) -> dict[str, object]:
        """Return JSON-safe fitted configuration and audit metadata."""
        if self._audit is None:
            raise ModelNotFittedError(
                f"{self.model_name}: metadata requires a fitted model"
            )
        return {
            "model_name": self.model_name,
            "fitted": True,
            "config": _json_safe(self._config_as_dict()),
            "fit_audit": self._audit.as_dict(),
            "intercept": (
                None
                if self._audit.estimator_intercept is None
                else float(self._audit.estimator_intercept)
            ),
            "feature_names": list(self._audit.feature_names),
        }
