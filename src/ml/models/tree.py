"""Configurable HistGradientBoosting regression with time-safe validation."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import platform
from typing import Mapping

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from src.ml.models.base import (
    ModelConfigError,
    ModelDataError,
    ModelFeatureImportanceUnavailableError,
    ModelFitAudit,
    ModelFitError,
    ModelNotFittedError,
    ModelParameterSpec,
    RegressionModelAdapter,
    _numeric_frame,
    _numeric_target,
)
from src.ml.models.linear import (
    _bool_value,
    _choice,
    _config_values,
    _float_value,
    _int_value,
    _positive_int,
)


_LOSSES = (
    "absolute_error",
    "gamma",
    "poisson",
    "quantile",
    "squared_error",
)
_FIT_PARAMETERS = frozenset(
    inspect.signature(HistGradientBoostingRegressor.fit).parameters
)
_SUPPORTS_EXTERNAL_VALIDATION = {"X_val", "y_val"} <= _FIT_PARAMETERS


@dataclass(frozen=True)
class HistGradientBoostingModelConfig:
    """Validated project parameters for sklearn histogram gradient boosting."""

    loss: str = "squared_error"
    quantile: float | None = None
    learning_rate: float = 0.1
    max_iter: int = 100
    max_leaf_nodes: int | None = 31
    max_depth: int | None = None
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    max_features: float = 1.0
    max_bins: int = 255
    early_stopping: bool = False
    n_iter_no_change: int = 10
    tol: float = 1e-7
    warm_start: bool = False
    random_state: int | None = 42
    verbose: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "loss",
            _choice(
                self.loss,
                model_name="hist_gradient_boosting",
                field_name="loss",
                choices=_LOSSES,
            ),
        )
        if self.quantile is None:
            quantile = None
        else:
            quantile = _float_value(
                self.quantile,
                model_name="hist_gradient_boosting",
                field_name="quantile",
            )
        object.__setattr__(self, "quantile", quantile)
        if self.loss == "quantile":
            if quantile is None or not 0.0 < quantile < 1.0:
                raise ModelConfigError(
                    "hist_gradient_boosting: parameter 'quantile' must satisfy "
                    "0 < quantile < 1 when loss='quantile'"
                )
        elif quantile is not None:
            raise ModelConfigError(
                "hist_gradient_boosting: parameter 'quantile' must be None "
                "unless loss='quantile'"
            )
        object.__setattr__(
            self,
            "learning_rate",
            _float_value(
                self.learning_rate,
                model_name="hist_gradient_boosting",
                field_name="learning_rate",
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "max_iter",
            _positive_int(
                self.max_iter,
                model_name="hist_gradient_boosting",
                field_name="max_iter",
            ),
        )
        max_leaf_nodes = _int_value(
            self.max_leaf_nodes,
            model_name="hist_gradient_boosting",
            field_name="max_leaf_nodes",
            optional=True,
        )
        if max_leaf_nodes is not None and max_leaf_nodes < 2:
            raise ModelConfigError(
                "hist_gradient_boosting: parameter 'max_leaf_nodes' must be "
                "None or an integer >= 2"
            )
        object.__setattr__(self, "max_leaf_nodes", max_leaf_nodes)
        object.__setattr__(
            self,
            "max_depth",
            _positive_int(
                self.max_depth,
                model_name="hist_gradient_boosting",
                field_name="max_depth",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "min_samples_leaf",
            _positive_int(
                self.min_samples_leaf,
                model_name="hist_gradient_boosting",
                field_name="min_samples_leaf",
            ),
        )
        object.__setattr__(
            self,
            "l2_regularization",
            _float_value(
                self.l2_regularization,
                model_name="hist_gradient_boosting",
                field_name="l2_regularization",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "max_features",
            _float_value(
                self.max_features,
                model_name="hist_gradient_boosting",
                field_name="max_features",
                minimum=0.0,
                minimum_inclusive=False,
                maximum=1.0,
            ),
        )
        max_bins = _int_value(
            self.max_bins,
            model_name="hist_gradient_boosting",
            field_name="max_bins",
        )
        if max_bins is None or not 2 <= max_bins <= 255:
            raise ModelConfigError(
                "hist_gradient_boosting: parameter 'max_bins' must be an "
                "integer between 2 and 255"
            )
        object.__setattr__(self, "max_bins", max_bins)
        for field_name in ("early_stopping", "warm_start"):
            object.__setattr__(
                self,
                field_name,
                _bool_value(
                    getattr(self, field_name),
                    model_name="hist_gradient_boosting",
                    field_name=field_name,
                ),
            )
        object.__setattr__(
            self,
            "n_iter_no_change",
            _positive_int(
                self.n_iter_no_change,
                model_name="hist_gradient_boosting",
                field_name="n_iter_no_change",
            ),
        )
        object.__setattr__(
            self,
            "tol",
            _float_value(
                self.tol,
                model_name="hist_gradient_boosting",
                field_name="tol",
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "random_state",
            _int_value(
                self.random_state,
                model_name="hist_gradient_boosting",
                field_name="random_state",
                optional=True,
            ),
        )
        verbose = _int_value(
            self.verbose,
            model_name="hist_gradient_boosting",
            field_name="verbose",
        )
        if verbose is None or verbose < 0:
            raise ModelConfigError(
                "hist_gradient_boosting: parameter 'verbose' must be a "
                "non-negative integer"
            )
        object.__setattr__(self, "verbose", verbose)

    @classmethod
    def from_dict(
        cls, values: Mapping[str, object] | None
    ) -> "HistGradientBoostingModelConfig":
        """Resolve validated user overrides over all project defaults."""
        return cls(
            **_config_values(
                cls,
                values,
                model_name="hist_gradient_boosting",
            )
        )

    def as_dict(self) -> dict[str, object]:
        """Return every public resolved parameter as JSON-safe values."""
        return {
            "loss": self.loss,
            "quantile": self.quantile,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "max_features": self.max_features,
            "max_bins": self.max_bins,
            "early_stopping": self.early_stopping,
            "n_iter_no_change": self.n_iter_no_change,
            "tol": self.tol,
            "warm_start": self.warm_start,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    @classmethod
    def parameter_schema(cls) -> tuple[ModelParameterSpec, ...]:
        """Return a stable UI-ready schema for every public parameter."""
        return (
            ModelParameterSpec(
                "loss",
                "Loss",
                "choice",
                "squared_error",
                "Regression loss optimized by the estimator.",
                choices=_LOSSES,
                ui_control="select",
            ),
            ModelParameterSpec(
                "quantile",
                "Quantile",
                "optional_float",
                None,
                "Required strictly between 0 and 1 only for quantile loss.",
                advanced=True,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
            ),
            ModelParameterSpec(
                "learning_rate",
                "Learning rate",
                "float",
                0.1,
                "Shrinkage applied to every boosting iteration.",
                minimum=0.0,
                step=0.01,
            ),
            ModelParameterSpec(
                "max_iter",
                "Maximum iterations",
                "int",
                100,
                "Maximum number of boosting iterations.",
                minimum=1,
                step=10,
            ),
            ModelParameterSpec(
                "max_leaf_nodes",
                "Maximum leaf nodes",
                "optional_int",
                31,
                "Maximum leaves per tree, or None for no leaf limit.",
                minimum=2,
                step=1,
            ),
            ModelParameterSpec(
                "max_depth",
                "Maximum depth",
                "optional_int",
                None,
                "Maximum tree depth, or None for no depth limit.",
                minimum=1,
                step=1,
            ),
            ModelParameterSpec(
                "min_samples_leaf",
                "Minimum samples per leaf",
                "int",
                20,
                "Minimum training samples in each leaf.",
                advanced=True,
                minimum=1,
                step=1,
            ),
            ModelParameterSpec(
                "l2_regularization",
                "L2 regularization",
                "float",
                0.0,
                "L2 penalty applied to leaf values.",
                advanced=True,
                minimum=0.0,
                step=0.1,
            ),
            ModelParameterSpec(
                "max_features",
                "Maximum feature fraction",
                "float",
                1.0,
                "Fraction of features considered at each node.",
                advanced=True,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
            ),
            ModelParameterSpec(
                "max_bins",
                "Maximum bins",
                "int",
                255,
                "Maximum histogram bins for each feature.",
                advanced=True,
                minimum=2,
                maximum=255,
                step=1,
            ),
            ModelParameterSpec(
                "early_stopping",
                "Early stopping",
                "bool",
                False,
                "Use only a caller-provided time validation set.",
                ui_control="checkbox",
            ),
            ModelParameterSpec(
                "n_iter_no_change",
                "Iterations without improvement",
                "int",
                10,
                "Early-stopping patience for the external validation set.",
                advanced=True,
                minimum=1,
                step=1,
            ),
            ModelParameterSpec(
                "tol",
                "Tolerance",
                "float",
                1e-7,
                "Minimum score improvement for early stopping.",
                advanced=True,
                minimum=0.0,
                step=1e-8,
            ),
            ModelParameterSpec(
                "warm_start",
                "Warm start",
                "bool",
                False,
                "Accepted by sklearn, but each adapter fit creates a new "
                "estimator and never reuses trees across calls.",
                advanced=True,
                ui_control="checkbox",
            ),
            ModelParameterSpec(
                "random_state",
                "Random state",
                "optional_int",
                42,
                "Optional seed used by the estimator.",
                advanced=True,
                step=1,
            ),
            ModelParameterSpec(
                "verbose",
                "Verbosity",
                "int",
                0,
                "Non-negative sklearn training verbosity.",
                advanced=True,
                minimum=0,
                step=1,
            ),
        )

    def to_estimator_params(self) -> dict[str, object]:
        """Return supported sklearn parameters with internal splitting disabled."""
        return {
            **self.as_dict(),
            "validation_fraction": None,
        }


class HistGradientBoostingModelAdapter(RegressionModelAdapter):
    """HistGradientBoosting adapter using native NaN and external validation."""

    _MODEL_NAME = "hist_gradient_boosting"

    def __init__(
        self, config: HistGradientBoostingModelConfig | None = None
    ) -> None:
        if config is not None and not isinstance(
            config, HistGradientBoostingModelConfig
        ):
            raise ModelConfigError(
                "hist_gradient_boosting: config must be "
                "HistGradientBoostingModelConfig or None"
            )
        self._config = (
            HistGradientBoostingModelConfig() if config is None else config
        )
        super().__init__()

    @property
    def config(self) -> HistGradientBoostingModelConfig:
        """Return the immutable resolved tree-model configuration."""
        return self._config

    def _build_estimator(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            **self._config.to_estimator_params()
        )

    def get_parameter_schema(self) -> tuple[ModelParameterSpec, ...]:
        """Return the HistGradientBoosting parameter schema."""
        return self._config.parameter_schema()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> ModelFitAudit:
        """Fit native-missing boosting with optional external early stopping."""
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
        valid_y: pd.Series | None = None
        if X_valid is not None and y_valid is not None:
            valid_x = _numeric_frame(
                X_valid,
                model_name=self.model_name,
                context="X_valid",
                minimum_rows=1,
                expected_features=tuple(train_x.columns),
            )
            valid_y = _numeric_target(
                y_valid,
                model_name=self.model_name,
                context="y_valid",
                expected_index=valid_x.index,
            )
        if self._config.early_stopping:
            if valid_x is None or valid_y is None:
                raise ModelConfigError(
                    f"{self.model_name}: early_stopping=True requires external "
                    "X_valid and y_valid"
                )
            if not _SUPPORTS_EXTERNAL_VALIDATION:
                raise ModelConfigError(
                    f"{self.model_name}: installed sklearn does not support "
                    "external X_val and y_val; internal validation is forbidden"
                )
            if len(train_x.index.intersection(valid_x.index)) > 0:
                raise ModelDataError(
                    f"{self.model_name}: early-stopping training and validation "
                    "indices must not overlap"
                )

        estimator = self._build_estimator()
        fit_kwargs: dict[str, object] = {}
        if self._config.early_stopping:
            fit_kwargs = {"X_val": valid_x, "y_val": valid_y}
        try:
            estimator.fit(train_x, train_y, **fit_kwargs)
            feature_names = tuple(train_x.columns)
            constant_features = tuple(
                name
                for name in feature_names
                if train_x[name].dropna().nunique() == 1
            )
            n_iterations_value = getattr(estimator, "n_iter_", None)
            n_iterations = (
                None
                if n_iterations_value is None
                else int(n_iterations_value)
            )
            best_iteration_value = getattr(estimator, "best_iteration_", None)
            best_iteration = (
                None
                if best_iteration_value is None
                else int(best_iteration_value)
            )
            audit = ModelFitAudit(
                model_name=self.model_name,
                estimator_class=type(estimator).__name__,
                feature_names=feature_names,
                n_train_rows=len(train_x),
                n_validation_rows=0 if valid_x is None else len(valid_x),
                n_features=train_x.shape[1],
                train_missing_counts=tuple(
                    (name, int(train_x[name].isna().sum()))
                    for name in feature_names
                ),
                validation_missing_counts=(
                    ()
                    if valid_x is None
                    else tuple(
                        (name, int(valid_x[name].isna().sum()))
                        for name in feature_names
                    )
                ),
                imputation_values=(),
                scaler_means=(),
                scaler_scales=(),
                constant_features=constant_features,
                validation_provided=valid_x is not None,
                validation_used_for_fit=self._config.early_stopping,
                resolved_parameters=tuple(self._config.as_dict().items()),
                preprocessing_parameters=(
                    ("native_missing_support", True),
                    ("imputer", None),
                    ("scaler", None),
                ),
                estimator_intercept=None,
                python_version=platform.python_version(),
                numpy_version=np.__version__,
                sklearn_version=sklearn.__version__,
                native_missing_support=True,
                imputer_enabled=False,
                scaler_enabled=False,
                best_iteration=best_iteration,
                n_iterations=n_iterations,
            )
        except ModelError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"{self.model_name}: model fit failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._pipeline = estimator  # stores the estimator, never a Pipeline
        self._audit = audit
        self._importance = None
        return audit

    def get_feature_importance(self) -> pd.DataFrame:
        """Reject unavailable native importance without fabricating values."""
        if not self.is_fitted:
            raise ModelNotFittedError(
                f"{self.model_name}: feature importance requires a fitted model"
            )
        raise ModelFeatureImportanceUnavailableError(
            "hist_gradient_boosting does not expose a supported native feature "
            "importance; permutation importance will be implemented in V3-F"
        )
