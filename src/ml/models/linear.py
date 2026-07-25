"""Validated Ridge and ElasticNet regression adapters."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Integral, Real
from typing import Mapping

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge

from src.ml.models.base import (
    ModelConfigError,
    ModelParameterSpec,
    RegressionModelAdapter,
)


_RIDGE_SOLVERS = (
    "auto",
    "svd",
    "cholesky",
    "lsqr",
    "sparse_cg",
    "sag",
    "saga",
    "lbfgs",
)
_ELASTIC_NET_SELECTIONS = ("cyclic", "random")


def _float_value(
    value: object,
    *,
    model_name: str,
    field_name: str,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ModelConfigError(
            f"{model_name}: parameter {field_name!r} must be a finite number"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ModelConfigError(
            f"{model_name}: parameter {field_name!r} must be finite"
        )
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ModelConfigError(
                f"{model_name}: parameter {field_name!r} must be {operator} {minimum}"
            )
    if maximum is not None and result > maximum:
        raise ModelConfigError(
            f"{model_name}: parameter {field_name!r} must be <= {maximum}"
        )
    return result


def _bool_value(value: object, *, model_name: str, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ModelConfigError(
            f"{model_name}: parameter {field_name!r} must be bool"
        )
    return value


def _int_value(
    value: object,
    *,
    model_name: str,
    field_name: str,
    optional: bool = False,
) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        qualifier = "None or " if optional else ""
        raise ModelConfigError(
            f"{model_name}: parameter {field_name!r} must be {qualifier}an integer"
        )
    return int(value)


def _positive_int(
    value: object,
    *,
    model_name: str,
    field_name: str,
    optional: bool = False,
) -> int | None:
    result = _int_value(
        value,
        model_name=model_name,
        field_name=field_name,
        optional=optional,
    )
    if result is not None and result <= 0:
        raise ModelConfigError(
            f"{model_name}: parameter {field_name!r} must be a positive integer"
        )
    return result


def _choice(
    value: object,
    *,
    model_name: str,
    field_name: str,
    choices: tuple[str, ...],
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelConfigError(
            f"{model_name}: parameter {field_name!r} must be a non-empty string"
        )
    result = value.strip().lower()
    if result not in choices:
        raise ModelConfigError(
            f"{model_name}: parameter {field_name!r} must be one of {choices!r}"
        )
    return result


def _config_values(
    config_class: type,
    values: Mapping[str, object] | None,
    *,
    model_name: str,
) -> dict[str, object]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ModelConfigError(
            f"{model_name}: parameters must be a mapping or None"
        )
    allowed = {field.name for field in fields(config_class)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ModelConfigError(
            f"{model_name}: unknown parameter(s) {unknown!r}; "
            f"valid parameters are {sorted(allowed)!r}"
        )
    return dict(values)


@dataclass(frozen=True)
class RidgeModelConfig:
    """Validated public Ridge parameters for YAML, CLI, and UI overrides."""

    alpha: float = 1.0
    fit_intercept: bool = True
    solver: str = "auto"
    tol: float = 1e-4
    max_iter: int | None = None
    positive: bool = False
    random_state: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "alpha",
            _float_value(
                self.alpha,
                model_name="ridge",
                field_name="alpha",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "fit_intercept",
            _bool_value(
                self.fit_intercept,
                model_name="ridge",
                field_name="fit_intercept",
            ),
        )
        object.__setattr__(
            self,
            "solver",
            _choice(
                self.solver,
                model_name="ridge",
                field_name="solver",
                choices=_RIDGE_SOLVERS,
            ),
        )
        object.__setattr__(
            self,
            "tol",
            _float_value(
                self.tol,
                model_name="ridge",
                field_name="tol",
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "max_iter",
            _positive_int(
                self.max_iter,
                model_name="ridge",
                field_name="max_iter",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "positive",
            _bool_value(
                self.positive,
                model_name="ridge",
                field_name="positive",
            ),
        )
        object.__setattr__(
            self,
            "random_state",
            _int_value(
                self.random_state,
                model_name="ridge",
                field_name="random_state",
                optional=True,
            ),
        )
        if self.solver == "lbfgs" and not self.positive:
            raise ModelConfigError(
                "ridge: parameter combination solver='lbfgs' requires positive=True"
            )
        if self.positive and self.solver not in {"auto", "lbfgs"}:
            raise ModelConfigError(
                "ridge: parameter positive=True requires solver 'auto' or 'lbfgs'"
            )

    @classmethod
    def from_dict(
        cls, values: Mapping[str, object] | None
    ) -> "RidgeModelConfig":
        """Resolve validated overrides on top of all Ridge defaults."""
        return cls(**_config_values(cls, values, model_name="ridge"))

    def as_dict(self) -> dict[str, object]:
        """Return every resolved Ridge parameter as JSON-safe Python values."""
        return {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "solver": self.solver,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "positive": self.positive,
            "random_state": self.random_state,
        }

    @classmethod
    def parameter_schema(cls) -> tuple[ModelParameterSpec, ...]:
        """Return the stable public Ridge parameter schema."""
        return (
            ModelParameterSpec(
                "alpha",
                "Alpha",
                "float",
                1.0,
                "L2 regularization strength.",
                minimum=0.0,
                step=0.1,
            ),
            ModelParameterSpec(
                "fit_intercept",
                "Fit intercept",
                "bool",
                True,
                "Whether to estimate an intercept.",
                advanced=True,
                ui_control="checkbox",
            ),
            ModelParameterSpec(
                "solver",
                "Solver",
                "choice",
                "auto",
                "Numerical solver used by Ridge.",
                advanced=True,
                choices=_RIDGE_SOLVERS,
                ui_control="select",
            ),
            ModelParameterSpec(
                "tol",
                "Tolerance",
                "float",
                1e-4,
                "Solver stopping tolerance.",
                advanced=True,
                minimum=0.0,
                step=1e-5,
            ),
            ModelParameterSpec(
                "max_iter",
                "Maximum iterations",
                "optional_int",
                None,
                "Optional solver iteration limit.",
                advanced=True,
                minimum=1,
                step=1,
            ),
            ModelParameterSpec(
                "positive",
                "Positive coefficients",
                "bool",
                False,
                "Constrain coefficients to be non-negative.",
                advanced=True,
                ui_control="checkbox",
            ),
            ModelParameterSpec(
                "random_state",
                "Random state",
                "optional_int",
                None,
                "Optional random seed for stochastic solvers.",
                advanced=True,
                step=1,
            ),
        )

    def to_estimator_params(self) -> dict[str, object]:
        """Map only parameters supported by sklearn 1.9 Ridge."""
        return self.as_dict()


@dataclass(frozen=True)
class ElasticNetModelConfig:
    """Validated public ElasticNet parameters for external override layers."""

    alpha: float = 1.0
    l1_ratio: float = 0.5
    fit_intercept: bool = True
    max_iter: int = 5000
    tol: float = 1e-4
    selection: str = "cyclic"
    random_state: int | None = 42
    positive: bool = False
    warm_start: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "alpha",
            _float_value(
                self.alpha,
                model_name="elastic_net",
                field_name="alpha",
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "l1_ratio",
            _float_value(
                self.l1_ratio,
                model_name="elastic_net",
                field_name="l1_ratio",
                minimum=0.0,
                minimum_inclusive=False,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "fit_intercept",
            _bool_value(
                self.fit_intercept,
                model_name="elastic_net",
                field_name="fit_intercept",
            ),
        )
        max_iter = _positive_int(
            self.max_iter,
            model_name="elastic_net",
            field_name="max_iter",
        )
        object.__setattr__(self, "max_iter", max_iter)
        object.__setattr__(
            self,
            "tol",
            _float_value(
                self.tol,
                model_name="elastic_net",
                field_name="tol",
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "selection",
            _choice(
                self.selection,
                model_name="elastic_net",
                field_name="selection",
                choices=_ELASTIC_NET_SELECTIONS,
            ),
        )
        object.__setattr__(
            self,
            "random_state",
            _int_value(
                self.random_state,
                model_name="elastic_net",
                field_name="random_state",
                optional=True,
            ),
        )
        for field_name in ("positive", "warm_start"):
            object.__setattr__(
                self,
                field_name,
                _bool_value(
                    getattr(self, field_name),
                    model_name="elastic_net",
                    field_name=field_name,
                ),
            )

    @classmethod
    def from_dict(
        cls, values: Mapping[str, object] | None
    ) -> "ElasticNetModelConfig":
        """Resolve validated overrides on top of all ElasticNet defaults."""
        return cls(**_config_values(cls, values, model_name="elastic_net"))

    def as_dict(self) -> dict[str, object]:
        """Return every resolved ElasticNet parameter as JSON-safe values."""
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "selection": self.selection,
            "random_state": self.random_state,
            "positive": self.positive,
            "warm_start": self.warm_start,
        }

    @classmethod
    def parameter_schema(cls) -> tuple[ModelParameterSpec, ...]:
        """Return the stable public ElasticNet parameter schema."""
        return (
            ModelParameterSpec(
                "alpha",
                "Alpha",
                "float",
                1.0,
                "Combined L1 and L2 regularization strength.",
                minimum=0.0,
                step=0.1,
            ),
            ModelParameterSpec(
                "l1_ratio",
                "L1 ratio",
                "float",
                0.5,
                "Share of L1 regularization in the combined penalty.",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
            ),
            ModelParameterSpec(
                "fit_intercept",
                "Fit intercept",
                "bool",
                True,
                "Whether to estimate an intercept.",
                advanced=True,
                ui_control="checkbox",
            ),
            ModelParameterSpec(
                "max_iter",
                "Maximum iterations",
                "int",
                5000,
                "Maximum coordinate-descent iterations.",
                advanced=True,
                minimum=1,
                step=100,
            ),
            ModelParameterSpec(
                "tol",
                "Tolerance",
                "float",
                1e-4,
                "Optimization stopping tolerance.",
                advanced=True,
                minimum=0.0,
                step=1e-5,
            ),
            ModelParameterSpec(
                "selection",
                "Coordinate selection",
                "choice",
                "cyclic",
                "Coordinate update order.",
                advanced=True,
                choices=_ELASTIC_NET_SELECTIONS,
                ui_control="select",
            ),
            ModelParameterSpec(
                "random_state",
                "Random state",
                "optional_int",
                42,
                "Optional seed used for random coordinate selection.",
                advanced=True,
                step=1,
            ),
            ModelParameterSpec(
                "positive",
                "Positive coefficients",
                "bool",
                False,
                "Constrain coefficients to be non-negative.",
                advanced=True,
                ui_control="checkbox",
            ),
            ModelParameterSpec(
                "warm_start",
                "Warm start",
                "bool",
                False,
                "Allow sklearn to reuse a prior estimator fit.",
                advanced=True,
                ui_control="checkbox",
            ),
        )

    def to_estimator_params(self) -> dict[str, object]:
        """Map only parameters supported by sklearn 1.9 ElasticNet."""
        return self.as_dict()


class RidgeModelAdapter(RegressionModelAdapter):
    """Uniform Ridge regression adapter."""

    _MODEL_NAME = "ridge"

    def __init__(self, config: RidgeModelConfig | None = None) -> None:
        if config is not None and not isinstance(config, RidgeModelConfig):
            raise ModelConfigError(
                "ridge: config must be RidgeModelConfig or None"
            )
        self._config = RidgeModelConfig() if config is None else config
        super().__init__()

    @property
    def config(self) -> RidgeModelConfig:
        """Return the immutable resolved Ridge configuration."""
        return self._config

    def _build_estimator(self) -> Ridge:
        return Ridge(**self._config.to_estimator_params())

    def get_parameter_schema(self) -> tuple[ModelParameterSpec, ...]:
        """Return the Ridge parameter schema."""
        return self._config.parameter_schema()


class ElasticNetModelAdapter(RegressionModelAdapter):
    """Uniform ElasticNet regression adapter."""

    _MODEL_NAME = "elastic_net"

    def __init__(self, config: ElasticNetModelConfig | None = None) -> None:
        if config is not None and not isinstance(config, ElasticNetModelConfig):
            raise ModelConfigError(
                "elastic_net: config must be ElasticNetModelConfig or None"
            )
        self._config = ElasticNetModelConfig() if config is None else config
        super().__init__()

    @property
    def config(self) -> ElasticNetModelConfig:
        """Return the immutable resolved ElasticNet configuration."""
        return self._config

    def _build_estimator(self) -> ElasticNet:
        return ElasticNet(**self._config.to_estimator_params())

    def get_parameter_schema(self) -> tuple[ModelParameterSpec, ...]:
        """Return the ElasticNet parameter schema."""
        return self._config.parameter_schema()
