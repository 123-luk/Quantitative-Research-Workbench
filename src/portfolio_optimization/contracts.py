"""Backend-neutral minimum-variance optimization contracts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Protocol, runtime_checkable

import numpy as np

from .errors import PortfolioOptimizationValidationError


def _safe(value: object, *, context: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _safe(value[key], context=context) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_safe(item, context=context) for item in value]
    raise PortfolioOptimizationValidationError(f"{context} must be JSON-safe and finite.")


class MinimumVarianceProblem:
    __slots__ = ("_covariance", "_initial_weights", "_lower_bounds", "_upper_bounds", "_name")

    def __init__(self, covariance: object, initial_weights: object, lower_bounds: object, upper_bounds: object, name: str = "minimum_variance") -> None:
        covariance_array = np.array(covariance, dtype=np.float64, copy=True)
        initial = np.array(initial_weights, dtype=np.float64, copy=True)
        lower = np.array(lower_bounds, dtype=np.float64, copy=True)
        upper = np.array(upper_bounds, dtype=np.float64, copy=True)
        if covariance_array.ndim != 2 or covariance_array.shape[0] != covariance_array.shape[1]:
            raise PortfolioOptimizationValidationError("covariance must be square.")
        count = covariance_array.shape[0]
        if any(item.shape != (count,) for item in (initial, lower, upper)) or not all(
            np.isfinite(item).all() for item in (covariance_array, initial, lower, upper)
        ):
            raise PortfolioOptimizationValidationError("problem arrays have invalid shape or values.")
        if bool((lower > upper).any()) or not np.isclose(initial.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise PortfolioOptimizationValidationError("problem bounds or initial equality are invalid.")
        if bool((initial < lower).any()) or bool((initial > upper).any()):
            raise PortfolioOptimizationValidationError("initial weights violate bounds.")
        if not isinstance(name, str) or not name:
            raise PortfolioOptimizationValidationError("problem name must be non-empty.")
        for item in (covariance_array, initial, lower, upper): item.setflags(write=False)
        self._covariance, self._initial_weights = covariance_array, initial
        self._lower_bounds, self._upper_bounds, self._name = lower, upper, name

    @property
    def covariance(self) -> np.ndarray: return self._covariance.copy()
    @property
    def initial_weights(self) -> np.ndarray: return self._initial_weights.copy()
    @property
    def lower_bounds(self) -> np.ndarray: return self._lower_bounds.copy()
    @property
    def upper_bounds(self) -> np.ndarray: return self._upper_bounds.copy()
    @property
    def name(self) -> str: return self._name
    def objective(self, weights: np.ndarray) -> float: return float(0.5 * weights @ self._covariance @ weights)
    def gradient(self, weights: np.ndarray) -> np.ndarray: return self._covariance @ weights


class OptimizationResult:
    __slots__ = ("_weights", "success", "status", "message", "objective_value", "iterations", "_diagnostics")

    def __init__(self, *, weights: object, success: bool, status: int, message: str, objective_value: float, iterations: int, diagnostics: Mapping[str, object] | None = None) -> None:
        array = np.array(weights, dtype=np.float64, copy=True)
        if array.ndim != 1 or not np.isfinite(array).all() or type(success) is not bool or type(status) is not int or not isinstance(message, str) or isinstance(objective_value, bool) or not isinstance(objective_value, (int, float)) or not math.isfinite(float(objective_value)) or type(iterations) is not int or iterations < 0:
            raise PortfolioOptimizationValidationError("optimization result fields are invalid.")
        safe = _safe(diagnostics or {}, context="optimization diagnostics")
        assert isinstance(safe, dict)
        array.setflags(write=False)
        self._weights = array
        self.success, self.status, self.message = success, status, message
        self.objective_value, self.iterations, self._diagnostics = float(objective_value), iterations, safe

    @property
    def weights(self) -> np.ndarray:
        result = self._weights.copy(); result.setflags(write=False); return result
    @property
    def diagnostics(self) -> dict[str, object]: return deepcopy(self._diagnostics)


@runtime_checkable
class OptimizerBackend(Protocol):
    def solve(self, problem: MinimumVarianceProblem) -> OptimizationResult: ...
