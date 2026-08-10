"""Stable contracts and centralized numerical policy for historical risk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
import json
import math
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .errors import RiskModelConfigError, RiskModelValidationError


COVARIANCE_SYMMETRY_TOLERANCE = 1e-12
PSD_RELATIVE_TOLERANCE = 1e-10


def _normalize_date(value: object, *, field: str) -> pd.Timestamp:
    if value is None or value is pd.NaT:
        raise RiskModelValidationError(f"{field} must be a valid date.")
    if isinstance(value, str):
        if not value or value != value.strip():
            raise RiskModelValidationError(f"{field} has invalid format.")
        date_format = "%Y%m%d" if len(value) == 8 and value.isdigit() else "%Y-%m-%d"
        try:
            return pd.Timestamp(datetime.strptime(value, date_format).date())
        except ValueError as exc:
            raise RiskModelValidationError(f"{field} has invalid format.") from exc
    if not isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        raise RiskModelValidationError(f"{field} must be date-like.")
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tz is not None:
        raise RiskModelValidationError(f"{field} must be timezone-naive and valid.")
    return result.normalize()


def _json_safe(value: object, *, context: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise RiskModelConfigError(f"{context} must contain finite JSON values.")
        return result
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise RiskModelConfigError(f"{context} keys must be strings.")
        return {key: _json_safe(value[key], context=context) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, context=context) for item in value]
    raise RiskModelConfigError(f"{context} must contain only JSON-safe values.")


class _FrozenMapping(Mapping[str, object]):
    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, object]) -> None:
        self._data = deepcopy(dict(value))

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RiskModelConfigError(f"{field} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class RiskModelConfig:
    estimator: str
    params: Mapping[str, object]
    lookback_trading_days: int
    min_observations: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimator", _name(self.estimator, field="estimator"))
        safe = _json_safe(self.params, context="risk estimator params")
        if not isinstance(safe, dict):
            raise RiskModelConfigError("risk estimator params must be a mapping.")
        object.__setattr__(self, "params", _FrozenMapping(safe))
        if type(self.lookback_trading_days) is not int or type(self.min_observations) is not int:
            raise RiskModelConfigError("lookback and min_observations must be strict ints.")
        if self.lookback_trading_days < 2 or not 2 <= self.min_observations <= self.lookback_trading_days:
            raise RiskModelConfigError(
                "risk model requires lookback >= 2 and 2 <= min_observations <= lookback."
            )

    @classmethod
    def from_dict(cls, value: object) -> "RiskModelConfig":
        if not isinstance(value, Mapping):
            raise RiskModelConfigError("risk model config must be a mapping.")
        safe = _json_safe(value, context="risk model config")
        assert isinstance(safe, dict)
        expected = {"estimator", "params", "lookback_trading_days", "min_observations"}
        if set(safe) != expected:
            raise RiskModelConfigError("risk model config fields are invalid.")
        return cls(**safe)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        result = {
            "estimator": self.estimator,
            "params": deepcopy(dict(self.params)),
            "lookback_trading_days": self.lookback_trading_days,
            "min_observations": self.min_observations,
        }
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result


@dataclass(frozen=True)
class RiskModelRequest:
    formation_date: pd.Timestamp
    assets: tuple[str, ...]
    config: RiskModelConfig

    def __init__(self, formation_date: object, assets: Sequence[str], config: RiskModelConfig) -> None:
        try:
            formation = _normalize_date(formation_date, field="formation_date")
        except ValueError as exc:
            raise RiskModelValidationError(str(exc)) from exc
        if isinstance(assets, (str, bytes)):
            raise RiskModelValidationError("assets must be a sequence of identifiers.")
        values = tuple(assets)
        if not values or len(values) != len(set(values)) or any(
            not isinstance(item, str) or not item or item != item.strip() for item in values
        ):
            raise RiskModelValidationError("assets must be unique non-empty trimmed strings.")
        if not isinstance(config, RiskModelConfig):
            raise RiskModelValidationError("config must be RiskModelConfig.")
        object.__setattr__(self, "formation_date", formation)
        object.__setattr__(self, "assets", values)
        object.__setattr__(self, "config", config)


@dataclass(frozen=True)
class RiskEstimate:
    covariance: np.ndarray
    diagnostics: Mapping[str, object]


@runtime_checkable
class RiskEstimator(Protocol):
    name: str

    def parse_params(self, raw_params: Mapping[str, object]) -> object: ...

    def estimate(self, aligned_returns: np.ndarray, parsed_params: object) -> RiskEstimate: ...


class RiskModelResult:
    __slots__ = ("_formation_date", "_risk_cutoff", "_assets", "_covariance", "_volatility", "_observation_count", "_estimator", "_diagnostics")

    def __init__(self, *, formation_date: object, risk_cutoff: object, assets: Sequence[str], covariance: object, observation_count: int, estimator: object, diagnostics: Mapping[str, object]) -> None:
        try:
            formation = _normalize_date(formation_date, field="formation_date")
            cutoff = _normalize_date(risk_cutoff, field="risk_cutoff")
        except ValueError as exc:
            raise RiskModelValidationError(str(exc)) from exc
        values = tuple(assets)
        if not values or len(values) != len(set(values)):
            raise RiskModelValidationError("result assets must be unique and non-empty.")
        matrix = np.array(covariance, dtype=np.float64, copy=True)
        if matrix.shape != (len(values), len(values)) or not np.isfinite(matrix).all():
            raise RiskModelValidationError("covariance shape or finite contract failed.")
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=COVARIANCE_SYMMETRY_TOLERANCE):
            raise RiskModelValidationError("covariance must be symmetric.")
        diagonal = np.diag(matrix)
        if bool((diagonal <= 0.0).any()):
            raise RiskModelValidationError("covariance diagonal must be strictly positive.")
        eigenvalues = np.linalg.eigvalsh(matrix)
        scale = max(1.0, float(np.max(np.abs(diagonal))))
        if float(eigenvalues[0]) < -PSD_RELATIVE_TOLERANCE * scale:
            raise RiskModelValidationError("covariance must be positive semidefinite.")
        if type(observation_count) is not int or observation_count < 2:
            raise RiskModelValidationError("observation_count must be a strict int >= 2.")
        name = _name(estimator, field="estimator")
        safe = _json_safe(diagnostics, context="risk diagnostics")
        assert isinstance(safe, dict)
        volatility = np.sqrt(diagonal)
        matrix.setflags(write=False)
        volatility.setflags(write=False)
        self._formation_date, self._risk_cutoff, self._assets = formation, cutoff, values
        self._covariance, self._volatility = matrix, volatility
        self._observation_count, self._estimator, self._diagnostics = observation_count, name, safe

    @property
    def formation_date(self) -> pd.Timestamp: return self._formation_date
    @property
    def risk_cutoff(self) -> pd.Timestamp: return self._risk_cutoff
    @property
    def assets(self) -> tuple[str, ...]: return self._assets
    @property
    def covariance(self) -> np.ndarray:
        result = self._covariance.copy(); result.setflags(write=False); return result
    @property
    def volatility(self) -> np.ndarray:
        result = self._volatility.copy(); result.setflags(write=False); return result
    @property
    def observation_count(self) -> int: return self._observation_count
    @property
    def estimator(self) -> str: return self._estimator
    @property
    def diagnostics(self) -> dict[str, object]: return deepcopy(self._diagnostics)


@runtime_checkable
class RiskModelService(Protocol):
    def estimate(self, request: RiskModelRequest) -> RiskModelResult: ...
