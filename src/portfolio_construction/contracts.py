"""Stable, persistence-independent portfolio-construction contracts."""

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

from .errors import (
    PortfolioConstructionConfigError,
    PortfolioConstructionDataError,
    PortfolioConstructionValidationError,
)


WEIGHT_ABSOLUTE_TOLERANCE = 1e-12
CANDIDATE_COLUMNS = ("ts_code", "score", "rank", "selection_position")
WEIGHT_COLUMNS = ("ts_code", "target_weight")
RETURN_COLUMNS = ("trade_date", "ts_code", "return")


class _FrozenMapping(Mapping[str, object]):
    """Small detached immutable mapping that remains deepcopy-compatible."""

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, object]) -> None:
        self._data = dict(value)

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __deepcopy__(self, memo):
        del memo
        return self


def normalize_date(value: object, *, field_name: str) -> pd.Timestamp:
    """Apply the V6 timezone-naive, normalized-date convention."""
    if value is None or value is pd.NaT:
        raise PortfolioConstructionValidationError(
            f"{field_name} must be a valid date."
        )
    if isinstance(value, str):
        if not value or value != value.strip():
            raise PortfolioConstructionValidationError(
                f"{field_name} must use YYYY-MM-DD or YYYYMMDD format."
            )
        date_format = "%Y%m%d" if len(value) == 8 and value.isdigit() else "%Y-%m-%d"
        try:
            return pd.Timestamp(datetime.strptime(value, date_format).date())
        except ValueError as exc:
            raise PortfolioConstructionValidationError(
                f"{field_name} must use YYYY-MM-DD or YYYYMMDD format."
            ) from exc
    if not isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        raise PortfolioConstructionValidationError(
            f"{field_name} must be a supported date-like value."
        )
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortfolioConstructionValidationError(
            f"{field_name} must be a valid date."
        ) from exc
    if pd.isna(result) or result.tz is not None:
        raise PortfolioConstructionValidationError(
            f"{field_name} must be timezone-naive and valid."
        )
    return result.normalize()


def json_safe(value: object, *, context: str) -> object:
    """Return a detached deterministic JSON value or reject it."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PortfolioConstructionConfigError(
                f"{context} must contain only finite JSON values."
            )
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PortfolioConstructionConfigError(
                f"{context} mapping keys must be strings."
            )
        return {
            key: json_safe(value[key], context=context)
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item, context=context) for item in value]
    raise PortfolioConstructionConfigError(
        f"{context} must contain only JSON-safe values."
    )


def strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PortfolioConstructionConfigError(f"{context} must be a mapping.")
    safe = json_safe(value, context=context)
    assert isinstance(safe, dict)
    json.dumps(safe, allow_nan=False, sort_keys=True)
    return safe


def canonical_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PortfolioConstructionConfigError(
            f"{field_name} must be a non-empty trimmed string."
        )
    return value


@dataclass(frozen=True)
class ConstraintSpec:
    """Generic serialized constraint declaration."""

    type: str
    params: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", canonical_name(self.type, field_name="type"))
        values = strict_mapping(self.params, context="constraint params")
        object.__setattr__(self, "params", _FrozenMapping(values))

    @classmethod
    def from_dict(cls, value: object) -> ConstraintSpec:
        values = strict_mapping(value, context="constraint")
        if set(values) != {"type", "params"}:
            raise PortfolioConstructionConfigError(
                "constraint fields must be exactly ('type', 'params')."
            )
        return cls(
            type=values["type"], params=values["params"]  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "params": deepcopy(dict(self.params))}


@dataclass(frozen=True)
class PortfolioConstructionConfig:
    """Generic extensible strategy configuration."""

    method: str
    params: Mapping[str, object]
    constraints: tuple[ConstraintSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "method", canonical_name(self.method, field_name="method")
        )
        params = strict_mapping(self.params, context="strategy params")
        object.__setattr__(self, "params", _FrozenMapping(params))
        if not isinstance(self.constraints, tuple):
            raise PortfolioConstructionConfigError("constraints must be a tuple.")
        specs = tuple(
            item if isinstance(item, ConstraintSpec) else ConstraintSpec.from_dict(item)
            for item in self.constraints
        )
        names = tuple(item.type for item in specs)
        if len(names) != len(set(names)):
            raise PortfolioConstructionConfigError(
                "duplicate constraint types are not supported."
            )
        object.__setattr__(self, "constraints", specs)

    @classmethod
    def from_dict(cls, value: object) -> PortfolioConstructionConfig:
        values = strict_mapping(value, context="portfolio construction config")
        if set(values) != {"method", "params", "constraints"}:
            raise PortfolioConstructionConfigError(
                "config fields must be exactly ('method', 'params', 'constraints')."
            )
        raw_constraints = values["constraints"]
        if not isinstance(raw_constraints, list):
            raise PortfolioConstructionConfigError("constraints must be a list.")
        return cls(
            method=values["method"],  # type: ignore[arg-type]
            params=values["params"],  # type: ignore[arg-type]
            constraints=tuple(
                ConstraintSpec.from_dict(item) for item in raw_constraints
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "params": deepcopy(dict(self.params)),
            "constraints": [item.to_dict() for item in self.constraints],
        }


def _candidate_frame(value: object) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise PortfolioConstructionValidationError(
            "candidates must be a non-empty DataFrame."
        )
    if (
        isinstance(value.columns, pd.MultiIndex)
        or tuple(value.columns) != CANDIDATE_COLUMNS
    ):
        raise PortfolioConstructionValidationError(
            f"candidates must contain exactly {CANDIDATE_COLUMNS!r}."
        )
    frame = value.copy(deep=True)
    codes = frame["ts_code"]
    if (
        codes.isna().any()
        or not codes.map(lambda item: isinstance(item, (str, np.str_))).all()
        or codes.map(lambda item: not str(item) or str(item) != str(item).strip()).any()
        or codes.duplicated().any()
    ):
        raise PortfolioConstructionValidationError(
            "candidate ts_code values must be unique non-empty trimmed strings."
        )
    frame["ts_code"] = codes.astype("string")
    score = frame["score"]
    if pd.api.types.is_bool_dtype(
        score.dtype
    ) or not pd.api.types.is_numeric_dtype(score.dtype):
        raise PortfolioConstructionValidationError(
            "candidate score must be real numeric data."
        )
    try:
        score_values = score.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise PortfolioConstructionValidationError(
            "candidate score must be real numeric data."
        ) from exc
    if not np.isfinite(score_values).all():
        raise PortfolioConstructionValidationError("candidate score must be finite.")
    frame["score"] = score_values
    for name in ("rank", "selection_position"):
        values = frame[name].tolist()
        if any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            for item in values
        ):
            raise PortfolioConstructionValidationError(
                f"candidate {name} must contain positive integer data."
            )
        converted = np.asarray(values, dtype=np.int64)
        if bool((converted <= 0).any()):
            raise PortfolioConstructionValidationError(
                f"candidate {name} must contain positive integer data."
            )
        frame[name] = converted
    positions = frame["selection_position"].to_numpy()
    expected = np.arange(1, len(frame) + 1, dtype=np.int64)
    if len(set(positions.tolist())) != len(frame) or set(positions) != set(expected):
        raise PortfolioConstructionValidationError(
            "selection_position must be unique contiguous 1..K."
        )
    return frame.sort_values("selection_position", kind="mergesort", ignore_index=True)


class PortfolioConstructionRequest:
    """One immutable selected-candidate snapshot for a formation date."""

    __slots__ = ("_formation_date", "_candidates")

    def __init__(self, formation_date: object, candidates: pd.DataFrame) -> None:
        self._formation_date = normalize_date(
            formation_date, field_name="formation_date"
        )
        self._candidates = _candidate_frame(candidates)

    @property
    def formation_date(self) -> pd.Timestamp:
        return self._formation_date

    @property
    def candidates(self) -> pd.DataFrame:
        return self._candidates.copy(deep=True)

    @property
    def ts_codes(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self._candidates["ts_code"])


class StrategyConstructionOutput:
    """Untrusted constructor output that the engine must validate."""

    __slots__ = ("_weights", "_diagnostics")

    def __init__(
        self, weights: object, diagnostics: Mapping[str, object] | None = None
    ) -> None:
        if not isinstance(weights, pd.DataFrame):
            raise PortfolioConstructionValidationError(
                "strategy weights must be a DataFrame."
            )
        self._weights = weights.copy(deep=True)
        safe = json_safe(diagnostics or {}, context="strategy diagnostics")
        assert isinstance(safe, dict)
        self._diagnostics = safe

    @property
    def weights(self) -> pd.DataFrame:
        return self._weights.copy(deep=True)

    @property
    def diagnostics(self) -> dict[str, object]:
        return deepcopy(self._diagnostics)


class PortfolioConstructionResult:
    """Canonical defensively isolated target weights and diagnostics."""

    __slots__ = ("_weights", "_diagnostics")

    def __init__(
        self, weights: pd.DataFrame, diagnostics: Mapping[str, object]
    ) -> None:
        if (
            not isinstance(weights, pd.DataFrame)
            or weights.empty
            or isinstance(weights.columns, pd.MultiIndex)
            or tuple(weights.columns) != WEIGHT_COLUMNS
        ):
            raise PortfolioConstructionValidationError(
                f"result weights must contain exactly {WEIGHT_COLUMNS!r}."
            )
        frame = weights.copy(deep=True)
        codes = frame["ts_code"]
        if (
            codes.isna().any()
            or not codes.map(lambda item: isinstance(item, (str, np.str_))).all()
            or codes.duplicated().any()
        ):
            raise PortfolioConstructionValidationError(
                "result ts_code values must be unique strings."
            )
        values = frame["target_weight"]
        if (
            pd.api.types.is_bool_dtype(values.dtype)
            or not pd.api.types.is_numeric_dtype(values.dtype)
        ):
            raise PortfolioConstructionValidationError(
                "result target_weight must be real numeric data."
            )
        numeric = values.to_numpy(dtype=np.float64, na_value=np.nan)
        if (
            not np.isfinite(numeric).all()
            or bool((numeric < 0.0).any())
            or not np.isclose(
                float(numeric.sum()),
                1.0,
                rtol=0.0,
                atol=WEIGHT_ABSOLUTE_TOLERANCE,
            )
        ):
            raise PortfolioConstructionValidationError(
                "result weights must be finite, nonnegative, and sum to one."
            )
        frame["target_weight"] = numeric
        self._weights = frame
        safe = json_safe(diagnostics, context="result diagnostics")
        assert isinstance(safe, dict)
        self._diagnostics = safe

    @property
    def weights(self) -> pd.DataFrame:
        return self._weights.copy(deep=True)

    @property
    def diagnostics(self) -> dict[str, object]:
        return deepcopy(self._diagnostics)


class HistoricalReturnWindow:
    """Validated sparse resolved return history with an explicit cutoff."""

    __slots__ = ("_risk_cutoff", "_returns")

    def __init__(self, risk_cutoff: object, returns: pd.DataFrame) -> None:
        try:
            self._risk_cutoff = normalize_date(risk_cutoff, field_name="risk_cutoff")
        except PortfolioConstructionValidationError as exc:
            raise PortfolioConstructionDataError(str(exc)) from exc
        if not isinstance(returns, pd.DataFrame):
            raise PortfolioConstructionDataError("returns must be a DataFrame.")
        if (
            isinstance(returns.columns, pd.MultiIndex)
            or tuple(returns.columns) != RETURN_COLUMNS
        ):
            raise PortfolioConstructionDataError(
                f"returns must contain exactly {RETURN_COLUMNS!r}."
            )
        frame = returns.copy(deep=True)
        dates: list[pd.Timestamp] = []
        for value in frame["trade_date"].tolist():
            try:
                dates.append(normalize_date(value, field_name="trade_date"))
            except PortfolioConstructionValidationError as exc:
                raise PortfolioConstructionDataError(str(exc)) from exc
        frame["trade_date"] = pd.Series(dates, dtype="datetime64[ns]")
        codes = frame["ts_code"]
        if (
            codes.isna().any()
            or not codes.map(
                lambda item: isinstance(item, (str, np.str_))
                and str(item).strip() == str(item)
                and bool(str(item))
            ).all()
        ):
            raise PortfolioConstructionDataError("return ts_code values are invalid.")
        frame["ts_code"] = codes.astype("string")
        values = frame["return"]
        if not frame.empty and (
            pd.api.types.is_bool_dtype(values.dtype)
            or not pd.api.types.is_numeric_dtype(values.dtype)
        ):
            raise PortfolioConstructionDataError("returns must be finite real data.")
        try:
            numeric = values.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise PortfolioConstructionDataError(
                "returns must be finite real data."
            ) from exc
        if not np.isfinite(numeric).all():
            raise PortfolioConstructionDataError("returns must be finite real data.")
        frame["return"] = numeric
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise PortfolioConstructionDataError("return keys must be unique.")
        if not frame.empty and bool((frame["trade_date"] > self._risk_cutoff).any()):
            raise PortfolioConstructionDataError(
                "return rows must not exceed risk_cutoff."
            )
        self._returns = frame.sort_values(
            ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
        )

    @property
    def risk_cutoff(self) -> pd.Timestamp:
        return self._risk_cutoff

    @property
    def returns(self) -> pd.DataFrame:
        return self._returns.copy(deep=True)


@runtime_checkable
class HistoricalReturnService(Protocol):
    """Provider-agnostic resolved historical-return service."""

    def load_window(
        self,
        ts_codes: Sequence[str],
        formation_date: pd.Timestamp,
        lookback_trading_days: int,
    ) -> HistoricalReturnWindow:
        """Return a bounded canonical sparse history for the exact request."""


@dataclass(frozen=True)
class PortfolioConstructionServices:
    """Stable dependency container for portfolio constructors."""

    historical_returns: HistoricalReturnService | None = None
