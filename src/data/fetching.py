"""Registry-driven provider fetch and completeness validation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from src.data.canonical_store import CanonicalDataError, normalize_frame
from src.data.contracts import CoverageKind, DatasetSpec, FetchStrategy
from src.data.coverage_planner import FetchTask


class ProviderFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchResult:
    task: FetchTask
    frame: pd.DataFrame


class FetchStrategyRegistry:
    def __init__(self, completeness: "CompletenessStrategyRegistry | None" = None) -> None:
        self._strategies: dict[FetchStrategy, Callable[[DatasetSpec, FetchTask, object], pd.DataFrame]] = {}
        self.completeness = completeness or create_default_completeness_registry()

    def register(self, strategy: FetchStrategy, handler: Callable[[DatasetSpec, FetchTask, object], pd.DataFrame]) -> None:
        if strategy in self._strategies:
            raise ValueError(f"Fetch strategy {strategy.value!r} is already registered.")
        self._strategies[strategy] = handler

    def fetch(self, spec: DatasetSpec, task: FetchTask, client: object) -> FetchResult:
        try:
            handler = self._strategies[spec.fetch_strategy]
        except KeyError as exc:
            raise KeyError(f"No handler for {spec.fetch_strategy.value!r}.") from exc
        try:
            frame = handler(spec, task, client)
        except Exception as exc:
            if isinstance(exc, (ProviderFetchError, CanonicalDataError)):
                raise
            raise ProviderFetchError(f"Provider call failed for dataset {spec.dataset_id!r}.") from exc
        return FetchResult(task, self.completeness.validate(spec, task, frame))


class CompletenessStrategyRegistry:
    def __init__(self) -> None:
        self._validators: dict[str, Callable[[DatasetSpec, FetchTask, pd.DataFrame], pd.DataFrame]] = {}

    def register(self, name: str, validator: Callable[[DatasetSpec, FetchTask, pd.DataFrame], pd.DataFrame]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("completeness strategy name must not be empty.")
        if name in self._validators:
            raise ValueError(f"Completeness strategy {name!r} is already registered.")
        self._validators[name] = validator

    def validate(self, spec: DatasetSpec, task: FetchTask, frame: pd.DataFrame) -> pd.DataFrame:
        try:
            validator = self._validators[spec.completeness_strategy]
        except KeyError as exc:
            raise KeyError(f"No completeness validator for {spec.completeness_strategy!r}.") from exc
        normalized = _base_complete(spec, frame)
        return validator(spec, task, normalized)


def _method(client: object, spec: DatasetSpec) -> Callable[..., pd.DataFrame]:
    value = getattr(client, f"get_{spec.endpoint}", None)
    if not callable(value):
        raise TypeError(f"Provider client has no get_{spec.endpoint} method.")
    return value


def _market_by_date(spec: DatasetSpec, task: FetchTask, client: object) -> pd.DataFrame:
    return _method(client, spec)(ts_code=None, trade_date=task.start.replace("-", ""), start_date=None, end_date=None)


def _entity_range(spec: DatasetSpec, task: FetchTask, client: object) -> pd.DataFrame:
    scope = dict(task.scope)
    if spec.coverage_kind is CoverageKind.CALENDAR_DATE:
        return _method(client, spec)(start_date=task.start.replace("-", ""), end_date=task.end.replace("-", ""))
    code = scope.get("index_code") or scope.get("ts_code")
    if not code:
        raise ValueError("entity date-range fetch requires index_code or ts_code scope.")
    return _method(client, spec)(ts_code=code, trade_date=None, start_date=task.start.replace("-", ""), end_date=task.end.replace("-", ""))


def _entity_month(spec: DatasetSpec, task: FetchTask, client: object) -> pd.DataFrame:
    code = dict(task.scope).get("index_code")
    if not code:
        raise ValueError("entity month fetch requires index_code scope.")
    year, month = (int(item) for item in task.start.split("-"))
    start = date(year, month, 1)
    next_month = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    end = next_month - timedelta(days=1)
    return _method(client, spec)(index_code=code, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))


def _reference(spec: DatasetSpec, task: FetchTask, client: object) -> pd.DataFrame:
    frames = [_method(client, spec)(list_status=status) for status in ("L", "D", "P")]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=spec.required_fields)


def create_default_fetch_strategy_registry() -> FetchStrategyRegistry:
    registry = FetchStrategyRegistry()
    registry.register(FetchStrategy.MARKET_SNAPSHOT_BY_DATE, _market_by_date)
    registry.register(FetchStrategy.ENTITY_DATE_RANGE, _entity_range)
    registry.register(FetchStrategy.ENTITY_MONTH_SNAPSHOT, _entity_month)
    registry.register(FetchStrategy.REFERENCE_SNAPSHOT, _reference)
    return registry


def _base_complete(spec: DatasetSpec, frame: pd.DataFrame) -> pd.DataFrame:
    if (
        isinstance(frame, pd.DataFrame)
        and frame.empty
        and not len(frame.columns)
        and spec.allow_empty_complete
    ):
        # Event endpoints legitimately prove a zero-row snapshot. TuShare may
        # represent that response with no columns, so materialize the canonical
        # empty schema before the ordinary strict normalizer runs.
        frame = pd.DataFrame(columns=spec.required_fields)
    normalized = normalize_frame(spec, frame)
    if normalized.empty:
        if spec.allow_empty_complete:
            return normalized
        raise CanonicalDataError(f"{spec.dataset_id} returned no rows; completeness is unproven.")
    if spec.provider_row_limit is not None and len(normalized) >= spec.provider_row_limit:
        raise CanonicalDataError(f"{spec.dataset_id} may be truncated at the provider row limit.")
    return normalized


def _calendar_complete(spec: DatasetSpec, task: FetchTask, normalized: pd.DataFrame) -> pd.DataFrame:
    scope = dict(task.scope)
    observed = set(normalized["cal_date"])
    if observed != set(task.units):
        raise CanonicalDataError("calendar result does not exactly match requested units.")
    if "exchange" in scope and not set(normalized["exchange"]).issubset({scope["exchange"], ""}):
        raise CanonicalDataError("calendar result is outside requested exchange scope.")
    return normalized


def _market_complete(spec: DatasetSpec, task: FetchTask, normalized: pd.DataFrame) -> pd.DataFrame:
    if not normalized.empty and set(normalized["trade_date"]) != set(task.units):
        raise CanonicalDataError("market snapshot does not match requested trade date.")
    return normalized


def _entity_complete(spec: DatasetSpec, task: FetchTask, normalized: pd.DataFrame) -> pd.DataFrame:
    scope = dict(task.scope)
    code = scope.get("index_code") or scope.get("ts_code")
    if set(normalized["ts_code"]) != {code} or set(normalized["trade_date"]) != set(task.units):
        raise CanonicalDataError("entity series result does not exactly match requested scope/units.")
    return normalized


def _entity_month_complete(spec: DatasetSpec, task: FetchTask, normalized: pd.DataFrame) -> pd.DataFrame:
    code = dict(task.scope).get("index_code")
    if set(normalized["index_code"]) != {code} or set(normalized["trade_date"].str[:7]) != set(task.units):
        raise CanonicalDataError("entity month result does not match requested scope/month.")
    return normalized


def _reference_complete(spec: DatasetSpec, task: FetchTask, normalized: pd.DataFrame) -> pd.DataFrame:
    return normalized


def create_default_completeness_registry() -> CompletenessStrategyRegistry:
    registry = CompletenessStrategyRegistry()
    registry.register("calendar_days", _calendar_complete)
    registry.register("reference_snapshot", _reference_complete)
    registry.register("market_date_snapshot", _market_complete)
    registry.register("event_date_snapshot", _market_complete)
    registry.register("entity_dates", _entity_complete)
    registry.register("entity_month_snapshot", _entity_month_complete)
    return registry


def validate_complete(spec: DatasetSpec, task: FetchTask, frame: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible convenience entrypoint using fresh validators."""
    return create_default_completeness_registry().validate(spec, task, frame)
