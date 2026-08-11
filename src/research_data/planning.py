"""Deterministic ResearchInputPlan and forward-label availability contracts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Iterable

import pandas as pd

from src.data.contracts import DataRequirement, ResearchFrequency, canonical_date
from src.factors.forward_returns import ForwardReturnConfig
from src.factors.frequency import FactorDependencyPlanner, FactorFrequencySpec
from src.research_data.adjusted_prices import AdjustedPriceService
from src.research_data.calendar import ResearchCalendar
from src.universe import UniverseService, UniverseSpec


class ResearchInputError(ValueError):
    pass


class ResearchInputDataUnavailable(ResearchInputError):
    pass


@dataclass(frozen=True)
class ForwardReturnSpec:
    horizon: int
    horizon_type: str = "TRADING_PERIODS"
    entry_lag_periods: int = 1
    entry_semantics: str = "close on the entry open date"
    exit_semantics: str = "close on the exit open date"
    return_source: str = "ADJUSTED_CLOSE"
    label_availability_rule: str = "available_at = exit_trade_date close"
    price_column: str = "close"
    return_column: str = "forward_return"
    require_positive_prices: bool = True

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or self.horizon < 1:
            raise ResearchInputError("forward horizon must be a strict positive integer.")
        if type(self.entry_lag_periods) is not int or self.entry_lag_periods < 0:
            raise ResearchInputError("entry_lag_periods must be a strict non-negative integer.")
        if self.horizon_type != "TRADING_PERIODS":
            raise ResearchInputError("horizon_type must be TRADING_PERIODS.")
        if self.return_source != "ADJUSTED_CLOSE":
            raise ResearchInputError("P4C3 return_source must be ADJUSTED_CLOSE.")
        if type(self.require_positive_prices) is not bool:
            raise ResearchInputError("require_positive_prices must be a bool.")
        for name in ("entry_semantics", "exit_semantics", "label_availability_rule", "price_column", "return_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ResearchInputError(f"{name} must be a non-empty trimmed string.")

    @classmethod
    def from_config(cls, config: ForwardReturnConfig) -> "ForwardReturnSpec":
        if not isinstance(config, ForwardReturnConfig):
            raise TypeError("config must be a ForwardReturnConfig.")
        return cls(config.holding_periods, entry_lag_periods=config.entry_lag_periods, price_column=config.price_col, return_column=config.return_col, require_positive_prices=config.require_positive_prices)

    def to_config(self) -> ForwardReturnConfig:
        return ForwardReturnConfig(price_col=self.price_column, return_col=self.return_column, entry_lag_periods=self.entry_lag_periods, holding_periods=self.horizon, require_positive_prices=self.require_positive_prices)

    def to_dict(self) -> dict[str, object]:
        return {
            "horizon": self.horizon,
            "horizon_type": self.horizon_type,
            "entry_lag_periods": self.entry_lag_periods,
            "entry_semantics": self.entry_semantics,
            "exit_semantics": self.exit_semantics,
            "return_source": self.return_source,
            "label_availability_rule": self.label_availability_rule,
            "price_column": self.price_column,
            "return_column": self.return_column,
            "require_positive_prices": self.require_positive_prices,
        }


def _requirement_dict(value: DataRequirement) -> dict[str, object]:
    return {
        "dataset_id": value.dataset_id,
        "scope": dict(value.scope),
        "required_start": value.required_start,
        "required_end": value.required_end,
        "required_fields": list(value.required_fields),
        "reason": value.reason,
        "as_of_cutoff": value.as_of_cutoff,
    }


def compose_requirements(requirements: Iterable[DataRequirement]) -> tuple[DataRequirement, ...]:
    """Union fields/ranges for requirements with the same semantic cutoff."""
    grouped: dict[tuple[str, tuple[tuple[str, str], ...], str | None], list[DataRequirement]] = {}
    for requirement in requirements:
        if not isinstance(requirement, DataRequirement):
            raise TypeError("requirements must contain DataRequirement values.")
        grouped.setdefault((requirement.dataset_id, requirement.scope, requirement.as_of_cutoff), []).append(requirement)
    result: list[DataRequirement] = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2] or "")):
        values = grouped[key]
        result.append(
            DataRequirement.create(
                key[0],
                scope=dict(key[1]),
                required_start=min(item.required_start for item in values),
                required_end=max(item.required_end for item in values),
                required_fields=tuple(sorted({field for item in values for field in item.required_fields})),
                reason="; ".join(sorted({reason for item in values for reason in item.reason.split("; ")})),
                as_of_cutoff=key[2],
            )
        )
    return tuple(result)


_TARGETS = (
    "factor_input.parquet",
    "price_panel.parquet",
    "score_panel.parquet",
    "modeling_factor_panel.parquet",
    "modeling_forward_returns.parquet",
    "labels_with_availability.parquet",
)


@dataclass(frozen=True)
class ResearchInputPlan:
    research_frequency: ResearchFrequency
    start_date: str
    end_date: str
    formation_dates: tuple[str, ...]
    universe_spec: UniverseSpec
    factor_ids: tuple[str, ...]
    factor_frequency_specs: tuple[tuple[str, FactorFrequencySpec], ...]
    forward_return_spec: ForwardReturnSpec
    requirements: tuple[DataRequirement, ...]
    materialization_targets: tuple[str, ...] = _TARGETS

    def __post_init__(self) -> None:
        if not isinstance(self.research_frequency, ResearchFrequency):
            raise TypeError("research_frequency must be a ResearchFrequency.")
        start, end = canonical_date(self.start_date), canonical_date(self.end_date)
        formations = tuple(canonical_date(item) for item in self.formation_dates)
        factors = tuple(self.factor_ids)
        specs = tuple(self.factor_frequency_specs)
        requirements = tuple(self.requirements)
        if start > end or not formations or formations != tuple(sorted(set(formations))) or formations[0] < start or formations[-1] > end:
            raise ResearchInputError("formation_dates must be ordered, unique, and inside the research interval.")
        if not isinstance(self.universe_spec, UniverseSpec):
            raise TypeError("universe_spec must be a UniverseSpec.")
        if not factors or len(factors) != len(set(factors)) or any(not isinstance(item, str) or not item.strip() for item in factors):
            raise ResearchInputError("factor_ids must contain unique non-empty IDs.")
        if tuple(name for name, _ in specs) != factors or any(not isinstance(spec, FactorFrequencySpec) or spec.research_frequency is not self.research_frequency for _, spec in specs):
            raise ResearchInputError("factor_frequency_specs must exactly match factor_ids and frequency.")
        if not isinstance(self.forward_return_spec, ForwardReturnSpec):
            raise TypeError("forward_return_spec must be a ForwardReturnSpec.")
        if any(not isinstance(item, DataRequirement) for item in requirements):
            raise TypeError("requirements must contain DataRequirement values.")
        if tuple(self.materialization_targets) != _TARGETS:
            raise ResearchInputError("materialization_targets must use the frozen P4C3 filenames.")
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)
        object.__setattr__(self, "formation_dates", formations)
        object.__setattr__(self, "factor_ids", factors)
        object.__setattr__(self, "factor_frequency_specs", specs)
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "materialization_targets", _TARGETS)

    def to_dict(self) -> dict[str, object]:
        return {
            "research_frequency": self.research_frequency.value,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "formation_dates": list(self.formation_dates),
            "universe_spec": self.universe_spec.to_dict(),
            "factor_ids": list(self.factor_ids),
            "factor_frequency_specs": {name: spec.to_dict() for name, spec in self.factor_frequency_specs},
            "forward_return_spec": self.forward_return_spec.to_dict(),
            "requirements": [_requirement_dict(item) for item in self.requirements],
            "materialization_targets": list(self.materialization_targets),
        }

    @property
    def plan_id(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


class ResearchInputPlanner:
    def __init__(self, *, calendar: ResearchCalendar, universe_service: UniverseService, factor_registry: object, calendar_scope: object = (("exchange", "SSE"),), market_scope: object = "CN_A") -> None:
        self.calendar = calendar
        self.universe_service = universe_service
        self.factor_registry = factor_registry
        self.calendar_scope = calendar_scope
        self.market_scope = market_scope

    def build(self, *, research_frequency: ResearchFrequency, start_date: object, end_date: object, universe_spec: UniverseSpec, factor_ids: Iterable[str], forward_return_spec: ForwardReturnSpec) -> ResearchInputPlan:
        start, end = canonical_date(start_date), canonical_date(end_date)
        formations = self.calendar.formation_dates(research_frequency, start, end)
        if not formations:
            raise ResearchInputError("research interval contains no formation date.")
        factors = tuple(factor_ids)
        specs: list[tuple[str, FactorFrequencySpec]] = []
        earliest = formations[0]
        needs_adjusted_features = False
        for name in factors:
            factor = self.factor_registry.get(name)
            spec = factor.metadata.frequency_spec(research_frequency)
            specs.append((name, spec))
            earliest = min(earliest, self.calendar.resolve_history(formations[0], spec.history_requirement).start_date)
            needs_adjusted_features = needs_adjusted_features or any(field.startswith("adj_") for field in factor.metadata.source_fields)
        exit_date = self.calendar.shift_open_date(formations[-1], forward_return_spec.entry_lag_periods + forward_return_spec.horizon)
        requirements: list[DataRequirement] = list(self.universe_service.requirements(universe_spec, start=start, end=end, frequency=research_frequency))
        requirements.extend(FactorDependencyPlanner(self.factor_registry, self.calendar).requirements(factors, frequency=research_frequency, start_date=start, end_date=end, scope=self.market_scope))
        if needs_adjusted_features:
            requirements.extend(AdjustedPriceService.requirements(start_date=earliest, end_date=formations[-1], price_fields=("close",), scope=self.market_scope))
        requirements.extend(AdjustedPriceService.requirements(start_date=formations[0], end_date=exit_date, price_fields=("close",), scope=self.market_scope))
        requirements.append(DataRequirement.create("trade_cal", scope=self.calendar_scope, required_start=earliest, required_end=exit_date, required_fields=("cal_date", "is_open"), reason="research formations, factor warmup, and forward-label horizon", as_of_cutoff=exit_date))
        return ResearchInputPlan(research_frequency, start, end, formations, universe_spec, factors, tuple(specs), forward_return_spec, compose_requirements(requirements))


class TrainingLabelAvailabilityGuard:
    """Exclude labels whose exit-close realization is after a training cutoff."""

    @staticmethod
    def available(labels: pd.DataFrame, cutoff: object) -> pd.DataFrame:
        if not isinstance(labels, pd.DataFrame):
            raise TypeError("labels must be a pandas DataFrame.")
        required = {"trade_date", "ts_code", "available_at"}
        missing = sorted(required - set(labels.columns))
        if missing:
            raise ResearchInputError(f"labels are missing availability fields: {missing!r}.")
        result = labels.copy(deep=True)
        available = pd.to_datetime(result["available_at"], errors="coerce")
        invalid = result["available_at"].notna() & available.isna()
        if invalid.any():
            raise ResearchInputError("available_at contains invalid dates.")
        cutoff_date = pd.Timestamp(canonical_date(cutoff))
        return result.loc[available.notna() & available.le(cutoff_date)].sort_values(["trade_date", "ts_code"], kind="mergesort", ignore_index=True)
