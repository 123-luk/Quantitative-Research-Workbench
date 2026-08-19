"""Frozen Data Layer 2.0 contracts for canonical L0/L1 market data."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime
from enum import Enum
from collections.abc import Mapping
from typing import Iterable


class ResearchFrequency(str, Enum):
    DAILY = "DAILY"
    MONTHLY = "MONTHLY"


class NativeFrequency(str, Enum):
    CALENDAR_DAY = "CALENDAR_DAY"
    TRADING_DAY = "TRADING_DAY"
    MONTHLY_SNAPSHOT = "MONTHLY_SNAPSHOT"
    REFERENCE_SNAPSHOT = "REFERENCE_SNAPSHOT"


class ScopeKind(str, Enum):
    MARKET_SNAPSHOT = "MARKET_SNAPSHOT"
    ENTITY_SERIES = "ENTITY_SERIES"
    ENTITY_MONTH_SNAPSHOT = "ENTITY_MONTH_SNAPSHOT"
    REFERENCE_SNAPSHOT = "REFERENCE_SNAPSHOT"


class CoverageKind(str, Enum):
    GLOBAL_SNAPSHOT = "GLOBAL_SNAPSHOT"
    CALENDAR_DATE = "CALENDAR_DATE"
    TRADE_DATE = "TRADE_DATE"
    ENTITY_TRADE_DATE = "ENTITY_TRADE_DATE"
    ENTITY_MONTH = "ENTITY_MONTH"
    REFERENCE_EFFECTIVE_THROUGH = "REFERENCE_EFFECTIVE_THROUGH"


GLOBAL_SNAPSHOT_UNIT = "GLOBAL"


class FetchStrategy(str, Enum):
    MARKET_SNAPSHOT_BY_DATE = "MARKET_SNAPSHOT_BY_DATE"
    ENTITY_DATE_RANGE = "ENTITY_DATE_RANGE"
    ENTITY_MONTH_SNAPSHOT = "ENTITY_MONTH_SNAPSHOT"
    REFERENCE_SNAPSHOT = "REFERENCE_SNAPSHOT"


class RevisionPolicy(str, Enum):
    MISSING_ONLY = "MISSING_ONLY"
    EXPLICIT_REFRESH = "EXPLICIT_REFRESH"


class IdentifierContract(str, Enum):
    NONE = "NONE"
    CANONICAL_TRADABLE = "CANONICAL_TRADABLE"
    PROVIDER_REFERENCE = "PROVIDER_REFERENCE"


def canonical_date(value: object) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("date must be YYYY-MM-DD or YYYYMMDD.")
    fmt = "%Y%m%d" if len(value) == 8 and value.isdigit() else "%Y-%m-%d"
    try:
        return datetime.strptime(value, fmt).date().isoformat()
    except ValueError as exc:
        raise ValueError("date must be YYYY-MM-DD or YYYYMMDD.") from exc


def normalize_scope(scope: object) -> tuple[tuple[str, str], ...]:
    if scope is None:
        return ()
    if isinstance(scope, str):
        text = scope.strip()
        if not text:
            raise ValueError("scope must not be empty.")
        return (("scope", text),)
    if isinstance(scope, Mapping):
        items = tuple(scope.items())
    elif isinstance(scope, (tuple, list)):
        try:
            items = tuple((item[0], item[1]) for item in scope)
        except (TypeError, IndexError) as exc:
            raise TypeError("scope pairs are invalid.") from exc
        if len({item[0] for item in items}) != len(items):
            raise ValueError("scope keys must be unique.")
    else:
        raise TypeError("scope must be a mapping, string, pair sequence, or None.")
    result: list[tuple[str, str]] = []
    for key, value in items:
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            raise ValueError("scope keys must be non-empty trimmed strings.")
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError("scope values must be non-empty trimmed strings.")
        result.append((key, value))
    return tuple(sorted(result))


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    provider: str
    endpoint: str
    native_frequency: NativeFrequency
    scope_kind: ScopeKind
    primary_key: tuple[str, ...]
    required_fields: tuple[str, ...]
    coverage_kind: CoverageKind
    storage_partition: tuple[str, ...]
    fetch_strategy: FetchStrategy
    completeness_strategy: str
    schema_version: str
    revision_policy: RevisionPolicy
    availability_semantics: str
    allow_empty_complete: bool = False
    provider_row_limit: int | None = None
    identifier_contract: IdentifierContract = IdentifierContract.NONE

    def __post_init__(self) -> None:
        enum_fields = {
            "native_frequency": NativeFrequency,
            "scope_kind": ScopeKind,
            "coverage_kind": CoverageKind,
            "fetch_strategy": FetchStrategy,
            "revision_policy": RevisionPolicy,
            "identifier_contract": IdentifierContract,
        }
        for name, enum_type in enum_fields.items():
            if not isinstance(getattr(self, name), enum_type):
                raise TypeError(f"{name} must be a {enum_type.__name__}.")
        for name in ("dataset_id", "provider", "endpoint", "completeness_strategy", "schema_version", "availability_semantics"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a non-empty trimmed string.")
        for name in ("primary_key", "required_fields", "storage_partition"):
            values = tuple(getattr(self, name))
            if not values or len(values) != len(set(values)) or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{name} must contain unique non-empty names.")
            object.__setattr__(self, name, values)
        if not set(self.primary_key).issubset(self.required_fields):
            raise ValueError("primary_key must be included in required_fields.")
        if self.provider_row_limit is not None and (type(self.provider_row_limit) is not int or self.provider_row_limit < 1):
            raise ValueError("provider_row_limit must be positive.")


@dataclass(frozen=True)
class DataRequirement:
    dataset_id: str
    scope: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    required_start: str = ""
    required_end: str = ""
    required_fields: tuple[str, ...] = field(default_factory=tuple)
    reason: str = "unspecified"
    as_of_cutoff: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip() or self.dataset_id != self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty trimmed string.")
        start = canonical_date(self.required_start)
        end = canonical_date(self.required_end)
        if start > end:
            raise ValueError("required_start must not be after required_end.")
        fields = tuple(sorted(set(self.required_fields)))
        if any(not isinstance(item, str) or not item.strip() or item != item.strip() for item in fields):
            raise ValueError("required_fields must contain trimmed names.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must not be empty.")
        object.__setattr__(self, "scope", normalize_scope(self.scope))
        object.__setattr__(self, "required_start", start)
        object.__setattr__(self, "required_end", end)
        object.__setattr__(self, "required_fields", fields)
        if self.as_of_cutoff is not None:
            object.__setattr__(self, "as_of_cutoff", canonical_date(self.as_of_cutoff))

    @classmethod
    def create(cls, dataset_id: str, *, scope: object = None, required_start: object, required_end: object, required_fields: Iterable[str] = (), reason: str = "unspecified", as_of_cutoff: object | None = None) -> "DataRequirement":
        return cls(dataset_id, normalize_scope(scope), canonical_date(required_start), canonical_date(required_end), tuple(required_fields), reason, None if as_of_cutoff is None else canonical_date(as_of_cutoff))


def coalesce_requirements(requirements: Iterable[DataRequirement]) -> tuple[DataRequirement, ...]:
    grouped: dict[tuple[str, tuple[tuple[str, str], ...], tuple[str, ...], str | None], list[DataRequirement]] = {}
    for requirement in requirements:
        if not isinstance(requirement, DataRequirement):
            raise TypeError("requirements must contain DataRequirement values.")
        key = (requirement.dataset_id, requirement.scope, requirement.required_fields, requirement.as_of_cutoff)
        grouped.setdefault(key, []).append(requirement)
    result: list[DataRequirement] = []
    for key in sorted(grouped):
        merged: list[DataRequirement] = []
        for requirement in sorted(grouped[key], key=lambda item: (item.required_start, item.required_end, item.reason)):
            if not merged or requirement.required_start > merged[-1].required_end:
                merged.append(requirement)
                continue
            current = merged[-1]
            merged[-1] = replace(current, required_end=max(current.required_end, requirement.required_end), reason="; ".join(sorted(set(current.reason.split("; ") + requirement.reason.split("; ")))))
        result.extend(merged)
    return tuple(result)


def formation_dates(frequency: ResearchFrequency, open_dates: Iterable[object]) -> tuple[str, ...]:
    dates = tuple(sorted(set(canonical_date(item) for item in open_dates)))
    if frequency is ResearchFrequency.DAILY:
        return dates
    by_month: dict[str, str] = {}
    for item in dates:
        by_month[item[:7]] = item
    return tuple(by_month[key] for key in sorted(by_month))
