"""Exact coverage-unit expansion and missing-only task planning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
from typing import Callable, Iterable

from src.data.contracts import GLOBAL_SNAPSHOT_UNIT, CoverageKind, DataRequirement, DatasetSpec, canonical_date, coalesce_requirements
from src.data.coverage_ledger import CoverageLedger
from src.data.dataset_registry import DatasetRegistry


def scope_key(scope: tuple[tuple[str, str], ...]) -> str:
    return json.dumps(dict(scope), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def coalesce_coverage_requirements(
    registry: DatasetRegistry, requirements: Iterable[DataRequirement]
) -> tuple[DataRequirement, ...]:
    """Coalesce global snapshots across dates, fields, reasons, and cutoffs."""
    ordinary: list[DataRequirement] = []
    snapshots: dict[tuple[str, tuple[tuple[str, str], ...]], list[DataRequirement]] = {}
    for requirement in coalesce_requirements(requirements):
        if registry.get(requirement.dataset_id).coverage_kind is CoverageKind.GLOBAL_SNAPSHOT:
            snapshots.setdefault((requirement.dataset_id, requirement.scope), []).append(requirement)
        else:
            ordinary.append(requirement)
    for values in snapshots.values():
        cutoffs = tuple(item.as_of_cutoff for item in values if item.as_of_cutoff is not None)
        ordinary.append(DataRequirement.create(
            values[0].dataset_id,
            scope=values[0].scope,
            required_start=min(item.required_start for item in values),
            required_end=max(item.required_end for item in values),
            required_fields=tuple(sorted({field for item in values for field in item.required_fields})),
            reason="; ".join(sorted({reason for item in values for reason in item.reason.split("; ")})),
            as_of_cutoff=max(cutoffs) if cutoffs else None,
        ))
    return tuple(sorted(ordinary, key=lambda item: (
        item.dataset_id, item.scope, item.required_start, item.required_end,
        item.required_fields, item.as_of_cutoff or "",
    )))


@dataclass(frozen=True)
class FetchTask:
    dataset_id: str
    scope: tuple[tuple[str, str], ...]
    units: tuple[str, ...]
    start: str
    end: str


@dataclass(frozen=True)
class MissingDataPlan:
    requirement: DataRequirement
    required_units: tuple[str, ...]
    complete_units: tuple[str, ...]
    missing_units: tuple[str, ...]
    grouped_fetch_tasks: tuple[FetchTask, ...]

    @property
    def ready(self) -> bool:
        return not self.missing_units


class MissingDataPlanner:
    def __init__(self, registry: DatasetRegistry, ledger: CoverageLedger, *, open_dates: Callable[[str, str], Iterable[object]] | None = None, unit_verifier: Callable[[str, tuple[tuple[str, str], ...], str, tuple[str, ...]], bool] | None = None, complete_units_resolver: Callable[[str, tuple[tuple[str, str], ...], tuple[str, ...], tuple[str, ...]], frozenset[str]] | None = None) -> None:
        self.registry = registry
        self.ledger = ledger
        self.open_dates = open_dates
        self.unit_verifier = unit_verifier
        self.complete_units_resolver = complete_units_resolver

    def _units(self, requirement: DataRequirement, spec: DatasetSpec) -> tuple[str, ...]:
        if spec.coverage_kind is CoverageKind.GLOBAL_SNAPSHOT:
            return (GLOBAL_SNAPSHOT_UNIT,)
        start = date.fromisoformat(requirement.required_start)
        end = date.fromisoformat(requirement.required_end)
        if spec.coverage_kind is CoverageKind.CALENDAR_DATE:
            count = (end - start).days + 1
            return tuple((start + timedelta(days=index)).isoformat() for index in range(count))
        if spec.coverage_kind in {CoverageKind.TRADE_DATE, CoverageKind.ENTITY_TRADE_DATE}:
            if self.open_dates is None:
                raise ValueError(f"Dataset {spec.dataset_id!r} requires an injected trade-calendar resolver.")
            values = tuple(sorted(set(canonical_date(item) for item in self.open_dates(requirement.required_start, requirement.required_end))))
            if any(item < requirement.required_start or item > requirement.required_end for item in values):
                raise ValueError("trade-calendar resolver returned an out-of-range date.")
            return values
        if spec.coverage_kind is CoverageKind.ENTITY_MONTH:
            cursor = start.replace(day=1)
            months: list[str] = []
            while cursor <= end:
                months.append(cursor.strftime("%Y-%m"))
                cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
            return tuple(months)
        return (requirement.required_end,)

    @staticmethod
    def _tasks(requirement: DataRequirement, spec: DatasetSpec, missing: tuple[str, ...], required: tuple[str, ...]) -> tuple[FetchTask, ...]:
        if not missing:
            return ()
        if spec.fetch_strategy.value in {"MARKET_SNAPSHOT_BY_DATE", "ENTITY_MONTH_SNAPSHOT"}:
            groups = tuple((item,) for item in missing)
        elif spec.fetch_strategy.value == "REFERENCE_SNAPSHOT":
            groups = (missing,)
        else:
            positions = {unit: index for index, unit in enumerate(required)}
            pending: list[list[str]] = []
            for item in missing:
                if not pending or positions[item] != positions[pending[-1][-1]] + 1:
                    pending.append([item])
                else:
                    pending[-1].append(item)
            groups = tuple(tuple(group) for group in pending)
        return tuple(FetchTask(requirement.dataset_id, requirement.scope, group, group[0], group[-1]) for group in groups)

    def plan(self, requirements: Iterable[DataRequirement]) -> tuple[MissingDataPlan, ...]:
        plans: list[MissingDataPlan] = []
        for requirement in coalesce_coverage_requirements(self.registry, requirements):
            spec = self.registry.get(requirement.dataset_id)
            if requirement.required_fields and not set(requirement.required_fields).issubset(spec.required_fields):
                raise ValueError(f"Requirement fields are outside dataset {spec.dataset_id!r} schema.")
            required = self._units(requirement, spec)
            if self.complete_units_resolver is not None:
                complete_set = self.complete_units_resolver(
                    spec.dataset_id, requirement.scope, required, requirement.required_fields
                )
            else:
                complete_set = self.ledger.complete_units(spec.dataset_id, scope_key(requirement.scope), required)
            if self.unit_verifier is not None and self.complete_units_resolver is None:
                complete_set = frozenset(
                    unit for unit in complete_set
                    if self.unit_verifier(spec.dataset_id, requirement.scope, unit, requirement.required_fields)
                )
            complete = tuple(item for item in required if item in complete_set)
            missing = tuple(item for item in required if item not in complete_set)
            plans.append(MissingDataPlan(requirement, required, complete, missing, self._tasks(requirement, spec, missing, required)))
        return tuple(plans)
