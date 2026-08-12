"""Generic missing-only L0/L1 data preparation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Callable, Iterable

from src.data.canonical_store import PartitionedParquetStore, RawParquetStore, content_hash
from src.data.contracts import DataRequirement, coalesce_requirements
from src.data.coverage_ledger import CoverageLedger, CoverageRecord
from src.data.coverage_planner import MissingDataPlan, MissingDataPlanner, scope_key
from src.data.dataset_registry import DatasetRegistry, create_default_dataset_registry
from src.data.fetching import FetchStrategyRegistry, create_default_fetch_strategy_registry
from src.data.tushare_client import TushareClient


class MissingCredentialError(RuntimeError):
    pass


class DataUnavailableError(RuntimeError):
    """Safe structured preparation failure; provider text is never exposed."""

    def __init__(
        self,
        message: str,
        *,
        dataset_id: str | None = None,
        units: tuple[str, ...] = (),
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.dataset_id = dataset_id
        self.units = tuple(units)
        self.safe_cause = cause


@dataclass(frozen=True)
class DataPreparationResult:
    status: str
    plans: tuple[MissingDataPlan, ...]
    provider_calls: int
    rows: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class CuratedTradingCalendarResolver:
    """Resolve open dates only from ledger-proven canonical trade_cal units."""

    def __init__(self, registry: DatasetRegistry, ledger: CoverageLedger, store: PartitionedParquetStore, *, scope: object = "SSE") -> None:
        from src.data.contracts import normalize_scope
        self.registry = registry
        self.ledger = ledger
        self.store = store
        self.scope = normalize_scope(scope)

    def __call__(self, start: str, end: str) -> tuple[str, ...]:
        requirement = DataRequirement.create("trade_cal", scope=dict(self.scope), required_start=start, required_end=end, reason="trading calendar truth")
        plan = MissingDataPlanner(self.registry, self.ledger).plan((requirement,))[0]
        if not plan.ready:
            raise DataUnavailableError("Canonical trade calendar coverage is incomplete for the requested interval.")
        spec = self.registry.get("trade_cal")
        result: list[str] = []
        for unit in plan.required_units:
            rows = self.store.rows_for_unit(spec, unit=unit, scope=self.scope)
            if len(rows) != 1:
                raise DataUnavailableError("Canonical trade calendar unit is missing or ambiguous.")
            if int(rows.iloc[0]["is_open"]) == 1:
                result.append(unit)
        return tuple(result)


class DataPreparationService:
    def __init__(self, *, registry: DatasetRegistry | None = None, ledger: CoverageLedger, curated_store: PartitionedParquetStore, raw_store: RawParquetStore, fetch_registry: FetchStrategyRegistry | None = None, open_dates: Callable[[str, str], Iterable[object]] | None = None, client_factory: Callable[[str], object] | None = None) -> None:
        self.registry = registry or create_default_dataset_registry()
        self.ledger = ledger
        self.curated_store = curated_store
        self.raw_store = raw_store
        self.fetch_registry = fetch_registry or create_default_fetch_strategy_registry()
        self.open_dates = open_dates
        self.client_factory = client_factory or (lambda token: TushareClient(token))

    @staticmethod
    def _fingerprint(dataset_id: str, task_scope: tuple[tuple[str, str], ...], units: tuple[str, ...]) -> str:
        payload = json.dumps({"dataset_id": dataset_id, "scope": task_scope, "units": units}, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def _planner(self) -> MissingDataPlanner:
        return MissingDataPlanner(self.registry, self.ledger, open_dates=self.open_dates)

    def inspect(self, requirements: Iterable[DataRequirement]) -> tuple[MissingDataPlan, ...]:
        """Plan exact missing coverage without provider calls or data writes."""
        normalized = tuple(
            sorted(
                coalesce_requirements(requirements),
                key=lambda item: (
                    self.registry.get(item.dataset_id).coverage_kind.value != "CALENDAR_DATE",
                    item.dataset_id,
                    item.scope,
                    item.required_start,
                ),
            )
        )
        return self._planner().plan(normalized)

    def ensure(self, requirements: Iterable[DataRequirement], credential: str | None = None, *, client: object | None = None) -> DataPreparationResult:
        normalized = tuple(sorted(coalesce_requirements(requirements), key=lambda item: (self.registry.get(item.dataset_id).coverage_kind.value != "CALENDAR_DATE", item.dataset_id, item.scope, item.required_start)))
        calls = 0
        rows_total = 0
        for requirement in normalized:
            plan = self._planner().plan((requirement,))[0]
            if plan.ready:
                continue
            if client is None:
                if credential is None or not isinstance(credential, str) or not credential.strip():
                    raise MissingCredentialError("Required canonical data is missing and no TuShare credential was provided.")
                try:
                    client = self.client_factory(credential.strip())
                except Exception:
                    raise DataUnavailableError("TuShare client initialization failed.") from None
            spec = self.registry.get(plan.requirement.dataset_id)
            for task in plan.grouped_fetch_tasks:
                fetch_id = self.ledger.start_fetch(spec.dataset_id, scope_key(task.scope), task.units, _now())
                try:
                    result = self.fetch_registry.fetch(spec, task, client)
                    calls += 1
                    frame = result.frame
                    self.raw_store.save(spec.dataset_id, fetch_id, frame)
                    self.curated_store.merge(spec, frame, units=task.units, scope=task.scope)
                    records: list[CoverageRecord] = []
                    for unit in task.units:
                        unit_rows = self.curated_store.rows_for_unit(spec, unit=unit, scope=task.scope)
                        if unit_rows.empty and not spec.allow_empty_complete:
                            raise DataUnavailableError(f"Canonical unit {spec.dataset_id}/{unit} is empty after merge.")
                        records.append(CoverageRecord(spec.dataset_id, scope_key(task.scope), unit, "COMPLETE", len(unit_rows), spec.schema_version, content_hash(spec, unit_rows), self._fingerprint(spec.dataset_id, task.scope, task.units), _now()))
                    with self.ledger.transaction() as connection:
                        self.ledger.mark_complete(records, connection)
                        connection.execute("UPDATE fetch_events SET finished_at=?,status='COMPLETE',rows=?,error_type=NULL WHERE fetch_id=?", (_now(), len(frame), fetch_id))
                    rows_total += len(frame)
                except Exception as exc:
                    self.ledger.finish_fetch(fetch_id, status="FAILED", finished_at=_now(), error_type=type(exc).__name__)
                    raise DataUnavailableError(
                        f"Data preparation failed for dataset {spec.dataset_id!r}.",
                        dataset_id=spec.dataset_id,
                        units=task.units,
                        cause=exc,
                    ) from exc
        final_plans = tuple(self._planner().plan((requirement,))[0] for requirement in normalized)
        if not all(plan.ready for plan in final_plans):
            raise DataUnavailableError("Data preparation finished without complete required coverage.")
        return DataPreparationResult("READY", final_plans, calls, rows_total)

    def verify_unit(self, dataset_id: str, *, scope: object, unit: str) -> bool:
        from src.data.contracts import normalize_scope
        normalized_scope = normalize_scope(scope)
        spec = self.registry.get(dataset_id)
        records = [record for record in self.ledger.records(dataset_id) if record.scope_key == scope_key(normalized_scope) and record.unit_key == unit and record.status == "COMPLETE"]
        if len(records) != 1:
            return False
        rows = self.curated_store.rows_for_unit(spec, unit=unit, scope=normalized_scope)
        return records[0].row_count == len(rows) and records[0].content_hash == content_hash(spec, rows)
