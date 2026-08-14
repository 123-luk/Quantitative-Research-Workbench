"""Generic missing-only L0/L1 data preparation orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import random
import re
from time import sleep
from typing import Callable, Iterable, Mapping

from src.data.canonical_store import PartitionedParquetStore, RawParquetStore, content_hash
from src.data.contracts import GLOBAL_SNAPSHOT_UNIT, CoverageKind, DataRequirement, DatasetSpec
from src.data.coverage_ledger import CoverageLedger, CoverageRecord
from src.data.coverage_planner import FetchTask, MissingDataPlan, MissingDataPlanner, coalesce_coverage_requirements, scope_key
from src.data.dataset_registry import DatasetRegistry, create_default_dataset_registry
from src.data.fetching import FetchStrategyRegistry, create_default_fetch_strategy_registry, provider_request_parameters
from src.data.provider_contracts import CoverageGranularity, EndpointContract, ProviderContractRegistry
from src.data.tushare_client import TushareClient
from src.data.provider_quality import validate_quality


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
        origin: str = "provider",
        scope: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.dataset_id = dataset_id
        self.units = tuple(units)
        self.safe_cause = cause
        self.scope = scope
        if origin not in {"provider", "local"}:
            raise ValueError("preparation failure origin is invalid")
        self.origin = origin


@dataclass(frozen=True)
class DataPreparationResult:
    status: str
    plans: tuple[MissingDataPlan, ...]
    provider_calls: int
    rows: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sdk_version(client: object) -> str:
    declared = getattr(client, "sdk_version", None)
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    try:
        return version("tushare")
    except PackageNotFoundError:
        return "UNKNOWN"


def _fresh_global_snapshot(completed_at: str) -> bool:
    """Application cache policy: reuse a verified snapshot only on its UTC day."""
    try:
        value = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).date() == datetime.now(timezone.utc).date()


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
        plan = MissingDataPlanner(
            self.registry,
            self.ledger,
            complete_units_resolver=lambda dataset_id, scope, units, fields: _verified_complete_units(
                self.registry, self.ledger, self.store, dataset_id, scope, units, fields
            ),
        ).plan((requirement,))[0]
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



def _verified_complete_units(
    registry: DatasetRegistry,
    ledger: object,
    store: PartitionedParquetStore,
    dataset_id: str,
    scope: tuple[tuple[str, str], ...],
    units: tuple[str, ...],
    required_fields: tuple[str, ...],
) -> frozenset[str]:
    """Return only Ledger COMPLETE units whose local canonical proof verifies."""
    spec = registry.get(dataset_id)
    records = {
        record.unit_key: record
        for record in ledger.records(dataset_id)  # type: ignore[attr-defined]
        if record.scope_key == scope_key(scope) and record.status == "COMPLETE" and record.unit_key in units
    }
    partitions: dict[object, object] = {}
    verified: set[str] = set()
    for unit in units:
        record = records.get(unit)
        if record is None:
            continue
        legacy_daily_basic = dataset_id == "daily_basic" and record.schema_version == "1.0" and "dv_ttm" not in required_fields
        if record.schema_version != spec.schema_version and not legacy_daily_basic:
            continue
        if spec.coverage_kind is CoverageKind.GLOBAL_SNAPSHOT and not _fresh_global_snapshot(record.completed_at):
            continue
        path = store.partition_path(spec, unit=unit, scope=scope)
        if path not in partitions:
            try:
                partitions[path] = store.load_partition(path, spec)
            except Exception:
                partitions[path] = None
        frame = partitions[path]
        if frame is None:
            continue
        if spec.coverage_kind in {CoverageKind.GLOBAL_SNAPSHOT, CoverageKind.REFERENCE_EFFECTIVE_THROUGH}:
            rows = frame
        elif spec.coverage_kind is CoverageKind.ENTITY_MONTH:
            rows = frame.loc[frame["trade_date"].str[:7].eq(unit)]
        else:
            column = "cal_date" if spec.coverage_kind is CoverageKind.CALENDAR_DATE else "trade_date"
            rows = frame.loc[frame[column].eq(unit)]
        if rows.empty and spec.allow_empty_complete and not store.has_empty_marker(spec, unit=unit, scope=scope):
            continue
        hash_spec = replace(spec, required_fields=tuple(field for field in spec.required_fields if field != "dv_ttm"), schema_version="1.0") if legacy_daily_basic else spec
        hash_rows = rows.loc[:, list(hash_spec.required_fields)]
        try:
            matches = record.row_count == len(rows) and record.content_hash == content_hash(hash_spec, hash_rows)
        except Exception:
            matches = False
        if matches:
            verified.add(unit)
    return frozenset(verified)


class DataPreparationService:
    def __init__(self, *, registry: DatasetRegistry | None = None, ledger: CoverageLedger, curated_store: PartitionedParquetStore, raw_store: RawParquetStore, fetch_registry: FetchStrategyRegistry | None = None, open_dates: Callable[[str, str], Iterable[object]] | None = None, client_factory: Callable[[str], object] | None = None, network_attempts: int = 3, retry_sleep: Callable[[float], None] = sleep, retry_jitter: Callable[[], float] = random.random) -> None:
        self.registry = registry or create_default_dataset_registry()
        self.ledger = ledger
        self.curated_store = curated_store
        self.raw_store = raw_store
        self.fetch_registry = fetch_registry or create_default_fetch_strategy_registry()
        self.open_dates = open_dates
        self.client_factory = client_factory or (lambda token: TushareClient(token))
        if type(network_attempts) is not int or network_attempts < 1:
            raise ValueError("network_attempts must be a positive integer.")
        self.network_attempts = network_attempts
        self.retry_sleep = retry_sleep
        self.retry_jitter = retry_jitter
        self.provider_contracts = ProviderContractRegistry()

    def _preflight(self, spec: DatasetSpec, task: FetchTask) -> tuple[EndpointContract, tuple[Mapping[str, object], ...]]:
        contract = self.provider_contracts.get(self.ledger.provider_id, spec.dataset_id)
        expected = {
            CoverageKind.GLOBAL_SNAPSHOT: CoverageGranularity.GLOBAL_SNAPSHOT,
            CoverageKind.CALENDAR_DATE: CoverageGranularity.CALENDAR_DATE_PARTITIONED,
            CoverageKind.TRADE_DATE: CoverageGranularity.TRADE_DATE_PARTITIONED,
            CoverageKind.ENTITY_TRADE_DATE: CoverageGranularity.TRADE_DATE_PARTITIONED,
            CoverageKind.ENTITY_MONTH: CoverageGranularity.MONTH_PARTITIONED,
        }.get(spec.coverage_kind)
        parameters = tuple(provider_request_parameters(spec, task))
        if expected is not None and contract.coverage_granularity is not expected:
            raise DataUnavailableError(
                "Dataset/provider coverage contracts are inconsistent.",
                dataset_id=spec.dataset_id, units=task.units, origin="local", scope=task.scope,
            )
        if spec.coverage_kind is CoverageKind.GLOBAL_SNAPSHOT:
            date_names = {"trade_date", "start_date", "end_date"}
            carries_date = any(
                name in item and item[name] is not None
                for item in parameters
                for name in date_names
            )
            if task.units != (GLOBAL_SNAPSHOT_UNIT,) or task.start != GLOBAL_SNAPSHOT_UNIT or task.end != GLOBAL_SNAPSHOT_UNIT or carries_date:
                raise DataUnavailableError(
                    "GLOBAL_SNAPSHOT endpoint was planned as a dated request.",
                    dataset_id=spec.dataset_id, units=task.units, origin="local", scope=task.scope,
                )
        return contract, parameters

    @staticmethod
    def _transient_network_failure(exc: BaseException) -> bool:
        chain: list[str] = []
        current: BaseException | None = exc
        while current is not None and len(chain) < 8:
            chain.append(f"{type(current).__name__} {current}")
            current = current.__cause__ or current.__context__
        text = " ".join(chain).lower()
        deterministic = r"token|auth|permission|points?|积分|字段|schema|structure|rate.?limit|frequency|限流|频率"
        transient = r"connect|timeout|timed out|dns|socket|proxy|network|网络|连接|超时"
        return not re.search(deterministic, text) and bool(re.search(transient, text))

    def _fetch_with_retry(self, spec: object, task: object, client: object) -> object:
        for attempt in range(1, self.network_attempts + 1):
            try:
                return self.fetch_registry.fetch(spec, task, client)  # type: ignore[arg-type]
            except Exception as exc:
                if attempt >= self.network_attempts or not self._transient_network_failure(exc):
                    setattr(exc, "provider_attempts", attempt)
                    raise
                delay = (0.5 * (2 ** (attempt - 1))) * (0.9 + 0.2 * self.retry_jitter())
                self.retry_sleep(delay)
        raise AssertionError("unreachable retry state")

    def _fingerprint(self, dataset_id: str, task_scope: tuple[tuple[str, str], ...], units: tuple[str, ...]) -> str:
        payload = json.dumps({"provider_id": self.ledger.provider_id, "dataset_id": dataset_id, "scope": task_scope, "units": units}, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def _planner(self) -> MissingDataPlanner:
        return MissingDataPlanner(
            self.registry,
            self.ledger,
            open_dates=self.open_dates,
            complete_units_resolver=lambda dataset_id, scope, units, fields: _verified_complete_units(
                self.registry, self.ledger, self.curated_store, dataset_id, scope, units, fields
            ),
        )

    def inspect(self, requirements: Iterable[DataRequirement]) -> tuple[MissingDataPlan, ...]:
        """Plan exact missing coverage without provider calls or data writes."""
        normalized = tuple(
            sorted(
                coalesce_coverage_requirements(self.registry, requirements),
                key=lambda item: (
                    self.registry.get(item.dataset_id).coverage_kind.value != "CALENDAR_DATE",
                    item.dataset_id,
                    item.scope,
                    item.required_start,
                ),
            )
        )
        return self._planner().plan(normalized)

    def ensure(self, requirements: Iterable[DataRequirement], credential: str | None = None, *, client: object | None = None, progress: Callable[[str, str, int, int], None] | None = None) -> DataPreparationResult:
        normalized = tuple(sorted(coalesce_coverage_requirements(self.registry, requirements), key=lambda item: (self.registry.get(item.dataset_id).coverage_kind.value != "CALENDAR_DATE", item.dataset_id, item.scope, item.required_start)))
        calls = 0
        rows_total = 0
        try:
            initial_plans = tuple(self._planner().plan((requirement,))[0] for requirement in normalized)
        except DataUnavailableError:
            initial_plans = ()
        total_units = sum(len(plan.required_units) for plan in initial_plans)
        completed_units = sum(len(plan.complete_units) for plan in initial_plans)
        dynamic_totals = not initial_plans
        for requirement in normalized:
            plan = self._planner().plan((requirement,))[0]
            if dynamic_totals:
                total_units += len(plan.required_units)
                completed_units += len(plan.complete_units)
            if plan.ready:
                continue
            spec = self.registry.get(plan.requirement.dataset_id)
            # Keep existing non-dividend research and offline fixtures compatible
            # with daily_basic 1.0. A request that explicitly needs dv_ttm uses
            # the strict 1.1 provider/canonical contract and triggers targeted
            # repair of only those units.
            if spec.dataset_id == "daily_basic" and "dv_ttm" not in plan.requirement.required_fields:
                spec = replace(
                    spec,
                    required_fields=tuple(field for field in spec.required_fields if field != "dv_ttm"),
                    schema_version="1.0",
                )
            preflight = tuple(self._preflight(spec, task) for task in plan.grouped_fetch_tasks)
            if client is None:
                if credential is None or not isinstance(credential, str) or not credential.strip():
                    raise MissingCredentialError("Required canonical data is missing and no TuShare credential was provided.")
                try:
                    client = self.client_factory(credential.strip())
                except Exception:
                    raise DataUnavailableError("TuShare client initialization failed.") from None
            for task, (contract, request_parameters) in zip(plan.grouped_fetch_tasks, preflight):
                if progress is not None:
                    progress(spec.dataset_id, task.units[0], completed_units, total_units)
                fetch_id = self.ledger.start_fetch(
                    spec.dataset_id, scope_key(task.scope), task.units, _now(),
                    endpoint=contract.api_name,
                    request_parameters=request_parameters,
                    contract_version=contract.contract_version,
                )
                failure_origin = "provider"
                try:
                    result = self._fetch_with_retry(spec, task, client)
                    calls += 1
                    frame = result.frame
                    retrieved_at = _now()
                    failure_origin = "local"
                    issues = validate_quality(spec, frame)
                    if issues:
                        categories = ",".join(sorted({item.category for item in issues}))
                        raise DataUnavailableError(
                            f"Provider data quality validation failed: {categories}."
                        )
                    raw_path = self.raw_store.save(spec.dataset_id, fetch_id, frame)
                    provenance = {
                        "endpoint": contract.api_name,
                        "contract_version": contract.contract_version,
                        "request_statuses": [
                            item["list_status"] for item in request_parameters
                            if "list_status" in item
                        ],
                        "request_parameters": list(request_parameters),
                        "retrieved_at": retrieved_at,
                        "raw_reference": str(raw_path),
                        "sdk_version": _sdk_version(client),
                        "quality_conclusion": "PASSED",
                    }
                    canonical_paths = self.curated_store.merge(
                        spec, frame, units=task.units, scope=task.scope,
                        provenance=provenance,
                    )
                    records: list[CoverageRecord] = []
                    for unit in task.units:
                        unit_rows = self.curated_store.rows_for_unit(spec, unit=unit, scope=task.scope)
                        if unit_rows.empty and not spec.allow_empty_complete:
                            raise DataUnavailableError(f"Canonical unit {spec.dataset_id}/{unit} is empty after merge.")
                        if unit_rows.empty:
                            self.curated_store.write_empty_marker(spec, unit=unit, scope=task.scope)
                        records.append(CoverageRecord(spec.dataset_id, scope_key(task.scope), unit, "COMPLETE", len(unit_rows), spec.schema_version, content_hash(spec, unit_rows), self._fingerprint(spec.dataset_id, task.scope, task.units), _now(), self.ledger.provider_id))
                    with self.ledger.transaction() as connection:
                        self.ledger.mark_complete(records, connection)
                        canonical_hash = (
                            records[0].content_hash if len(records) == 1
                            else sha256("|".join(record.content_hash for record in records).encode()).hexdigest()
                        )
                        manifests = tuple(str(path.with_suffix(path.suffix + ".manifest.json")) for path in canonical_paths)
                        self.ledger.finish_fetch(
                            fetch_id, status="COMPLETE", finished_at=_now(), rows=len(frame),
                            connection=connection, retrieved_at=retrieved_at,
                            schema_version=spec.schema_version, canonical_hash=canonical_hash,
                            raw_reference=str(raw_path),
                            canonical_reference=json.dumps(tuple(str(path) for path in canonical_paths)),
                            manifest_reference=json.dumps(manifests),
                            sdk_version=_sdk_version(client), quality_conclusion="PASSED",
                        )
                    rows_total += len(frame)
                    for unit in task.units:
                        completed_units += 1
                        if progress is not None:
                            progress(spec.dataset_id, unit, completed_units, total_units)
                except Exception as exc:
                    self.ledger.finish_fetch(fetch_id, status="FAILED", finished_at=_now(), error_type=type(exc).__name__)
                    raise DataUnavailableError(
                        f"Data preparation failed for dataset {spec.dataset_id!r}.",
                        dataset_id=spec.dataset_id,
                        units=task.units,
                        cause=exc,
                        origin=failure_origin,
                        scope=task.scope,
                    ) from exc
        final_plans = tuple(self._planner().plan((requirement,))[0] for requirement in normalized)
        if not all(plan.ready for plan in final_plans):
            raise DataUnavailableError("Data preparation finished without complete required coverage.", origin="local")
        return DataPreparationResult("READY", final_plans, calls, rows_total)

    def verify_unit(self, dataset_id: str, *, scope: object, unit: str, required_fields: tuple[str, ...] = ()) -> bool:
        from src.data.contracts import normalize_scope
        normalized_scope = normalize_scope(scope)
        return unit in _verified_complete_units(
            self.registry, self.ledger, self.curated_store,
            dataset_id, normalized_scope, (unit,), required_fields,
        )
