from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import CoverageLedger, PartitionedParquetStore, RawParquetStore
from src.data.contracts import (
    GLOBAL_SNAPSHOT_UNIT,
    CoverageKind,
    DataRequirement,
    ScopeKind,
)
from src.data.coverage_ledger import coverage_identity_key
from src.data.coverage_planner import MissingDataPlanner, scope_key
from src.data.dataset_registry import create_default_dataset_registry
from src.data.preparation import DataPreparationService, DataUnavailableError
from src.universe.data import STOCK_BASIC_SCOPE


STOCK_FIELDS = (
    "ts_code", "symbol", "name", "area", "industry", "market", "exchange",
    "curr_type", "list_status", "list_date", "delist_date",
)


def _stock(code: str, status: str) -> dict[str, object]:
    return {
        "ts_code": code,
        "symbol": code[:6],
        "name": code,
        "area": "China",
        "industry": "Test",
        "market": "Main Board",
        "exchange": "SSE",
        "curr_type": "CNY",
        "list_status": status,
        "list_date": "20100101",
        "delist_date": None,
    }


class DiagnosticSnapshotProvider:
    sdk_version = "offline-diagnostic"

    def __init__(self, code: str) -> None:
        self.code = code

    def get_stock_basic(self, *, list_status: str) -> pd.DataFrame:
        rows = [_stock(self.code, list_status)] if list_status == "L" else []
        return pd.DataFrame(rows, columns=STOCK_FIELDS)


def _service(tmp_path: Path) -> DataPreparationService:
    return DataPreparationService(
        ledger=CoverageLedger(tmp_path / "metadata" / "catalog.sqlite"),
        curated_store=PartitionedParquetStore(tmp_path / "curated"),
        raw_store=RawParquetStore(tmp_path / "raw"),
    )


def _requirement() -> DataRequirement:
    return DataRequirement.create(
        "stock_basic",
        scope=STOCK_BASIC_SCOPE,
        required_start="2023-01-01",
        required_end="2023-02-01",
        reason="coverage transaction diagnostics",
    )


def test_quality_failure_records_exact_safe_stopping_state(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(DataUnavailableError):
        service.ensure((_requirement(),), client=DiagnosticSnapshotProvider("INVALID"))

    event = service.ledger.fetch_events()[0]
    transitions = service.ledger.coverage_transitions(str(event["fetch_id"]))
    provider_calls = [
        item for item in transitions
        if item.state == "FETCH_SUCCEEDED"
        and item.operation is not None
        and item.operation.startswith("PROVIDER_CALL:stock_basic")
    ]
    assert [item.operation.rpartition("=")[2] for item in provider_calls] == ["L", "D", "P", "G"]
    assert [item.rows for item in provider_calls] == [1, 0, 0, 0]
    assert all(json.loads(item.fields) == list(STOCK_FIELDS) for item in provider_calls)
    terminal = transitions[-1]
    assert terminal.state == "FETCH_FAILED"
    assert terminal.operation == "QUALITY_VALIDATION"
    assert terminal.error_code == "QUALITY_VALIDATION_FAILED"
    assert terminal.exception_type == "DataUnavailableError"
    assert terminal.safe_message == "INVALID_SECURITY_CODE"
    assert terminal.rows == 1
    assert event["status"] == "FAILED"
    assert event["error_type"] == "DataUnavailableError"
    assert service.ledger.records("stock_basic") == ()
    assert not (tmp_path / "raw" / "tushare_official" / "stock_basic").exists()
    assert not (tmp_path / "curated" / "stock_basic").exists()


def test_success_records_fetch_to_readback_transaction_and_manifest_identity(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.ensure((_requirement(),), client=DiagnosticSnapshotProvider("600000.SH"))

    event = service.ledger.fetch_events()[0]
    transitions = service.ledger.coverage_transitions(str(event["fetch_id"]))
    states = {item.state for item in transitions}
    assert {
        "PLANNED", "FETCH_STARTED", "FETCH_SUCCEEDED", "RAW_STAGED",
        "CANONICAL_VALIDATED", "CANONICAL_COMMITTED", "LEDGER_COMMITTED",
        "READBACK_VERIFIED",
    }.issubset(states)
    assert transitions[-1].state == "READBACK_VERIFIED"
    record = service.ledger.records("stock_basic")[0]
    assert record.unit_key == GLOBAL_SNAPSHOT_UNIT
    expected_identity = json.loads(
        coverage_identity_key("stock_basic", scope_key(STOCK_BASIC_SCOPE), GLOBAL_SNAPSHOT_UNIT)
    )
    manifest_path = Path(json.loads(str(event["manifest_reference"]))[0])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["coverage_identities"] == [expected_identity]
    assert event["status"] == "COMPLETE"


def _scope_for(kind: ScopeKind) -> object:
    if kind is ScopeKind.REFERENCE_SNAPSHOT:
        return STOCK_BASIC_SCOPE
    if kind is ScopeKind.ENTITY_MONTH_SNAPSHOT:
        return {"index_code": "000300.SH"}
    if kind is ScopeKind.ENTITY_SERIES:
        return {"exchange": "SSE"}
    return {"scope": "CN_A"}


@pytest.mark.parametrize(
    "dataset_id",
    create_default_dataset_registry().list_ids(),
)
def test_registered_coverage_identity_is_registry_driven_and_layer_stable(
    tmp_path: Path, dataset_id: str
) -> None:
    registry = create_default_dataset_registry()
    spec = registry.get(dataset_id)
    scope = _scope_for(spec.scope_kind)
    if dataset_id == "index_daily":
        scope = {"index_code": "000300.SH"}
    requirement = DataRequirement.create(
        dataset_id,
        scope=scope,
        required_start="2024-01-02",
        required_end="2024-01-03",
        reason="registry identity invariant",
    )
    ledger = CoverageLedger(tmp_path / dataset_id / "catalog.sqlite")
    planner = MissingDataPlanner(
        registry,
        ledger,
        open_dates=lambda _start, _end: ("2024-01-02", "2024-01-03"),
    )
    plan = planner.plan((requirement,))[0]
    assert tuple(unit for task in plan.grouped_fetch_tasks for unit in task.units) == plan.missing_units
    assert plan.required_units == plan.missing_units
    if spec.coverage_kind is CoverageKind.GLOBAL_SNAPSHOT:
        assert plan.required_units == (GLOBAL_SNAPSHOT_UNIT,)

    store = PartitionedParquetStore(tmp_path / dataset_id / "curated")
    for task in plan.grouped_fetch_tasks:
        fetch_id = ledger.start_fetch(
            spec.dataset_id,
            scope_key(task.scope),
            task.units,
            "2026-08-14T00:00:00+00:00",
            endpoint=spec.endpoint,
        )
        planned = ledger.coverage_transitions(fetch_id)
        assert [item.unit_key for item in planned] == list(task.units)
        for item in planned:
            assert item.coverage_identity == coverage_identity_key(
                spec.dataset_id, scope_key(task.scope), item.unit_key
            )
            assert store.partition_path(
                spec, unit=item.unit_key, scope=task.scope
            ).parent.name
