from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import CoverageLedger, PartitionedParquetStore, RawParquetStore
from src.data.contracts import GLOBAL_SNAPSHOT_UNIT, DataRequirement, FetchStrategy
from src.data.coverage_ledger import CoverageRecord
from src.data.coverage_planner import MissingDataPlanner, scope_key
from src.data.dataset_registry import DatasetRegistry, create_default_dataset_registry
from src.data.preparation import DataPreparationService, DataUnavailableError
from src.universe import CanonicalUniverseDataSource, UniverseService, UniverseSpec
from src.universe.data import STOCK_BASIC_SCOPE


STOCK_FIELDS = (
    "ts_code", "symbol", "name", "area", "industry", "market", "exchange",
    "curr_type", "list_status", "list_date", "delist_date",
)


def _stock(code: str, status: str, listed: str, delisted: str | None = None) -> dict[str, object]:
    return {
        "ts_code": code, "symbol": code[:6], "name": code, "area": "China",
        "industry": "Current-only", "market": "主板",
        "exchange": "SSE" if code.endswith("SH") else "SZSE",
        "curr_type": "CNY", "list_status": status,
        "list_date": listed, "delist_date": delisted,
    }


class SnapshotProvider:
    sdk_version = "fake-sdk-1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_stock_basic(self, *, list_status: str) -> pd.DataFrame:
        self.calls.append(list_status)
        rows = {
            "L": [
                _stock("600001.SH", "L", "20100101"),
                _stock("600002.SH", "L", "20100101"),
            ],
            # Terminal status wins the documented application dedup rule for
            # a code exposed by two sequential status-scoped calls.
            "D": [_stock("600002.SH", "D", "20100101", "20200601")],
            "P": [_stock("000001.SZ", "P", "20150101")],
            "G": [_stock("000002.SZ", "G", "20250101")],
        }[list_status]
        return pd.DataFrame(rows, columns=STOCK_FIELDS)


def _components(tmp_path: Path):
    registry = create_default_dataset_registry()
    ledger = CoverageLedger(tmp_path / "metadata" / "catalog.sqlite")
    curated = PartitionedParquetStore(tmp_path / "curated")
    raw = RawParquetStore(tmp_path / "raw")
    return registry, ledger, curated, raw


def _requirement(start: str, end: str, fields: tuple[str, ...] = ()) -> DataRequirement:
    return DataRequirement.create(
        "stock_basic", scope=STOCK_BASIC_SCOPE,
        required_start=start, required_end=end, required_fields=fields,
        reason="UAT-032 snapshot dependency",
    )


def test_stock_basic_is_one_global_unit_for_any_interval_and_coalesces_fields(tmp_path: Path) -> None:
    registry = create_default_dataset_registry()
    planner = MissingDataPlanner(registry, CoverageLedger(tmp_path / "catalog.sqlite"))
    plans = planner.plan((
        _requirement("2010-01-01", "2010-01-02", ("ts_code", "list_date")),
        _requirement("1990-01-01", "2026-12-31", ("ts_code", "delist_date")),
    ))
    assert len(plans) == 1
    assert plans[0].required_units == (GLOBAL_SNAPSHOT_UNIT,)
    assert plans[0].missing_units == (GLOBAL_SNAPSHOT_UNIT,)
    assert len(plans[0].grouped_fetch_tasks) == 1
    assert set(plans[0].requirement.required_fields) == {"ts_code", "list_date", "delist_date"}


def test_four_status_snapshot_has_no_date_calls_and_persists_provenance(tmp_path: Path) -> None:
    registry, ledger, curated, raw = _components(tmp_path)
    provider = SnapshotProvider()
    service = DataPreparationService(
        registry=registry, ledger=ledger, curated_store=curated, raw_store=raw,
    )
    result = service.ensure((_requirement("1990-01-01", "2026-12-31"),), client=provider)

    assert provider.calls == ["L", "D", "P", "G"]
    assert result.provider_calls == 1
    assert result.plans[0].required_units == (GLOBAL_SNAPSHOT_UNIT,)
    record = ledger.records("stock_basic")[0]
    assert record.unit_key == GLOBAL_SNAPSHOT_UNIT and record.row_count == 4
    event = ledger.fetch_events()[0]
    parameters = json.loads(str(event["request_parameters"]))
    assert json.loads(str(event["requested_statuses"])) == ["L", "D", "P", "G"]
    assert [item["list_status"] for item in parameters] == ["L", "D", "P", "G"]
    assert not any(
        name in item for item in parameters
        for name in ("trade_date", "start_date", "end_date")
    )
    assert event["endpoint"] == "stock_basic"
    assert event["schema_version"] == "1.2"
    assert event["contract_version"] == "1.1"
    assert event["quality_conclusion"] == "PASSED"
    assert Path(str(event["raw_reference"])).is_file()
    manifest_paths = json.loads(str(event["manifest_reference"]))
    manifest = json.loads(Path(manifest_paths[0]).read_text(encoding="utf-8"))
    assert manifest["request_statuses"] == ["L", "D", "P", "G"]
    assert manifest["content_hash"] == record.content_hash


def test_snapshot_materialization_supports_historical_lifecycle_boundaries(tmp_path: Path) -> None:
    registry, ledger, curated, raw = _components(tmp_path)
    service = DataPreparationService(
        registry=registry, ledger=ledger, curated_store=curated, raw_store=raw,
    )
    service.ensure((_requirement("2019-01-01", "2026-12-31"),), client=SnapshotProvider())
    source = CanonicalUniverseDataSource(
        registry=registry, ledger=ledger, store=curated,
        stock_basic_as_of="2020-06-01", index_weight_start="2020-01-01",
    )
    assert source.stock_basic().source_as_of == datetime.now(timezone.utc).date().isoformat()
    universe = UniverseSpec.custom(("600001.SH", "600002.SH", "000002.SZ"))
    before = UniverseService().resolve(universe, "2020-05-31", source)
    boundary = UniverseService().resolve(universe, "2020-06-01", source)

    assert before.securities == ("600001.SH", "600002.SH")
    assert boundary.securities == ("600001.SH",)
    assert before.diagnostics["lifecycle_boundary"] == "list_date <= T < delist_date"


def test_old_dated_stock_unit_is_ignored_while_other_complete_units_survive(tmp_path: Path) -> None:
    registry = create_default_dataset_registry()
    ledger = CoverageLedger(tmp_path / "catalog.sqlite")
    now = datetime.now(timezone.utc).isoformat()
    daily_scope = (("scope", "CN_A"),)
    ledger.mark_complete((
        CoverageRecord("stock_basic", scope_key(STOCK_BASIC_SCOPE), "2024-01-30", "COMPLETE", 1, "1.1", "old", "old", now),
        CoverageRecord("daily", scope_key(daily_scope), "2024-01-30", "COMPLETE", 1, "1.0", "kept", "kept", now),
    ))
    planner = MissingDataPlanner(
        registry, ledger, open_dates=lambda _start, _end: ("2024-01-30",)
    )
    plans = planner.plan((
        _requirement("2024-01-01", "2024-01-30"),
        DataRequirement.create("daily", scope=daily_scope, required_start="2024-01-30", required_end="2024-01-30"),
    ))
    by_dataset = {item.requirement.dataset_id: item for item in plans}
    assert by_dataset["stock_basic"].missing_units == (GLOBAL_SNAPSHOT_UNIT,)
    assert by_dataset["daily"].ready
    assert {record.unit_key for record in ledger.records("stock_basic")} == {"2024-01-30"}


def test_global_snapshot_preflight_rejects_dated_strategy_before_client_creation(tmp_path: Path) -> None:
    default = create_default_dataset_registry().get("stock_basic")
    registry = DatasetRegistry()
    registry.register(replace(default, fetch_strategy=FetchStrategy.MARKET_SNAPSHOT_BY_DATE))
    ledger = CoverageLedger(tmp_path / "catalog.sqlite")
    created: list[str] = []
    service = DataPreparationService(
        registry=registry, ledger=ledger,
        curated_store=PartitionedParquetStore(tmp_path / "curated"),
        raw_store=RawParquetStore(tmp_path / "raw"),
        client_factory=lambda _token: created.append("created") or SnapshotProvider(),
    )
    with pytest.raises(DataUnavailableError, match="GLOBAL_SNAPSHOT") as raised:
        service.ensure((_requirement("2024-01-01", "2024-01-31"),), credential="fake")
    assert raised.value.origin == "local"
    assert created == []
    assert ledger.fetch_events() == ()
