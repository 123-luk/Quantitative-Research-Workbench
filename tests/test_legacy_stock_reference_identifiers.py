from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.services.first_run_service import (
    WorkbenchErrorCode,
    classify_data_unavailable_error,
    classify_data_unavailable_stage,
)
from src.data import CoverageLedger, PartitionedParquetStore, RawParquetStore
from src.data.contracts import DataRequirement, GLOBAL_SNAPSHOT_UNIT
from src.data.preparation import DataPreparationService, DataUnavailableError
from src.data.security_identifiers import (
    LEGACY_REFERENCE_RULE_ID,
    UNSUPPORTED_LEGACY_RULE_ID,
)
from src.universe.data import STOCK_BASIC_SCOPE
from src.universe.data import CanonicalUniverseDataSource
from src.universe.contracts import UnsupportedLegacySecurityIdentifier


STOCK_FIELDS = (
    "ts_code", "symbol", "name", "area", "industry", "market", "exchange",
    "curr_type", "list_status", "list_date", "delist_date",
)
REQUIRED_START = "2022-11-17"
REQUIRED_END = "2023-02-08"


def _stock(
    code: str,
    status: str,
    *,
    list_date: str | None,
    delist_date: str | None,
    market: str | None = "主板",
) -> dict[str, object]:
    return {
        "ts_code": code,
        "symbol": code.split(".")[0],
        "name": "Provider reference fixture",
        "area": None,
        "industry": None,
        "market": market,
        "exchange": "SSE",
        "curr_type": "CNY",
        "list_status": status,
        "list_date": list_date,
        "delist_date": delist_date,
    }


class SnapshotProvider:
    sdk_version = "offline-legacy-reference"

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def get_stock_basic(self, *, list_status: str) -> pd.DataFrame:
        selected = [row for row in self.rows if row["list_status"] == list_status]
        return pd.DataFrame(selected, columns=STOCK_FIELDS)


def _service(root: Path) -> DataPreparationService:
    return DataPreparationService(
        ledger=CoverageLedger(root / "metadata" / "catalog.sqlite"),
        curated_store=PartitionedParquetStore(root / "curated"),
        raw_store=RawParquetStore(root / "raw"),
    )


def _requirement() -> DataRequirement:
    return DataRequirement.create(
        "stock_basic",
        scope=STOCK_BASIC_SCOPE,
        required_start=REQUIRED_START,
        required_end=REQUIRED_END,
        reason="complete research interval including warmup and forward holding",
    )


def _terminal_evidence(service: DataPreparationService) -> tuple[object, dict[str, object]]:
    event = service.ledger.fetch_events()[0]
    terminal = service.ledger.coverage_transitions(str(event["fetch_id"]))[-1]
    return terminal, json.loads(terminal.quality_evidence)


def test_real_legacy_samples_outside_complete_interval_pass_with_exclusion_evidence(tmp_path: Path) -> None:
    rows = [
        _stock("T600018.SH", "D", list_date="2000-07-19", delist_date="2006-10-20", market=None),
        _stock("TS0018.SH", "D", list_date="2001-01-01", delist_date="2006-10-20"),
    ]
    service = _service(tmp_path)

    service.ensure((_requirement(),), client=SnapshotProvider(rows))

    stored = service.curated_store.rows_for_unit(
        service.registry.get("stock_basic"),
        unit=GLOBAL_SNAPSHOT_UNIT,
        scope=STOCK_BASIC_SCOPE,
    )
    assert set(stored["ts_code"].astype(str)) == {"T600018.SH", "TS0018.SH"}
    merged = next(
        item for item in service.ledger.coverage_transitions()
        if item.operation == "FETCH_MERGED"
    )
    evidence = json.loads(merged.quality_evidence)
    assert evidence["invalid_count"] == 0
    assert evidence["excluded_count"] == 2
    assert evidence["reason_counts"] == {LEGACY_REFERENCE_RULE_ID: 2}
    assert {item["classification"] for item in evidence["samples"]} == {"LEGACY_REFERENCE"}
    assert all(item["required_start"] == REQUIRED_START for item in evidence["samples"])
    assert all(item["required_end"] == REQUIRED_END for item in evidence["samples"])
    tradable = CanonicalUniverseDataSource(
        registry=service.registry,
        ledger=service.ledger,
        store=service.curated_store,
        stock_basic_as_of=REQUIRED_END,
        index_weight_start=REQUIRED_START,
        stock_basic_required_start=REQUIRED_START,
        stock_basic_required_end=REQUIRED_END,
    ).stock_basic()
    assert tradable.frame.empty
    with pytest.raises(UnsupportedLegacySecurityIdentifier):
        CanonicalUniverseDataSource(
            registry=service.registry,
            ledger=service.ledger,
            store=service.curated_store,
            stock_basic_as_of="2005-12-31",
            index_weight_start="2000-01-01",
            stock_basic_required_start="2005-01-01",
            stock_basic_required_end="2005-12-31",
        ).stock_basic()


def test_overlapping_legacy_reference_blocks_with_precise_error(tmp_path: Path) -> None:
    service = _service(tmp_path)
    row = _stock(
        "TS0018.SH", "D",
        list_date="2022-12-01", delist_date="2023-01-20",
    )

    with pytest.raises(DataUnavailableError):
        service.ensure((_requirement(),), client=SnapshotProvider([row]))

    terminal, evidence = _terminal_evidence(service)
    assert terminal.operation == "QUALITY_VALIDATION"
    assert terminal.safe_message == UNSUPPORTED_LEGACY_RULE_ID
    assert evidence["invalid_count"] == 1
    assert evidence["samples"][0]["classification"] == "INVALID"
    assert evidence["samples"][0]["rule_id"] == UNSUPPORTED_LEGACY_RULE_ID


def test_canonical_numeric_delisted_security_is_retained_inside_history(tmp_path: Path) -> None:
    service = _service(tmp_path)
    row = _stock(
        "600018.SH", "D",
        list_date="2000-07-19", delist_date="2023-01-20",
    )

    service.ensure((_requirement(),), client=SnapshotProvider([row]))

    stored = service.curated_store.rows_for_unit(
        service.registry.get("stock_basic"),
        unit=GLOBAL_SNAPSHOT_UNIT,
        scope=STOCK_BASIC_SCOPE,
    )
    assert stored.loc[0, "ts_code"] == "600018.SH"
    assert stored.loc[0, "list_status"] == "D"
    assert stored.loc[0, "delist_date"] == "2023-01-20"


def test_noncanonical_live_pending_or_unproven_reference_still_blocks(tmp_path: Path) -> None:
    cases = (
        ("L", "2000-01-01", None),
        ("P", "2000-01-01", None),
        ("G", "2000-01-01", None),
        ("D", None, "2006-10-20"),
    )
    for index, (status, listed, delisted) in enumerate(cases):
        service = _service(tmp_path / str(index))
        row = _stock(
            f"LEGACY{index}.SH", status,
            list_date=listed, delist_date=delisted,
        )
        with pytest.raises(DataUnavailableError):
            service.ensure((_requirement(),), client=SnapshotProvider([row]))
        terminal, evidence = _terminal_evidence(service)
        assert terminal.safe_message == "INVALID_SECURITY_CODE"
        assert evidence["invalid_count"] == 1
        assert evidence["samples"][0]["classification"] == "INVALID"


def test_minimal_stock_basic_global_canonical_ledger_readback(tmp_path: Path) -> None:
    service = _service(tmp_path)
    row = _stock("600000.SH", "L", list_date="1999-11-10", delist_date=None)

    result = service.ensure((_requirement(),), client=SnapshotProvider([row]))

    event = service.ledger.fetch_events()[0]
    states = {item.state for item in service.ledger.coverage_transitions(str(event["fetch_id"]))}
    assert result.status == "READY"
    assert service.verify_unit(
        "stock_basic", scope=STOCK_BASIC_SCOPE, unit=GLOBAL_SNAPSHOT_UNIT
    )
    assert {"RAW_STAGED", "CANONICAL_COMMITTED", "LEDGER_COMMITTED", "READBACK_VERIFIED"} <= states


def test_provider_quality_maps_to_quality_stage_and_user_error() -> None:
    ordinary = DataUnavailableError(
        "safe", origin="provider_quality", diagnosis_code="INVALID_SECURITY_CODE"
    )
    unsupported = DataUnavailableError(
        "safe", origin="provider_quality",
        diagnosis_code=UNSUPPORTED_LEGACY_RULE_ID,
    )

    assert classify_data_unavailable_error(ordinary) is WorkbenchErrorCode.PROVIDER_DATA_QUALITY
    assert classify_data_unavailable_error(unsupported) is WorkbenchErrorCode.UNSUPPORTED_LEGACY_SECURITY_IDENTIFIER
    assert classify_data_unavailable_stage(ordinary, "download") == "quality_validation"


def test_failed_quarantine_raw_cannot_satisfy_canonical_or_reuse(tmp_path: Path) -> None:
    service = _service(tmp_path)
    row = _stock(
        "TS0018.SH", "D",
        list_date="2022-12-01", delist_date="2023-01-20",
    )

    with pytest.raises(DataUnavailableError):
        service.ensure((_requirement(),), client=SnapshotProvider([row]))

    event = service.ledger.fetch_events()[0]
    raw_path = Path(str(event["raw_reference"]))
    assert raw_path.is_file()
    assert event["quality_conclusion"] == UNSUPPORTED_LEGACY_RULE_ID
    assert pd.read_parquet(raw_path).loc[0, "ts_code"] == "TS0018.SH"
    assert service.ledger.records("stock_basic") == ()
    assert not (tmp_path / "curated" / "stock_basic").exists()
    assert not service.inspect((_requirement(),))[0].ready
