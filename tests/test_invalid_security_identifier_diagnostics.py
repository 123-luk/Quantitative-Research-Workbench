from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.views.runs import _quality_diagnostic_summary
from src.data import CoverageLedger, PartitionedParquetStore, RawParquetStore
from src.data.contracts import DataRequirement
from src.data.preparation import DataPreparationService, DataUnavailableError
from src.data.provider_quality import (
    MAX_INVALID_IDENTIFIER_SAMPLES,
    SECURITY_IDENTIFIER_RULE_ID,
    invalid_security_identifier_evidence,
    sanitize_identifier_evidence,
)
from src.universe.data import STOCK_BASIC_SCOPE


STOCK_FIELDS = (
    "ts_code", "symbol", "name", "area", "industry", "market", "exchange",
    "curr_type", "list_status", "list_date", "delist_date",
)


def _stock(code: str, status: str = "L") -> dict[str, object]:
    return {
        "ts_code": code,
        "symbol": code.split(".")[0],
        "name": "Public security",
        "area": "China",
        "industry": "Test",
        "market": "Main Board",
        "exchange": "SSE",
        "curr_type": "CNY",
        "list_status": status,
        "list_date": "20100101",
        "delist_date": None,
    }


def test_invalid_identifier_samples_are_bounded_deduplicated_and_stably_sorted() -> None:
    codes = [f"BAD-{index:02d}" for index in reversed(range(22))]
    rows = pd.DataFrame([_stock(code) for code in codes] + [_stock("BAD-00")])
    raw = pd.DataFrame([_stock(code) for code in codes], columns=STOCK_FIELDS)

    evidence = invalid_security_identifier_evidence(
        rows,
        raw_frames={"L": raw, "D": pd.DataFrame(), "P": pd.DataFrame(), "G": pd.DataFrame()},
        status_row_counts={"L": 22, "D": 0, "P": 0, "G": 0},
        pre_merge_rows=23,
    )

    samples = evidence["samples"]
    assert evidence["invalid_count"] == 23
    assert len(samples) == MAX_INVALID_IDENTIFIER_SAMPLES
    assert [item["ts_code"] for item in samples] == [f"BAD-{index:02d}" for index in range(20)]
    assert len({json.dumps(item, sort_keys=True) for item in samples}) == len(samples)


def test_identifier_evidence_never_persists_token_or_provider_payload() -> None:
    secret = "token-super-secret"
    evidence = sanitize_identifier_evidence({
        "invalid_count": 1,
        "token": secret,
        "provider_raw_response": {"secret": secret},
        "samples": [{
            "ts_code": "BAD",
            "symbol": "BAD",
            "list_status": "L",
            "market": "Main Board",
            "exchange": "SSE",
            "raw_ts_code": "BAD",
            "normalized_ts_code": "BAD",
            "rule_id": SECURITY_IDENTIFIER_RULE_ID,
            "request_headers": secret,
        }],
        "status_row_counts": {"L": 1},
        "pre_merge_rows": 1,
        "merged_rows": 1,
        "deduplicated_rows": 0,
    })

    payload = json.dumps(evidence, sort_keys=True)
    assert secret not in payload
    assert "provider_raw_response" not in payload
    assert "request_headers" not in payload


def test_gui_quality_summary_is_whitelisted_and_bounded() -> None:
    samples = [
        {
            "ts_code": f"BAD-{index:02d}", "symbol": str(index),
            "list_status": "L", "market": "Main Board", "exchange": "SSE",
            "raw_ts_code": f"BAD-{index:02d}",
            "normalized_ts_code": f"BAD-{index:02d}",
            "rule_id": SECURITY_IDENTIFIER_RULE_ID,
            "provider_payload": "must-not-display",
        }
        for index in reversed(range(25))
    ]
    summary = _quality_diagnostic_summary({
        "invalid_count": 25,
        "samples": samples,
        "status_row_counts": {"L": 25},
        "pre_merge_rows": 25,
        "merged_rows": 25,
        "deduplicated_rows": 0,
        "token": "must-not-display",
    })

    assert len(summary["samples"]) == MAX_INVALID_IDENTIFIER_SAMPLES
    assert "must-not-display" not in json.dumps(summary, sort_keys=True)
    assert list(summary["status_row_counts"]) == ["L", "D", "P", "G"]


class InvalidSnapshotProvider:
    sdk_version = "offline-diagnostic"

    def get_stock_basic(self, *, list_status: str) -> pd.DataFrame:
        rows = [_stock("INVALID", list_status)] if list_status == "L" else []
        return pd.DataFrame(rows, columns=STOCK_FIELDS)


def test_quality_failure_persists_identifier_evidence_without_changing_state_chain(tmp_path: Path) -> None:
    service = DataPreparationService(
        ledger=CoverageLedger(tmp_path / "metadata" / "catalog.sqlite"),
        curated_store=PartitionedParquetStore(tmp_path / "curated"),
        raw_store=RawParquetStore(tmp_path / "raw"),
    )
    requirement = DataRequirement.create(
        "stock_basic",
        scope=STOCK_BASIC_SCOPE,
        required_start="2023-01-01",
        required_end="2023-02-01",
        reason="bounded invalid identifier evidence",
    )

    with pytest.raises(DataUnavailableError):
        service.ensure((requirement,), client=InvalidSnapshotProvider())

    event = service.ledger.fetch_events()[0]
    transitions = service.ledger.coverage_transitions(str(event["fetch_id"]))
    terminal = transitions[-1]
    evidence = json.loads(terminal.quality_evidence)
    assert terminal.state == "FETCH_FAILED"
    assert terminal.operation == "QUALITY_VALIDATION"
    assert terminal.error_code == "QUALITY_VALIDATION_FAILED"
    assert terminal.safe_message == "INVALID_SECURITY_CODE"
    assert evidence["invalid_count"] == 1
    assert evidence["status_row_counts"] == {"L": 1, "D": 0, "P": 0, "G": 0}
    assert evidence["pre_merge_rows"] == evidence["merged_rows"] == 1
    assert evidence["deduplicated_rows"] == 0
    assert evidence["samples"] == [{
        "exchange": "SSE",
        "list_status": "L",
        "market": "Main Board",
        "normalized_ts_code": "INVALID",
        "raw_ts_code": "INVALID",
        "rule_id": SECURITY_IDENTIFIER_RULE_ID,
        "symbol": "INVALID",
        "ts_code": "INVALID",
    }]
    assert event["status"] == "FAILED"
    assert service.ledger.records("stock_basic") == ()
    assert not (tmp_path / "raw" / "tushare_official" / "stock_basic").exists()
    assert not (tmp_path / "curated" / "stock_basic").exists()
