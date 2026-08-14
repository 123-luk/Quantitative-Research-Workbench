from __future__ import annotations

import json

import pandas as pd
import pytest

from app.services.credential_service import CredentialService
from app.services.research_task_service import ResearchTaskService, TASK_SCHEMA_VERSION
from src.data import CoverageLedger, PartitionedParquetStore
from src.data.dataset_registry import create_default_dataset_registry
from src.data.provider_contracts import CoverageGranularity, OFFICIAL_NOT_STATED, PROXY_RULE_UNKNOWN, ProviderContractRegistry
from src.data.provider_quality import compare_providers, validate_quality
from src.data.provider_registry import PROXY_HTTPS_ENDPOINT, ProviderCompatibilityError, TushareProxyClient


class _Api:
    def __init__(self) -> None:
        self._DataApi__token = "initial"
        self._DataApi__http_url = "https://api.tushare.pro"


def test_contract_registry_keeps_unknown_rules_and_proxy_claim_separate() -> None:
    registry = ProviderContractRegistry()
    suspend = registry.get("tushare_official", "suspend_d")
    proxy = registry.get("tushare_proxy", "suspend_d")
    assert suspend.doc_id == 214
    assert suspend.minimum_points == OFFICIAL_NOT_STATED
    assert suspend.calls_per_minute == OFFICIAL_NOT_STATED
    assert proxy.calls_per_minute == PROXY_RULE_UNKNOWN
    assert "NOT_OFFICIALLY_VERIFIED" in str(proxy.minimum_points)
    assert {item.dataset_id for item in registry.for_provider("tushare_official")} == {
        "stock_basic", "trade_cal", "daily", "daily_basic", "adj_factor",
        "index_weight", "stk_limit", "suspend_d", "index_daily", "monthly",
    }
    assert registry.get("tushare_official", "index_daily").doc_id == OFFICIAL_NOT_STATED
    assert registry.get("tushare_official", "monthly").official_url == OFFICIAL_NOT_STATED
    stock = registry.get("tushare_official", "stock_basic")
    assert stock.coverage_granularity is CoverageGranularity.GLOBAL_SNAPSHOT
    assert stock.contract_version == "1.1" and stock.max_rows == 6000
    assert registry.get("tushare_proxy", "stock_basic").coverage_granularity is CoverageGranularity.GLOBAL_SNAPSHOT


def test_proxy_private_sdk_fields_are_fixed_and_incompatibility_is_explicit() -> None:
    client = object.__new__(TushareProxyClient)
    api = _Api()
    assert client._configure_client(api, "proxy-secret") is api
    assert api._DataApi__token == "proxy-secret"
    assert api._DataApi__http_url == PROXY_HTTPS_ENDPOINT
    with pytest.raises(ProviderCompatibilityError, match="不兼容"):
        client._configure_client(object(), "proxy-secret")


def test_credentials_are_provider_specific_and_proxy_never_reads_official_environment() -> None:
    class Environment:
        def tushare_token(self):
            return "official-secret"
    service = CredentialService(environment=Environment())
    assert service.resolve(None, provider_id="tushare_official").available
    assert not service.resolve(None, provider_id="tushare_proxy").available


def test_ledger_empty_marker_and_manifest_are_provider_bound(tmp_path) -> None:
    official = CoverageLedger(tmp_path / "official.sqlite", provider_id="tushare_official")
    proxy = CoverageLedger(tmp_path / "proxy.sqlite", provider_id="tushare_proxy")
    assert official.provider_id != proxy.provider_id
    spec = create_default_dataset_registry().get("suspend_d")
    off_store = PartitionedParquetStore(tmp_path / "official", provider_id="tushare_official")
    proxy_store = PartitionedParquetStore(tmp_path / "proxy", provider_id="tushare_proxy")
    off_store.write_empty_marker(spec, unit="2023-11-16", scope=(("scope", "CN_A"),))
    assert off_store.has_empty_marker(spec, unit="2023-11-16", scope=(("scope", "CN_A"),))
    assert not proxy_store.has_empty_marker(spec, unit="2023-11-16", scope=(("scope", "CN_A"),))


def test_quality_comparison_detects_numeric_and_ohlc_differences() -> None:
    spec = create_default_dataset_registry().get("daily")
    row = {"ts_code": "000001.SZ", "trade_date": "2024-01-02", "open": 10, "high": 11, "low": 9, "close": 10.5, "pre_close": 10, "change": .5, "pct_chg": 5, "vol": 100, "amount": 1000}
    left = pd.DataFrame([row])
    right = pd.DataFrame([{**row, "close": 10.6}])
    report = compare_providers(spec, left, right)
    assert not report.consistent
    assert any(item.field == "close" for item in report.issues)
    invalid = pd.DataFrame([{**row, "high": 8}])
    assert any(item.category == "OHLC_INVARIANT" for item in validate_quality(spec, invalid))


def test_task_1_0_read_migration_is_in_memory_and_adds_formal_diagnostics(tmp_path) -> None:
    service = ResearchTaskService(tmp_path)
    service.root.mkdir(parents=True)
    task_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    record = {
        "schema_version": "1.0", "task_id": task_id, "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00", "status": "failed", "current_stage": "download",
        "completed_stages": [], "config_summary": {}, "retry_of": None,
    }
    service._path(task_id).write_text(json.dumps(record), encoding="utf-8")
    task = service.get(task_id)
    assert task.provider_id == "tushare_official" and task.ledger_status is None
    assert task.transaction_state is None and task.transaction_fields == ()
    assert json.loads(service._path(task_id).read_text())["schema_version"] == "1.0"
    assert TASK_SCHEMA_VERSION == "1.3"


def test_legacy_retry_reuses_successful_migrated_child_without_rewriting_source(tmp_path) -> None:
    service = ResearchTaskService(tmp_path)
    service.root.mkdir(parents=True)
    source_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    child_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    source = {
        "schema_version": "1.0", "task_id": source_id,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:01+00:00", "status": "failed",
        "current_stage": "download", "completed_stages": [],
        "failure_dataset": "stock_basic",
        "failure_range": ["2024-01-30", "2024-01-30"],
        "config_summary": {}, "retry_of": None,
    }
    child = {
        "schema_version": TASK_SCHEMA_VERSION, "task_id": child_id,
        "created_at": "2024-01-01T00:01:00+00:00",
        "updated_at": "2024-01-01T00:02:00+00:00", "status": "succeeded",
        "current_stage": "complete", "completed_stages": ["complete"],
        "config_summary": {}, "retry_of": source_id,
        "run_id": "20240101_000100_snapshot_retry", "result_ready": True,
    }
    service._path(source_id).write_text(json.dumps(source), encoding="utf-8")
    service._path(child_id).write_text(json.dumps(child), encoding="utf-8")

    result = service.retry(source_id, credential=None)

    assert result.task_id == child_id
    assert len(tuple(service.root.glob("*.json"))) == 2
    assert json.loads(service._path(source_id).read_text(encoding="utf-8")) == source
