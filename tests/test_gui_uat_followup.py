from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
import json
from pathlib import Path
from threading import Event
from time import perf_counter, sleep

import pytest
from streamlit.testing.v1 import AppTest

from app.components.navigation import page_path
from app.services.capability_catalog_service import CapabilityCatalogService
from app.services.credential_service import ProviderErrorKind, classify_provider_error
from app.services.first_run_service import ProgressEvent, WorkbenchRuntime, WorkbenchRunDraft, WorkbenchRunError, WorkbenchErrorCode
from app.services.first_run_service import classify_data_unavailable_error
from app.services.research_date_service import ResearchDateError, shanghai_today, validate_research_dates
from app.services.research_task_service import ResearchTaskService, TaskClearError
from app.services.ui_metadata_service import DISPLAY_VALUES, PARAMETER_HELP_ZH, PARAMETER_NAMES, assert_registry_display_metadata, display_value, factor_label
from src.data.contracts import ResearchFrequency
from src.data.canonical_store import PartitionedParquetStore, RawParquetStore
from src.data.coverage_ledger import CoverageLedger
from src.data.preparation import DataPreparationService
from src.data.preparation import DataUnavailableError
from src.data.canonical_store import CanonicalDataError
from src.pipeline.config import PipelineConfig
from src.universe import UniverseSpec


ROOT = Path(__file__).resolve().parents[1]


def _draft(tmp_path: Path, *, start: str = "2024-01-02", end: str = "2024-01-31") -> WorkbenchRunDraft:
    config = PipelineConfig(
        backtest_start=start, backtest_end=end, train_years=0, max_lookback_months=0,
        stock_pool="CUSTOM:600000.SH", benchmark="000300.SH", strategy_name="uat",
        selected_factors=["bp"], rebalance_frequency="D", top_n=1, transaction_cost=0.0,
        data_root=str(tmp_path / "data"), raw_data_dir=str(tmp_path / "data" / "raw"),
        processed_data_dir=str(tmp_path / "data" / "processed"), cache_dir=str(tmp_path / "data" / "cache"),
        output_dir=str(tmp_path / "output"), parquet_engine="pyarrow", required_datasets=[],
    )
    return WorkbenchRunDraft(config, UniverseSpec.custom(("600000.SH",)), ResearchFrequency.DAILY)


class _GateOrchestrator:
    def __init__(self, gate: Event) -> None:
        self.gate = gate

    def run(self, draft: WorkbenchRunDraft, *, credential: str | None, progress):
        progress(ProgressEvent("download", "STARTED"))
        self.gate.wait(5)
        raise WorkbenchRunError(WorkbenchErrorCode.PIPELINE_ERROR, "pipeline", user_message="safe")


def _wait(service: ResearchTaskService, task_id: str, status: str) -> None:
    deadline = perf_counter() + 15
    while perf_counter() < deadline:
        if service.get(task_id).status == status:
            return
        sleep(0.01)
    raise AssertionError(status)


def test_stable_page_keys_include_research_tasks() -> None:
    assert page_path("runs") == "views/runs.py"
    with pytest.raises(ValueError):
        page_path("unknown")


def test_shanghai_date_rules_and_service_reject_without_writes(tmp_path: Path) -> None:
    assert shanghai_today(datetime(2026, 8, 11, 16, 30, tzinfo=timezone.utc)) == date(2026, 8, 12)
    assert validate_research_dates("2024-01-02", "2024-01-03", today=date(2024, 1, 3)).valid
    assert validate_research_dates("2024-01-02", "2024-01-02", today=date(2024, 1, 3)).code == "order"
    service = ResearchTaskService(tmp_path / "output")
    with pytest.raises(ResearchDateError):
        service.submit(_draft(tmp_path, start="2035-01-01", end="2035-01-02"), credential=None)
    assert not service.root.exists()


def test_submit_idempotency_and_active_clear_guard(tmp_path: Path) -> None:
    gate = Event()
    service = ResearchTaskService(tmp_path / "output", orchestrator_factory=lambda: _GateOrchestrator(gate))
    first = service.submit(_draft(tmp_path), credential="SESSION_SECRET")
    second = service.submit(_draft(tmp_path), credential="SESSION_SECRET")
    assert first.task_id == second.task_id
    with pytest.raises(TaskClearError):
        service.clear(first.task_id)
    gate.set()
    _wait(service, first.task_id, "failed")
    assert service.get(first.task_id).failure_code == WorkbenchErrorCode.PIPELINE_ERROR.value


def test_clear_failed_is_safe_idempotent_and_scoped(tmp_path: Path) -> None:
    service = ResearchTaskService(tmp_path / "output")
    service.root.mkdir(parents=True)
    shared = tmp_path / "data" / "metadata" / "catalog.sqlite"
    shared.parent.mkdir(parents=True)
    shared.write_text("shared", encoding="utf-8")
    task_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    other_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    base = {
        "schema_version": "1.0", "name": "x", "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00", "status": "failed", "current_stage": "download",
        "completed_stages": [], "result_ready": False,
    }
    for value in (task_id, other_id):
        (service.root / f"{value}.json").write_text(json.dumps({**base, "task_id": value}), encoding="utf-8")
    assert service.clear(task_id).removed
    assert not service.clear(task_id).removed
    assert (service.root / f"{other_id}.json").exists()
    assert shared.read_text(encoding="utf-8") == "shared"


def test_clear_success_preserves_exact_run(tmp_path: Path) -> None:
    service = ResearchTaskService(tmp_path / "output")
    service.root.mkdir(parents=True)
    run_id = "20240101_010101_uat_custom"
    run_dir = tmp_path / "output" / "runs" / run_id
    run_dir.mkdir(parents=True)
    marker = run_dir / "artifact.marker"
    marker.write_text("keep", encoding="utf-8")
    task_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    record = {
        "schema_version": "1.0", "task_id": task_id, "name": "x", "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00", "status": "succeeded", "current_stage": "complete",
        "completed_stages": ["complete"], "result_ready": True, "run_id": run_id,
    }
    (service.root / f"{task_id}.json").write_text(json.dumps(record), encoding="utf-8")
    result = service.clear(task_id)
    assert result.results_preserved and result.run_id == run_id
    assert marker.read_text(encoding="utf-8") == "keep"


def test_clear_historical_is_idempotent_hide_only(tmp_path: Path) -> None:
    service = ResearchTaskService(tmp_path / "output")
    run_id = "20240101_010101_legacy_custom"
    run_dir = tmp_path / "output" / "runs" / run_id
    run_dir.mkdir(parents=True)
    marker = run_dir / "legacy.marker"
    marker.write_text("preserved", encoding="utf-8")
    assert service.clear_historical_run(run_id)
    assert not service.clear_historical_run(run_id)
    assert run_id in service.hidden_run_ids()
    assert marker.read_text(encoding="utf-8") == "preserved"
    with pytest.raises((FileNotFoundError, ValueError)):
        service.clear_historical_run("..\\escape")


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("invalid token", ProviderErrorKind.AUTHENTICATION_INVALID),
        ("permission denied", ProviderErrorKind.PERMISSION_INSUFFICIENT),
        ("insufficient points", ProviderErrorKind.POINTS_INSUFFICIENT),
        ("rate limit exceeded", ProviderErrorKind.RATE_LIMITED),
        ("connection timeout", ProviderErrorKind.NETWORK_ERROR),
        ("missing required columns", ProviderErrorKind.RESPONSE_INVALID),
        ("provider unavailable", ProviderErrorKind.PROVIDER_ERROR),
    ],
)
def test_provider_failure_categories(message: str, kind: ProviderErrorKind) -> None:
    assert classify_provider_error(RuntimeError(message)) is kind


def test_malformed_provider_response_is_not_classified_as_network() -> None:
    failure = DataUnavailableError(
        "safe", dataset_id="suspend_d", units=("2023-11-16",),
        cause=CanonicalDataError("suspend_d rows are missing required columns"),
    )
    assert classify_data_unavailable_error(failure) is WorkbenchErrorCode.PROVIDER_RESPONSE_INVALID


def test_local_ledger_failure_is_distinct_from_provider_structure() -> None:
    failure = DataUnavailableError(
        "safe", dataset_id="suspend_d", units=("2023-11-16",),
        cause=CanonicalDataError("canonical partition verification failed"), origin="local",
    )
    assert classify_data_unavailable_error(failure) is WorkbenchErrorCode.COVERAGE_VALIDATION


def test_registry_display_metadata_is_complete_and_no_internal_chinese_labels() -> None:
    catalog = CapabilityCatalogService()
    assert_registry_display_metadata(
        models=catalog.list_model_names(), portfolios=catalog.list_portfolio_methods(), risks=catalog.list_risk_estimators()
    )
    for value in ("research_workbench", "CUSTOM", "elastic_net", "equal_weight", "descending"):
        assert display_value(value, "zh-CN") not in {value, value.replace("_", " ").title()}
        assert display_value(value, "en")
    parameter_names = {item.name for model in catalog.list_model_names() for item in catalog.get_model_parameter_schema(model)}
    assert parameter_names <= set(PARAMETER_NAMES)
    assert parameter_names <= set(PARAMETER_HELP_ZH)
    assert factor_label("bp", "zh-CN") == "市净率倒数（BP）"


def test_first_run_runtime_uses_canonical_empty_event_path(tmp_path: Path) -> None:
    class EmptyProvider:
        calls = 0

        def get_suspend_d(self, **kwargs: object):
            import pandas as pd
            self.calls += 1
            return pd.DataFrame()

    provider = EmptyProvider()
    runtime = WorkbenchRuntime(
        _draft(tmp_path).pipeline_config,
        root=tmp_path,
        client_factory=lambda _token: provider,
    )
    service = runtime.preparation(open_dates=lambda _start, _end: ("2024-01-02",))
    from src.data.contracts import DataRequirement
    requirement = DataRequirement.create("suspend_d", scope="CN_A", required_start="2024-01-02", required_end="2024-01-02")
    assert service.ensure((requirement,), credential="memory-only").provider_calls == 1
    assert service.ensure((requirement,), credential=None).provider_calls == 0
    assert provider.calls == 1


def test_token_never_enters_task_or_shared_stores(tmp_path: Path) -> None:
    secret = "SESSION_ONLY_TOKEN_NEVER_PERSIST_UAT025"
    gate = Event()
    service = ResearchTaskService(tmp_path / "output", orchestrator_factory=lambda: _GateOrchestrator(gate))
    task = service.submit(_draft(tmp_path), credential=secret)
    _wait(service, task.task_id, "running")
    payload = (service.root / f"{task.task_id}.json").read_text(encoding="utf-8")
    assert secret not in payload
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".log", ".txt", ".sqlite", ".yaml", ".yml"}:
            assert secret.encode() not in path.read_bytes()
    gate.set()
    _wait(service, task.task_id, "failed")


def test_new_run_apptest_date_bounds_localized_values_and_disabled_submit() -> None:
    app = AppTest.from_file(str(ROOT / "app" / "views" / "new_run.py"), default_timeout=30).run()
    assert not app.exception
    assert app.date_input[0].max == shanghai_today()
    assert app.date_input[1].max == shanghai_today()
    app.date_input[1].set_value(app.date_input[0].value).run()
    assert any("严格晚于" in item.value for item in app.error)
    run_button = next(item for item in app.button if item.label == "运行研究")
    assert run_button.disabled
    visible = " ".join(str(item.label) for item in (*app.selectbox, *app.checkbox))
    for leaked in ("elastic_net", "equal_weight", "descending", "allow_partial"):
        assert leaked not in visible


def test_runs_apptest_has_clear_action_and_sanitized_diagnostics(tmp_path: Path) -> None:
    service = ResearchTaskService(tmp_path / "output")
    service.root.mkdir(parents=True)
    task_id = "dddddddd-dddd-dddd-dddd-dddddddddddd"
    record = {
        "schema_version": "1.0", "task_id": task_id, "name": "research_workbench", "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00", "status": "failed", "current_stage": "download",
        "completed_stages": ["check"], "result_ready": False, "failure_code": "NETWORK_ERROR", "failure_stage": "download",
        "failure_message": "C:\\secret\\trace TOKEN=never-show", "failure_dataset": "suspend_d",
        "failure_range": ["2023-11-16", "2023-11-16"],
        "config_summary": {"research_start": "2023-11-16", "research_end": "2024-01-01", "universe_type": "CUSTOM", "model": "elastic_net", "portfolio_method": "equal_weight", "factors": ["bp"]},
    }
    (service.root / f"{task_id}.json").write_text(json.dumps(record), encoding="utf-8")
    wrapper = tmp_path / "runs_wrapper.py"
    wrapper.write_text(
        "import streamlit as st\n"
        "from app.services.research_task_service import ResearchTaskService\n"
        "from app.views.runs import render\n"
        f"render(st, service=ResearchTaskService({str(tmp_path / 'output')!r}))\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(str(wrapper), default_timeout=30).run()
    assert not app.exception
    visible = " ".join(str(getattr(item, "value", "")) + str(getattr(item, "label", "")) for item in (*app.markdown, *app.error, *app.button))
    assert "TOKEN=never-show" not in visible and "C:\\secret" not in visible
    assert "弹性网络" in visible and "等权重" in visible
    assert any(item.label == "清除记录" for item in app.button)
    next(item for item in app.button if item.label == "清除记录").click().run()
    assert any(item.label == "确认清除" for item in app.button)
    next(item for item in app.button if item.label == "确认清除").click().run()
    assert not (service.root / f"{task_id}.json").exists()


def test_runs_apptest_disables_clear_for_running_task(tmp_path: Path) -> None:
    service = ResearchTaskService(tmp_path / "output")
    service.root.mkdir(parents=True)
    task_id = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
    record = {
        "schema_version": "1.0", "task_id": task_id, "name": "x",
        "created_at": "2024-01-01T00:00:00+00:00", "updated_at": "2024-01-01T00:00:00+00:00",
        "status": "running", "current_stage": "download", "completed_stages": [], "result_ready": False,
        "config_summary": {"research_start": "2023-01-01", "research_end": "2024-01-01"},
    }
    (service.root / f"{task_id}.json").write_text(json.dumps(record), encoding="utf-8")
    wrapper = tmp_path / "running_wrapper.py"
    wrapper.write_text(
        "import streamlit as st\nfrom app.services.research_task_service import ResearchTaskService\n"
        "from app.views.runs import render\n"
        "class ActiveTaskService(ResearchTaskService):\n"
        "    def list_tasks(self, *, reconcile_interrupted=True):\n"
        "        return super().list_tasks(reconcile_interrupted=False)\n"
        f"render(st, service=ActiveTaskService({str(tmp_path / 'output')!r}))\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(str(wrapper), default_timeout=30).run()
    clear = next(item for item in app.button if item.label == "清除记录")
    assert clear.disabled
    assert any("不能清除" in item.value for item in app.caption)
