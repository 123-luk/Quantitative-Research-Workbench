from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from time import perf_counter, sleep

import pytest

from app.services.first_run_service import (
    FirstRunResult,
    ProgressEvent,
    WorkbenchErrorCode,
    WorkbenchRunDraft,
    WorkbenchRunError,
)
from app.services.research_task_service import ResearchTaskService
from app.services.run_service import RunOutcome
from app.services.ui_metadata_service import PARAMETERS, dataset_label, factor_explanations
from app.services.first_run_service import create_workbench_factor_registry
from src.data.contracts import ResearchFrequency
from src.data.dataset_registry import create_default_dataset_registry
from src.data.coverage_planner import FetchTask
from src.data.fetching import validate_complete
from src.pipeline.config import PipelineConfig
from src.universe import UniverseSpec


def _draft(tmp_path: Path) -> WorkbenchRunDraft:
    config = PipelineConfig(
        backtest_start="2024-01-02", backtest_end="2024-01-31", train_years=0,
        max_lookback_months=0, stock_pool="CUSTOM:600000.SH", benchmark="000300.SH",
        strategy_name="uat", selected_factors=["bp"], rebalance_frequency="D",
        top_n=1, transaction_cost=0.0, data_root=str(tmp_path / "data"),
        raw_data_dir=str(tmp_path / "data" / "raw"), processed_data_dir=str(tmp_path / "data" / "processed"),
        cache_dir=str(tmp_path / "data" / "cache"), output_dir=str(tmp_path / "output"),
        parquet_engine="pyarrow", required_datasets=[],
    )
    return WorkbenchRunDraft(config, UniverseSpec.custom(("600000.SH",)), ResearchFrequency.DAILY)


class _BlockingOrchestrator:
    def __init__(self, gate: Event, *, fail: WorkbenchRunError | None = None) -> None:
        self.gate = gate
        self.fail = fail

    def run(self, draft: WorkbenchRunDraft, *, credential: str | None, progress):
        progress(ProgressEvent("download", "STARTED", completed=0, total=2, detail="missing only"))
        assert self.gate.wait(5)
        if self.fail:
            raise self.fail
        progress(ProgressEvent("download", "COMPLETE", completed=2, total=2))
        raise WorkbenchRunError(WorkbenchErrorCode.PIPELINE_ERROR, "pipeline", user_message="fixture stopped before pipeline")


def _wait(service: ResearchTaskService, task_id: str, status: str) -> object:
    deadline = perf_counter() + 15
    while perf_counter() < deadline:
        task = service.get(task_id)
        if task.status == status:
            return task
        sleep(0.01)
    raise AssertionError(f"task did not reach {status}")


def test_submit_returns_immediately_is_idempotent_and_persists_real_progress(tmp_path: Path) -> None:
    gate = Event()
    service = ResearchTaskService(tmp_path / "output", orchestrator_factory=lambda: _BlockingOrchestrator(gate))
    draft = _draft(tmp_path)
    started = perf_counter()
    first = service.submit(draft, credential="SESSION_ONLY_SECRET")
    assert perf_counter() - started < 0.5
    second = service.submit(draft, credential="SESSION_ONLY_SECRET")
    assert second.task_id == first.task_id
    try:
        running = _wait(service, first.task_id, "running")
        assert running.current_stage == "download"
        assert (running.progress_completed, running.progress_total) == (0, 2)
    finally:
        gate.set()
    failed = _wait(service, first.task_id, "failed")
    assert failed.failure_stage == "pipeline" and not failed.can_open_results


def test_task_record_is_atomic_json_and_never_persists_token(tmp_path: Path) -> None:
    secret = "SESSION_ONLY_SECRET_UAT"
    gate = Event()
    service = ResearchTaskService(tmp_path / "output", orchestrator_factory=lambda: _BlockingOrchestrator(gate))
    task = service.submit(_draft(tmp_path), credential=secret)
    _wait(service, task.task_id, "running")
    paths = tuple((tmp_path / "output" / "workbench_tasks").iterdir())
    assert len([path for path in paths if path.suffix == ".json"]) == 1
    payload = (tmp_path / "output" / "workbench_tasks" / f"{task.task_id}.json").read_text(encoding="utf-8")
    assert secret not in payload
    assert json.loads(payload)["schema_version"] == "1.0"
    gate.set()


@pytest.mark.parametrize(
    ("code", "stage"),
    [
        (WorkbenchErrorCode.AUTHENTICATION_INVALID, "download"),
        (WorkbenchErrorCode.PERMISSION_INSUFFICIENT, "download"),
        (WorkbenchErrorCode.RATE_LIMITED, "download"),
        (WorkbenchErrorCode.NETWORK_ERROR, "download"),
        (WorkbenchErrorCode.PROVIDER_EMPTY, "download"),
    ],
)
def test_failure_mapping_is_persistent_and_never_result_ready(tmp_path: Path, code: WorkbenchErrorCode, stage: str) -> None:
    gate = Event()
    failure = WorkbenchRunError(code, stage, dataset_id="daily_basic", missing_range=("2024-01-02", "2024-01-03"), user_message="safe diagnostic")
    service = ResearchTaskService(tmp_path / "output", orchestrator_factory=lambda: _BlockingOrchestrator(gate, fail=failure))
    task = service.submit(_draft(tmp_path), credential="secret")
    gate.set()
    failed = _wait(service, task.task_id, "failed")
    assert failed.failure_code == code.value
    assert failed.failure_message == "safe diagnostic"
    assert not failed.result_ready and not failed.can_open_results


def test_units_risk_free_boundary_and_factor_registry_metadata() -> None:
    assert PARAMETERS["annual_risk_free_rate"].input_scale == "2 means 2%"
    assert 2.0 / 100.0 == 0.02
    assert PARAMETERS["initial_nav"].unit_en == "dimensionless"
    assert dataset_label("daily_basic", "zh-CN") == "每日基本面指标"
    registry = create_workbench_factor_registry()
    rows = factor_explanations(registry, ResearchFrequency.DAILY)
    assert {item.code for item in rows} == {
        item.name for item in registry.list_metadata() if any(spec.research_frequency is ResearchFrequency.DAILY for spec in item.frequency_specs)
    }
    bp = next(item for item in rows if item.code == "bp")
    assert bp.formula == "1 / pb (positive values only)"
    assert bp.source_fields == ("pb",)


def test_empty_event_snapshot_is_valid_complete_with_canonical_schema() -> None:
    spec = create_default_dataset_registry().get("suspend_d")
    task = FetchTask("suspend_d", (("scope", "CN_A"),), ("2024-01-02",), "2024-01-02", "2024-01-02")
    result = validate_complete(spec, task, __import__("pandas").DataFrame())
    assert result.empty
    assert tuple(result.columns) == spec.required_fields
