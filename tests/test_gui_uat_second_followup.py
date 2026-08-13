from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.services.first_run_service import ProgressEvent
from app.services.research_task_service import ResearchTaskService
from app.views.runs import _diagnostic_value, _display_time, _elapsed
from src.data.canonical_store import PartitionedParquetStore, RawParquetStore, content_hash
from src.data.contracts import DataRequirement
from src.data.coverage_ledger import CoverageLedger, CoverageRecord
from src.data.coverage_planner import scope_key
from src.data.dataset_registry import create_default_dataset_registry
from src.data.preparation import DataPreparationService, DataUnavailableError


def _service(tmp_path: Path, client: object, *, attempts: int = 3, delays: list[float] | None = None) -> DataPreparationService:
    return DataPreparationService(
        ledger=CoverageLedger(tmp_path / "metadata" / "catalog.sqlite"),
        curated_store=PartitionedParquetStore(tmp_path / "curated"),
        raw_store=RawParquetStore(tmp_path / "raw"),
        open_dates=lambda _start, _end: ("2023-11-16",),
        client_factory=lambda _token: client,
        network_attempts=attempts,
        retry_sleep=(delays.append if delays is not None else lambda _value: None),
        retry_jitter=lambda: 0.5,
    )


def _suspend_requirement() -> DataRequirement:
    return DataRequirement.create("suspend_d", scope="CN_A", required_start="2023-11-16", required_end="2023-11-16")


def test_empty_event_persists_marker_and_ledger_only_after_marker(tmp_path: Path) -> None:
    class EmptyClient:
        def get_suspend_d(self, **_kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()

    service = _service(tmp_path, EmptyClient())
    requirement = _suspend_requirement()
    service.ensure((requirement,), credential="memory-only")
    spec = service.registry.get("suspend_d")
    marker = service.curated_store.empty_marker_path(spec, unit="2023-11-16", scope=requirement.scope)
    assert marker.is_file()
    assert service.verify_unit("suspend_d", scope="CN_A", unit="2023-11-16")


def test_complete_ledger_without_empty_marker_is_targeted_for_repair(tmp_path: Path) -> None:
    class EmptyClient:
        calls = 0

        def get_suspend_d(self, **_kwargs: object) -> pd.DataFrame:
            self.calls += 1
            return pd.DataFrame()

    client = EmptyClient()
    service = _service(tmp_path, client)
    requirement = _suspend_requirement()
    spec = service.registry.get("suspend_d")
    empty = pd.DataFrame(columns=spec.required_fields)
    service.ledger.mark_complete((CoverageRecord(
        "suspend_d", scope_key(requirement.scope), "2023-11-16", "COMPLETE", 0,
        spec.schema_version, content_hash(spec, empty), "legacy", "2026-01-01T00:00:00+00:00",
    ),))
    assert service.inspect((requirement,))[0].missing_units == ("2023-11-16",)
    service.ensure((requirement,), credential="memory-only")
    assert client.calls == 1
    assert service.verify_unit("suspend_d", scope="CN_A", unit="2023-11-16")


def test_write_marker_failure_never_marks_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyClient:
        def get_suspend_d(self, **_kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()

    service = _service(tmp_path, EmptyClient())
    monkeypatch.setattr(service.curated_store, "write_empty_marker", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(DataUnavailableError):
        service.ensure((_suspend_requirement(),), credential="memory-only")
    assert service.ledger.records("suspend_d") == ()


def test_transient_network_retries_bounded_with_exponential_backoff(tmp_path: Path) -> None:
    delays: list[float] = []

    class FlakyClient:
        calls = 0

        def get_suspend_d(self, **_kwargs: object) -> pd.DataFrame:
            self.calls += 1
            if self.calls < 3:
                raise TimeoutError("read timeout")
            return pd.DataFrame()

    client = FlakyClient()
    service = _service(tmp_path, client, attempts=3, delays=delays)
    service.ensure((_suspend_requirement(),), credential="memory-only")
    assert client.calls == 3
    assert delays == [0.5, 1.0]


def test_exhausted_retry_reports_exact_attempt_count(tmp_path: Path) -> None:
    class OfflineClient:
        calls = 0

        def get_suspend_d(self, **_kwargs: object) -> pd.DataFrame:
            self.calls += 1
            raise TimeoutError("read timeout")

    client = OfflineClient()
    with pytest.raises(DataUnavailableError) as failure:
        _service(tmp_path, client).ensure((_suspend_requirement(),), credential="memory-only")
    assert client.calls == 3
    assert getattr(failure.value.safe_cause, "provider_attempts") == 3


@pytest.mark.parametrize("message", ["invalid token", "permission denied", "rate limit exceeded", "missing required columns"])
def test_deterministic_provider_errors_are_not_retried(tmp_path: Path, message: str) -> None:
    class BadClient:
        calls = 0

        def get_suspend_d(self, **_kwargs: object) -> pd.DataFrame:
            self.calls += 1
            raise RuntimeError(message)

    client = BadClient()
    with pytest.raises(DataUnavailableError):
        _service(tmp_path, client).ensure((_suspend_requirement(),), credential="memory-only")
    assert client.calls == 1


def test_progress_is_monotonic_across_refresh_failure_and_retry(tmp_path: Path) -> None:
    service = ResearchTaskService(tmp_path / "output")
    service.root.mkdir(parents=True)
    task_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    record = {
        "schema_version": "1.0", "task_id": task_id, "name": "x",
        "created_at": "2026-08-12T12:29:13.291333+00:00", "updated_at": "2026-08-12T12:29:13.291333+00:00",
        "status": "running", "current_stage": "download", "completed_stages": [], "result_ready": False,
        "progress_completed": 8, "progress_total": 10, "progress_detail": "daily · 2024-01-10",
    }
    service._write(record)
    service._update_progress(task_id, ProgressEvent("download", "STARTED", completed=3, total=10, detail="retry"))
    reloaded = ResearchTaskService(tmp_path / "output").get(task_id)
    assert (reloaded.progress_completed, reloaded.progress_total) == (8, 10)


def test_active_elapsed_is_recomputed_after_refresh(tmp_path: Path) -> None:
    service = ResearchTaskService(tmp_path / "output")
    service.root.mkdir(parents=True)
    task_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    service._write({
        "schema_version": "1.0", "task_id": task_id, "name": "x",
        "created_at": "2024-01-01T00:00:00+00:00", "updated_at": "2024-01-01T00:00:00+00:00",
        "started_at": "2024-01-01T00:00:00+00:00", "status": "running", "current_stage": "download",
        "completed_stages": [], "result_ready": False, "elapsed_seconds": None,
    })
    assert service.get(task_id).elapsed_seconds is not None
    assert service.get(task_id).elapsed_seconds > 0


def test_time_and_duration_display_are_localized_and_safe() -> None:
    assert _display_time("2026-08-12T12:29:13.291333+00:00", "zh-CN") == "2026-08-12 20:29:13"
    assert _display_time("2026-08-12T12:29:13.291333+00:00", "en").endswith("CST")
    assert _display_time("2026-08-12T12:29:13.291333", "zh-CN") == "2026-08-12 20:29:13"
    assert _display_time("broken", "zh-CN") == "—"
    assert _elapsed(682, "zh-CN") == "11 分 22 秒"
    assert _elapsed(682, "en") == "11 min 22 s"


def test_failure_diagnostics_are_natural_chinese_without_internal_terms() -> None:
    values = (
        _diagnostic_value("ledger", "MISSING", "zh-CN"),
        _diagnostic_value("canonical_status", "MISSING", "zh-CN"),
        _diagnostic_value("canonical_status", "READABLE_ROWS:12", "zh-CN"),
        _diagnostic_value("consistency", "ledger/canonical mismatch", "zh-CN"),
        _diagnostic_value(
            "repair",
            "refetch missing unit and publish canonical proof before marking COMPLETE",
            "zh-CN",
        ),
    )
    assert all("Provider" not in value and "Coverage Ledger" not in value for value in values)
    assert values[2] == "可读取（12 行）"


def test_runs_page_uses_fragment_polling_only_for_active_tasks(tmp_path: Path) -> None:
    output = tmp_path / "output"
    service = ResearchTaskService(output)
    service.root.mkdir(parents=True)
    task_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    (service.root / f"{task_id}.json").write_text(json.dumps({
        "schema_version": "1.0", "task_id": task_id, "name": "x",
        "created_at": "2026-08-12T12:29:13.291333+00:00", "updated_at": "2026-08-12T12:29:13.291333+00:00",
        "status": "running", "current_stage": "download", "completed_stages": [], "result_ready": False,
        "progress_completed": 2, "progress_total": 10, "config_summary": {},
    }), encoding="utf-8")
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "from app.views.runs import render\nfrom app.services.research_task_service import ResearchTaskService\n"
        "import streamlit as st\n"
        "class S(ResearchTaskService):\n"
        " def list_tasks(self, **kwargs): return super().list_tasks(reconcile_interrupted=False)\n"
        f"render(st, service=S({str(output)!r}))\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(str(wrapper), default_timeout=30).run()
    assert not app.exception
    assert any("2026-08-12 20:29:13" in str(item.value) for item in app.caption)
    terminal = json.loads((service.root / f"{task_id}.json").read_text(encoding="utf-8"))
    terminal.update(status="failed", current_stage="download", failure_code="NETWORK_ERROR", failure_stage="download")
    service._write(terminal)
    app.run()
    assert not app.exception
    assert len(tuple(service.root.glob("*.json"))) == 1
    source = (Path(__file__).resolve().parents[1] / "app" / "views" / "runs.py").read_text(encoding="utf-8")
    assert '@st.fragment(run_every="3s")' in source
    assert 'if not any(task.active for task in current)' in source


def test_daily_basic_legacy_schema_reused_unless_dv_ttm_is_required(tmp_path: Path) -> None:
    registry = create_default_dataset_registry()
    service = _service(tmp_path, object())
    spec = registry.get("daily_basic")
    old_spec = replace(spec, required_fields=tuple(field for field in spec.required_fields if field != "dv_ttm"), schema_version="1.0")
    scope = (("scope", "CN_A"),)
    frame = pd.DataFrame([{field: ("600000.SH" if field == "ts_code" else "20231116" if field == "trade_date" else 1.0) for field in old_spec.required_fields}])
    service.curated_store.merge(old_spec, frame, units=("2023-11-16",), scope=scope)
    rows = service.curated_store.rows_for_unit(spec, unit="2023-11-16", scope=scope)
    service.ledger.mark_complete((CoverageRecord("daily_basic", scope_key(scope), "2023-11-16", "COMPLETE", len(rows), "1.0", content_hash(old_spec, rows.loc[:, list(old_spec.required_fields)]), "old", "2026-01-01T00:00:00+00:00"),))
    base = DataRequirement.create("daily_basic", scope="CN_A", required_start="2023-11-16", required_end="2023-11-16", required_fields=("pb",))
    dividend = DataRequirement.create("daily_basic", scope="CN_A", required_start="2023-11-16", required_end="2023-11-16", required_fields=("dv_ttm",))
    assert service.inspect((base,))[0].ready
    assert service.inspect((dividend,))[0].missing_units == ("2023-11-16",)


def test_daily_basic_v11_fetch_supplies_dividend_factor_field(tmp_path: Path) -> None:
    class DividendClient:
        def get_daily_basic(self, **kwargs: object) -> pd.DataFrame:
            fields = create_default_dataset_registry().get("daily_basic").required_fields
            return pd.DataFrame([{
                field: "600000.SH" if field == "ts_code" else kwargs["trade_date"] if field == "trade_date" else 2.0
                for field in fields
            }], columns=fields)

    service = _service(tmp_path, DividendClient())
    requirement = DataRequirement.create(
        "daily_basic", scope="CN_A", required_start="2023-11-16", required_end="2023-11-16",
        required_fields=("dv_ttm",),
    )
    service.ensure((requirement,), credential="memory-only")
    record = service.ledger.records("daily_basic")[0]
    assert record.schema_version == "1.1"
    rows = service.curated_store.rows_for_unit(service.registry.get("daily_basic"), unit="2023-11-16", scope=requirement.scope)
    assert rows["dv_ttm"].tolist() == [2.0]
