"""Persistent, non-blocking lifecycle for Workbench research tasks."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Lock, RLock
from typing import Callable, Mapping
from uuid import uuid4

from app.services.first_run_service import (
    FirstRunOrchestrator,
    ProgressEvent,
    WorkbenchRunDraft,
    WorkbenchRunError,
)
from app.services.result_service import ResultService
from app.services.research_date_service import require_valid_research_dates
from src.data.contracts import ResearchFrequency
from src.pipeline.config import PipelineConfig
from src.pipeline.experiment import ExperimentManager
from src.universe import UniverseSpec


TASK_SCHEMA_VERSION = "1.0"
ACTIVE_STATUSES = frozenset({"created", "running"})
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _elapsed(started_at: str | None, finished_at: str | None = None) -> float | None:
    if not started_at:
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat((finished_at or _now()).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (end - start).total_seconds())


def _safe_name(draft: WorkbenchRunDraft) -> str:
    config = draft.pipeline_config
    return f"{config.strategy_name} · {config.backtest_start} – {config.backtest_end}"


def _summary(draft: WorkbenchRunDraft) -> dict[str, object]:
    config = draft.pipeline_config
    portfolio = config.holdings.portfolio_construction
    return {
        "universe_type": draft.universe_spec.universe_type.value,
        "universe": draft.universe_spec.to_dict(),
        "research_start": config.backtest_start,
        "research_end": config.backtest_end,
        "frequency": draft.research_frequency.value,
        "factors": list(config.selected_factors),
        "model": config.ml_experiment.experiment.training_config.model_name
        if config.ml_experiment.enabled and config.ml_experiment.experiment is not None
        else None,
        "top_n": config.holdings.top_n,
        "portfolio_method": portfolio.method,
        "benchmark": config.research_backtest.benchmark.benchmark_code
        if config.research_backtest.enabled
        else config.benchmark,
    }


@dataclass(frozen=True)
class ResearchTask:
    task_id: str
    name: str
    created_at: str
    updated_at: str
    status: str
    current_stage: str
    completed_stages: tuple[str, ...]
    progress_completed: int | None
    progress_total: int | None
    progress_detail: str | None
    run_id: str | None
    failure_code: str | None
    failure_stage: str | None
    failure_message: str | None
    failure_dataset: str | None
    failure_range: tuple[str, str] | None
    result_ready: bool
    config_summary: Mapping[str, object]
    started_at: str | None
    finished_at: str | None
    elapsed_seconds: float | None
    retry_of: str | None
    historical: bool = False

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def can_open_results(self) -> bool:
        return self.status == "succeeded" and self.result_ready and bool(self.run_id)

    @property
    def can_clear(self) -> bool:
        return not self.active


@dataclass(frozen=True)
class TaskClearResult:
    task_id: str
    removed: bool
    run_id: str | None
    results_preserved: bool


class TaskClearError(RuntimeError):
    pass


_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-workbench")
_LOCK = Lock()
_IO_LOCK = RLock()
_SUBMIT_LOCK = RLock()
_ACTIVE: dict[str, Future[object]] = {}


class ResearchTaskService:
    """Create, persist, resume diagnostics for, and query background tasks."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        orchestrator_factory: Callable[[], FirstRunOrchestrator] | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.root = self.output_root / "workbench_tasks"
        self.hidden_runs_root = self.root / "hidden_runs"
        self._orchestrator_factory = orchestrator_factory or FirstRunOrchestrator

    def _path(self, task_id: str) -> Path:
        if not isinstance(task_id, str) or not task_id or any(char not in "0123456789abcdef-" for char in task_id):
            raise ValueError("Invalid task_id.")
        root = self.root.resolve()
        path = (root / f"{task_id}.json").resolve()
        if path.parent != root:
            raise ValueError("Task path escaped the task storage root.")
        return path

    def _write(self, record: Mapping[str, object]) -> None:
        with _IO_LOCK:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self._path(str(record["task_id"]))
            temp = self.root / f".{path.name}.{uuid4().hex}.tmp"
            payload = json.dumps(dict(record), ensure_ascii=False, sort_keys=True, indent=2)
            try:
                temp.write_text(payload, encoding="utf-8")
                os.replace(temp, path)
            finally:
                if temp.exists():
                    temp.unlink()

    def _read_raw(self, task_id: str) -> dict[str, object]:
        path = self._path(task_id)
        with _IO_LOCK:
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(task_id)
            value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != TASK_SCHEMA_VERSION:
            raise ValueError("Unsupported task record schema.")
        return value

    @staticmethod
    def _view(value: Mapping[str, object]) -> ResearchTask:
        return ResearchTask(
            task_id=str(value["task_id"]),
            name=str(value.get("name") or "Historical research task"),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            status=str(value["status"]),
            current_stage=str(value.get("current_stage") or "created"),
            completed_stages=tuple(str(item) for item in value.get("completed_stages", ())),
            progress_completed=value.get("progress_completed") if type(value.get("progress_completed")) is int else None,
            progress_total=value.get("progress_total") if type(value.get("progress_total")) is int else None,
            progress_detail=value.get("progress_detail") if isinstance(value.get("progress_detail"), str) else None,
            run_id=value.get("run_id") if isinstance(value.get("run_id"), str) else None,
            failure_code=value.get("failure_code") if isinstance(value.get("failure_code"), str) else None,
            failure_stage=value.get("failure_stage") if isinstance(value.get("failure_stage"), str) else None,
            failure_message=value.get("failure_message") if isinstance(value.get("failure_message"), str) else None,
            failure_dataset=value.get("failure_dataset") if isinstance(value.get("failure_dataset"), str) else None,
            failure_range=tuple(value["failure_range"]) if isinstance(value.get("failure_range"), list) and len(value["failure_range"]) == 2 else None,  # type: ignore[arg-type]
            result_ready=value.get("result_ready") is True,
            config_summary=value.get("config_summary") if isinstance(value.get("config_summary"), Mapping) else {},
            started_at=value.get("started_at") if isinstance(value.get("started_at"), str) else None,
            finished_at=value.get("finished_at") if isinstance(value.get("finished_at"), str) else None,
            elapsed_seconds=float(value["elapsed_seconds"]) if isinstance(value.get("elapsed_seconds"), (int, float)) else None,
            retry_of=value.get("retry_of") if isinstance(value.get("retry_of"), str) else None,
        )

    def get(self, task_id: str) -> ResearchTask:
        return self._view(self._read_raw(task_id))

    def list_tasks(self, *, reconcile_interrupted: bool = True) -> tuple[ResearchTask, ...]:
        if not self.root.is_dir():
            return ()
        tasks: list[ResearchTask] = []
        for path in self.root.iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix != ".json" or path.name.startswith("."):
                continue
            try:
                task = self.get(path.stem)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if reconcile_interrupted and task.active:
                with _LOCK:
                    future = _ACTIVE.get(task.task_id)
                if future is None:
                    raw = self._read_raw(task.task_id)
                    finished = _now()
                    raw.update({
                        "status": "failed",
                        "current_stage": task.current_stage,
                        "failure_code": "PROCESS_INTERRUPTED",
                        "failure_stage": task.current_stage,
                        "failure_message": "The Workbench process stopped before the task finished. Re-enter the token if missing data still needs downloading, then retry.",
                        "updated_at": finished,
                        "finished_at": finished,
                        "elapsed_seconds": _elapsed(task.started_at, finished),
                    })
                    self._write(raw)
                    task = self._view(raw)
            tasks.append(task)
        return tuple(sorted(tasks, key=lambda item: (item.created_at, item.task_id), reverse=True))

    @staticmethod
    def _fingerprint(draft: WorkbenchRunDraft) -> str:
        payload = {
            "pipeline_config": draft.pipeline_config.to_dict(),
            "universe_spec": draft.universe_spec.to_dict(),
            "research_frequency": draft.research_frequency.value,
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    def submit(self, draft: WorkbenchRunDraft, *, credential: str | None, retry_of: str | None = None) -> ResearchTask:
        if not isinstance(draft, WorkbenchRunDraft):
            raise TypeError("draft must be a WorkbenchRunDraft.")
        require_valid_research_dates(
            draft.pipeline_config.backtest_start,
            draft.pipeline_config.backtest_end,
        )
        fingerprint = self._fingerprint(draft)
        with _SUBMIT_LOCK:
            for task in self.list_tasks(reconcile_interrupted=False):
                if task.active:
                    raw = self._read_raw(task.task_id)
                    if raw.get("request_fingerprint") == fingerprint:
                        return task
            task_id = str(uuid4())
            created = _now()
            record: dict[str, object] = {
            "schema_version": TASK_SCHEMA_VERSION,
            "task_id": task_id,
            "name": _safe_name(draft),
            "created_at": created,
            "updated_at": created,
            "started_at": None,
            "finished_at": None,
            "elapsed_seconds": None,
            "status": "created",
            "current_stage": "created",
            "completed_stages": [],
            "progress_completed": None,
            "progress_total": None,
            "progress_detail": None,
            "run_id": None,
            "result_ready": False,
            "failure_code": None,
            "failure_stage": None,
            "failure_message": None,
            "failure_dataset": None,
            "failure_range": None,
            "config_summary": _summary(draft),
            "pipeline_config": draft.pipeline_config.to_dict(),
            "universe_spec": draft.universe_spec.to_dict(),
            "research_frequency": draft.research_frequency.value,
            "request_fingerprint": fingerprint,
            "retry_of": retry_of,
            }
            self._write(record)
            future = _EXECUTOR.submit(self._run, task_id, draft, credential)
            with _LOCK:
                _ACTIVE[task_id] = future
            future.add_done_callback(lambda _: self._forget(task_id))
        return self.get(task_id)

    @staticmethod
    def _forget(task_id: str) -> None:
        with _LOCK:
            _ACTIVE.pop(task_id, None)

    def retry(self, task_id: str, *, credential: str | None) -> ResearchTask:
        raw = self._read_raw(task_id)
        if str(raw.get("status")) not in TERMINAL_STATUSES:
            return self._view(raw)
        draft = WorkbenchRunDraft(
            PipelineConfig.from_dict(raw["pipeline_config"]),  # type: ignore[arg-type]
            UniverseSpec.from_dict(raw["universe_spec"]),  # type: ignore[arg-type]
            ResearchFrequency(str(raw["research_frequency"])),
        )
        return self.submit(draft, credential=credential, retry_of=task_id)

    def clear(self, task_id: str) -> TaskClearResult:
        """Remove only one terminal task record; exact run artifacts are retained."""
        try:
            raw = self._read_raw(task_id)
        except FileNotFoundError:
            return TaskClearResult(task_id, False, None, True)
        task = self._view(raw)
        if task.active:
            raise TaskClearError("Active research tasks cannot be cleared.")
        path = self._path(task_id)
        with _IO_LOCK:
            if path.is_symlink():
                raise TaskClearError("Task record must not be a symbolic link.")
            try:
                path.unlink()
            except FileNotFoundError:
                return TaskClearResult(task_id, False, task.run_id, True)
            except OSError as exc:
                raise TaskClearError("The task record could not be cleared safely.") from exc
        return TaskClearResult(task_id, True, task.run_id, True)

    def _hidden_run_path(self, run_id: str) -> Path:
        ExperimentManager(self.output_root).resolve_run_dir(run_id)
        root = self.hidden_runs_root.resolve()
        path = (root / f"{run_id}.json").resolve()
        if path.parent != root:
            raise ValueError("Hidden-run marker escaped its storage root.")
        return path

    def hidden_run_ids(self) -> frozenset[str]:
        if not self.hidden_runs_root.is_dir() or self.hidden_runs_root.is_symlink():
            return frozenset()
        result: set[str] = set()
        for path in self.hidden_runs_root.iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                continue
            try:
                self._hidden_run_path(path.stem)
            except (FileNotFoundError, ValueError):
                continue
            result.add(path.stem)
        return frozenset(result)

    def clear_historical_run(self, run_id: str) -> bool:
        """Hide one validated historical run record without deleting its files."""
        path = self._hidden_run_path(run_id)
        with _IO_LOCK:
            if path.exists():
                if path.is_symlink() or not path.is_file():
                    raise TaskClearError("Historical record marker is unsafe.")
                return False
            self.hidden_runs_root.mkdir(parents=True, exist_ok=True)
            temp = self.hidden_runs_root / f".{path.name}.{uuid4().hex}.tmp"
            try:
                temp.write_text(json.dumps({"run_id": run_id, "hidden_at": _now()}), encoding="utf-8")
                os.replace(temp, path)
            except OSError as exc:
                raise TaskClearError("The historical record could not be cleared safely.") from exc
            finally:
                if temp.exists():
                    temp.unlink()
        return True

    def _update_progress(self, task_id: str, event: ProgressEvent) -> None:
        raw = self._read_raw(task_id)
        completed = list(raw.get("completed_stages", ()))
        if event.status in {"COMPLETE", "SKIPPED"} and event.stage not in completed:
            completed.append(event.stage)
        raw.update({
            "status": "running",
            "current_stage": event.stage,
            "completed_stages": completed,
            "progress_completed": event.completed,
            "progress_total": event.total,
            "progress_detail": event.detail,
            "updated_at": _now(),
        })
        self._write(raw)

    def _run(self, task_id: str, draft: WorkbenchRunDraft, credential: str | None) -> None:
        started = _now()
        raw = self._read_raw(task_id)
        raw.update({"status": "running", "current_stage": "validating", "started_at": started, "updated_at": started})
        self._write(raw)
        try:
            result = self._orchestrator_factory().run(
                draft,
                credential=credential,
                progress=lambda event: self._update_progress(task_id, event),
            )
            run_id = result.run.run_id
            ready = bool(run_id)
            if run_id:
                ResultService(draft.pipeline_config.output_dir).load(run_id)
            finished = _now()
            raw = self._read_raw(task_id)
            raw.update({
                "status": "succeeded",
                "current_stage": "complete",
                "run_id": run_id,
                "result_ready": ready,
                "provider_calls": result.provider_calls,
                "updated_at": finished,
                "finished_at": finished,
                "elapsed_seconds": _elapsed(started, finished),
            })
            self._write(raw)
        except WorkbenchRunError as exc:
            finished = _now()
            raw = self._read_raw(task_id)
            raw.update({
                "status": "failed",
                "current_stage": exc.stage,
                "run_id": exc.run_id,
                "result_ready": False,
                "failure_code": exc.code.value,
                "failure_stage": exc.stage,
                "failure_message": exc.user_message,
                "failure_dataset": exc.dataset_id,
                "failure_range": list(exc.missing_range) if exc.missing_range else None,
                "updated_at": finished,
                "finished_at": finished,
                "elapsed_seconds": _elapsed(started, finished),
            })
            self._write(raw)
        except Exception:
            finished = _now()
            raw = self._read_raw(task_id)
            raw.update({
                "status": "failed",
                "failure_code": "INTERNAL_ERROR",
                "failure_stage": str(raw.get("current_stage") or "unknown"),
                "failure_message": "An internal error stopped this task. Technical details were withheld; retry or review application logs.",
                "updated_at": finished,
                "finished_at": finished,
                "elapsed_seconds": _elapsed(started, finished),
            })
            self._write(raw)
