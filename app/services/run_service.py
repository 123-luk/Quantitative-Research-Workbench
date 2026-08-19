"""Thin execution boundary from validated configuration to the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping
import re

from src.pipeline.config import PipelineConfig
from src.pipeline.runner import run_pipeline


@dataclass(frozen=True)
class SafeRunError:
    exception_class: str
    message: str
    stage: str | None = None
    run_id: str | None = None
    cause_class: str | None = None
    cause_message: str | None = None
    input_shape: Mapping[str, object] | None = None
    output_shape: Mapping[str, object] | None = None
    retryable: bool = True
    retry_stage: str = "validate"


_SHAPE_KEYS = (
    "input_rows",
    "output_rows",
    "row_count",
    "trade_date_count",
    "date_count",
    "min_trade_date",
    "max_trade_date",
    "first_trade_date",
    "last_trade_date",
    "column_count",
    "columns",
)


def _sanitize_message(value: object) -> str:
    message = str(value)
    if re.search(
        r"(?i)(token|secret|credential|private[ _-]?key|\.env)", message
    ):
        return "Sensitive backend details were redacted."
    return message


def _artifact_shape(directory: Path) -> dict[str, object] | None:
    """Read only bounded row/column/date metadata from one exact Artifact."""
    values: dict[str, object] = {}
    for filename in ("audit.json", "manifest.json"):
        path = directory / filename
        if not path.is_file() or path.is_symlink():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        for key in _SHAPE_KEYS:
            value = payload.get(key)
            if type(value) is int and value >= 0:
                values[key] = value
            elif isinstance(value, str) and len(value) <= 64:
                values[key] = value
            elif key == "columns" and isinstance(value, list):
                columns = [item for item in value if isinstance(item, str)]
                if len(columns) == len(value) and len(columns) <= 64:
                    values[key] = columns
    return values or None


def _stage_shapes(
    config: PipelineConfig,
    run_id: str | None,
    stage: str | None,
) -> tuple[Mapping[str, object] | None, Mapping[str, object] | None]:
    if not run_id or not stage:
        return None, None
    run_dir = Path(config.output_dir).resolve() / "runs" / run_id
    if not run_dir.is_dir() or run_dir.is_symlink():
        return None, None
    experiment_id = config.ml_experiment.experiment_id
    if not isinstance(experiment_id, str) or not experiment_id:
        experiment_id = "__unavailable_experiment__"
    locations = {
        "modeling": (
            run_dir / config.factor_research.artifact_subdir,
            run_dir / config.modeling_panel.output.artifact_subdir,
        ),
        "ml": (
            run_dir / config.modeling_panel.output.artifact_subdir,
            run_dir / config.ml_experiment.artifact_root / experiment_id,
        ),
        "signal": (
            run_dir / config.ml_experiment.artifact_root / experiment_id,
            run_dir / config.signal.artifact_subdir,
        ),
        "portfolio": (
            run_dir / config.signal.artifact_subdir,
            run_dir / config.holdings.artifact_subdir,
        ),
        "research_backtest": (
            run_dir / config.holdings.artifact_subdir,
            run_dir / config.research_backtest.artifact_subdir,
        ),
    }
    pair = locations.get(stage)
    if pair is None:
        return None, None
    return _artifact_shape(pair[0]), _artifact_shape(pair[1])


@dataclass(frozen=True)
class RunOutcome:
    success: bool
    run_id: str | None
    experiment_id: str | None
    status: str
    elapsed_seconds: float
    stage_summary: Mapping[str, object] | None = None
    artifact_summary: Mapping[str, object] | None = None
    error: SafeRunError | None = None


class RunService:
    """Run the existing pipeline once and preserve only its exact returned identity."""

    def __init__(
        self,
        runner: Callable[..., dict[str, object]] | None = None,
        *,
        supports_identity_hook: bool | None = None,
    ) -> None:
        self._runner = run_pipeline if runner is None else runner
        self._supports_identity_hook = runner is None if supports_identity_hook is None else supports_identity_hook

    def run(
        self,
        config: PipelineConfig,
        *,
        stage_callback: Callable[[str, str], None] | None = None,
    ) -> RunOutcome:
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be a validated PipelineConfig.")
        started = perf_counter()
        exact_run_id: str | None = None
        active_stage: str | None = None

        def remember_run(run_dir: Path) -> None:
            nonlocal exact_run_id
            exact_run_id = run_dir.name

        def observe_stage(stage: str, status: str) -> None:
            nonlocal active_stage
            if status == "STARTED":
                active_stage = stage
            if stage_callback is not None:
                stage_callback(stage, status)

        try:
            if self._supports_identity_hook:
                kwargs: dict[str, object] = {"run_created_callback": remember_run}
                if stage_callback is not None:
                    kwargs["stage_callback"] = observe_stage
                summary = self._runner(config, **kwargs)
            else:
                summary = self._runner(config)
            if not isinstance(summary, dict):
                raise TypeError("run_pipeline must return a summary mapping.")
            run_dir = summary.get("run_dir")
            if not isinstance(run_dir, str) or not run_dir.strip():
                raise ValueError("run_pipeline returned no exact run_dir identity.")
            run_id = Path(run_dir).name
            if exact_run_id is not None and run_id != exact_run_id:
                raise ValueError("run_pipeline returned an inconsistent run identity.")
            stage_summary = {
                key: value
                for key, value in summary.items()
                if key not in {"run_dir", "signal", "holdings", "research_backtest"}
            }
            artifact_summary = {
                key: value
                for key, value in summary.items()
                if key in {"signal", "holdings", "research_backtest"}
            }
            experiment_id = None
            ml_summary = summary.get("ml_experiment")
            if isinstance(ml_summary, Mapping):
                value = ml_summary.get("experiment_id")
                experiment_id = value if isinstance(value, str) else None
            if experiment_id is None and config.ml_experiment.enabled:
                experiment_id = config.ml_experiment.experiment_id
            return RunOutcome(
                success=True,
                run_id=run_id,
                experiment_id=experiment_id,
                status=str(summary.get("status", "completed")),
                elapsed_seconds=perf_counter() - started,
                stage_summary=stage_summary,
                artifact_summary=artifact_summary,
            )
        except Exception as exc:
            message = _sanitize_message(exc)
            stage = getattr(exc, "stage", None)
            safe_stage = (
                stage
                if isinstance(stage, str) and stage.strip()
                else active_stage
            )
            failed_run_id = getattr(exc, "run_id", None)
            safe_run_id = (
                failed_run_id
                if isinstance(failed_run_id, str) and failed_run_id.strip()
                else exact_run_id
            )
            cause = exc.__cause__
            input_shape, output_shape = _stage_shapes(
                config, safe_run_id, safe_stage
            )
            return RunOutcome(
                success=False,
                run_id=safe_run_id,
                experiment_id=None,
                status="failed",
                elapsed_seconds=perf_counter() - started,
                error=SafeRunError(
                    exception_class=type(exc).__name__,
                    message=message,
                    stage=safe_stage,
                    run_id=safe_run_id,
                    cause_class=None if cause is None else type(cause).__name__,
                    cause_message=None if cause is None else _sanitize_message(cause),
                    input_shape=input_shape,
                    output_shape=output_shape,
                ),
            )
