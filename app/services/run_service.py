"""Thin execution boundary from validated configuration to the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
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

    def run(self, config: PipelineConfig) -> RunOutcome:
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be a validated PipelineConfig.")
        started = perf_counter()
        exact_run_id: str | None = None

        def remember_run(run_dir: Path) -> None:
            nonlocal exact_run_id
            exact_run_id = run_dir.name

        try:
            summary = (
                self._runner(config, run_created_callback=remember_run)
                if self._supports_identity_hook
                else self._runner(config)
            )
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
            message = str(exc)
            if re.search(
                r"(?i)(token|secret|credential|private[ _-]?key|\.env)", message
            ):
                message = "Sensitive backend details were redacted."
            stage = getattr(exc, "stage", None)
            safe_stage = stage if isinstance(stage, str) and stage.strip() else None
            failed_run_id = getattr(exc, "run_id", None)
            safe_run_id = (
                failed_run_id
                if isinstance(failed_run_id, str) and failed_run_id.strip()
                else exact_run_id
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
                ),
            )
