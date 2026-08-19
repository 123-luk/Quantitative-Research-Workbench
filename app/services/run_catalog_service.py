"""Deterministic, canonical-metadata-backed run catalog."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import yaml

from src.pipeline.config import PipelineConfig
from src.pipeline.experiment import ExperimentManager
from src.research_backtest.analytics import PERFORMANCE_METRIC_KEYS
from src.research_backtest.artifacts import ResearchBacktestArtifactStore


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    created_at: str | None
    status: str | None
    model: str | None
    top_n: int | None
    portfolio_method: str | None
    benchmark: str | None
    backtest_status: str
    net_total_return: float | None
    net_sharpe_ratio: float | None


def _json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _config(path: Path) -> tuple[dict[str, object], PipelineConfig] | None:
    if not path.exists() or path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
        if not isinstance(raw, dict):
            return None
        return raw, PipelineConfig.from_dict(raw)
    except Exception:
        return None


def _canonical_created_at(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value


def _created_at_sort_key(value: str | None) -> datetime:
    if value is None:
        return datetime.min
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


class RunCatalogService:
    """Enumerate exact runs without mtime, latest aliases, or fallback paths."""

    def __init__(self, output_root: str | Path) -> None:
        self._manager = ExperimentManager(output_root)

    def list_runs(self) -> tuple[RunSummary, ...]:
        summaries: list[RunSummary] = []
        for run_id in self._manager.list_run_ids():
            run_dir = self._manager.resolve_run_dir(run_id)
            info = _json(run_dir / "run_info.json") or {}
            config_result = _config(run_dir / "config_snapshot.yaml")
            raw = None if config_result is None else config_result[0]
            config = None if config_result is None else config_result[1]
            model: str | None = None
            if isinstance(raw, Mapping):
                ml = raw.get("ml_experiment")
                if isinstance(ml, Mapping):
                    experiment = ml.get("experiment")
                    if isinstance(experiment, Mapping):
                        training = experiment.get("training")
                        if isinstance(training, Mapping) and isinstance(training.get("model_name"), str):
                            model = training["model_name"]  # type: ignore[assignment]
            backtest_status = "not_configured"
            net_return: float | None = None
            sharpe: float | None = None
            if config is not None and config.research_backtest.enabled:
                backtest_status = "unavailable"
                rb_dir = run_dir / config.research_backtest.artifact_subdir
                if rb_dir.exists():
                    try:
                        validation = ResearchBacktestArtifactStore().validate(rb_dir)
                        metric_values = _json(rb_dir / "metrics.json")
                    except Exception:
                        validation = None
                        metric_values = None
                    if (
                        validation is not None
                        and validation.is_valid
                        and metric_values is not None
                        and set(metric_values) == set(PERFORMANCE_METRIC_KEYS)
                    ):
                        backtest_status = "available"
                        value = metric_values.get("net_total_return")
                        net_return = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
                        value = metric_values.get("net_sharpe_ratio")
                        sharpe = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
                    else:
                        backtest_status = "invalid"
            summaries.append(RunSummary(
                run_id=run_id,
                created_at=_canonical_created_at(info.get("created_at")),
                status=info.get("status") if isinstance(info.get("status"), str) else None,
                model=model,
                top_n=None if config is None else config.holdings.top_n,
                portfolio_method=None if config is None else config.holdings.portfolio_construction.method,
                benchmark=None if config is None else config.benchmark,
                backtest_status=backtest_status,
                net_total_return=net_return,
                net_sharpe_ratio=sharpe,
            ))
        return tuple(sorted(
            summaries,
            key=lambda item: (
                item.created_at is not None,
                _created_at_sort_key(item.created_at),
                item.run_id,
            ),
            reverse=True,
        ))
