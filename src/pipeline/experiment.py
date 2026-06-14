"""Experiment run directory management for pipeline executions."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.config import PipelineConfig


class ExperimentManager:
    """Create and persist artifacts for one pipeline experiment run."""

    def __init__(self, output_root: str | Path) -> None:
        """Initialize the experiment manager.

        Args:
            output_root: Base output directory. Run folders are created below
                ``<output_root>/runs``.
        """
        self.output_root = Path(output_root)
        self.runs_root = self.output_root / "runs"

    def create_run_dir(self, strategy_name: str, stock_pool: str) -> Path:
        """Create a unique run directory without overwriting older results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_run_id = "_".join(
            [
                timestamp,
                slugify(strategy_name),
                slugify(stock_pool),
            ]
        )
        self.runs_root.mkdir(parents=True, exist_ok=True)

        run_dir = self.runs_root / base_run_id
        suffix = 1
        while run_dir.exists():
            run_dir = self.runs_root / f"{base_run_id}_{suffix:03d}"
            suffix += 1

        run_dir.mkdir(parents=False, exist_ok=False)
        return run_dir

    def save_config_snapshot(self, run_dir: Path, config: PipelineConfig) -> Path:
        """Save the pipeline configuration snapshot as YAML."""
        path = run_dir / "config_snapshot.yaml"
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                config.to_dict(),
                file,
                allow_unicode=True,
                sort_keys=False,
            )
        return path

    def save_run_info(self, run_dir: Path, run_info: dict[str, Any]) -> Path:
        """Save basic run metadata as JSON."""
        path = run_dir / "run_info.json"
        payload = dict(run_info)
        payload.setdefault("created_at", datetime.now().replace(microsecond=0).isoformat())
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return path

    def save_metrics(self, run_dir: Path, metrics: dict[str, Any]) -> Path:
        """Save metrics JSON, using placeholders when metrics are not ready."""
        path = run_dir / "metrics.json"
        payload = dict(metrics)
        payload.setdefault("status", "placeholder")
        payload.setdefault("metrics_ready", False)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return path


def slugify(value: str) -> str:
    """Convert a string into a filesystem-friendly lowercase slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_") or "run"
