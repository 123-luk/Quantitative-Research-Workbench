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

    def list_run_ids(self) -> tuple[str, ...]:
        """Return valid direct-child run identities in stable lexical order."""
        if not self.runs_root.exists():
            return ()
        if self.runs_root.is_symlink() or not self.runs_root.is_dir():
            raise ValueError("runs root must be a regular directory.")
        result: list[str] = []
        for entry in self.runs_root.iterdir():
            if (
                entry.is_dir()
                and not entry.is_symlink()
                and _RUN_ID_RE.fullmatch(entry.name)
            ):
                result.append(entry.name)
        return tuple(sorted(result))

    def resolve_run_dir(self, run_id: str) -> Path:
        """Resolve one exact existing direct-child run directory safely."""
        if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError("run_id is not a valid canonical run identity.")
        root = self.runs_root.resolve()
        candidate = root / run_id
        if candidate.is_symlink():
            raise ValueError("run directory must not be a symbolic link.")
        run_dir = candidate.resolve()
        if run_dir.parent != root:
            raise ValueError("run_id must resolve to a direct child of runs root.")
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise ValueError("run directory must be a regular directory.")
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


_RUN_ID_RE = re.compile(r"\d{8}_\d{6}_[a-z0-9]+(?:_[a-z0-9]+)+")
