"""Service helpers for running the research pipeline from Streamlit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def build_pipeline_command(
    project_root: Path,
    start: str = "20240101",
    end: str = "20241231",
    universe: str = "hs300",
    max_stocks: int = 50,
    top_n: int = 10,
    n_groups: int = 5,
    transaction_cost: float = 0.0005,
    sleep: float = 0.5,
    skip_fetch: bool = True,
    skip_plot: bool = False,
) -> list[str]:
    """Build a command for scripts/run_research_pipeline.py."""
    script_path = project_root / "scripts" / "run_research_pipeline.py"
    command = [
        sys.executable,
        str(script_path),
        "--start",
        start,
        "--end",
        end,
        "--universe",
        universe,
        "--max-stocks",
        str(max_stocks),
        "--top-n",
        str(top_n),
        "--n-groups",
        str(n_groups),
        "--transaction-cost",
        str(transaction_cost),
        "--sleep",
        str(sleep),
    ]
    if skip_fetch:
        command.append("--skip-fetch")
    if skip_plot:
        command.append("--skip-plot")
    return command


def command_to_display(command: list[str]) -> str:
    """Convert a command list into a readable display string without tokens."""
    display_parts = []
    for part in command:
        if " " in part:
            display_parts.append(f'"{part}"')
        else:
            display_parts.append(part)
    return " ".join(display_parts)


def run_pipeline_command(
    command: list[str],
    project_root: Path,
    timeout: int | None = None,
) -> dict[str, object]:
    """Run a prepared pipeline command and capture its output."""
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "success": completed.returncode == 0,
        }
    except Exception as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "success": False,
        }


def run_research_pipeline_from_app(
    project_root: Path,
    start: str = "20240101",
    end: str = "20241231",
    universe: str = "hs300",
    max_stocks: int = 50,
    top_n: int = 10,
    n_groups: int = 5,
    transaction_cost: float = 0.0005,
    sleep: float = 0.5,
    skip_fetch: bool = True,
    skip_plot: bool = False,
    timeout: int | None = None,
) -> dict[str, object]:
    """Run the historical research pipeline through the existing CLI script."""
    command = build_pipeline_command(
        project_root=project_root,
        start=start,
        end=end,
        universe=universe,
        max_stocks=max_stocks,
        top_n=top_n,
        n_groups=n_groups,
        transaction_cost=transaction_cost,
        sleep=sleep,
        skip_fetch=skip_fetch,
        skip_plot=skip_plot,
    )
    result = run_pipeline_command(command, project_root=project_root, timeout=timeout)
    result["command"] = command
    result["command_display"] = command_to_display(command)
    return result
