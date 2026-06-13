"""Lightweight project health checks for quant-factor-system."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "AGENT.md",
    "app/streamlit_app.py",
    "scripts/run_research_pipeline.py",
    "scripts/fetch_tushare_data.py",
    "scripts/build_factor_panel.py",
    "scripts/preprocess_factor_panel.py",
    "scripts/run_factor_test.py",
    "scripts/run_scoring_model.py",
    "scripts/run_backtest.py",
    "scripts/plot_backtest.py",
    "run_app.bat",
    "run_app.ps1",
]

CORE_MODULES = [
    "app.services.dashboard_service",
    "app.services.pipeline_runner_service",
    "app.services.stock_query_service",
    "app.services.stock_rating_service",
    "app.services.stock_report_service",
    "app.services.stock_chart_service",
    "app.services.stock_price_service",
    "app.services.portfolio_report_service",
    "app.services.backtest_report_service",
    "app.services.factor_report_service",
]

OPTIONAL_OUTPUT_FILES = [
    "data/processed/factor_panel.csv",
    "data/processed/factor_panel_clean.csv",
    "reports/tables/ic_summary.csv",
    "reports/tables/factor_score.csv",
    "reports/tables/selected_portfolio.csv",
    "reports/tables/backtest_metrics.csv",
    "reports/tables/backtest_nav.csv",
    "reports/figures/nav_curve.png",
    "reports/figures/monthly_return_bar.png",
    "reports/figures/drawdown_curve.png",
]


def get_project_root() -> Path:
    """Return the project root based on this script location."""
    return Path(__file__).resolve().parents[1]


def check_file_exists(
    project_root: Path,
    relative_path: str,
    required: bool = True,
) -> dict[str, object]:
    """Check whether a project file exists and return a structured result."""
    path = project_root / relative_path
    exists = path.exists()
    if exists:
        status = "OK"
        message = "File exists."
    elif required:
        status = "ERROR"
        message = "Required file is missing."
    else:
        status = "WARN"
        message = "Optional output file is missing."

    return {
        "path": relative_path,
        "exists": exists,
        "required": required,
        "status": status,
        "message": message,
    }


def check_module_import(module_name: str, required: bool = True) -> dict[str, object]:
    """Try importing a module and return a structured health-check result."""
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - health checks should catch import failures.
        status = "ERROR" if required else "WARN"
        message = f"Import failed: {type(exc).__name__}"
        exists = False
    else:
        status = "OK"
        message = "Module import succeeded."
        exists = True

    return {
        "path": module_name,
        "exists": exists,
        "required": required,
        "status": status,
        "message": message,
    }


def run_health_checks(project_root: Path | None = None) -> list[dict[str, object]]:
    """Run project file, module import, and optional output checks."""
    root = project_root if project_root is not None else get_project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    results: list[dict[str, object]] = []
    for relative_path in REQUIRED_FILES:
        results.append(check_file_exists(root, relative_path, required=True))
    for module_name in CORE_MODULES:
        results.append(check_module_import(module_name, required=True))
    for relative_path in OPTIONAL_OUTPUT_FILES:
        results.append(check_file_exists(root, relative_path, required=False))
    return results


def print_health_report(results: list[dict[str, object]]) -> int:
    """Print a clear health report and return a process exit code."""
    counts = {"OK": 0, "WARN": 0, "ERROR": 0}
    print("quant-factor-system project health check")
    print("=" * 48)
    for result in results:
        status = str(result.get("status", "ERROR"))
        counts[status] = counts.get(status, 0) + 1
        path = result.get("path", "N/A")
        message = result.get("message", "")
        print(f"[{status}] {path} - {message}")

    print("=" * 48)
    print(f"OK: {counts.get('OK', 0)}")
    print(f"WARN: {counts.get('WARN', 0)}")
    print(f"ERROR: {counts.get('ERROR', 0)}")
    return 1 if counts.get("ERROR", 0) > 0 else 0


def main() -> None:
    """Run the project health check command-line entrypoint."""
    project_root = get_project_root()
    results = run_health_checks(project_root)
    exit_code = print_health_report(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
