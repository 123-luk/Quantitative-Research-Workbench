"""Data loading helpers for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


TABLE_FILES = {
    "ic_summary": "reports/tables/ic_summary.csv",
    "backtest_metrics": "reports/tables/backtest_metrics.csv",
    "backtest_nav": "reports/tables/backtest_nav.csv",
    "selected_portfolio": "reports/tables/selected_portfolio.csv",
    "factor_score": "reports/tables/factor_score.csv",
    "backtest_turnover": "reports/tables/backtest_turnover.csv",
    "group_return": "reports/tables/group_return.csv",
    "long_short_return": "reports/tables/long_short_return.csv",
}

FIGURE_FILES = {
    "nav_curve": "reports/figures/nav_curve.png",
    "monthly_return_bar": "reports/figures/monthly_return_bar.png",
    "drawdown_curve": "reports/figures/drawdown_curve.png",
}


def get_project_root() -> Path:
    """Return the project root by walking upward from this file location."""
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "README.md").exists() or (parent / "AGENT.md").exists():
            return parent
    return Path.cwd()


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file safely, returning an empty DataFrame on failure."""
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_dashboard_data(project_root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all dashboard CSV tables from local result files."""
    root = project_root or get_project_root()
    return {key: safe_read_csv(root / relative_path) for key, relative_path in TABLE_FILES.items()}


def get_figure_paths(project_root: Path | None = None) -> dict[str, Path]:
    """Return expected local figure paths for dashboard display."""
    root = project_root or get_project_root()
    return {key: root / relative_path for key, relative_path in FIGURE_FILES.items()}


def get_latest_portfolio(selected_portfolio: pd.DataFrame) -> pd.DataFrame:
    """Return the latest available model-selected portfolio rows."""
    if selected_portfolio.empty or "date" not in selected_portfolio.columns:
        return pd.DataFrame()

    latest_date = selected_portfolio["date"].max()
    latest = selected_portfolio[selected_portfolio["date"] == latest_date].copy()
    if "score_rank" in latest.columns:
        latest = latest.sort_values("score_rank")
    return latest.reset_index(drop=True)


def format_percent(value: float | int | None) -> str:
    """Format a decimal number as a percentage string for dashboard metrics."""
    if value is None:
        return "N/A"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if pd.isna(numeric_value):
        return "N/A"
    return f"{numeric_value:.2%}"


def get_metric_value(metrics_df: pd.DataFrame, column: str) -> Any:
    """Return the first-row value for a metric column, or None if unavailable."""
    if metrics_df.empty or column not in metrics_df.columns:
        return None
    return metrics_df.iloc[0][column]
