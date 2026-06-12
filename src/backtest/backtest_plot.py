"""Plot historical backtest figures from saved NAV results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_backtest_nav(path: str | Path) -> pd.DataFrame:
    """Load historical backtest NAV data and validate required columns."""
    nav_path = Path(path)
    nav_df = pd.read_csv(nav_path)
    required_cols = {"date", "nav", "net_return"}
    missing_cols = required_cols.difference(nav_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in backtest NAV file: {sorted(missing_cols)}")

    nav_df = nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df = nav_df.sort_values("date").reset_index(drop=True)
    return nav_df


def add_drawdown(nav_df: pd.DataFrame, nav_col: str = "nav") -> pd.DataFrame:
    """Add historical drawdown columns without modifying the input DataFrame."""
    result = nav_df.copy()
    result[nav_col] = pd.to_numeric(result[nav_col], errors="coerce")
    result["cumulative_max_nav"] = result[nav_col].cummax()
    result["drawdown"] = result[nav_col] / result["cumulative_max_nav"] - 1.0
    return result


def plot_nav_curve(
    nav_df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "date",
    nav_col: str = "nav",
) -> None:
    """Plot and save the historical backtest NAV curve."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nav_df[date_col], nav_df[nav_col])
    ax.set_title("Historical Backtest NAV Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_monthly_return_bar(
    nav_df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "date",
    ret_col: str = "net_return",
) -> None:
    """Plot and save historical monthly net returns as a bar chart."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(nav_df[date_col], nav_df[ret_col])
    ax.set_title("Historical Monthly Net Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Return")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_drawdown_curve(
    nav_df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "date",
    drawdown_col: str = "drawdown",
) -> None:
    """Plot and save the historical drawdown curve."""
    data = nav_df.copy()
    if drawdown_col not in data.columns:
        data = add_drawdown(data)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data[date_col], data[drawdown_col])
    ax.set_title("Historical Drawdown Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_all_backtest_figures(
    nav_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Create all historical backtest figures from a NAV CSV file."""
    output_path = Path(output_dir)
    nav_df = add_drawdown(load_backtest_nav(nav_path))

    figure_paths = {
        "nav_curve": output_path / "nav_curve.png",
        "monthly_return_bar": output_path / "monthly_return_bar.png",
        "drawdown_curve": output_path / "drawdown_curve.png",
    }
    plot_nav_curve(nav_df, figure_paths["nav_curve"])
    plot_monthly_return_bar(nav_df, figure_paths["monthly_return_bar"])
    plot_drawdown_curve(nav_df, figure_paths["drawdown_curve"])
    return figure_paths
