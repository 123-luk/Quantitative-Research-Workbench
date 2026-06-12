"""Backtest metric summary helpers for historical research output."""

from __future__ import annotations

from typing import Any

import pandas as pd


BACKTEST_DISCLAIMER = "以上结果仅为历史样本回测和量化研究参考，不代表未来表现，不构成投资建议。"
METRIC_COLUMNS = [
    "cumulative_return",
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "average_turnover",
    "n_periods",
]


def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible, returning None for missing values."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric_value):
        return None
    return numeric_value


def safe_percent(value: Any, default: str = "N/A") -> str:
    """Format a decimal value as a percentage string with two decimals."""
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return default
    return f"{numeric_value:.2%}"


def safe_number(value: Any, digits: int = 2, default: str = "N/A") -> str:
    """Format a numeric value with a fixed number of decimal places."""
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return default
    return f"{numeric_value:.{digits}f}"


def normalize_metrics(metrics_df: pd.DataFrame) -> dict[str, object]:
    """Normalize a backtest metrics table into a dictionary of known metrics."""
    if metrics_df.empty:
        return {}

    first_row = metrics_df.iloc[0]
    return {column: first_row[column] for column in METRIC_COLUMNS if column in metrics_df.columns}


def classify_backtest_performance(metrics: dict[str, object]) -> dict[str, str]:
    """Classify core historical backtest metrics into readable labels."""
    cumulative_return = _safe_float(metrics.get("cumulative_return"))
    max_drawdown = _safe_float(metrics.get("max_drawdown"))
    sharpe_ratio = _safe_float(metrics.get("sharpe_ratio"))
    win_rate = _safe_float(metrics.get("win_rate"))

    if cumulative_return is None:
        cumulative_return_label = "N/A"
    elif cumulative_return >= 0.30:
        cumulative_return_label = "历史累计收益较高"
    elif cumulative_return >= 0.10:
        cumulative_return_label = "历史累计收益偏强"
    elif cumulative_return > -0.10:
        cumulative_return_label = "历史累计收益中性"
    elif cumulative_return > -0.30:
        cumulative_return_label = "历史累计收益偏弱"
    else:
        cumulative_return_label = "历史累计收益较弱"

    if max_drawdown is None:
        max_drawdown_label = "N/A"
    elif max_drawdown >= -0.05:
        max_drawdown_label = "历史回撤较低"
    elif max_drawdown >= -0.15:
        max_drawdown_label = "历史回撤中等"
    else:
        max_drawdown_label = "历史回撤较高"

    if sharpe_ratio is None:
        sharpe_label = "N/A"
    elif sharpe_ratio >= 1.5:
        sharpe_label = "历史夏普较高"
    elif sharpe_ratio >= 0.8:
        sharpe_label = "历史夏普中等偏高"
    elif sharpe_ratio >= 0.0:
        sharpe_label = "历史夏普一般"
    else:
        sharpe_label = "历史夏普较弱"

    if win_rate is None:
        win_rate_label = "N/A"
    elif win_rate >= 0.6:
        win_rate_label = "历史胜率较高"
    elif win_rate >= 0.45:
        win_rate_label = "历史胜率中等"
    else:
        win_rate_label = "历史胜率偏低"

    return {
        "cumulative_return_label": cumulative_return_label,
        "max_drawdown_label": max_drawdown_label,
        "sharpe_label": sharpe_label,
        "win_rate_label": win_rate_label,
    }


def build_backtest_research_comment(metrics: dict[str, object], labels: dict[str, str]) -> str:
    """Build a neutral Chinese summary for historical backtest metrics."""
    if not metrics:
        return f"暂无可用回测指标数据。{BACKTEST_DISCLAIMER}"

    n_periods = safe_number(metrics.get("n_periods"), digits=0)
    cumulative_return = safe_percent(metrics.get("cumulative_return"))
    annual_return = safe_percent(metrics.get("annual_return"))
    annual_volatility = safe_percent(metrics.get("annual_volatility"))
    max_drawdown = safe_percent(metrics.get("max_drawdown"))
    sharpe_ratio = safe_number(metrics.get("sharpe_ratio"), digits=2)
    win_rate = safe_percent(metrics.get("win_rate"))
    average_turnover = safe_percent(metrics.get("average_turnover"))

    cumulative_label = labels.get("cumulative_return_label", "N/A")
    drawdown_label = labels.get("max_drawdown_label", "N/A")
    sharpe_label = labels.get("sharpe_label", "N/A")
    win_rate_label = labels.get("win_rate_label", "N/A")

    return (
        f"本次历史回测共包含 {n_periods} 个调仓周期，累计收益为 {cumulative_return}，"
        f"年化收益为 {annual_return}，年化波动为 {annual_volatility}，"
        f"最大回撤为 {max_drawdown}，夏普比率为 {sharpe_ratio}，"
        f"胜率为 {win_rate}，平均换手率为 {average_turnover}。"
        f"表现标签方面，累计收益为“{cumulative_label}”，最大回撤为“{drawdown_label}”，"
        f"夏普比率为“{sharpe_label}”，胜率为“{win_rate_label}”。"
        f"{BACKTEST_DISCLAIMER}"
    )


def prepare_backtest_summary_data(metrics_df: pd.DataFrame) -> dict[str, object]:
    """Prepare normalized metrics, labels, and a research comment for backtest summary."""
    metrics = normalize_metrics(metrics_df)
    labels = classify_backtest_performance(metrics)
    research_comment = build_backtest_research_comment(metrics, labels)
    return {
        "metrics": metrics,
        "labels": labels,
        "research_comment": research_comment,
    }


def prepare_nav_curve(nav_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the historical backtest NAV curve table for display."""
    if nav_df.empty or "date" not in nav_df.columns:
        return pd.DataFrame()

    output_cols = [
        "date",
        "gross_return",
        "cost",
        "net_return",
        "nav",
        "drawdown",
        "turnover",
    ]
    curve = nav_df.copy()
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
    curve = curve.dropna(subset=["date"]).sort_values("date")
    available_cols = [column for column in output_cols if column in curve.columns]
    if not available_cols:
        return pd.DataFrame()
    return curve.loc[:, available_cols].reset_index(drop=True)


def prepare_monthly_return_series(nav_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the historical monthly net return series from a backtest NAV table."""
    nav_curve = prepare_nav_curve(nav_df)
    if nav_curve.empty or "net_return" not in nav_curve.columns:
        return pd.DataFrame()
    return nav_curve.loc[:, ["date", "net_return"]].sort_values("date").reset_index(drop=True)


def prepare_drawdown_series(nav_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the historical drawdown series, calculating it from NAV when needed."""
    nav_curve = prepare_nav_curve(nav_df)
    if nav_curve.empty:
        return pd.DataFrame()

    if "drawdown" not in nav_curve.columns:
        if "nav" not in nav_curve.columns:
            return pd.DataFrame()
        nav_curve = nav_curve.copy()
        nav_values = pd.to_numeric(nav_curve["nav"], errors="coerce")
        cumulative_max = nav_values.cummax()
        nav_curve["drawdown"] = nav_values / cumulative_max - 1

    return nav_curve.loc[:, ["date", "drawdown"]].sort_values("date").reset_index(drop=True)


def prepare_turnover_series(
    turnover_df: pd.DataFrame | None = None,
    nav_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prepare the historical portfolio turnover series from turnover or NAV data."""
    source_df = turnover_df
    if source_df is None or source_df.empty or "turnover" not in source_df.columns:
        source_df = nav_df

    if source_df is None or source_df.empty or "date" not in source_df.columns or "turnover" not in source_df.columns:
        return pd.DataFrame()

    turnover = source_df.copy()
    turnover["date"] = pd.to_datetime(turnover["date"], errors="coerce")
    turnover["turnover"] = pd.to_numeric(turnover["turnover"], errors="coerce")
    turnover = turnover.dropna(subset=["date"]).sort_values("date")
    return turnover.loc[:, ["date", "turnover"]].reset_index(drop=True)


def prepare_backtest_report_data(
    metrics_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    turnover_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Prepare summary and time-series data for the historical backtest report."""
    summary_data = prepare_backtest_summary_data(metrics_df)
    return {
        "metrics": summary_data["metrics"],
        "labels": summary_data["labels"],
        "research_comment": summary_data["research_comment"],
        "nav_curve": prepare_nav_curve(nav_df),
        "monthly_return": prepare_monthly_return_series(nav_df),
        "drawdown": prepare_drawdown_series(nav_df),
        "turnover": prepare_turnover_series(turnover_df, nav_df),
    }
