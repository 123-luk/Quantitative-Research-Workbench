"""Streamlit dashboard for Quant Factor System research outputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.dashboard_service import (  # noqa: E402
    format_percent,
    get_figure_paths,
    get_latest_portfolio,
    get_metric_value,
    get_project_root,
    load_dashboard_data,
)


RISK_NOTICE = "本应用展示的是历史样本回测和量化研究结果，不代表未来表现，不构成投资建议。"
MISSING_FIGURE_HINT = "Please run scripts/plot_backtest.py or scripts/run_research_pipeline.py first."


def show_risk_notice() -> None:
    """Display the standard research and historical backtest risk notice."""
    st.warning(RISK_NOTICE)


def format_metric_value(value: Any, percent: bool = False, decimals: int = 2) -> str:
    """Format dashboard metric values with graceful missing-value handling."""
    if percent:
        return format_percent(value)
    if value is None:
        return "N/A"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(numeric_value):
        return "N/A"
    return f"{numeric_value:.{decimals}f}"


def show_metric_cards(metrics_df: pd.DataFrame) -> None:
    """Render historical backtest metric cards."""
    if metrics_df.empty:
        st.info("Backtest metrics are empty. Please run scripts/run_backtest.py or scripts/run_research_pipeline.py first.")
        return

    metric_specs = [
        ("Cumulative Return", "cumulative_return", True),
        ("Annual Return", "annual_return", True),
        ("Annual Volatility", "annual_volatility", True),
        ("Sharpe Ratio", "sharpe_ratio", False),
        ("Max Drawdown", "max_drawdown", True),
        ("Win Rate", "win_rate", True),
        ("Average Turnover", "average_turnover", True),
    ]
    columns = st.columns(4)
    for index, (label, column, is_percent) in enumerate(metric_specs):
        value = get_metric_value(metrics_df, column)
        columns[index % len(columns)].metric(
            label,
            format_metric_value(value, percent=is_percent),
        )


def show_figure(path: Path, caption: str) -> None:
    """Display a local figure when available, otherwise show a friendly hint."""
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(MISSING_FIGURE_HINT)


def show_figures(figure_paths: dict[str, Path]) -> None:
    """Render the three historical backtest figures."""
    show_figure(figure_paths["nav_curve"], "Historical Backtest NAV Curve")
    show_figure(figure_paths["monthly_return_bar"], "Historical Monthly Net Return")
    show_figure(figure_paths["drawdown_curve"], "Historical Drawdown Curve")


def show_table_or_hint(df: pd.DataFrame, hint: str) -> None:
    """Display a table or a non-blocking hint when it is empty."""
    if df.empty:
        st.info(hint)
    else:
        st.dataframe(df, use_container_width=True)


def render_dashboard_page(data: dict[str, pd.DataFrame], figure_paths: dict[str, Path]) -> None:
    """Render the home dashboard page."""
    st.title("Quant Factor System")
    st.subheader("A-share multi-factor research dashboard")
    show_risk_notice()
    show_metric_cards(data["backtest_metrics"])
    st.divider()
    show_figures(figure_paths)


def render_portfolio_page(data: dict[str, pd.DataFrame]) -> None:
    """Render the model-selected portfolio page."""
    st.title("模型选股结果 / Model-selected Portfolio")
    st.info("当前页面展示最新一期模型选出的 Top N 股票，仅作为量化研究信号，不构成投资建议。")

    selected_portfolio = data["selected_portfolio"]
    latest_portfolio = get_latest_portfolio(selected_portfolio)
    preferred_cols = [
        "date",
        "ts_code",
        "name",
        "industry",
        "composite_score",
        "score_rank",
        "score_pct_rank",
        "return_next",
    ]
    if latest_portfolio.empty:
        st.info("No selected portfolio data found. Please run scripts/run_scoring_model.py or scripts/run_research_pipeline.py first.")
    else:
        visible_cols = [col for col in preferred_cols if col in latest_portfolio.columns]
        st.dataframe(latest_portfolio.loc[:, visible_cols], use_container_width=True)

    with st.expander("查看全部历史模型选股结果"):
        show_table_or_hint(selected_portfolio, "No historical model-selected portfolio data found.")


def render_backtest_page(data: dict[str, pd.DataFrame], figure_paths: dict[str, Path]) -> None:
    """Render the historical backtest page."""
    st.title("历史回测结果 / Historical Backtest")
    st.info("以下内容为 historical backtest 历史样本测算，不代表未来收益。")

    st.subheader("Backtest Metrics")
    show_table_or_hint(data["backtest_metrics"], "No backtest metrics found.")

    st.subheader("Backtest NAV")
    show_table_or_hint(data["backtest_nav"], "No backtest NAV data found.")

    st.subheader("Backtest Turnover")
    show_table_or_hint(data["backtest_turnover"], "No backtest turnover data found.")

    st.subheader("Historical Backtest Figures")
    show_figures(figure_paths)


def render_factor_research_page(data: dict[str, pd.DataFrame]) -> None:
    """Render the factor research page."""
    st.title("因子研究 / Factor Research")
    st.info("Factor research outputs are historical research statistics and do not represent future performance.")

    ic_summary = data["ic_summary"]
    if ic_summary.empty:
        st.info("No IC summary found. Please run scripts/run_factor_test.py or scripts/run_research_pipeline.py first.")
    else:
        if "mean_ic" in ic_summary.columns:
            ic_summary = ic_summary.sort_values("mean_ic", ascending=False)
        st.dataframe(ic_summary, use_container_width=True)

    with st.expander("Group Return"):
        show_table_or_hint(data["group_return"], "No group return data found.")

    with st.expander("Long-short Return"):
        show_table_or_hint(data["long_short_return"], "No long-short return data found.")


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.set_page_config(page_title="Quant Factor System", layout="wide")
    project_root = get_project_root()
    data = load_dashboard_data(project_root)
    figure_paths = get_figure_paths(project_root)

    st.sidebar.title("Quant Factor System")
    page = st.sidebar.radio(
        "Navigation",
        ["首页 Dashboard", "推荐投资组合", "回测结果", "因子研究"],
    )
    st.sidebar.markdown("当前版本：V6-A Portfolio Dashboard")
    st.sidebar.markdown("数据来源：TuShare + local CSV")
    st.sidebar.info("如数据为空，请先运行 scripts/run_research_pipeline.py")

    if page == "首页 Dashboard":
        render_dashboard_page(data, figure_paths)
    elif page == "推荐投资组合":
        render_portfolio_page(data)
    elif page == "回测结果":
        render_backtest_page(data, figure_paths)
    elif page == "因子研究":
        render_factor_research_page(data)


if __name__ == "__main__":
    main()
