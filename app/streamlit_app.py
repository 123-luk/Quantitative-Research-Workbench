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
from app.services.stock_query_service import (  # noqa: E402
    build_stock_lookup,
    calculate_selection_frequency,
    get_latest_stock_snapshot,
    get_stock_factor_history,
    get_stock_selection_history,
    normalize_ts_code,
    search_stock,
)


RISK_NOTICE = "本应用展示的是历史样本回测和量化研究结果，不代表未来表现，不构成投资建议。"
SINGLE_STOCK_NOTICE = (
    "本页面展示的是历史样本中的模型评分、排名和入选情况，仅作为量化研究参考，"
    "不代表未来表现，不构成投资建议。"
)
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
        columns[index % len(columns)].metric(label, format_metric_value(value, percent=is_percent))


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
        st.caption("return_next 是历史下一期收益标签，仅用于回测检验，不是未来收益预测。")

    with st.expander("查看全部历史模型选股结果"):
        show_table_or_hint(selected_portfolio, "No historical model-selected portfolio data found.")


def choose_stock_from_matches(matches: pd.DataFrame) -> str | None:
    """Return the selected ts_code from one or more search matches."""
    if matches.empty:
        return None
    if len(matches) == 1:
        return str(matches.iloc[0]["ts_code"])

    options = []
    for _, row in matches.iterrows():
        name = row.get("name", "")
        industry = row.get("industry", "")
        options.append(f"{row['ts_code']} | {name} | {industry}")
    selected_label = st.selectbox("请选择一个匹配股票", options)
    return selected_label.split("|", 1)[0].strip()


def render_stock_snapshot(snapshot: dict[str, object]) -> None:
    """Render the latest single-stock research snapshot."""
    if not snapshot:
        st.info("No factor score history found for this stock.")
        return

    st.subheader("最新一期模型评分快照")
    cols = st.columns(4)
    cols[0].metric("ts_code", str(snapshot.get("ts_code", "N/A")))
    cols[1].metric("name", str(snapshot.get("name", "N/A")))
    cols[2].metric("industry", str(snapshot.get("industry", "N/A")))
    cols[3].metric("latest_date", str(snapshot.get("latest_date", "N/A")))

    cols = st.columns(4)
    cols[0].metric("Composite Score", format_metric_value(snapshot.get("composite_score")))
    cols[1].metric("Score Rank", format_metric_value(snapshot.get("score_rank"), decimals=0))
    cols[2].metric("Score Percentile", format_metric_value(snapshot.get("score_pct_rank"), percent=True))
    cols[3].metric("Selected Latest", "是" if snapshot.get("is_selected_latest") else "否")

    st.caption("return_next 是历史下一期收益标签，仅用于回测检验，不是未来收益预测。")
    st.metric("return_next 历史标签", format_metric_value(snapshot.get("return_next"), percent=True))


def render_selection_frequency(frequency: dict[str, object]) -> None:
    """Render selection frequency metrics for one stock."""
    st.subheader("历史入选频率")
    cols = st.columns(3)
    cols[0].metric("Total Periods", str(frequency.get("total_periods", 0)))
    cols[1].metric("Selected Periods", str(frequency.get("selected_periods", 0)))
    cols[2].metric("Selection Frequency", format_percent(frequency.get("selection_frequency")))


def render_stock_history_charts(history: pd.DataFrame) -> None:
    """Render simple score and rank history charts for one stock."""
    if history.empty or "date" not in history.columns:
        return

    chart_data = history.copy()
    chart_data["date"] = pd.to_datetime(chart_data["date"], errors="coerce")
    chart_data = chart_data.dropna(subset=["date"]).set_index("date")
    if "composite_score" in chart_data.columns:
        st.subheader("模型评分走势")
        st.line_chart(chart_data[["composite_score"]])
    if "score_rank" in chart_data.columns:
        st.subheader("模型排名走势")
        st.caption("score_rank 越低表示该期模型排名越靠前。")
        st.line_chart(chart_data[["score_rank"]])


def render_single_stock_page(data: dict[str, pd.DataFrame]) -> None:
    """Render the single-stock analysis page."""
    st.title("单只股票分析 / Single Stock Analysis")
    st.warning(SINGLE_STOCK_NOTICE)

    factor_score = data["factor_score"]
    selected_portfolio = data["selected_portfolio"]
    lookup = build_stock_lookup(factor_score, selected_portfolio)

    query = st.text_input("股票名称或代码，例如：贵州茅台、600519、600519.SH")
    if not query:
        st.info("请输入股票名称或代码进行查询。")
        return

    matches = search_stock(query, lookup)
    if matches.empty:
        normalized_code = normalize_ts_code(query)
        st.info(f"未找到匹配股票，请检查股票名称或代码。规范化代码：{normalized_code}")
        return

    selected_ts_code = choose_stock_from_matches(matches)
    if not selected_ts_code:
        st.info("请选择一个股票查看单股分析。")
        return

    snapshot = get_latest_stock_snapshot(selected_ts_code, factor_score, selected_portfolio)
    render_stock_snapshot(snapshot)

    frequency = calculate_selection_frequency(selected_ts_code, selected_portfolio, factor_score)
    render_selection_frequency(frequency)

    history = get_stock_factor_history(selected_ts_code, factor_score)
    st.subheader("历史评分记录")
    history_cols = [
        "date",
        "composite_score",
        "score_rank",
        "score_pct_rank",
        "return_next",
        "momentum_1m",
        "momentum_3m",
        "volatility_6m",
        "ep",
        "bp",
        "ps_inverse",
    ]
    if history.empty:
        st.info("该股票暂无历史评分记录。")
    else:
        visible_cols = [col for col in history_cols if col in history.columns]
        st.dataframe(history.loc[:, visible_cols], use_container_width=True)
        st.caption("return_next 为历史下一期收益标签，仅用于回测检验，不作为未来收益预测。")
        render_stock_history_charts(history)

    selection_history = get_stock_selection_history(selected_ts_code, selected_portfolio)
    with st.expander("查看历史入选模型组合记录"):
        if selection_history.empty:
            st.info("该股票在当前样本期没有进入模型选股组合。")
        else:
            st.dataframe(selection_history, use_container_width=True)


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
        ["首页 Dashboard", "推荐投资组合", "单只股票分析", "回测结果", "因子研究"],
    )
    st.sidebar.markdown("当前版本：V6-A Portfolio Dashboard")
    st.sidebar.markdown("数据来源：TuShare + local CSV")
    st.sidebar.info("如数据为空，请先运行 scripts/run_research_pipeline.py")

    if page == "首页 Dashboard":
        render_dashboard_page(data, figure_paths)
    elif page == "推荐投资组合":
        render_portfolio_page(data)
    elif page == "单只股票分析":
        render_single_stock_page(data)
    elif page == "回测结果":
        render_backtest_page(data, figure_paths)
    elif page == "因子研究":
        render_factor_research_page(data)


if __name__ == "__main__":
    main()
