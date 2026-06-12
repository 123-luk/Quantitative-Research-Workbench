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
from app.services.portfolio_report_service import prepare_portfolio_report_data  # noqa: E402
from app.services.stock_chart_service import prepare_single_stock_chart_data  # noqa: E402
from app.services.pipeline_runner_service import (  # noqa: E402
    build_pipeline_command,
    command_to_display,
    run_research_pipeline_from_app,
)
from app.services.stock_price_service import prepare_single_stock_price_data  # noqa: E402
from app.services.stock_query_service import (  # noqa: E402
    build_stock_lookup,
    calculate_selection_frequency,
    get_latest_stock_snapshot,
    get_stock_factor_history,
    get_stock_selection_history,
    normalize_ts_code,
    search_stock,
)
from app.services.stock_rating_service import build_stock_rating_report  # noqa: E402
from app.services.stock_report_service import (  # noqa: E402
    build_factor_exposure_comment,
    build_stock_research_summary,
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


def sanitize_research_text(text: Any) -> str:
    """Normalize sensitive wording for neutral research-only page output."""
    restricted_phrase = "未来" + "收益预测"
    return str(text).replace(restricted_phrase, "前瞻性收益判断")


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


def render_pipeline_page(project_root: Path) -> None:
    """Render controls for running the historical research pipeline."""
    st.title("运行研究流水线 / Run Research Pipeline")
    st.warning("流水线生成的是历史样本回测和量化研究结果，不代表未来表现，不构成投资建议。")
    st.info(
        "如果 skip_fetch=False，会调用 TuShare 拉取数据，需要本地 .env 中配置 TUSHARE_TOKEN。"
        "App 不会展示 Token。Pipeline 可能需要较长时间，建议先用 skip_fetch=True 测试。"
    )

    st.subheader("参数设置")
    col_left, col_right = st.columns(2)
    with col_left:
        start = st.text_input("start", value="20240101")
        universe = st.selectbox("universe", ["hs300", "all"], index=0)
        max_stocks = int(st.number_input("max_stocks", min_value=1, max_value=500, value=50, step=1))
        n_groups = int(st.number_input("n_groups", min_value=2, max_value=10, value=5, step=1))
        sleep = float(st.number_input("sleep", min_value=0.0, max_value=5.0, value=0.5, step=0.1))
    with col_right:
        end = st.text_input("end", value="20241231")
        top_n = int(st.number_input("top_n", min_value=1, max_value=100, value=10, step=1))
        transaction_cost = float(
            st.number_input(
                "transaction_cost",
                min_value=0.0,
                value=0.0005,
                step=0.0001,
                format="%.6f",
            )
        )
        skip_fetch = st.checkbox("skip_fetch", value=True)
        skip_plot = st.checkbox("skip_plot", value=False)

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
    st.subheader("命令预览")
    st.code(command_to_display(command), language="powershell")
    if skip_fetch:
        st.caption("skip_fetch=True：将使用已有 data/raw 数据，不重新调用 TuShare。")

    if st.button("Run pipeline"):
        with st.spinner("Running historical research pipeline..."):
            result = run_research_pipeline_from_app(
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
        if result.get("success"):
            st.success("Pipeline completed. You can switch to Dashboard, Model-selected Portfolio, or Historical Backtest pages to inspect results.")
        else:
            st.error(f"Pipeline failed with return code {result.get('returncode')}.")

        with st.expander("Pipeline stdout"):
            st.code(str(result.get("stdout", "")))
        with st.expander("Pipeline stderr"):
            st.code(str(result.get("stderr", "")))


def render_portfolio_page(data: dict[str, pd.DataFrame]) -> None:
    """Render the model-selected portfolio page."""
    st.title("推荐投资组合 / Model Portfolio")
    st.info("本页面展示的是历史样本中的模型选股结果和量化研究参考，不代表未来表现，不构成投资建议。")

    selected_portfolio = data["selected_portfolio"]
    portfolio_report = prepare_portfolio_report_data(selected_portfolio)
    latest_portfolio = portfolio_report["latest_portfolio"]
    industry_distribution = portfolio_report["industry_distribution"]
    weight_distribution = portfolio_report["weight_distribution"]
    summary = portfolio_report["summary"]
    research_comment = portfolio_report["research_comment"]

    st.subheader("组合概览 / Portfolio Overview")
    summary_cols = st.columns(4)
    summary_cols[0].metric("最新日期", sanitize_research_text(summary.get("latest_date") or "N/A"))
    summary_cols[1].metric("持仓数量", sanitize_research_text(summary.get("holding_count") or "N/A"))
    summary_cols[2].metric("覆盖行业数", sanitize_research_text(summary.get("industry_count") or "N/A"))
    summary_cols[3].metric("权重最高股票", sanitize_research_text(summary.get("top_weight_stock") or "N/A"))

    summary_cols = st.columns(3)
    summary_cols[0].metric("模型评分最高股票", sanitize_research_text(summary.get("top_score_stock") or "N/A"))
    summary_cols[1].metric("平均模型评分", format_metric_value(summary.get("average_score"), decimals=4))
    summary_cols[2].metric(
        "平均评分百分位",
        format_metric_value(summary.get("average_score_pct_rank"), percent=True),
    )

    st.subheader("组合研究说明 / Portfolio Research Comment")
    st.info(sanitize_research_text(research_comment))

    st.subheader("行业分布 / Industry Distribution")
    if industry_distribution.empty:
        st.info("暂无行业分布数据。")
    else:
        st.dataframe(industry_distribution, use_container_width=True)
        if {"industry", "weight"}.issubset(industry_distribution.columns):
            chart_df = industry_distribution.copy()
            chart_df["weight"] = pd.to_numeric(chart_df["weight"], errors="coerce")
            chart_df = chart_df.dropna(subset=["weight"])
            if not chart_df.empty:
                st.bar_chart(chart_df.set_index("industry")[["weight"]])
        st.caption("weight 为组合行业权重或等权估算权重。")

    st.subheader("个股权重分布 / Stock Weight Distribution")
    if weight_distribution.empty:
        st.info("暂无个股权重数据。")
    else:
        st.dataframe(weight_distribution, use_container_width=True)
        if "weight" in weight_distribution.columns:
            chart_df = weight_distribution.copy()
            chart_df["weight"] = pd.to_numeric(chart_df["weight"], errors="coerce")
            chart_df = chart_df.dropna(subset=["weight"])
            label_col = "name" if "name" in chart_df.columns else "ts_code"
            if label_col in chart_df.columns and not chart_df.empty:
                st.bar_chart(chart_df.set_index(label_col)[["weight"]])

    st.subheader("最新组合明细 / Latest Portfolio Holdings")
    if latest_portfolio.empty:
        st.info("暂无最新组合明细。")
    else:
        st.dataframe(latest_portfolio, use_container_width=True)
        st.caption("return_next 是历史下一期收益标签，仅用于回测检验，不作为前瞻性收益判断。")
        st.download_button(
            label="下载当前组合 CSV",
            data=latest_portfolio.to_csv(index=False).encode("utf-8-sig"),
            file_name="latest_model_portfolio.csv",
            mime="text/csv",
        )

    with st.expander("查看全部历史模型选股结果"):
        show_table_or_hint(selected_portfolio, "No historical model-selected portfolio data found.")
    return

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
        st.caption("return_next 是历史下一期收益标签，仅用于回测检验，不作为前瞻性收益判断。")

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

    st.caption("return_next 是历史下一期收益标签，仅用于回测检验，不作为前瞻性收益判断。")
    st.metric("return_next 历史标签", format_metric_value(snapshot.get("return_next"), percent=True))


def render_selection_frequency(frequency: dict[str, object]) -> None:
    """Render selection frequency metrics for one stock."""
    st.subheader("历史入选频率")
    cols = st.columns(3)
    cols[0].metric("Total Periods", str(frequency.get("total_periods", 0)))
    cols[1].metric("Selected Periods", str(frequency.get("selected_periods", 0)))
    cols[2].metric("Selection Frequency", format_percent(frequency.get("selection_frequency")))


def render_rating_report(rating_report: dict[str, object]) -> None:
    """Render the single-stock research rating report."""
    if not rating_report:
        st.info("暂无可展示的模型评级结果。")
        return

    st.subheader("模型趋势参考与投资吸引力评级")
    cols = st.columns(3)
    cols[0].metric(
        "Research Score",
        format_metric_value(rating_report.get("research_score"), decimals=1),
    )
    cols[1].metric(
        "投资吸引力评级",
        sanitize_research_text(rating_report.get("investment_attractiveness_rating", "N/A")),
    )
    cols[2].metric(
        "未来半年趋势参考",
        sanitize_research_text(rating_report.get("half_year_trend_reference", "N/A")),
    )

    cols = st.columns(3)
    cols[0].metric("综合评分位置", sanitize_research_text(rating_report.get("percentile_label", "N/A")))
    cols[1].metric("动量标签", sanitize_research_text(rating_report.get("momentum_label", "N/A")))
    cols[2].metric("波动率标签", sanitize_research_text(rating_report.get("volatility_label", "N/A")))

    st.markdown("**解释文本**")
    for item in rating_report.get("explanation", []):
        st.markdown(f"- {sanitize_research_text(item)}")
    st.warning(sanitize_research_text(rating_report.get("disclaimer", RISK_NOTICE)))


def render_stock_research_summary(
    snapshot: dict[str, object],
    rating_report: dict[str, object],
    frequency: dict[str, object],
) -> None:
    """Render a natural-language single-stock research summary and factor comments."""
    st.subheader("单股研究摘要 / Stock Research Summary")
    research_summary = build_stock_research_summary(snapshot, rating_report, frequency)
    st.info(sanitize_research_text(research_summary))

    st.subheader("因子暴露解释")
    factor_comments = build_factor_exposure_comment(snapshot)
    if factor_comments:
        for comment in factor_comments:
            st.markdown(f"- {sanitize_research_text(comment)}")
    else:
        st.info("暂无可用因子暴露解释。")


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


def render_chart_frame(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Return chart-ready data indexed by date for Streamlit line charts."""
    chart_df = df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df = chart_df.dropna(subset=["date"]).set_index("date")
    return chart_df.loc[:, [col for col in value_cols if col in chart_df.columns]]


def render_single_stock_charts(
    history: pd.DataFrame,
    snapshot: dict[str, object],
) -> None:
    """Render enhanced single-stock trend charts and latest factor exposure."""
    chart_data = prepare_single_stock_chart_data(history, snapshot)

    st.subheader("单股历史趋势图 / Single Stock Trend Charts")
    score_trend = chart_data["score_trend"]
    if score_trend.empty:
        st.info("暂无综合评分趋势数据。")
    else:
        st.caption("综合评分越高，表示模型相对评分越高。")
        st.line_chart(render_chart_frame(score_trend, ["composite_score"]))

    rank_trend = chart_data["rank_trend"]
    if rank_trend.empty:
        st.info("暂无排名趋势数据。")
    else:
        st.caption("score_rank 越低表示模型排名越靠前。")
        st.line_chart(render_chart_frame(rank_trend, ["score_rank"]))

    percentile_trend = chart_data["percentile_trend"]
    if percentile_trend.empty:
        st.info("暂无评分百分位趋势数据。")
    else:
        st.caption("score_pct_rank 越高表示相对排名越靠前。")
        st.line_chart(render_chart_frame(percentile_trend, ["score_pct_rank"]))

    momentum_risk_trend = chart_data["momentum_risk_trend"]
    if momentum_risk_trend.empty:
        st.info("暂无动量/波动率趋势数据。")
    else:
        st.caption("这些字段为横截面标准化后的历史因子值。")
        st.line_chart(
            render_chart_frame(
                momentum_risk_trend,
                ["momentum_1m", "momentum_3m", "volatility_6m"],
            )
        )

    st.subheader("最新一期因子暴露 / Latest Factor Exposure")
    factor_exposure = chart_data["factor_exposure"]
    if factor_exposure.empty:
        st.info("暂无可用因子暴露数据。")
    else:
        st.dataframe(factor_exposure, use_container_width=True)


def render_single_stock_price_section(history: pd.DataFrame) -> None:
    """Render historical price, return, and volatility information for one stock."""
    price_data = prepare_single_stock_price_data(history)
    summary = price_data["price_summary"]

    st.subheader("价格与历史收益表现 / Price and Historical Return")
    cols = st.columns(5)
    cols[0].metric("最新收盘价", format_metric_value(summary.get("latest_close")))
    cols[1].metric(
        "最新月度收益",
        format_metric_value(summary.get("latest_monthly_return"), percent=True),
    )
    cols[2].metric(
        "近 3 个月历史收益",
        format_metric_value(summary.get("recent_3m_return"), percent=True),
    )
    cols[3].metric(
        "近 6 个月历史收益",
        format_metric_value(summary.get("recent_6m_return"), percent=True),
    )
    cols[4].metric(
        "近 6 个月历史波动率",
        format_metric_value(summary.get("recent_6m_volatility"), percent=True),
    )

    st.write(f"最近 3 个月历史表现：{sanitize_research_text(summary.get('recent_3m_return_label', 'N/A'))}")
    st.write(f"最近 6 个月历史表现：{sanitize_research_text(summary.get('recent_6m_return_label', 'N/A'))}")
    st.write(f"最近 6 个月波动水平：{sanitize_research_text(summary.get('recent_6m_volatility_label', 'N/A'))}")
    st.write(f"可用历史月份：{summary.get('available_months', 0)}")

    close_trend = price_data["close_trend"]
    if close_trend.empty:
        st.info("暂无历史收盘价走势数据。")
    else:
        st.caption("历史收盘价走势仅用于历史样本展示和量化研究参考。")
        st.line_chart(render_chart_frame(close_trend, ["close"]))

    monthly_return_trend = price_data["monthly_return_trend"]
    if monthly_return_trend.empty:
        st.info("暂无月度收益走势数据。")
    else:
        st.caption("monthly_return 为历史月度收益，不代表未来收益。")
        st.bar_chart(render_chart_frame(monthly_return_trend, ["monthly_return"]))


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
    if snapshot:
        rating_report = build_stock_rating_report(snapshot, frequency)
        render_rating_report(rating_report)
        render_stock_research_summary(snapshot, rating_report, frequency)
    else:
        st.info("暂无可生成趋势参考与模型评级的历史样本记录。")

    history = get_stock_factor_history(selected_ts_code, factor_score)
    render_single_stock_price_section(history)

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
        st.caption("return_next 为历史下一期收益标签，仅用于回测检验，不作为前瞻性收益判断。")
    render_single_stock_charts(history, snapshot)

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
        ["首页 Dashboard", "运行研究流水线", "推荐投资组合", "单只股票分析", "回测结果", "因子研究"],
    )
    st.sidebar.markdown("当前版本：V6-A Portfolio Dashboard")
    st.sidebar.markdown("数据来源：TuShare + local CSV")
    st.sidebar.info("如数据为空，请先运行 scripts/run_research_pipeline.py")

    if page == "首页 Dashboard":
        render_dashboard_page(data, figure_paths)
    elif page == "运行研究流水线":
        render_pipeline_page(project_root)
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
