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
from app.services.backtest_report_service import prepare_backtest_report_data  # noqa: E402
from app.services.factor_report_service import prepare_factor_report_data  # noqa: E402
from app.services.portfolio_report_service import prepare_portfolio_report_data  # noqa: E402
from app.services.stock_chart_service import prepare_single_stock_chart_data  # noqa: E402
from app.services.pipeline_config_service import (  # noqa: E402
    EQUAL_WEIGHT_LABEL,
    ERROR_IF_INSUFFICIENT,
    HIGH_SCORE_FIRST,
    INVERSE_VOLATILITY_LABEL,
    LOW_SCORE_FIRST,
    RANK_WEIGHT_LABEL,
    SUGGESTED_INVERSE_VOLATILITY_LOOKBACK,
    SUGGESTED_INVERSE_VOLATILITY_MIN_OBSERVATIONS,
    SUGGESTED_MAX_WEIGHT_PERCENT,
    USE_ALL_VALID,
    build_effective_pipeline_config,
    build_selection_summary,
    get_default_holdings_top_n,
    get_default_research_backtest_enabled,
    load_canonical_base_config,
    SUGGESTED_ANNUAL_RISK_FREE_RATE,
    SUGGESTED_RESEARCH_BACKTEST_BENCHMARK,
    SUGGESTED_RESEARCH_BACKTEST_COST_BPS,
)
from app.services.pipeline_runner_service import (  # noqa: E402
    build_pipeline_command,
    command_to_display,
    run_canonical_pipeline,
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
from app.services.research_backtest_ui_service import (  # noqa: E402
    PERCENT_METRICS,
    ResearchBacktestDashboardError,
    build_research_backtest_details,
    format_research_backtest_metric,
    load_research_backtest_dashboard,
)
from src.pipeline.config import PipelineConfig  # noqa: E402


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
    """Render canonical V5 controls plus the explicitly separate legacy path."""
    st.title("Signal / Holdings Pipeline")
    st.warning("本页运行 canonical ML → Signal → Holdings 流水线；结果仅供量化研究参考。")
    st.caption("基础配置须使用 direct PipelineConfig YAML；files source 继续由 YAML/CLI 支持。")

    config_path_text = st.text_input(
        "Pipeline 配置文件",
        value="",
        placeholder="输入 direct canonical PipelineConfig YAML 路径",
    )
    col_left, col_right = st.columns(2)
    with col_left:
        top_n = st.number_input(
            "Top N 股票数量",
            min_value=1,
            value=get_default_holdings_top_n(),
            step=1,
        )
        direction_label = st.selectbox(
            "Signal 排序方向",
            [HIGH_SCORE_FIRST, LOW_SCORE_FIRST],
        )
    with col_right:
        insufficient_label = st.selectbox(
            "股票不足 N 只时",
            [ERROR_IF_INSUFFICIENT, USE_ALL_VALID],
        )
        st.text_input("权重方式", value=EQUAL_WEIGHT_LABEL, disabled=True)

    st.subheader("Portfolio Construction")
    portfolio_method_label = st.selectbox(
        "组合构建方法",
        [EQUAL_WEIGHT_LABEL, RANK_WEIGHT_LABEL, INVERSE_VOLATILITY_LABEL],
    )
    max_weight_enabled = st.checkbox("设置单股最大权重", value=False)
    max_weight_percent = SUGGESTED_MAX_WEIGHT_PERCENT
    if max_weight_enabled:
        max_weight_percent = float(
            st.number_input(
                "单股最大权重 (%)",
                min_value=0.01,
                value=SUGGESTED_MAX_WEIGHT_PERCENT,
                step=0.5,
            )
        )
    inverse_volatility_lookback = SUGGESTED_INVERSE_VOLATILITY_LOOKBACK
    inverse_volatility_min_observations = (
        SUGGESTED_INVERSE_VOLATILITY_MIN_OBSERVATIONS
    )
    if portfolio_method_label == INVERSE_VOLATILITY_LABEL:
        inverse_left, inverse_right = st.columns(2)
        with inverse_left:
            inverse_volatility_lookback = int(
                st.number_input(
                    "波动率回看交易日",
                    min_value=2,
                    value=SUGGESTED_INVERSE_VOLATILITY_LOOKBACK,
                    step=1,
                )
            )
        with inverse_right:
            inverse_volatility_min_observations = int(
                st.number_input(
                    "最少收益观测数",
                    min_value=2,
                    max_value=inverse_volatility_lookback,
                    value=min(
                        SUGGESTED_INVERSE_VOLATILITY_MIN_OBSERVATIONS,
                        inverse_volatility_lookback,
                    ),
                    step=1,
                )
            )

    st.subheader("Research Backtest")
    enable_research_backtest = st.checkbox(
        "Enable Research Backtest",
        value=get_default_research_backtest_enabled(),
        help="Historical portfolio evaluation through the canonical pipeline.",
    )
    cost_bps = SUGGESTED_RESEARCH_BACKTEST_COST_BPS
    benchmark_code = SUGGESTED_RESEARCH_BACKTEST_BENCHMARK
    annual_risk_free_rate = SUGGESTED_ANNUAL_RISK_FREE_RATE
    if enable_research_backtest:
        backtest_left, backtest_right = st.columns(2)
        with backtest_left:
            cost_bps = float(
                st.number_input(
                    "Transaction cost (bps)",
                    min_value=0.0,
                    value=SUGGESTED_RESEARCH_BACKTEST_COST_BPS,
                    step=0.1,
                    help="One-way traded security notional fee assumption.",
                )
            )
            benchmark_code = st.text_input(
                "Benchmark code", value=SUGGESTED_RESEARCH_BACKTEST_BENCHMARK
            )
        with backtest_right:
            annual_risk_free_rate = float(
                st.number_input(
                    "Annual risk-free rate",
                    value=SUGGESTED_ANNUAL_RISK_FREE_RATE,
                    step=0.001,
                    format="%.4f",
                    help="Annual decimal scalar; for example, 0.02 means 2%.",
                )
            )
            st.caption("Annualization: 252 trading days · Base NAV: 1.0")
        st.caption(
            "Source: current-run Holdings · Schedule: Holdings dates · "
            "Effective: next trading day · Return: adjusted close-to-close · "
            "Turnover: half-L1 pre-to-target"
        )

    effective_config = None
    if config_path_text.strip():
        try:
            config_path = Path(config_path_text.strip()).expanduser()
            if not config_path.is_absolute():
                config_path = project_root / config_path
            base_config = load_canonical_base_config(config_path)
            effective_config = build_effective_pipeline_config(
                base_config,
                top_n=top_n,
                signal_direction_label=direction_label,
                insufficient_policy_label=insufficient_label,
                portfolio_method_label=portfolio_method_label,
                inverse_volatility_lookback=inverse_volatility_lookback,
                inverse_volatility_min_observations=(
                    inverse_volatility_min_observations
                ),
                max_weight_enabled=max_weight_enabled,
                max_weight_percent=max_weight_percent,
                research_backtest_enabled=enable_research_backtest,
                research_backtest_cost_bps=cost_bps,
                research_backtest_benchmark=benchmark_code,
                annual_risk_free_rate=annual_risk_free_rate,
            )
            st.subheader("本次选股设置")
            st.json(build_selection_summary(effective_config))
        except Exception as exc:
            st.error(f"Pipeline 配置无效：{exc}")
    else:
        st.info("请输入一份 direct canonical PipelineConfig YAML 后运行。示例文件仅供复制和调整。")

    if st.button(
        "运行 Signal / Holdings Pipeline",
        disabled=effective_config is None,
    ):
        try:
            with st.spinner("Running canonical ML → Signal → Holdings pipeline..."):
                result = run_canonical_pipeline(effective_config)
        except Exception as exc:
            st.error(f"Canonical pipeline failed: {exc}")
        else:
            signal = result.get("signal", {})
            holdings = result.get("holdings", {})
            run_dir = str(result.get("run_dir", ""))
            st.success("Canonical Signal / Holdings pipeline completed.")
            st.subheader("运行结果")
            st.json(
                {
                    "run_id": Path(run_dir).name if run_dir else None,
                    "run directory": run_dir or None,
                    "Signal artifact": signal.get("artifact_dir"),
                    "Holdings artifact": holdings.get("artifact_dir"),
                    "Top N": holdings.get("requested_top_n"),
                    "Holdings rows": holdings.get("rows"),
                    "Holdings dates": holdings.get("trade_date_count"),
                    "Signal direction": signal.get("signal_direction"),
                    "weighting": holdings.get("weighting"),
                    "portfolio construction": (
                        effective_config.holdings.portfolio_construction.method
                    ),
                }
            )
            research_backtest = result.get("research_backtest")
            if effective_config.research_backtest.enabled:
                if not isinstance(research_backtest, dict):
                    st.error("Research Backtest result is missing from the pipeline summary.")
                else:
                    render_research_backtest_dashboard(
                        research_backtest, effective_config
                    )

    st.divider()
    with st.expander("Legacy research / backtest pipeline"):
        render_legacy_pipeline_controls(project_root)


def render_research_backtest_dashboard(
    result: dict[str, object], config: PipelineConfig
) -> None:
    """Render exact V6 summary metrics and validated Artifact NAV values."""
    try:
        payload = load_research_backtest_dashboard(result)
    except ResearchBacktestDashboardError as exc:
        st.error(str(exc))
        return

    st.subheader("Historical Research Performance")
    metric_rows = [
        (
            ("Net Total Return", "net_total_return"),
            ("Net Annualized Return", "net_annualized_return"),
            ("Net Sharpe Ratio", "net_sharpe_ratio"),
            ("Net Max Drawdown", "net_max_drawdown"),
        ),
        (
            ("Benchmark Total Return", "benchmark_total_return"),
            ("Excess Total Return", "excess_total_return"),
            ("Tracking Error", "tracking_error"),
            ("Information Ratio", "information_ratio"),
        ),
        (
            ("Rebalance Count", "rebalance_count"),
            ("Average Turnover", "average_turnover"),
            ("Total Transaction Cost", "total_transaction_cost"),
            ("Transaction Cost Return Drag", "transaction_cost_return_drag"),
        ),
    ]
    for row in metric_rows:
        columns = st.columns(4)
        for column, (label, key) in zip(columns, row):
            value = (
                result.get("rebalance_count")
                if key == "rebalance_count"
                else payload.metrics.get(key)
            )
            column.metric(
                label,
                format_research_backtest_metric(
                    value,
                    percent=key in PERCENT_METRICS,
                    decimals=0 if key == "rebalance_count" else 2,
                ),
            )

    st.subheader("Research Portfolio NAV")
    st.line_chart(payload.nav.set_index("trade_date"))
    with st.expander("Research Backtest Details"):
        st.json(build_research_backtest_details(result, config))
        st.markdown("**All backend metrics**")
        st.json(payload.metrics)

def render_legacy_pipeline_controls(project_root: Path) -> None:
    """Render the retained legacy subprocess research controls."""
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
    st.title("回测结果 / Backtest Results")
    st.info("本页面展示的是历史样本回测结果和量化研究参考，不代表未来表现，不构成投资建议。")

    backtest_metrics = data["backtest_metrics"]
    backtest_nav = data["backtest_nav"]
    backtest_turnover = data["backtest_turnover"]
    backtest_report = prepare_backtest_report_data(
        metrics_df=backtest_metrics,
        nav_df=backtest_nav,
        turnover_df=backtest_turnover,
    )
    metrics = backtest_report["metrics"]
    labels = backtest_report["labels"]
    research_comment = backtest_report["research_comment"]
    nav_curve = backtest_report["nav_curve"]
    monthly_return = backtest_report["monthly_return"]
    drawdown = backtest_report["drawdown"]
    turnover = backtest_report["turnover"]

    st.subheader("回测核心指标 / Key Backtest Metrics")
    metric_cols = st.columns(4)
    metric_cols[0].metric("累计收益", format_metric_value(metrics.get("cumulative_return"), percent=True))
    metric_cols[1].metric("年化收益", format_metric_value(metrics.get("annual_return"), percent=True))
    metric_cols[2].metric("年化波动", format_metric_value(metrics.get("annual_volatility"), percent=True))
    metric_cols[3].metric("夏普比率", format_metric_value(metrics.get("sharpe_ratio"), decimals=2))

    metric_cols = st.columns(4)
    metric_cols[0].metric("最大回撤", format_metric_value(metrics.get("max_drawdown"), percent=True))
    metric_cols[1].metric("胜率", format_metric_value(metrics.get("win_rate"), percent=True))
    metric_cols[2].metric("平均换手率", format_metric_value(metrics.get("average_turnover"), percent=True))
    metric_cols[3].metric("调仓周期数", format_metric_value(metrics.get("n_periods"), decimals=0))

    st.subheader("回测研究摘要 / Backtest Research Comment")
    st.info(sanitize_research_text(research_comment))

    st.subheader("表现标签 / Performance Labels")
    if labels:
        st.markdown(f"- cumulative_return_label：{sanitize_research_text(labels.get('cumulative_return_label', 'N/A'))}")
        st.markdown(f"- max_drawdown_label：{sanitize_research_text(labels.get('max_drawdown_label', 'N/A'))}")
        st.markdown(f"- sharpe_label：{sanitize_research_text(labels.get('sharpe_label', 'N/A'))}")
        st.markdown(f"- win_rate_label：{sanitize_research_text(labels.get('win_rate_label', 'N/A'))}")
    else:
        st.info("暂无表现标签。")

    st.subheader("净值曲线 / NAV Curve")
    if nav_curve.empty or "nav" not in nav_curve.columns:
        st.info("暂无净值曲线数据。")
    else:
        st.line_chart(nav_curve.set_index("date")[["nav"]])

    st.subheader("月度收益 / Monthly Return")
    if monthly_return.empty or "net_return" not in monthly_return.columns:
        st.info("暂无月度收益数据。")
    else:
        st.bar_chart(monthly_return.set_index("date")[["net_return"]])
        st.caption("net_return 为历史回测月度净收益，不代表未来收益。")

    st.subheader("回撤曲线 / Drawdown")
    if drawdown.empty or "drawdown" not in drawdown.columns:
        st.info("暂无回撤数据。")
    else:
        st.line_chart(drawdown.set_index("date")[["drawdown"]])

    st.subheader("换手率 / Turnover")
    if turnover.empty or "turnover" not in turnover.columns:
        st.info("暂无换手率数据。")
    else:
        st.line_chart(turnover.set_index("date")[["turnover"]])

    st.subheader("回测数据表 / Backtest Data Tables")
    st.markdown("**backtest_metrics**")
    show_table_or_hint(backtest_metrics, "No backtest metrics found.")
    st.markdown("**nav_curve**")
    show_table_or_hint(nav_curve, "No backtest NAV curve data found.")
    st.markdown("**turnover**")
    show_table_or_hint(turnover, "No backtest turnover data found.")

    if not nav_curve.empty:
        st.download_button(
            label="下载回测净值数据 CSV",
            data=nav_curve.to_csv(index=False).encode("utf-8-sig"),
            file_name="backtest_nav_curve.csv",
            mime="text/csv",
        )
    if not monthly_return.empty:
        st.download_button(
            label="下载月度收益数据 CSV",
            data=monthly_return.to_csv(index=False).encode("utf-8-sig"),
            file_name="backtest_monthly_return.csv",
            mime="text/csv",
        )

    with st.expander("Historical Backtest Figures"):
        show_figures(figure_paths)


def render_factor_research_page(data: dict[str, pd.DataFrame]) -> None:
    """Render the factor research page."""
    st.title("因子研究 / Factor Research")
    ic_summary = data["ic_summary"]
    group_return = data["group_return"]
    long_short_return = data["long_short_return"]
    factor_report = prepare_factor_report_data(
        ic_summary=ic_summary,
        group_return=group_return,
        long_short_return=long_short_return,
    )
    ic_table = factor_report["ic_table"]
    factor_rank_table = factor_report["factor_rank_table"]
    top_factors = factor_report["top_factors"]
    ic_research_comment = factor_report["ic_research_comment"]
    group_return_summary = factor_report["group_return_summary"]
    long_short_summary = factor_report["long_short_summary"]
    return_research_comment = factor_report["return_research_comment"]

    st.info("本页面展示的是历史样本因子研究结果和量化研究参考，不代表未来表现，不构成投资建议。")

    st.subheader("因子 IC 研究摘要 / Factor IC Research Comment")
    st.info(sanitize_research_text(ic_research_comment))

    st.subheader("Top 因子 / Top Factors")
    if top_factors.empty:
        st.info("暂无 Top 因子数据。")
    else:
        st.dataframe(top_factors, use_container_width=True)
        if {"factor", "abs_mean_ic"}.issubset(top_factors.columns):
            chart_df = top_factors.copy()
            chart_df["abs_mean_ic"] = pd.to_numeric(chart_df["abs_mean_ic"], errors="coerce")
            chart_df = chart_df.dropna(subset=["abs_mean_ic"])
            if not chart_df.empty:
                st.bar_chart(chart_df.set_index("factor")[["abs_mean_ic"]])

    st.subheader("因子 IC 排名 / Factor IC Ranking")
    if factor_rank_table.empty:
        st.info("暂无因子 IC 排名数据。")
    else:
        st.dataframe(factor_rank_table, use_container_width=True)

    st.subheader("因子收益研究摘要 / Factor Return Research Comment")
    st.info(sanitize_research_text(return_research_comment))

    st.subheader("历史多空收益摘要 / Long-Short Return Summary")
    if long_short_summary.empty:
        st.info("暂无多空收益摘要数据。")
    else:
        st.dataframe(long_short_summary, use_container_width=True)
        if {"factor", "mean_long_short_return"}.issubset(long_short_summary.columns):
            chart_df = long_short_summary.copy()
            chart_df["mean_long_short_return"] = pd.to_numeric(
                chart_df["mean_long_short_return"],
                errors="coerce",
            )
            chart_df = chart_df.dropna(subset=["mean_long_short_return"])
            if not chart_df.empty:
                st.bar_chart(chart_df.set_index("factor")[["mean_long_short_return"]])
        st.caption("mean_long_short_return 为历史样本中的平均多空收益，不代表未来收益。")

    st.subheader("历史分组收益摘要 / Group Return Summary")
    if group_return_summary.empty:
        st.info("暂无分组收益摘要数据。")
    else:
        st.dataframe(group_return_summary, use_container_width=True)
        st.caption("mean_group_return 为历史样本中的平均分组收益，不代表未来收益。")

    st.subheader("原始因子研究数据 / Raw Factor Research Tables")
    with st.expander("IC Summary 原表"):
        show_table_or_hint(ic_summary, "No IC summary found.")
    with st.expander("Group Return 原表"):
        show_table_or_hint(group_return, "No group return data found.")
    with st.expander("Long-Short Return 原表"):
        show_table_or_hint(long_short_return, "No long-short return data found.")

    if not factor_rank_table.empty:
        st.download_button(
            label="下载因子 IC 排名 CSV",
            data=factor_rank_table.to_csv(index=False).encode("utf-8-sig"),
            file_name="factor_rank_table.csv",
            mime="text/csv",
        )
    if not long_short_summary.empty:
        st.download_button(
            label="下载多空收益摘要 CSV",
            data=long_short_summary.to_csv(index=False).encode("utf-8-sig"),
            file_name="factor_long_short_summary.csv",
            mime="text/csv",
        )
    if not group_return_summary.empty:
        st.download_button(
            label="下载分组收益摘要 CSV",
            data=group_return_summary.to_csv(index=False).encode("utf-8-sig"),
            file_name="factor_group_return_summary.csv",
            mime="text/csv",
        )


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
    st.sidebar.markdown("当前能力：V6 Research Backtest Dashboard")
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
