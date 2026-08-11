"""Exact-run, Artifact-backed Results page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.components.errors import ErrorPresenter
from app.services.formatting import format_float, format_percentage
from app.services.result_service import ResultService, ResultServiceError
from app.services.run_service import SafeRunError
from src.pipeline.config import PipelineConfig


def _output_root() -> str:
    root = Path(__file__).resolve().parents[2]
    return PipelineConfig.from_yaml(root / "config" / "config.yaml").output_dir


def _metric_rows() -> tuple[tuple[tuple[str, str, bool], ...], ...]:
    return (
        (
            ("Net Total Return", "net_total_return", True),
            ("Net Annualized Return", "net_annualized_return", True),
            ("Net Sharpe Ratio", "net_sharpe_ratio", False),
            ("Net Max Drawdown", "net_max_drawdown", True),
            ("Net Annualized Volatility", "net_annualized_volatility", True),
        ),
        (
            ("Benchmark Total Return", "benchmark_total_return", True),
            ("Excess Total Return", "excess_total_return", True),
            ("Information Ratio", "information_ratio", False),
            ("Average Turnover", "average_turnover", True),
            ("Transaction Cost Drag", "transaction_cost_return_drag", True),
        ),
    )


def _render_metrics(st: object, metrics: object) -> None:
    for row in _metric_rows():
        columns = st.columns(len(row))
        for column, (label, key, percent) in zip(columns, row):
            value = metrics.get(key)  # type: ignore[union-attr]
            column.metric(label, format_percentage(value) if percent else format_float(value))


def render(st: object) -> None:
    st.title("Results")
    run_id = st.session_state.get("selected_run_id")
    if not isinstance(run_id, str) or not run_id:
        st.info("Select or complete an exact run before opening Results.")
        return
    try:
        bundle = ResultService(_output_root()).load(run_id)
    except ResultServiceError as exc:
        ErrorPresenter.render(st, SafeRunError(type(exc).__name__, str(exc), run_id=run_id))
        return

    st.caption(f"Exact run: {bundle.run_id} | Status: {bundle.status or 'N/A'}")
    overview, holdings_tab, returns_tab, config_tab, artifacts_tab = st.tabs(
        ("Overview", "Holdings", "Returns", "Config", "Artifacts")
    )
    with overview:
        if not bundle.research_backtest_available:
            st.info("Research Backtest results are not available for this run.")
        else:
            _render_metrics(st, bundle.metrics)
            st.subheader("Portfolio Net NAV vs Benchmark NAV")
            st.line_chart(bundle.nav.set_index("trade_date"))
            st.subheader("Drawdown")
            st.line_chart(bundle.drawdown.set_index("trade_date"))
            if bundle.drawdown_matches_metric is False:
                st.warning(
                    "Display-derived drawdown does not match the canonical metrics.json value. "
                    "The metric card remains the canonical truth."
                )
            cost_columns = st.columns(3)
            cost_columns[0].metric("Total Turnover", format_float(bundle.metrics.get("total_turnover")))
            cost_columns[1].metric("Total Traded Notional", format_float(bundle.metrics.get("total_traded_notional")))
            cost_columns[2].metric("Total Transaction Cost", format_float(bundle.metrics.get("total_transaction_cost")))

    with holdings_tab:
        if not bundle.holdings_available:
            st.info("Holdings results are not available for this run.")
        else:
            dates = tuple(bundle.holdings["trade_date"].sort_values(kind="stable").drop_duplicates())
            selected = st.selectbox("Formation Date", dates, format_func=lambda value: pd.Timestamp(value).date().isoformat())
            view = bundle.holdings.loc[
                bundle.holdings["trade_date"] == selected,
                ["rank", "ts_code", "target_weight", "score"],
            ].copy()
            view.columns = ["Rank", "Code", "Target Weight", "Score"]
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.subheader("Portfolio Weight Distribution")
            st.bar_chart(view.set_index("Code")[["Target Weight"]])

    with returns_tab:
        if not bundle.research_backtest_available:
            st.info("Daily and monthly returns are not available for this run.")
        else:
            st.subheader("Daily Returns")
            columns = (
                "trade_date", "gross_return", "transaction_cost", "net_return",
                "turnover", "traded_notional", "is_rebalance",
            )
            st.dataframe(bundle.daily_returns.loc[:, columns], use_container_width=True, hide_index=True)
            st.subheader("Monthly Return Table")
            st.caption("Derived display view compounded only from canonical daily net_return; not a backend metric.")
            st.dataframe(bundle.monthly_returns, use_container_width=True, hide_index=True)

    with config_tab:
        if not bundle.config_summary:
            st.info("Canonical run configuration is not available.")
        else:
            summary = pd.DataFrame(
                [(key, value) for key, value in bundle.config_summary.items()],
                columns=["Setting", "Value"],
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
        if bundle.raw_config is not None:
            with st.expander("Raw Config JSON"):
                st.json(dict(bundle.raw_config))

    with artifacts_tab:
        if not bundle.artifacts:
            st.info("No validated canonical Artifacts are available for this run.")
        else:
            st.dataframe(pd.DataFrame([
                {
                    "Artifact Type": item.artifact_type,
                    "Relative Path": item.relative_path,
                    "Schema Version": item.schema_version,
                    "Status": item.status,
                    "Upstream Lineage": dict(item.upstream),
                }
                for item in bundle.artifacts
            ]), use_container_width=True, hide_index=True)

