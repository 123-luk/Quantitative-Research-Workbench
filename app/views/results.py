"""Exact-run, Artifact-backed Results page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.components.errors import ErrorPresenter
from app.i18n import get_locale, t
from app.services.formatting import format_float, format_percentage
from app.services.result_service import ResultService, ResultServiceError
from app.services.run_service import SafeRunError
from src.pipeline.config import PipelineConfig


def _output_root() -> str:
    root = Path(__file__).resolve().parents[2]
    return PipelineConfig.from_yaml(root / "config" / "config.yaml").output_dir


def _metric_rows() -> tuple[tuple[tuple[str, bool], ...], ...]:
    return (
        (
            ("net_total_return", True), ("net_annualized_return", True),
            ("net_sharpe_ratio", False), ("net_max_drawdown", True),
            ("net_annualized_volatility", True),
        ),
        (
            ("benchmark_total_return", True), ("excess_total_return", True),
            ("information_ratio", False), ("average_turnover", True),
            ("transaction_cost_return_drag", True),
        ),
    )


def _render_metrics(st: object, metrics: object, locale: str) -> None:
    for row in _metric_rows():
        columns = st.columns(len(row))
        for column, (key, percent) in zip(columns, row):
            value = metrics.get(key)  # type: ignore[union-attr]
            column.metric(t(f"metric.{key}", locale=locale), format_percentage(value) if percent else format_float(value))


def render(st: object) -> None:
    locale = get_locale(st.session_state)
    st.title(t("results.title", locale=locale))
    run_id = st.session_state.get("selected_run_id")
    if not isinstance(run_id, str) or not run_id:
        st.info(t("results.empty", locale=locale))
        return
    try:
        bundle = ResultService(_output_root()).load(run_id)
    except ResultServiceError as exc:
        ErrorPresenter.render(st, SafeRunError(type(exc).__name__, str(exc), run_id=run_id))
        return

    st.caption(t("results.exact", locale=locale, run_id=bundle.run_id, status=bundle.status or "—"))
    overview, holdings_tab, returns_tab, config_tab, artifacts_tab = st.tabs(
        tuple(t(key, locale=locale) for key in ("results.overview", "results.holdings", "results.returns", "results.config", "results.artifacts"))
    )
    with overview:
        if not bundle.research_backtest_available:
            st.info(t("results.unavailable", locale=locale))
        else:
            _render_metrics(st, bundle.metrics, locale)
            st.subheader(t("results.nav", locale=locale))
            st.line_chart(bundle.nav.set_index("trade_date"))
            st.subheader(t("results.drawdown", locale=locale))
            st.line_chart(bundle.drawdown.set_index("trade_date"))
            if bundle.drawdown_matches_metric is False:
                st.warning(t("results.drawdown_warning", locale=locale))
            cost_columns = st.columns(3)
            cost_columns[0].metric(t("results.total_turnover", locale=locale), format_float(bundle.metrics.get("total_turnover")))
            cost_columns[1].metric(t("results.total_notional", locale=locale), format_float(bundle.metrics.get("total_traded_notional")))
            cost_columns[2].metric(t("results.total_cost", locale=locale), format_float(bundle.metrics.get("total_transaction_cost")))

    with holdings_tab:
        if not bundle.holdings_available:
            st.info(t("results.holdings_missing", locale=locale))
        else:
            dates = tuple(bundle.holdings["trade_date"].sort_values(kind="stable").drop_duplicates())
            selected = st.selectbox(t("results.formation", locale=locale), dates, format_func=lambda value: pd.Timestamp(value).date().isoformat())
            view = bundle.holdings.loc[
                bundle.holdings["trade_date"] == selected,
                ["rank", "ts_code", "target_weight", "score"],
            ].copy()
            view.columns = [t(key, locale=locale) for key in ("results.rank", "results.code", "results.target_weight", "results.score")]
            st.dataframe(view, width="stretch", hide_index=True)
            st.subheader(t("results.weights", locale=locale))
            st.bar_chart(view.set_index(t("results.code", locale=locale))[[t("results.target_weight", locale=locale)]])

    with returns_tab:
        if not bundle.research_backtest_available:
            st.info(t("results.returns_missing", locale=locale))
        else:
            st.subheader(t("results.daily", locale=locale))
            columns = (
                "trade_date", "gross_return", "transaction_cost", "net_return",
                "turnover", "traded_notional", "is_rebalance",
            )
            st.dataframe(bundle.daily_returns.loc[:, columns], width="stretch", hide_index=True)
            st.subheader(t("results.monthly", locale=locale))
            st.caption(t("results.monthly_note", locale=locale))
            st.dataframe(bundle.monthly_returns, width="stretch", hide_index=True)

    with config_tab:
        if not bundle.config_summary:
            st.info(t("results.config_missing", locale=locale))
        else:
            summary = pd.DataFrame(
                [(key, value) for key, value in bundle.config_summary.items()],
                columns=[t("results.setting", locale=locale), t("results.value", locale=locale)],
            )
            st.dataframe(summary, width="stretch", hide_index=True)
        if bundle.raw_config is not None:
            with st.expander(t("task.technical", locale=locale)):
                st.caption(t("results.raw_config", locale=locale))
                st.json(dict(bundle.raw_config))

    with artifacts_tab:
        if not bundle.artifacts:
            st.info(t("results.artifacts_missing", locale=locale))
        else:
            st.dataframe(pd.DataFrame([
                {
                    t("results.artifact_type", locale=locale): item.artifact_type,
                    t("results.relative_path", locale=locale): item.relative_path,
                    t("results.schema_version", locale=locale): item.schema_version,
                    t("results.status", locale=locale): item.status,
                    t("results.upstream", locale=locale): dict(item.upstream),
                }
                for item in bundle.artifacts
            ]), width="stretch", hide_index=True)


if __name__ == "__main__":
    import streamlit as st
    render(st)
