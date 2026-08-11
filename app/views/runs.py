"""Validated historical run catalog."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.components.navigation import open_results
from app.i18n import get_locale, t
from app.services.formatting import format_float, format_percentage
from app.services.run_catalog_service import RunCatalogService
from src.pipeline.config import PipelineConfig


def _output_root() -> str:
    root = Path(__file__).resolve().parents[2]
    return PipelineConfig.from_yaml(root / "config" / "config.yaml").output_dir


def render(st: object, *, navigate=None) -> None:
    locale = get_locale(st.session_state)
    st.title(t("runs.title", locale=locale))
    runs = RunCatalogService(_output_root()).list_runs()
    if not runs:
        st.info(t("runs.empty", locale=locale))
        return
    table = pd.DataFrame([
        {
            "Run ID": item.run_id,
            t("runs.created", locale=locale): item.created_at or "N/A",
            t("runs.status", locale=locale): item.status or "N/A",
            t("runs.model", locale=locale): item.model or "N/A",
            "Top N": str(item.top_n) if item.top_n is not None else "N/A",
            t("runs.portfolio", locale=locale): item.portfolio_method or "N/A",
            t("runs.benchmark", locale=locale): item.benchmark or "N/A",
            t("runs.backtest", locale=locale): item.backtest_status,
            t("runs.net_return", locale=locale): format_percentage(item.net_total_return),
            "Sharpe": format_float(item.net_sharpe_ratio),
        }
        for item in runs
    ])
    st.dataframe(table, width="stretch", hide_index=True)
    selected = st.selectbox(t("runs.select", locale=locale), tuple(item.run_id for item in runs))
    if st.button(t("runs.open", locale=locale), type="primary"):
        open_results(st.session_state, selected)
        if navigate is not None:
            navigate("results")


if __name__ == "__main__":
    import streamlit as st
    render(st, navigate=lambda name: st.switch_page({"results": "views/results.py"}[name]))
