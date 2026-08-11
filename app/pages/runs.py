"""Validated historical run catalog."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.components.navigation import open_results
from app.services.formatting import format_float, format_percentage
from app.services.run_catalog_service import RunCatalogService
from src.pipeline.config import PipelineConfig


def _output_root() -> str:
    root = Path(__file__).resolve().parents[2]
    return PipelineConfig.from_yaml(root / "config" / "config.yaml").output_dir


def render(st: object) -> None:
    st.title("Runs")
    runs = RunCatalogService(_output_root()).list_runs()
    if not runs:
        st.info("No validated canonical runs are available.")
        return
    table = pd.DataFrame([
        {
            "Run ID": item.run_id,
            "Created": item.created_at or "N/A",
            "Status": item.status or "N/A",
            "Model": item.model or "N/A",
            "Top N": item.top_n if item.top_n is not None else "N/A",
            "Portfolio Method": item.portfolio_method or "N/A",
            "Benchmark": item.benchmark or "N/A",
            "Backtest Status": item.backtest_status,
            "Net Return": format_percentage(item.net_total_return),
            "Sharpe": format_float(item.net_sharpe_ratio),
        }
        for item in runs
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)
    selected = st.selectbox("Select Run", tuple(item.run_id for item in runs))
    if st.button("Open Results", type="primary"):
        open_results(st.session_state, selected)
        st.rerun()

