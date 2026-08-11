"""Workbench overview page."""

from __future__ import annotations

from pathlib import Path

from app.components.navigation import open_results
from app.services.run_catalog_service import RunCatalogService
from src.pipeline.config import PipelineConfig


def render(st: object) -> None:
    st.title("Quant Factor System")
    st.subheader("Quant Research Workbench")
    st.caption("Registry-driven research configuration backed by the canonical pipeline.")
    columns = st.columns(3)
    capabilities = (
        ("Data Layer", "Local cache readiness and canonical market data inputs."),
        ("Factor Research", "Registered factors, preprocessing, composition, and evaluation."),
        ("ML", "Walk-forward training with registry-owned model schemas."),
        ("Signal", "Canonical prediction-to-signal handoff."),
        ("Portfolio Construction", "Registry-driven weighting, risk estimation, and constraints."),
        ("Research Backtest", "Artifact-backed accounting and performance metrics."),
    )
    for index, (name, description) in enumerate(capabilities):
        with columns[index % 3]:
            st.markdown(f"#### {name}")
            st.caption(description)
    root = Path(__file__).resolve().parents[2]
    output_root = PipelineConfig.from_yaml(root / "config" / "config.yaml").output_dir
    runs = RunCatalogService(output_root).list_runs()
    st.divider()
    summary = st.columns(2)
    summary[0].metric("Available Runs", len(runs))
    summary[1].metric("Runs With Canonical Status", sum(item.status is not None for item in runs))
    canonical = next((item for item in runs if item.created_at is not None), None)
    if canonical is not None:
        st.markdown(f"**Latest canonical run:** `{canonical.run_id}` — {canonical.created_at}")
        if st.button("Open Latest Canonical Run"):
            open_results(st.session_state, canonical.run_id)
            st.rerun()
    if st.button("New Research Run", type="primary"):
        st.session_state["current_page"] = "New Run"
        st.rerun()
