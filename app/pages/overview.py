"""Workbench overview page."""

from __future__ import annotations


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
    if st.button("New Research Run", type="primary"):
        st.session_state["current_page"] = "New Run"
        st.rerun()

