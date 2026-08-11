"""P1 exact-identity Results route contract."""

from __future__ import annotations


def render(st: object) -> None:
    st.title("Results")
    run_id = st.session_state.get("selected_run_id")
    if not run_id:
        st.info("Select or complete a run to open exact results.")
        return
    st.markdown("**Results for run:**")
    st.code(str(run_id))
    st.info("Artifact-backed analytics are available in V9-P2.")

