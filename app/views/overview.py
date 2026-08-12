"""State-aware Workbench start page with one stable primary action."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.components.navigation import open_results
from app.i18n import get_locale, t
from app.services.credential_service import CredentialService
from app.services.research_task_service import ResearchTaskService
from src.pipeline.config import PipelineConfig


def render(st: object, *, navigate: Callable[[str], None] | None = None) -> None:
    locale = get_locale(st.session_state)
    st.title(t("overview.title", locale=locale))
    st.caption(t("overview.subtitle", locale=locale))
    root = Path(__file__).resolve().parents[2]
    output_root = PipelineConfig.from_yaml(root / "config" / "config.yaml").output_dir
    tasks = ResearchTaskService(output_root).list_tasks()
    credential = CredentialService().resolve(st.session_state.get("tushare_session_token"))
    active = next((item for item in tasks if item.active), None)
    recent = tasks[0] if tasks else None

    token_col, task_col, recent_col = st.columns(3)
    token_col.metric(t("overview.credential", locale=locale), t("provider.available" if credential.available else "provider.missing", locale=locale))
    task_col.metric(t("overview.active_task", locale=locale), t("overview.yes" if active else "overview.no", locale=locale))
    recent_col.metric(t("overview.recent_status", locale=locale), t(f"task.status.{recent.status}", locale=locale) if recent else t("overview.never", locale=locale))

    st.subheader(t("overview.next", locale=locale))
    if active is not None:
        st.info(t("overview.running", locale=locale))
        label, target = t("overview.view_progress", locale=locale), "runs"
    elif recent is None:
        st.info(t("overview.first", locale=locale))
        label, target = t("overview.start", locale=locale), "new_run"
    elif recent.status == "succeeded" and recent.can_open_results:
        st.success(t("overview.succeeded", locale=locale))
        label, target = t("overview.view_result", locale=locale), "results"
    elif recent.status == "failed":
        st.error(t("overview.failed", locale=locale))
        label, target = t("overview.view_failure", locale=locale), "runs"
    else:
        st.info(t("overview.continue", locale=locale))
        label, target = t("overview.start", locale=locale), "new_run"

    disabled = target == "results" and (recent is None or not recent.can_open_results)
    if st.button(label, type="primary", disabled=disabled, width="stretch"):
        if target == "results" and recent is not None and recent.run_id:
            open_results(st.session_state, recent.run_id)
        if navigate is not None:
            navigate(target)
    if disabled:
        st.caption(t("task.result_not_ready", locale=locale))

    st.divider()
    shortcuts = st.columns(3)
    for column, (target_name, key) in zip(shortcuts, (("new_run", "overview.shortcut.new"), ("runs", "overview.shortcut.runs"), ("data", "overview.shortcut.data"))):
        if column.button(t(key, locale=locale), width="stretch") and navigate is not None:
            navigate(target_name)


if __name__ == "__main__":
    import streamlit as st
    render(st, navigate=lambda name: st.switch_page({"new_run": "views/new_run.py", "results": "views/results.py", "runs": "views/runs.py", "data": "views/data.py"}[name]))
