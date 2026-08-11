"""Localized Workbench overview backed by canonical run and data metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.components.navigation import open_results
from app.i18n import get_locale, t
from app.services.credential_service import CredentialService
from app.services.data_status_service import DataLayer2StatusService
from app.services.run_catalog_service import RunCatalogService
from src.pipeline.config import PipelineConfig


def render(st: object, *, navigate: Callable[[str], None] | None = None) -> None:
    locale = get_locale(st.session_state)
    st.title(t("overview.title", locale=locale))
    st.caption(t("overview.subtitle", locale=locale))
    root = Path(__file__).resolve().parents[2]
    output_root = PipelineConfig.from_yaml(root / "config" / "config.yaml").output_dir
    runs = RunCatalogService(output_root).list_runs()
    data = DataLayer2StatusService(root / "config" / "config.yaml").get_status()
    credential = CredentialService().resolve(st.session_state.get("tushare_session_token"))
    columns = st.columns(5)
    columns[0].metric(t("overview.available_runs", locale=locale), len(runs))
    columns[1].metric(t("overview.successful", locale=locale), sum(str(item.status).lower() in {"success", "completed", "ready"} for item in runs))
    columns[2].metric(t("overview.unknown", locale=locale), sum(item.status is None for item in runs))
    columns[3].metric(t("overview.data_ready", locale=locale), sum(item.complete_units for item in data.datasets))
    columns[4].metric(t("overview.credential", locale=locale), t("provider.available" if credential.available else "provider.missing", locale=locale))
    canonical = next((item for item in runs if item.created_at is not None), None)
    if canonical is None:
        st.info(t("overview.no_runs", locale=locale))
    else:
        st.markdown(f"**{t('overview.latest', locale=locale)}:** `{canonical.run_id}` — {canonical.created_at}")
        if st.button(t("overview.open_latest", locale=locale)):
            open_results(st.session_state, canonical.run_id)
            if navigate is not None:
                navigate("results")
    st.divider()
    shortcuts = st.columns(4)
    targets = (("new_run", "overview.shortcut.new"), ("results", "overview.shortcut.results"), ("runs", "overview.shortcut.runs"), ("data", "overview.shortcut.data"))
    for column, (target, key) in zip(shortcuts, targets):
        if column.button(t(key, locale=locale), width="stretch") and navigate is not None:
            navigate(target)


if __name__ == "__main__":
    import streamlit as st
    render(st, navigate=lambda name: st.switch_page({"new_run": "views/new_run.py", "results": "views/results.py", "runs": "views/runs.py", "data": "views/data.py"}[name]))
