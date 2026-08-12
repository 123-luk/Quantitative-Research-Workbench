"""User-facing research tasks with exact-result action gating."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.components.navigation import open_results
from app.i18n import get_locale, t
from app.services.credential_service import CredentialService
from app.services.research_task_service import ResearchTask, ResearchTaskService
from app.services.run_catalog_service import RunCatalogService
from app.services.ui_metadata_service import dataset_label
from src.pipeline.config import PipelineConfig


def _output_root() -> str:
    root = Path(__file__).resolve().parents[2]
    return PipelineConfig.from_yaml(root / "config" / "config.yaml").output_dir


def _status_key(status: str) -> str:
    return f"task.status.{status}" if status in {"created", "running", "succeeded", "failed", "cancelled"} else "task.status.historical"


def _elapsed(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 60:
        return f"{value:.1f} s"
    return f"{int(value // 60)} min {int(value % 60)} s"


def _task_card(st: object, task: ResearchTask, locale: str, service: ResearchTaskService, navigate: object) -> None:
    summary = task.config_summary
    with st.container(border=True):
        left, middle, right = st.columns((3, 2, 2))
        left.subheader(task.name)
        left.caption(task.created_at)
        middle.metric(t("runs.status", locale=locale), t(_status_key(task.status), locale=locale))
        middle.caption(f"{t('task.stage', locale=locale)}: {t(f'progress.{task.current_stage}', locale=locale)}")
        right.metric(t("task.elapsed", locale=locale), _elapsed(task.elapsed_seconds))
        right.caption(f"{summary.get('research_start', '—')} – {summary.get('research_end', '—')}")
        st.write(
            f"{t('task.universe', locale=locale)}: {summary.get('universe_type', '—')} · "
            f"{t('new.factors', locale=locale)}: {', '.join(summary.get('factors', ())) or '—'} · "
            f"{t('new.model', locale=locale)}: {summary.get('model') or '—'} · "
            f"{t('new.portfolio', locale=locale)}: {summary.get('portfolio_method') or '—'}"
        )
        if task.progress_total:
            st.progress(min(1.0, (task.progress_completed or 0) / task.progress_total), text=f"{task.progress_completed or 0}/{task.progress_total}")
        elif task.active:
            st.info(t("task.running_help", locale=locale))
        if task.status == "failed":
            st.error(task.failure_message or t("task.failure_unknown", locale=locale))
            st.write(f"{t('task.failed_stage', locale=locale)}: {t(f'progress.{task.failure_stage or task.current_stage}', locale=locale)}")
            if task.failure_dataset:
                st.write(f"{t('task.failed_dataset', locale=locale)}: {dataset_label(task.failure_dataset, locale)}")
            if task.failure_range:
                st.write(f"{t('task.failed_range', locale=locale)}: {task.failure_range[0]} – {task.failure_range[1]}")
            st.write(f"{t('task.blocking', locale=locale)}: {t('overview.yes', locale=locale)}")
            st.write(f"{t('task.attempted', locale=locale)}: {t('task.attempted_missing', locale=locale)}")
            st.write(t(f"task.recovery.{task.failure_code or 'INTERNAL_ERROR'}", locale=locale))
        action, details = st.columns((1, 4))
        if task.can_open_results:
            if action.button(t("task.open_results", locale=locale), key=f"open_{task.task_id}", type="primary"):
                open_results(st.session_state, task.run_id)
                if navigate is not None:
                    navigate("results")
        elif task.status == "failed":
            if action.button(t("task.retry", locale=locale), key=f"retry_{task.task_id}"):
                credential = CredentialService().resolve(st.session_state.get("tushare_session_token")).reveal_for_provider()
                retried = service.retry(task.task_id, credential=credential)
                st.session_state["current_task_id"] = retried.task_id
                st.rerun()
        else:
            action.button(t("task.open_results", locale=locale), key=f"disabled_{task.task_id}", disabled=True)
            action.caption(t("task.result_not_ready", locale=locale))
        with details.expander(t("task.technical", locale=locale)):
            st.code(f"task_id: {task.task_id}\nrun_id: {task.run_id or '—'}\nschema_version: 1.0")


def render(st: object, *, navigate=None) -> None:
    locale = get_locale(st.session_state)
    st.title(t("runs.title", locale=locale))
    st.caption(t("runs.subtitle", locale=locale))
    service = ResearchTaskService(_output_root())
    tasks = service.list_tasks()
    if tasks:
        for task in tasks:
            _task_card(st, task, locale, service, navigate)
    else:
        st.info(t("runs.empty", locale=locale))

    historical_ids = {task.run_id for task in tasks if task.run_id}
    historical = [item for item in RunCatalogService(_output_root()).list_runs() if item.run_id not in historical_ids]
    if historical:
        with st.expander(t("runs.historical", locale=locale)):
            st.caption(t("runs.historical_help", locale=locale))
            st.dataframe(pd.DataFrame([
                {
                    t("runs.created", locale=locale): item.created_at or "—",
                    t("runs.status", locale=locale): t("task.status.historical", locale=locale),
                    t("runs.model", locale=locale): item.model or "—",
                    t("runs.portfolio", locale=locale): item.portfolio_method or "—",
                    t("runs.backtest", locale=locale): item.backtest_status,
                }
                for item in historical
            ]), width="stretch", hide_index=True)
            available = tuple(item.run_id for item in historical if item.backtest_status == "available")
            if available:
                selected = st.selectbox(t("runs.select", locale=locale), available, format_func=lambda value: value)
                if st.button(t("runs.open", locale=locale), type="primary"):
                    open_results(st.session_state, selected)
                    if navigate is not None:
                        navigate("results")


if __name__ == "__main__":
    import streamlit as st
    render(st, navigate=lambda name: st.switch_page({"results": "views/results.py"}[name]))
