"""User-facing research tasks with exact-result action gating."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from app.components.navigation import open_results, page_path
from app.i18n import get_locale, t
from app.services.credential_service import CredentialService
from app.services.research_task_service import ResearchTask, ResearchTaskService, TaskClearError
from app.services.run_catalog_service import RunCatalogService
from app.services.ui_metadata_service import dataset_label, display_value, factor_label
from src.pipeline.config import PipelineConfig


def _output_root() -> str:
    root = Path(__file__).resolve().parents[2]
    return PipelineConfig.from_yaml(root / "config" / "config.yaml").output_dir


def _status_key(status: str) -> str:
    return f"task.status.{status}" if status in {"created", "running", "succeeded", "failed", "cancelled"} else "task.status.historical"


_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _display_time(value: str | None, locale: str) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        local = parsed.astimezone(_SHANGHAI)
    except (TypeError, ValueError, OverflowError):
        return "—"
    if locale == "zh-CN":
        return local.strftime("%Y-%m-%d %H:%M:%S")
    return local.strftime("%Y-%m-%d %H:%M:%S CST")


def _elapsed(value: float | None, locale: str) -> str:
    if value is None:
        return "—"
    if value < 60:
        return f"{value:.1f} 秒" if locale == "zh-CN" else f"{value:.1f} s"
    minutes, seconds = int(value // 60), int(value % 60)
    return f"{minutes} 分 {seconds} 秒" if locale == "zh-CN" else f"{minutes} min {seconds} s"


def _progress_detail(value: str | None, locale: str) -> str | None:
    if not value:
        return None
    fixed = {
        "Downloading only ledger-missing coverage.": "task.progress_preparing",
        "All required coverage is COMPLETE; provider calls are skipped.": "task.progress_ready",
    }
    if value in fixed:
        return t(fixed[value], locale=locale)
    dataset_id, separator, unit = value.partition(" · ")
    if separator and dataset_id and unit:
        return f"{dataset_label(dataset_id, locale)} · {unit}"
    return value


def _diagnostic_value(kind: str, value: str, locale: str) -> str:
    if kind == "canonical_status" and value.startswith("READABLE_ROWS:"):
        return t("task.canonical.READABLE_ROWS", locale=locale, rows=value.partition(":")[2])
    key = f"task.{kind}.{value}"
    translated = t(key, locale=locale)
    return value if translated == f"〔{key}〕" else translated


def _task_card(st: object, task: ResearchTask, locale: str, service: ResearchTaskService, navigate: object) -> None:
    summary = task.config_summary
    unavailable = t("task.diagnostic_unavailable", locale=locale)
    with st.container(border=True):
        left, middle, right = st.columns((3, 2, 2))
        left.subheader(t("task.display_name", locale=locale, start=summary.get("research_start", "—"), end=summary.get("research_end", "—")))
        left.caption(_display_time(task.created_at, locale))
        middle.metric(t("runs.status", locale=locale), t(_status_key(task.status), locale=locale))
        middle.caption(f"{t('task.stage', locale=locale)}: {t(f'progress.{task.current_stage}', locale=locale)}")
        right.metric(t("task.elapsed", locale=locale), _elapsed(task.elapsed_seconds, locale))
        right.caption(f"{summary.get('research_start', '—')} – {summary.get('research_end', '—')}")
        st.write(
            f"{t('task.universe', locale=locale)}: {display_value(summary.get('universe_type', '—'), locale)} · "
            f"{t('new.factors', locale=locale)}: {', '.join(factor_label(str(value), locale) for value in summary.get('factors', ())) or '—'} · "
            f"{t('new.model', locale=locale)}: {display_value(summary.get('model') or '—', locale)} · "
            f"{t('new.portfolio', locale=locale)}: {display_value(summary.get('portfolio_method') or '—', locale)}"
        )
        if task.progress_total:
            st.progress(min(1.0, (task.progress_completed or 0) / task.progress_total), text=f"{task.progress_completed or 0}/{task.progress_total}")
            if task.progress_detail:
                st.caption(_progress_detail(task.progress_detail, locale))
        elif task.active:
            st.info(t("task.running_help", locale=locale))
        if task.status == "failed":
            st.error(t("task.failure_summary", locale=locale))
            st.write(f"{t('task.failed_stage', locale=locale)}: {t(f'progress.{task.failure_stage or task.current_stage}', locale=locale)}")
            if task.failure_dataset:
                st.write(f"{t('task.failed_dataset', locale=locale)}: {dataset_label(task.failure_dataset, locale)}")
            if task.failure_range:
                st.write(f"{t('task.failed_range', locale=locale)}: {task.failure_range[0]} — {task.failure_range[1]}")
            st.write(f"{t('task.ledger_status', locale=locale)}: {_diagnostic_value('ledger', task.ledger_status, locale) if task.ledger_status else unavailable}")
            st.write(f"{t('task.canonical_status', locale=locale)}: {_diagnostic_value('canonical_status', task.canonical_status, locale) if task.canonical_status else unavailable}")
            st.write(f"{t('task.consistency_issue', locale=locale)}: {_diagnostic_value('consistency', task.consistency_issue, locale) if task.consistency_issue else unavailable}")
            st.write(f"{t('task.repair_action', locale=locale)}: {_diagnostic_value('repair', task.repair_action, locale) if task.repair_action else unavailable}")
            if task.provider_attempts is not None:
                st.write(f"{t('task.provider_attempts', locale=locale)}: {task.provider_attempts}")
            if task.network_category:
                st.write(f"{t('task.network_category', locale=locale)}: {t(f'task.network.{task.network_category}', locale=locale)}")
            failure_code = task.failure_code or "INTERNAL_ERROR"
            st.write(f"{t('task.error_category', locale=locale)}: {t(f'task.error.{failure_code}', locale=locale)}")
            st.write(f"{t('task.reason', locale=locale)}: {t(f'task.reason.{failure_code}', locale=locale)}")
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
            retry_key = f"retrying_{task.task_id}"
            retrying = st.session_state.get(retry_key) is True
            if action.button(t("task.retry", locale=locale), key=f"retry_{task.task_id}", disabled=retrying):
                st.session_state[retry_key] = True
                token_key = "tushare_official_session_token" if task.provider_id == "tushare_official" else "tushare_proxy_session_token"
                credential = CredentialService().resolve(st.session_state.get(token_key), provider_id=task.provider_id).reveal_for_provider()
                try:
                    retried = service.retry(task.task_id, credential=credential)
                    st.session_state["current_task_id"] = retried.task_id
                finally:
                    st.session_state.pop(retry_key, None)
                st.rerun()
        else:
            action.button(t("task.open_results", locale=locale), key=f"disabled_{task.task_id}", disabled=True)
            action.caption(t("task.result_not_ready", locale=locale))
        with details.expander(t("task.technical", locale=locale)):
            st.write(f"{t('task.technical_id', locale=locale)}: `{task.task_id}`")
            st.write(f"{t('task.technical_result', locale=locale)}: {t('overview.yes' if task.run_id else 'overview.no', locale=locale)}")
            st.write(f"{t('task.technical_completed', locale=locale)}: {', '.join(t(f'progress.{stage}', locale=locale) for stage in task.completed_stages) or '—'}")
            if task.failure_code:
                st.write(f"{t('task.error_category', locale=locale)}: {t(f'task.error.{task.failure_code}', locale=locale)}")

        clear_col, clear_help = st.columns((1, 4))
        clear_requested = st.session_state.get("confirm_clear_task") == task.task_id
        if clear_col.button(t("task.clear", locale=locale), key=f"clear_{task.task_id}", disabled=not task.can_clear):
            st.session_state["confirm_clear_task"] = task.task_id
            st.rerun()
        if not task.can_clear:
            clear_help.caption(t("task.clear_active", locale=locale))
        if clear_requested:
            message_key = "task.clear_confirm_success" if task.run_id else "task.clear_confirm_record"
            st.warning(t(message_key, locale=locale))
            confirm, cancel = st.columns(2)
            if confirm.button(t("task.clear_confirm", locale=locale), key=f"confirm_clear_{task.task_id}", type="primary"):
                try:
                    service.clear(task.task_id)
                except TaskClearError:
                    st.error(t("task.clear_failed", locale=locale))
                else:
                    st.session_state.pop("confirm_clear_task", None)
                    st.success(t("task.clear_done", locale=locale))
                    st.rerun()
            if cancel.button(t("task.clear_cancel", locale=locale), key=f"cancel_clear_{task.task_id}"):
                st.session_state.pop("confirm_clear_task", None)
                st.rerun()


def render(st: object, *, navigate=None, service: ResearchTaskService | None = None) -> None:
    locale = get_locale(st.session_state)
    st.title(t("runs.title", locale=locale))
    st.caption(t("runs.subtitle", locale=locale))
    service = service or ResearchTaskService(_output_root())
    def render_tasks() -> tuple[ResearchTask, ...]:
        tasks = service.list_tasks()
        if tasks:
            for task in tasks:
                _task_card(st, task, locale, service, navigate)
        else:
            st.info(t("runs.empty", locale=locale))
        return tasks

    initial_tasks = service.list_tasks()
    if any(task.active for task in initial_tasks) and callable(getattr(st, "fragment", None)):
        @st.fragment(run_every="3s")
        def live_tasks() -> tuple[ResearchTask, ...]:
            current = render_tasks()
            if not any(task.active for task in current):
                st.rerun(scope="app")
            return current
        tasks = live_tasks()
    else:
        tasks = render_tasks()

    historical_ids = {task.run_id for task in tasks if task.run_id}
    hidden_runs = service.hidden_run_ids()
    historical = [item for item in RunCatalogService(_output_root()).list_runs() if item.run_id not in historical_ids and item.run_id not in hidden_runs]
    if historical:
        with st.expander(t("runs.historical", locale=locale)):
            st.caption(t("runs.historical_help", locale=locale))
            st.dataframe(pd.DataFrame([
                {
                    t("runs.created", locale=locale): _display_time(item.created_at, locale),
                    t("runs.status", locale=locale): t("task.status.historical", locale=locale),
                    t("runs.model", locale=locale): display_value(item.model or "—", locale),
                    t("runs.portfolio", locale=locale): display_value(item.portfolio_method or "—", locale),
                    t("runs.backtest", locale=locale): t(f"runs.backtest.{item.backtest_status}", locale=locale),
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
            historical_run_ids = tuple(item.run_id for item in historical)
            selected_clear = st.selectbox(t("runs.clear_select", locale=locale), historical_run_ids, key="historical_clear_selection")
            if st.button(t("task.clear", locale=locale), key="clear_historical"):
                st.session_state["confirm_clear_historical"] = selected_clear
                st.rerun()
            if st.session_state.get("confirm_clear_historical") == selected_clear:
                st.warning(t("task.clear_confirm_historical", locale=locale))
                confirm, cancel = st.columns(2)
                if confirm.button(t("task.clear_confirm", locale=locale), key="confirm_clear_historical"):
                    try:
                        service.clear_historical_run(selected_clear)
                    except TaskClearError:
                        st.error(t("task.clear_failed", locale=locale))
                    else:
                        st.session_state.pop("confirm_clear_historical", None)
                        st.rerun()
                if cancel.button(t("task.clear_cancel", locale=locale), key="cancel_clear_historical"):
                    st.session_state.pop("confirm_clear_historical", None)
                    st.rerun()


if __name__ == "__main__":
    import streamlit as st
    render(st, navigate=lambda name: st.switch_page(page_path(name)))
