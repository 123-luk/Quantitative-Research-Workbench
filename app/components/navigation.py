"""Five-route workbench navigation and exact run handoff helpers."""

from __future__ import annotations

from typing import MutableMapping


NAVIGATION_ROUTES = ("Overview", "New Run", "Results", "Runs", "Data")


def open_results(state: MutableMapping[str, object], run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Results require an exact non-empty run_id.")
    state["selected_run_id"] = run_id.strip()
    state["current_page"] = "Results"


def initialize_session_state(state: MutableMapping[str, object]) -> None:
    state.setdefault("locale", "zh-CN")
    state.setdefault("current_page", "Overview")
    if state["current_page"] not in NAVIGATION_ROUTES:
        state["current_page"] = "Overview"
    state.setdefault("draft_config", None)
    state.setdefault("current_run_id", None)
    state.setdefault("selected_run_id", None)
    state.setdefault("last_run_status", None)
