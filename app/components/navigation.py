"""Five-route workbench navigation and exact run handoff helpers."""

from __future__ import annotations

from typing import MutableMapping


NAVIGATION_ROUTES = ("Overview", "New Run", "Results", "Runs", "Data")
PAGE_PATHS = {
    "overview": "views/overview.py",
    "new_run": "views/new_run.py",
    "results": "views/results.py",
    "runs": "views/runs.py",
    "data": "views/data.py",
}


def page_path(page_key: str) -> str:
    try:
        return PAGE_PATHS[page_key]
    except KeyError as exc:
        raise ValueError(f"Unknown page key: {page_key!r}") from exc


def open_results(
    state: MutableMapping[str, object],
    run_id: str,
    query_params: MutableMapping[str, object] | None = None,
) -> None:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Results require an exact non-empty run_id.")
    exact_run_id = run_id.strip()
    state["selected_run_id"] = exact_run_id
    state["current_page"] = "Results"
    if query_params is not None:
        query_params["run_id"] = exact_run_id


def initialize_session_state(state: MutableMapping[str, object]) -> None:
    state.setdefault("locale", "zh-CN")
    state.setdefault("current_page", "Overview")
    if state["current_page"] not in NAVIGATION_ROUTES:
        state["current_page"] = "Overview"
    state.setdefault("draft_config", None)
    state.setdefault("current_run_id", None)
    state.setdefault("selected_run_id", None)
    state.setdefault("last_run_status", None)
