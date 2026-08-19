from __future__ import annotations

from datetime import datetime, timezone

from src.data.contracts import DataRequirement
from src.data.coverage_ledger import CoverageLedger, CoverageRecord
from src.data.coverage_planner import MissingDataPlanner, scope_key
from src.data.dataset_registry import create_default_dataset_registry


def record(dataset: str, scope: tuple[tuple[str, str], ...], unit: str) -> CoverageRecord:
    return CoverageRecord(dataset, scope_key(scope), unit, "COMPLETE", 1, "1.0", "hash", "fingerprint", datetime.now(timezone.utc).isoformat())


def test_ledger_and_planner_find_internal_and_disjoint_gaps(tmp_path) -> None:
    ledger = CoverageLedger(tmp_path / "catalog.sqlite")
    registry = create_default_dataset_registry()
    requirement = DataRequirement.create("daily", scope="CN_A", required_start="2024-01-02", required_end="2024-01-08")
    open_dates = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08")
    ledger.mark_complete((record("daily", requirement.scope, "2024-01-02"), record("daily", requirement.scope, "2024-01-04")))
    plan = MissingDataPlanner(registry, ledger, open_dates=lambda _s, _e: open_dates).plan((requirement,))[0]
    assert plan.complete_units == ("2024-01-02", "2024-01-04")
    assert plan.missing_units == ("2024-01-03", "2024-01-05", "2024-01-08")
    assert tuple(task.units for task in plan.grouped_fetch_tasks) == (("2024-01-03",), ("2024-01-05",), ("2024-01-08",))


def test_entity_scope_and_month_units_are_independent(tmp_path) -> None:
    ledger = CoverageLedger(tmp_path / "catalog.sqlite")
    registry = create_default_dataset_registry()
    a = DataRequirement.create("index_weight", scope={"index_code": "A"}, required_start="2024-01-01", required_end="2024-02-29")
    b = DataRequirement.create("index_weight", scope={"index_code": "B"}, required_start="2024-01-01", required_end="2024-02-29")
    ledger.mark_complete((record("index_weight", a.scope, "2024-01"), record("index_weight", a.scope, "2024-02")))
    plans = MissingDataPlanner(registry, ledger).plan((a, b))
    by_scope = {dict(plan.requirement.scope)["index_code"]: plan for plan in plans}
    assert by_scope["A"].ready
    assert by_scope["B"].missing_units == ("2024-01", "2024-02")


def test_calendar_uses_every_natural_day_and_old_json_is_not_truth(tmp_path) -> None:
    (tmp_path / "data_status.json").write_text('{"trade_cal":{"start_date":"2000-01-01","end_date":"2099-12-31"}}')
    ledger = CoverageLedger(tmp_path / "catalog.sqlite")
    requirement = DataRequirement.create("trade_cal", scope={"exchange": "SSE"}, required_start="2024-01-05", required_end="2024-01-07")
    plan = MissingDataPlanner(create_default_dataset_registry(), ledger).plan((requirement,))[0]
    assert plan.required_units == ("2024-01-05", "2024-01-06", "2024-01-07")
    assert plan.missing_units == plan.required_units


def test_fetch_event_transaction_records_no_credentials(tmp_path) -> None:
    ledger = CoverageLedger(tmp_path / "catalog.sqlite")
    fetch_id = ledger.start_fetch("daily", "{}", ("2024-01-02",), "2024-01-01T00:00:00Z")
    ledger.finish_fetch(fetch_id, status="FAILED", finished_at="2024-01-01T00:00:01Z", error_type="ProviderError")
    event = ledger.fetch_events()[0]
    assert event["status"] == "FAILED"
    assert event["error_type"] == "ProviderError"
