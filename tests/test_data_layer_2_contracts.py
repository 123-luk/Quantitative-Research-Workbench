from __future__ import annotations

import pytest

from src.data.contracts import (
    CoverageKind,
    DataRequirement,
    DatasetSpec,
    FetchStrategy,
    NativeFrequency,
    ResearchFrequency,
    RevisionPolicy,
    ScopeKind,
    coalesce_requirements,
    formation_dates,
)
from src.data.dataset_registry import DatasetRegistry, create_default_dataset_registry


EXPECTED = {
    "trade_cal": (("exchange", "cal_date"), NativeFrequency.CALENDAR_DAY, ScopeKind.ENTITY_SERIES, CoverageKind.CALENDAR_DATE, ("year",), FetchStrategy.ENTITY_DATE_RANGE),
    "stock_basic": (("ts_code",), NativeFrequency.REFERENCE_SNAPSHOT, ScopeKind.REFERENCE_SNAPSHOT, CoverageKind.GLOBAL_SNAPSHOT, ("snapshot",), FetchStrategy.REFERENCE_SNAPSHOT),
    "daily": (("ts_code", "trade_date"), NativeFrequency.TRADING_DAY, ScopeKind.MARKET_SNAPSHOT, CoverageKind.TRADE_DATE, ("year", "month"), FetchStrategy.MARKET_SNAPSHOT_BY_DATE),
    "daily_basic": (("ts_code", "trade_date"), NativeFrequency.TRADING_DAY, ScopeKind.MARKET_SNAPSHOT, CoverageKind.TRADE_DATE, ("year", "month"), FetchStrategy.MARKET_SNAPSHOT_BY_DATE),
    "adj_factor": (("ts_code", "trade_date"), NativeFrequency.TRADING_DAY, ScopeKind.MARKET_SNAPSHOT, CoverageKind.TRADE_DATE, ("year", "month"), FetchStrategy.MARKET_SNAPSHOT_BY_DATE),
    "suspend_d": (("ts_code", "trade_date"), NativeFrequency.TRADING_DAY, ScopeKind.MARKET_SNAPSHOT, CoverageKind.TRADE_DATE, ("year", "month"), FetchStrategy.MARKET_SNAPSHOT_BY_DATE),
    "index_daily": (("ts_code", "trade_date"), NativeFrequency.TRADING_DAY, ScopeKind.ENTITY_SERIES, CoverageKind.ENTITY_TRADE_DATE, ("entity", "year", "month"), FetchStrategy.ENTITY_DATE_RANGE),
    "index_weight": (("index_code", "con_code", "trade_date"), NativeFrequency.MONTHLY_SNAPSHOT, ScopeKind.ENTITY_MONTH_SNAPSHOT, CoverageKind.ENTITY_MONTH, ("entity", "year"), FetchStrategy.ENTITY_MONTH_SNAPSHOT),
}


def test_registry_contains_exact_eight_frozen_specs() -> None:
    registry = create_default_dataset_registry()
    assert set(registry.list_ids()) == set(EXPECTED)
    for dataset_id, expected in EXPECTED.items():
        spec = registry.get(dataset_id)
        assert (spec.primary_key, spec.native_frequency, spec.scope_kind, spec.coverage_kind, spec.storage_partition, spec.fetch_strategy) == expected
        assert spec.schema_version == {
            "stock_basic": "1.2", "daily_basic": "1.1",
        }.get(dataset_id, "1.0")
        assert spec.revision_policy in {RevisionPolicy.MISSING_ONLY, RevisionPolicy.EXPLICIT_REFRESH}


def test_registry_is_fresh_duplicate_and_unknown_fail_closed() -> None:
    first = create_default_dataset_registry()
    second = create_default_dataset_registry()
    assert first is not second
    with pytest.raises(ValueError, match="already registered"):
        first.register(first.get("daily"))
    with pytest.raises(KeyError, match="Unknown"):
        first.get("not_real")


def test_custom_spec_plugin_needs_no_data_manager_dispatch() -> None:
    registry = DatasetRegistry()
    spec = DatasetSpec("custom", "fake", "custom", NativeFrequency.TRADING_DAY, ScopeKind.MARKET_SNAPSHOT, ("id", "trade_date"), ("id", "trade_date"), CoverageKind.TRADE_DATE, ("year",), FetchStrategy.MARKET_SNAPSHOT_BY_DATE, "custom", "1", RevisionPolicy.MISSING_ONLY, "test")
    registry.register(spec)
    assert registry.get("custom") is spec


def test_research_frequency_monthly_uses_last_real_open_day() -> None:
    dates = ("2024-01-30", "2024-01-31", "2024-02-27", "2024-02-29")
    assert formation_dates(ResearchFrequency.DAILY, dates) == dates
    assert formation_dates(ResearchFrequency.MONTHLY, dates) == ("2024-01-31", "2024-02-29")


def test_requirement_normalization_and_coalescing_are_deterministic() -> None:
    left = DataRequirement.create("index_daily", scope={"index_code": "000300.SH"}, required_start="20240102", required_end="2024-01-03", required_fields=("pct_chg",), reason="benchmark")
    right = DataRequirement.create("index_daily", scope={"index_code": "000300.SH"}, required_start="2024-01-01", required_end="2024-01-04", required_fields=("pct_chg",), reason="risk")
    result = coalesce_requirements((left, right))
    assert len(result) == 1
    assert result[0].required_start == "2024-01-01"
    assert result[0].required_end == "2024-01-04"
    assert result[0].reason == "benchmark; risk"


@pytest.mark.parametrize("start,end", [("bad", "2024-01-01"), ("2024-02-01", "2024-01-01")])
def test_requirement_invalid_dates_fail_closed(start: str, end: str) -> None:
    with pytest.raises(ValueError):
        DataRequirement.create("daily", required_start=start, required_end=end)
