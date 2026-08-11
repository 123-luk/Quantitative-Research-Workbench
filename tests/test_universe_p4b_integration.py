from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import CoverageLedger, PartitionedParquetStore, RawParquetStore
from src.data.contracts import ResearchFrequency
from src.data.dataset_registry import create_default_dataset_registry
from src.data.preparation import DataPreparationService
from src.universe import CanonicalUniverseDataSource, UniverseDataUnavailable, UniverseService, UniverseSpec


STOCK_FIELDS = (
    "ts_code", "symbol", "name", "area", "industry", "market", "exchange",
    "curr_type", "list_status", "list_date", "delist_date",
)


class FakeUniverseProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get_stock_basic(self, list_status: str = "L") -> pd.DataFrame:
        self.calls.append(("stock_basic", {"list_status": list_status}))
        if list_status != "L":
            return pd.DataFrame(columns=STOCK_FIELDS)
        return pd.DataFrame(
            [
                {"ts_code": "600001.SH", "symbol": "600001", "name": "A", "area": "China", "industry": "Test", "market": "主板", "exchange": "SSE", "curr_type": "CNY", "list_status": "L", "list_date": "20100101", "delist_date": None},
                {"ts_code": "000001.SZ", "symbol": "000001", "name": "B", "area": "China", "industry": "Test", "market": "主板", "exchange": "SZSE", "curr_type": "CNY", "list_status": "L", "list_date": "20100101", "delist_date": None},
            ],
            columns=STOCK_FIELDS,
        )

    def get_index_weight(self, index_code: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        self.calls.append(("index_weight", {"index_code": index_code, "start_date": start_date, "end_date": end_date}))
        assert end_date is not None
        month = end_date[:6]
        snapshot = "20231229" if month == "202312" else "20240131"
        members = ("600001.SH",) if month == "202312" else ("600001.SH", "000001.SZ")
        return pd.DataFrame(
            [{"index_code": index_code, "con_code": code, "trade_date": snapshot, "weight": 100.0 / len(members)} for code in members],
            columns=("index_code", "con_code", "trade_date", "weight"),
        )


def components(tmp_path: Path):
    registry = create_default_dataset_registry()
    ledger = CoverageLedger(tmp_path / "metadata" / "catalog.sqlite")
    curated = PartitionedParquetStore(tmp_path / "curated")
    raw = RawParquetStore(tmp_path / "raw")
    return registry, ledger, curated, raw


def test_index_requirements_prepare_missing_only_then_resolve_same_snapshot(tmp_path) -> None:
    registry, ledger, curated, raw = components(tmp_path)
    provider = FakeUniverseProvider()
    preparation = DataPreparationService(registry=registry, ledger=ledger, curated_store=curated, raw_store=raw)
    spec = UniverseSpec.index("000300.SH")
    requirements = UniverseService().requirements(spec, start="2024-01-01", end="2024-01-31", frequency=ResearchFrequency.MONTHLY)

    first = preparation.ensure(requirements, client=provider)
    assert first.provider_calls == 3
    assert [name for name, _ in provider.calls].count("index_weight") == 2
    assert [name for name, _ in provider.calls].count("stock_basic") == 3

    source = CanonicalUniverseDataSource(registry=registry, ledger=ledger, store=curated, stock_basic_as_of="2024-01-31", index_weight_start="2023-12-01")
    before = UniverseService().resolve(spec, "2024-01-31", source)
    assert before.securities == ("000001.SZ", "600001.SH")
    assert before.source_as_of == "2024-01-31"

    provider.calls.clear()
    second = preparation.ensure(requirements, client=provider)
    after = UniverseService().resolve(spec, "2024-01-31", source)
    assert second.provider_calls == 0
    assert provider.calls == []
    assert after == before


def test_canonical_source_fails_closed_when_ledger_units_are_absent(tmp_path) -> None:
    registry, ledger, curated, _raw = components(tmp_path)
    source = CanonicalUniverseDataSource(registry=registry, ledger=ledger, store=curated, stock_basic_as_of="2024-01-31", index_weight_start="2023-12-01")
    with pytest.raises(UniverseDataUnavailable, match="coverage is unavailable"):
        source.stock_basic()
