from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import CoverageLedger, PartitionedParquetStore, RawParquetStore
from src.data.dataset_registry import create_default_dataset_registry
from src.data.preparation import DataPreparationService
from src.research_data import AdjustedPriceDataUnavailable, AdjustedPriceRequest, AdjustedPriceService, CanonicalAdjustedPriceDataSource


DAILY_FIELDS = ("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount")
ADJ_FIELDS = ("ts_code", "trade_date", "adj_factor")


class FakeAdjustedProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_daily(self, **kwargs: object) -> pd.DataFrame:
        day = str(kwargs["trade_date"])
        self.calls.append(("daily", day))
        close = 10.0 if day == "20240102" else 11.0
        return pd.DataFrame(
            [{"ts_code": "600001.SH", "trade_date": day, "open": close - 1, "high": close + 1, "low": close - 2, "close": close, "pre_close": close - 0.5, "change": 0.5, "pct_chg": 5.0, "vol": 100.0, "amount": 200.0}],
            columns=DAILY_FIELDS,
        )

    def get_adj_factor(self, **kwargs: object) -> pd.DataFrame:
        day = str(kwargs["trade_date"])
        self.calls.append(("adj_factor", day))
        value = 2.0 if day == "20240102" else 2.1
        return pd.DataFrame([{"ts_code": "600001.SH", "trade_date": day, "adj_factor": value}], columns=ADJ_FIELDS)


def components(tmp_path: Path):
    registry = create_default_dataset_registry()
    ledger = CoverageLedger(tmp_path / "metadata" / "catalog.sqlite")
    curated = PartitionedParquetStore(tmp_path / "curated")
    raw = RawParquetStore(tmp_path / "raw")
    return registry, ledger, curated, raw


def test_adjusted_requirements_prepare_once_then_canonical_resolution_is_offline(tmp_path: Path) -> None:
    registry, ledger, curated, raw = components(tmp_path)
    dates = ("2024-01-02", "2024-01-03")
    provider = FakeAdjustedProvider()
    preparation = DataPreparationService(registry=registry, ledger=ledger, curated_store=curated, raw_store=raw, open_dates=lambda start, end: tuple(item for item in dates if start <= item <= end))
    requirements = AdjustedPriceService.requirements(start_date=dates[0], end_date=dates[-1], scope="CN_A")
    first = preparation.ensure(requirements, client=provider)
    assert first.provider_calls == 4
    assert provider.calls == [("adj_factor", "20240102"), ("adj_factor", "20240103"), ("daily", "20240102"), ("daily", "20240103")]

    source = CanonicalAdjustedPriceDataSource(registry=registry, ledger=ledger, store=curated, scope="CN_A")
    request = AdjustedPriceRequest(("600001.SH",), dates)
    before = AdjustedPriceService(source).compute(request)
    assert tuple(before.frame["adj_close"]) == (20.0, 23.1)

    provider.calls.clear()
    second = preparation.ensure(requirements, client=provider)
    after = AdjustedPriceService(source).compute(request)
    assert second.provider_calls == 0
    assert provider.calls == []
    pd.testing.assert_frame_equal(after.frame, before.frame, check_exact=True)
    assert after.source_identity == before.source_identity


def test_canonical_adjusted_source_requires_ledger_proof(tmp_path: Path) -> None:
    registry, ledger, curated, _raw = components(tmp_path)
    source = CanonicalAdjustedPriceDataSource(registry=registry, ledger=ledger, store=curated, scope="CN_A")
    with pytest.raises(AdjustedPriceDataUnavailable, match="coverage is unavailable"):
        source.daily(("2024-01-02",))
