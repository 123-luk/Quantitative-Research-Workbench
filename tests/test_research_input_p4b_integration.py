from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import CoverageLedger, PartitionedParquetStore, RawParquetStore
from src.data.contracts import ResearchFrequency
from src.data.dataset_registry import create_default_dataset_registry
from src.data.preparation import DataPreparationService
from src.factors import BP, FactorFrequencySpec, FactorMetadata, FactorRegistry, FunctionFactor
from src.factors.forward_returns import ForwardReturnConfig
from src.factors.preprocessing import PreprocessingConfig
from src.factors.research_pipeline import FactorResearchConfig, FactorResearchRunner
from src.research_data import AdjustedPriceService, CanonicalAdjustedPriceDataSource, ForwardReturnSpec, HistoryRequirement, ResearchCalendar, ResearchInputBuilder, ResearchInputPlanner, ResearchMaterializationStore
from src.universe import CanonicalUniverseDataSource, UniverseService, UniverseSpec


DAILY_FIELDS = ("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount")
BASIC_FIELDS = ("ts_code", "trade_date", "close", "turnover_rate", "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "total_mv", "circ_mv")
STOCK_FIELDS = ("ts_code", "symbol", "name", "area", "industry", "market", "exchange", "curr_type", "list_status", "list_date", "delist_date")


class Provider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append("trade_cal")
        return pd.DataFrame([{"exchange": "SSE", "cal_date": day.strftime("%Y%m%d"), "is_open": int(day.weekday() < 5), "pretrade_date": None} for day in pd.date_range(start_date, end_date, freq="D")])

    def get_stock_basic(self, list_status: str = "L") -> pd.DataFrame:
        self.calls.append("stock_basic")
        if list_status != "L":
            return pd.DataFrame(columns=STOCK_FIELDS)
        return pd.DataFrame([{"ts_code": "600001.SH", "symbol": "600001", "name": "A", "area": "China", "industry": "Test", "market": "主板", "exchange": "SSE", "curr_type": "CNY", "list_status": "L", "list_date": "20100101", "delist_date": None}], columns=STOCK_FIELDS)

    def get_daily(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append("daily")
        day = str(kwargs["trade_date"])
        value = float(pd.Timestamp(day).day)
        row = {field: value for field in DAILY_FIELDS}
        row.update(ts_code="600001.SH", trade_date=day, open=value - 1, high=value + 1, low=value - 2, close=value, vol=100.0, amount=200.0)
        return pd.DataFrame([row], columns=DAILY_FIELDS)

    def get_adj_factor(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append("adj_factor")
        return pd.DataFrame([{"ts_code": "600001.SH", "trade_date": kwargs["trade_date"], "adj_factor": 2.0}])

    def get_daily_basic(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append("daily_basic")
        day = str(kwargs["trade_date"])
        row = {field: 1.0 for field in BASIC_FIELDS}
        row.update(ts_code="600001.SH", trade_date=day, pb=2.0)
        return pd.DataFrame([row], columns=BASIC_FIELDS)


def factor_registry() -> FactorRegistry:
    specs = tuple(FactorFrequencySpec(frequency, ("daily", "adj_factor"), {"daily": ("ts_code", "trade_date", "close"), "adj_factor": ("ts_code", "trade_date", "adj_factor")}, HistoryRequirement.trading_days(3), "trailing adjusted close", "adjusted_momentum_3d") for frequency in ResearchFrequency)
    plugin = FunctionFactor(FactorMetadata("adjusted_momentum_3d", "momentum", 1, required_datasets=("daily", "adj_factor"), source_fields=("adj_close",), lookback_days=3, frequency_specs=specs), lambda frame: frame["adj_close"] / frame["adj_close"].shift(2) - 1)
    registry = FactorRegistry()
    registry.register(BP)
    registry.register(plugin)
    return registry


def test_plan_prepares_through_p4b_then_second_prepare_and_build_are_reused(tmp_path: Path) -> None:
    calendar_rows = pd.DataFrame([{"cal_date": day.strftime("%Y-%m-%d"), "is_open": int(day.weekday() < 5)} for day in pd.date_range("2024-01-01", "2024-01-31", freq="D")])
    calendar = ResearchCalendar(calendar_rows)
    registry = factor_registry()
    forward = ForwardReturnSpec.from_config(ForwardReturnConfig(entry_lag_periods=1, holding_periods=2))
    plan = ResearchInputPlanner(calendar=calendar, universe_service=UniverseService(), factor_registry=registry).build(research_frequency=ResearchFrequency.DAILY, start_date="2024-01-08", end_date="2024-01-08", universe_spec=UniverseSpec.custom(("600001.SH",)), factor_ids=("bp", "adjusted_momentum_3d"), forward_return_spec=forward)

    dataset_registry = create_default_dataset_registry()
    ledger = CoverageLedger(tmp_path / "metadata" / "catalog.sqlite")
    curated = PartitionedParquetStore(tmp_path / "curated")
    provider = Provider()
    preparation = DataPreparationService(registry=dataset_registry, ledger=ledger, curated_store=curated, raw_store=RawParquetStore(tmp_path / "raw"), open_dates=lambda start, end: tuple(day for day in calendar.open_dates if start <= day <= end))
    first = preparation.ensure(plan.requirements, client=provider)
    assert first.provider_calls > 0

    market = CanonicalAdjustedPriceDataSource(registry=dataset_registry, ledger=ledger, store=curated, scope="CN_A")
    universe = CanonicalUniverseDataSource(registry=dataset_registry, ledger=ledger, store=curated, stock_basic_as_of="2024-01-08", index_weight_start="2024-01-01")
    runner = FactorResearchRunner(registry, FactorResearchConfig(factor_names=plan.factor_ids, composition_method="none", evaluate_components=False, evaluate_composite=False), preprocessing_config=PreprocessingConfig(missing_method="none", winsor_method="none", standardize_method="none", min_cross_section_size=1), forward_return_config=forward.to_config())
    builder = ResearchInputBuilder(calendar=calendar, universe_service=UniverseService(), universe_data=universe, factor_registry=registry, dataset_source=market, adjusted_prices=AdjustedPriceService(market), factor_runner=runner, store=ResearchMaterializationStore(tmp_path / "research-cache"))
    materialized = builder.build(plan)
    assert materialized.reused is False

    provider.calls.clear()
    second = preparation.ensure(plan.requirements, client=provider)
    reused = builder.build(plan)
    assert second.provider_calls == 0
    assert provider.calls == []
    assert reused.reused is True
    assert reused.materialization_id == materialized.materialization_id
