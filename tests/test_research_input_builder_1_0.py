from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from src.data.contracts import ResearchFrequency
from src.factors import BP, FactorFrequencySpec, FactorMetadata, FactorRegistry, FunctionFactor
from src.factors.forward_returns import ForwardReturnConfig
from src.factors.preprocessing import PreprocessingConfig
from src.factors.research_pipeline import FactorResearchConfig, FactorResearchRunner
from src.research_data import (
    AdjustedPriceService,
    CanonicalMarketSlice,
    ForwardReturnSpec,
    HistoryRequirement,
    ResearchCalendar,
    ResearchInputBuilder,
    ResearchInputDataUnavailable,
    ResearchInputPlanner,
    ResearchMaterializationStore,
    TrainingLabelAvailabilityGuard,
)
from src.universe import CanonicalUniverseSlice, UniverseService, UniverseSpec


def research_calendar() -> ResearchCalendar:
    rows = []
    for day in pd.date_range("2023-11-01", "2024-04-30", freq="D"):
        text = day.strftime("%Y-%m-%d")
        rows.append({"cal_date": text, "is_open": int(day.weekday() < 5 and text not in {"2024-01-01", "2024-02-09"})})
    return ResearchCalendar(pd.DataFrame(rows))


@dataclass
class UniverseSource:
    def stock_basic(self) -> CanonicalUniverseSlice:
        rows = pd.DataFrame(
            [
                {"ts_code": "600001.SH", "symbol": "600001", "name": "A", "area": "China", "industry": "Test", "market": "主板", "exchange": "SSE", "curr_type": "CNY", "list_status": "L", "list_date": "2010-01-01", "delist_date": None},
                {"ts_code": "000001.SZ", "symbol": "000001", "name": "B", "area": "China", "industry": "Test", "market": "主板", "exchange": "SZSE", "curr_type": "CNY", "list_status": "L", "list_date": "2010-01-01", "delist_date": None},
            ]
        )
        return CanonicalUniverseSlice(rows, "stock_basic", "1.1", "2024-04-30", "stock:v1")

    def index_weight(self, index_code: str, through_date: str) -> CanonicalUniverseSlice:
        raise AssertionError("CUSTOM universe must not request index_weight")


class MarketSource:
    def __init__(self, calendar: ResearchCalendar) -> None:
        self.version = "v1"
        daily_rows = []
        basic_rows = []
        factor_rows = []
        for date_index, day in enumerate(calendar.open_dates):
            for code_index, code in enumerate(("600001.SH", "000001.SZ")):
                close = 20.0 + date_index + code_index * 3
                daily_rows.append({"ts_code": code, "trade_date": day, "open": close - 1, "high": close + 1, "low": close - 2, "close": close, "vol": 100.0, "amount": 200.0})
                factor_rows.append({"ts_code": code, "trade_date": day, "adj_factor": 1.5 + code_index * 0.1})
                basic_rows.append({"ts_code": code, "trade_date": day, "pb": 2.0 + code_index + date_index / 1000})
        self.frames = {
            "daily": pd.DataFrame(daily_rows),
            "adj_factor": pd.DataFrame(factor_rows),
            "daily_basic": pd.DataFrame(basic_rows),
        }

    def _slice(self, dataset: str) -> CanonicalMarketSlice:
        return CanonicalMarketSlice(self.frames[dataset], dataset, "1.0", f"{dataset}:{self.version}")

    def load(self, dataset_id: str, dates: tuple[str, ...]) -> CanonicalMarketSlice:
        return self._slice(dataset_id)

    def daily(self, dates: tuple[str, ...]) -> CanonicalMarketSlice:
        return self._slice("daily")

    def adj_factor(self, dates: tuple[str, ...]) -> CanonicalMarketSlice:
        return self._slice("adj_factor")


def registry_and_counter() -> tuple[FactorRegistry, dict[str, int]]:
    counter = {"calls": 0}

    def momentum(frame: pd.DataFrame) -> pd.Series:
        counter["calls"] += 1
        return frame["adj_close"] / frame["adj_close"].shift(2) - 1.0

    specs = tuple(
        FactorFrequencySpec(
            frequency,
            ("daily", "adj_factor"),
            {"daily": ("ts_code", "trade_date", "close"), "adj_factor": ("ts_code", "trade_date", "adj_factor")},
            HistoryRequirement.trading_days(3),
            "trailing three daily adjusted closes at formation",
            "adjusted_momentum_3d",
        )
        for frequency in ResearchFrequency
    )
    plugin = FunctionFactor(
        FactorMetadata("adjusted_momentum_3d", "momentum", 1, required_datasets=("daily", "adj_factor"), source_fields=("adj_close",), lookback_days=3, frequency_specs=specs),
        momentum,
    )
    registry = FactorRegistry()
    registry.register(BP)
    registry.register(plugin)
    return registry, counter


def runner(registry: FactorRegistry, forward: ForwardReturnSpec) -> FactorResearchRunner:
    return FactorResearchRunner(
        registry,
        FactorResearchConfig(factor_names=("bp", "adjusted_momentum_3d"), composition_method="none", evaluate_components=False, evaluate_composite=False),
        preprocessing_config=PreprocessingConfig(missing_method="none", winsor_method="none", standardize_method="none", min_cross_section_size=1),
        forward_return_config=forward.to_config(),
    )


def setup(tmp_path: Path, frequency: ResearchFrequency, start: str, end: str):
    calendar = research_calendar()
    registry, counter = registry_and_counter()
    forward = ForwardReturnSpec.from_config(ForwardReturnConfig(price_col="close", return_col="forward_return", entry_lag_periods=1, holding_periods=2))
    planner = ResearchInputPlanner(calendar=calendar, universe_service=UniverseService(), factor_registry=registry)
    plan = planner.build(research_frequency=frequency, start_date=start, end_date=end, universe_spec=UniverseSpec.custom(("600001.SH", "000001.SZ")), factor_ids=("bp", "adjusted_momentum_3d"), forward_return_spec=forward)
    market = MarketSource(calendar)
    builder = ResearchInputBuilder(calendar=calendar, universe_service=UniverseService(), universe_data=UniverseSource(), factor_registry=registry, dataset_source=market, adjusted_prices=AdjustedPriceService(market), factor_runner=runner(registry, forward), store=ResearchMaterializationStore(tmp_path / "research-cache"))
    return plan, builder, market, counter


def read(result, name: str) -> pd.DataFrame:
    return pd.read_parquet(result.paths[name])


def test_plan_is_deterministic_serializable_and_separates_warmup_horizon(tmp_path: Path) -> None:
    first, _builder, _market, _counter = setup(tmp_path, ResearchFrequency.DAILY, "2024-01-08", "2024-01-10")
    second, _builder2, _market2, _counter2 = setup(tmp_path / "other", ResearchFrequency.DAILY, "2024-01-08", "2024-01-10")
    assert first.to_dict() == second.to_dict()
    assert first.plan_id == second.plan_id
    assert first.formation_dates == ("2024-01-08", "2024-01-09", "2024-01-10")
    assert min(item.required_start for item in first.requirements) == "2024-01-04"
    assert max(item.required_end for item in first.requirements) == "2024-01-15"
    assert tuple(sorted(first.materialization_targets)) == tuple(sorted(("factor_input.parquet", "price_panel.parquet", "score_panel.parquet", "modeling_factor_panel.parquet", "modeling_forward_returns.parquet", "labels_with_availability.parquet")))
    assert not ({"suspend_d", "index_weight"} & {item.dataset_id for item in first.requirements})


def test_daily_e2e_exact_schemas_universe_forward_formula_and_ml_validation(tmp_path: Path) -> None:
    plan, builder, _market, _counter = setup(tmp_path, ResearchFrequency.DAILY, "2024-01-08", "2024-01-10")
    result = builder.build(plan)
    assert result.reused is False
    factors = read(result, "factor_input.parquet")
    prices = read(result, "price_panel.parquet")
    scores = read(result, "score_panel.parquet")
    model_factors = read(result, "modeling_factor_panel.parquet")
    model_returns = read(result, "modeling_forward_returns.parquet")
    labels = read(result, "labels_with_availability.parquet")
    assert tuple(factors.columns) == ("trade_date", "ts_code", "adj_close", "pb")
    assert tuple(prices.columns) == ("trade_date", "ts_code", "close")
    assert tuple(scores.columns) == ("trade_date", "ts_code")
    assert tuple(model_factors.columns) == ("trade_date", "ts_code", "bp", "adjusted_momentum_3d")
    assert tuple(model_returns.columns) == ("trade_date", "ts_code", "entry_trade_date", "exit_trade_date", "entry_price", "exit_price", "forward_return")
    assert tuple(labels.columns) == (*model_returns.columns, "available_at")
    assert len(scores) == 6 and set(scores["ts_code"]) == {"600001.SH", "000001.SZ"}
    expected = model_returns["exit_price"] / model_returns["entry_price"] - 1.0
    pd.testing.assert_series_equal(model_returns["forward_return"], expected, check_names=False)
    assert (labels["available_at"] == labels["exit_trade_date"]).all()
    assert result.diagnostics["score_panel_semantics"].startswith("formation/universe")


def test_monthly_e2e_uses_real_month_end_and_daily_history_without_mean(tmp_path: Path) -> None:
    plan, builder, _market, _counter = setup(tmp_path, ResearchFrequency.MONTHLY, "2024-01-01", "2024-02-29")
    assert plan.formation_dates == ("2024-01-31", "2024-02-29")
    result = builder.build(plan)
    scores = read(result, "score_panel.parquet")
    model_factors = read(result, "modeling_factor_panel.parquet")
    assert tuple(scores["trade_date"].dt.strftime("%Y-%m-%d").drop_duplicates()) == plan.formation_dates
    assert len(model_factors) == 4
    assert model_factors["adjusted_momentum_3d"].notna().all()
    assert model_factors["bp"].notna().all()


def test_training_cutoff_excludes_unrealized_labels_and_future_mutation(tmp_path: Path) -> None:
    plan, builder, _market, _counter = setup(tmp_path, ResearchFrequency.DAILY, "2024-01-08", "2024-01-08")
    labels = read(builder.build(plan), "labels_with_availability.parquet")
    before = TrainingLabelAvailabilityGuard.available(labels, "2024-01-10")
    assert before.empty
    realized = TrainingLabelAvailabilityGuard.available(labels, "2024-01-11")
    assert len(realized) == 2
    changed = labels.copy()
    changed["forward_return"] = 999999.0
    pd.testing.assert_frame_equal(before, TrainingLabelAvailabilityGuard.available(changed, "2024-01-10"))


def test_repeat_reuses_identity_without_factor_recomputation_and_source_change_invalidates(tmp_path: Path) -> None:
    plan, builder, market, counter = setup(tmp_path, ResearchFrequency.DAILY, "2024-01-08", "2024-01-10")
    first = builder.build(plan)
    calls = counter["calls"]
    second = builder.build(plan)
    assert second.reused is True and second.materialization_id == first.materialization_id
    assert counter["calls"] == calls
    market.version = "v2"
    future = market.frames["daily"]["trade_date"].gt("2024-01-08")
    market.frames["daily"].loc[future, "close"] *= 1000.0
    third = builder.build(plan)
    assert third.materialization_id != first.materialization_id
    old_features = read(first, "modeling_factor_panel.parquet")
    new_features = read(third, "modeling_factor_panel.parquet")
    old_t = old_features.loc[old_features["trade_date"].eq(pd.Timestamp("2024-01-08"))].reset_index(drop=True)
    new_t = new_features.loc[new_features["trade_date"].eq(pd.Timestamp("2024-01-08"))].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_t, new_t, check_exact=True)


def test_atomic_store_failure_leaves_old_materialization_and_no_partial_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ResearchMaterializationStore(tmp_path / "cache")
    frame = pd.DataFrame({"x": [1]})
    old = store.publish("a" * 64, {"one.parquet": frame}, {"diagnostics": {}})
    assert old.directory.is_dir()
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError, match="injected"):
        store.publish("b" * 64, {"two.parquet": frame}, {"diagnostics": {}})
    assert old.directory.is_dir()
    assert not (store.root / ("b" * 64)).exists()
    assert not tuple(store.root.glob(f".{('b' * 64)}.*"))


def test_missing_canonical_factor_observation_fails_without_changing_universe(tmp_path: Path) -> None:
    plan, builder, market, _counter = setup(tmp_path, ResearchFrequency.DAILY, "2024-01-08", "2024-01-08")
    market.frames["daily_basic"] = market.frames["daily_basic"].loc[market.frames["daily_basic"]["trade_date"].ne("2024-01-08")]
    market.version = "missing"
    with pytest.raises(ResearchInputDataUnavailable, match="modeling input validation failed"):
        builder.build(plan)
    assert not tuple((tmp_path / "research-cache").glob("*/manifest.json"))
