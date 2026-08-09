from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import src.pipeline.runner as runner_module
from src.pipeline import (
    BacktestSourceConfig,
    BenchmarkConfig,
    HoldingsPipelineConfig,
    PerformanceConfig,
    PipelineConfig,
    PredictionSourceConfig,
    ResearchBacktestPipelineConfig,
    SignalPipelineConfig,
    TransactionCostConfig,
    run_pipeline,
)


class _ReadyDataManager:
    def prepare_data(self, config: object) -> dict[str, object]:
        return {"cache_status": "ready", "missing_ranges": {}}


def _research(source: BacktestSourceConfig) -> ResearchBacktestPipelineConfig:
    return ResearchBacktestPipelineConfig(
        enabled=True,
        source=source,
        transaction_cost=TransactionCostConfig(cost_bps=12.0),
        benchmark=BenchmarkConfig(benchmark_code="TEST.IDX"),
        performance=PerformanceConfig(annual_risk_free_rate=0.02),
        artifact_subdir="research_backtest",
    )


def _base(tmp_path: Path, **updates: object) -> PipelineConfig:
    values: dict[str, object] = {
        "backtest_start": "2024-01-01",
        "backtest_end": "2024-01-08",
        "train_years": 1,
        "max_lookback_months": 1,
        "stock_pool": "test",
        "benchmark": "LEGACY.IDX",
        "strategy_name": "test",
        "selected_factors": [],
        "rebalance_frequency": "M",
        "top_n": 2,
        "transaction_cost": 0.123,
        "data_root": "data",
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "cache_dir": "data/cache",
        "output_dir": str(tmp_path / "output"),
        "parquet_engine": "auto",
        "required_datasets": [],
    }
    values.update(updates)
    return PipelineConfig.from_dict(values)


def test_old_config_defaults_research_backtest_disabled(tmp_path: Path) -> None:
    config = _base(tmp_path)
    assert not config.research_backtest.enabled
    snapshot = config.to_dict()
    assert snapshot["research_backtest"]["enabled"] is False  # type: ignore[index]
    assert PipelineConfig.from_dict(snapshot).to_dict() == snapshot


def test_pipeline_source_requires_explicit_enabled_holdings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires holdings"):
        _base(
            tmp_path,
            research_backtest=_research(BacktestSourceConfig(mode="pipeline")),
        )


@pytest.mark.parametrize("holdings_enabled", [False, True])
def test_files_source_is_independent_of_current_holdings(
    tmp_path: Path, holdings_enabled: bool
) -> None:
    config = _base(
        tmp_path,
        signal=SignalPipelineConfig(
            enabled=holdings_enabled,
            source=PredictionSourceConfig("files", tmp_path / "ml"),
        ),
        holdings=HoldingsPipelineConfig(
            enabled=holdings_enabled,
            top_n=2,
        ),
        research_backtest=_research(
            BacktestSourceConfig(mode="files", artifact_dir=tmp_path / "native")
        ),
    )
    assert config.research_backtest.enabled
    assert config.holdings.enabled is holdings_enabled


def test_research_config_does_not_inherit_legacy_business_fields(
    tmp_path: Path,
) -> None:
    config = _base(
        tmp_path,
        benchmark="LEGACY.IDX",
        transaction_cost=0.5,
        rebalance_frequency="W",
        research_backtest=_research(
            BacktestSourceConfig(mode="files", artifact_dir=tmp_path / "native")
        ),
    )
    research = config.research_backtest
    assert research.benchmark.benchmark_code == "TEST.IDX"  # type: ignore[union-attr]
    assert research.transaction_cost.cost_bps == 12.0  # type: ignore[union-attr]
    assert "frequency" not in research.to_dict()


def test_direct_config_roundtrip_and_input_immutability(tmp_path: Path) -> None:
    source = _base(tmp_path).to_dict()
    source["research_backtest"] = _research(
        BacktestSourceConfig(mode="files", artifact_dir=tmp_path / "native")
    ).to_dict()
    before = deepcopy(source)
    result = PipelineConfig.from_dict(source)
    assert source == before
    assert PipelineConfig.from_dict(result.to_dict()).to_dict() == result.to_dict()


class _StageResult:
    def __init__(self, name: str) -> None:
        self.name = name

    def as_dict(self) -> dict[str, object]:
        return {"enabled": True, "name": self.name}


class _BacktestResult:
    def to_dict(self) -> dict[str, object]:
        return {"enabled": True, "artifact_dir": "backtest", "metrics": {}}


def _patch_runner(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    observed: dict[str, object],
) -> _StageResult:
    signal_result = _StageResult("signal")
    holdings_result = _StageResult("holdings")

    class SignalExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(self, run_dir: object, **kwargs: object) -> _StageResult:
            events.append("signal")
            return signal_result

    class HoldingsExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(self, run_dir: object, **kwargs: object) -> _StageResult:
            events.append("holdings")
            return holdings_result

    client = object()

    class BacktestExecutor:
        def __init__(self, config: object, received_client: object) -> None:
            observed["client"] = received_client

        def execute(self, **kwargs: object) -> _BacktestResult:
            events.append("backtest")
            observed.update(kwargs)
            return _BacktestResult()

    monkeypatch.setattr(runner_module, "DataManager", _ReadyDataManager)
    monkeypatch.setattr(runner_module, "SignalPipelineExecutor", SignalExecutor)
    monkeypatch.setattr(runner_module, "HoldingsPipelineExecutor", HoldingsExecutor)
    monkeypatch.setattr(
        runner_module, "ResearchBacktestPipelineExecutor", BacktestExecutor
    )
    monkeypatch.setattr(runner_module, "TushareClient", lambda: client)
    return holdings_result


def test_runner_pipeline_mode_exact_handoff_order_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    observed: dict[str, object] = {}
    holdings_result = _patch_runner(monkeypatch, events, observed)
    config = _base(
        tmp_path,
        signal=SignalPipelineConfig(
            enabled=True,
            source=PredictionSourceConfig("files", tmp_path / "ml"),
        ),
        holdings=HoldingsPipelineConfig(enabled=True, top_n=2),
        research_backtest=_research(BacktestSourceConfig(mode="pipeline")),
    )
    summary = run_pipeline(config)
    run_dir = Path(summary["run_dir"])
    assert events == ["signal", "holdings", "backtest"]
    assert observed["holdings_result"] is holdings_result
    assert observed["end_date"] == config.backtest_end
    assert observed["artifact_dir"] == run_dir / "research_backtest"
    assert "research_backtest" in summary
    json.dumps(summary, allow_nan=False)


def test_runner_files_mode_never_passes_current_holdings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    observed: dict[str, object] = {}
    _patch_runner(monkeypatch, events, observed)
    config = _base(
        tmp_path,
        signal=SignalPipelineConfig(
            enabled=True,
            source=PredictionSourceConfig("files", tmp_path / "ml"),
        ),
        holdings=HoldingsPipelineConfig(enabled=True, top_n=2),
        research_backtest=_research(
            BacktestSourceConfig(mode="files", artifact_dir=tmp_path / "native")
        ),
    )
    run_pipeline(config)
    assert observed["holdings_result"] is None


def test_disabled_runner_omits_summary_and_market_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "DataManager", _ReadyDataManager)

    def forbidden() -> object:
        raise AssertionError("disabled research backtest constructed a client")

    monkeypatch.setattr(runner_module, "TushareClient", forbidden)
    summary = run_pipeline(_base(tmp_path))
    assert "research_backtest" not in summary


def test_runner_propagates_backtest_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    observed: dict[str, object] = {}
    _patch_runner(monkeypatch, events, observed)

    class FailingExecutor:
        def __init__(self, config: object, client: object) -> None:
            pass

        def execute(self, **kwargs: object) -> object:
            raise RuntimeError("backtest failed")

    monkeypatch.setattr(
        runner_module, "ResearchBacktestPipelineExecutor", FailingExecutor
    )
    config = _base(
        tmp_path,
        research_backtest=_research(
            BacktestSourceConfig(mode="files", artifact_dir=tmp_path / "native")
        ),
    )
    with pytest.raises(RuntimeError, match="backtest failed"):
        run_pipeline(config)
