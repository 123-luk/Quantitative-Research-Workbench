"""Tests for strict standalone V6 research-backtest configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import json
from pathlib import Path

import numpy as np
import pytest

from src.pipeline import ResearchBacktestPipelineConfig
from src.pipeline.research_backtest_config import (
    BacktestScheduleConfig,
    BacktestSourceConfig,
    BenchmarkConfig,
    PerformanceConfig,
    PortfolioAccountingConfig,
    ResearchBacktestConfigError,
    ReturnAlignmentConfig,
    TransactionCostConfig,
)


def _enabled_mapping() -> dict[str, object]:
    return {
        "enabled": True,
        "source": {"mode": "pipeline", "artifact_dir": None},
        "schedule": {"mode": "holdings_dates"},
        "return_alignment": {
            "effective_rule": "next_trading_day",
            "return_convention": "adjusted_close_to_close",
        },
        "portfolio": {
            "initial_nav": 1.0,
            "turnover_definition": "half_l1_pre_to_target",
        },
        "transaction_cost": {
            "cost_bps": 5.25,
            "rate_basis": "one_way_traded_notional",
        },
        "benchmark": {
            "benchmark_code": "000300.SH",
            "alignment_policy": "strict_common_calendar",
        },
        "performance": {
            "annualization_days": 252,
            "annual_risk_free_rate": 0.0,
        },
        "artifact_subdir": "research_backtest",
    }


def test_source_defaults_pipeline_without_filesystem_access() -> None:
    source = BacktestSourceConfig()
    assert source.to_dict() == {"mode": "pipeline", "artifact_dir": None}
    assert BacktestSourceConfig.from_dict(source.to_dict()) == source


def test_files_source_accepts_nonexistent_explicit_native_directory() -> None:
    path = Path("does/not/need/to/exist")
    source = BacktestSourceConfig.from_dict(
        {"mode": " FILES ", "artifact_dir": path}
    )
    assert source.mode == "files"
    assert source.artifact_dir == path
    assert source.to_dict()["artifact_dir"] == str(path)


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "pipeline", "artifact_dir": "holdings"},
        {"mode": "files"},
        {"mode": "files", "artifact_dir": ""},
        {"mode": "files", "artifact_dir": " holdings "},
        {"mode": "latest"},
        {"holdings_path": "holdings.parquet"},
        {"parquet_path": "holdings.parquet"},
        {"run_id": "latest"},
        {1: "files"},
    ],
)
def test_source_rejects_invalid_or_unknown_fields(value: object) -> None:
    with pytest.raises(ResearchBacktestConfigError):
        BacktestSourceConfig.from_dict(value)  # type: ignore[arg-type]


def test_source_input_is_not_mutated() -> None:
    raw = {"mode": "files", "artifact_dir": Path("native/holdings")}
    before = dict(raw)
    config = BacktestSourceConfig.from_dict(raw)
    assert raw == before
    assert BacktestSourceConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("mode", ["monthly", "weekly", "daily", "custom"])
def test_schedule_is_owned_only_by_holdings_dates(mode: str) -> None:
    with pytest.raises(ResearchBacktestConfigError):
        BacktestScheduleConfig(mode=mode)
    assert BacktestScheduleConfig().mode == "holdings_dates"
    assert not hasattr(BacktestScheduleConfig(), "frequency")


def test_alignment_defaults_and_roundtrip() -> None:
    config = ReturnAlignmentConfig()
    assert config.to_dict() == {
        "effective_rule": "next_trading_day",
        "return_convention": "adjusted_close_to_close",
    }
    assert ReturnAlignmentConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "value",
    [
        {"effective_rule": "same_day"},
        {"effective_rule": "next_open"},
        {"return_convention": "raw_close"},
        {"return_convention": "qfq"},
        {"return_convention": "hfq"},
        {"provider": "tushare"},
        {"missing_policy": "fill"},
        {"qfq": True},
    ],
)
def test_alignment_rejects_unfrozen_or_unknown_choices(value: object) -> None:
    with pytest.raises(ResearchBacktestConfigError):
        ReturnAlignmentConfig.from_dict(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, 1.5, np.float64(2.0)])
def test_portfolio_accepts_positive_finite_real(value: object) -> None:
    assert PortfolioAccountingConfig(initial_nav=value).initial_nav == float(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value", [0, -1, True, False, float("nan"), float("inf"), "1"]
)
def test_portfolio_rejects_invalid_initial_nav(value: object) -> None:
    with pytest.raises(ResearchBacktestConfigError):
        PortfolioAccountingConfig(initial_nav=value)  # type: ignore[arg-type]


def test_portfolio_turnover_contract_is_exact() -> None:
    assert PortfolioAccountingConfig().turnover_definition == "half_l1_pre_to_target"
    with pytest.raises(ResearchBacktestConfigError):
        PortfolioAccountingConfig(turnover_definition="target_to_target")


@pytest.mark.parametrize("value", [0, 5, 10, 0.125, 10**12, np.float64(3.5)])
def test_cost_accepts_nonnegative_unbounded_finite_reals(value: object) -> None:
    config = TransactionCostConfig(cost_bps=value)  # type: ignore[arg-type]
    assert config.cost_bps == float(value)
    assert config.rate_basis == "one_way_traded_notional"


@pytest.mark.parametrize(
    "value", [-0.1, True, False, "5", float("nan"), float("inf")]
)
def test_cost_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ResearchBacktestConfigError):
        TransactionCostConfig(cost_bps=value)  # type: ignore[arg-type]


def test_cost_has_no_enabled_field_and_requires_explicit_bps() -> None:
    assert "enabled" not in {item.name for item in fields(TransactionCostConfig)}
    with pytest.raises(ResearchBacktestConfigError, match="requires cost_bps"):
        TransactionCostConfig.from_dict({})
    with pytest.raises(ResearchBacktestConfigError):
        TransactionCostConfig.from_dict({"cost_bps": 5, "enabled": True})
    with pytest.raises(ResearchBacktestConfigError):
        TransactionCostConfig(cost_bps=5, rate_basis="round_trip")


@pytest.mark.parametrize("code", ["000300.SH", "000905.SH", "CUSTOM"])
def test_benchmark_requires_and_preserves_explicit_code(code: str) -> None:
    config = BenchmarkConfig(benchmark_code=f" {code} ")
    assert config.benchmark_code == code
    assert config.alignment_policy == "strict_common_calendar"


@pytest.mark.parametrize("value", ["", "   ", 300, None])
def test_benchmark_rejects_invalid_codes(value: object) -> None:
    with pytest.raises(ResearchBacktestConfigError):
        BenchmarkConfig(benchmark_code=value)  # type: ignore[arg-type]


def test_benchmark_has_no_hidden_code_default() -> None:
    with pytest.raises(TypeError):
        BenchmarkConfig()  # type: ignore[call-arg]
    with pytest.raises(ResearchBacktestConfigError):
        BenchmarkConfig.from_dict({})
    with pytest.raises(ResearchBacktestConfigError):
        BenchmarkConfig("000300.SH", alignment_policy="forward_fill")


@pytest.mark.parametrize("days", [1, 252, 365, 1000])
def test_performance_accepts_explicit_strict_annualization_days(days: int) -> None:
    assert PerformanceConfig(0.0, annualization_days=days).annualization_days == days


@pytest.mark.parametrize("days", [0, -1, True, False, 252.0, np.int64(252)])
def test_performance_rejects_invalid_annualization_days(days: object) -> None:
    with pytest.raises(ResearchBacktestConfigError):
        PerformanceConfig(0.0, annualization_days=days)  # type: ignore[arg-type]


@pytest.mark.parametrize("rate", [0, 0.03, -0.01, np.float64(0.02)])
def test_performance_accepts_finite_risk_free_rate(rate: object) -> None:
    config = PerformanceConfig(annual_risk_free_rate=rate)  # type: ignore[arg-type]
    assert config.annual_risk_free_rate == float(rate)
    assert config.annualization_days == 252


@pytest.mark.parametrize("rate", [True, False, "0", float("nan"), float("inf")])
def test_performance_rejects_invalid_risk_free_rate(rate: object) -> None:
    with pytest.raises(ResearchBacktestConfigError):
        PerformanceConfig(annual_risk_free_rate=rate)  # type: ignore[arg-type]


def test_performance_from_dict_requires_explicit_risk_free_rate() -> None:
    with pytest.raises(ResearchBacktestConfigError, match="requires"):
        PerformanceConfig.from_dict({"annualization_days": 252})


def test_disabled_top_level_has_no_hidden_business_assumptions() -> None:
    config = ResearchBacktestPipelineConfig()
    assert config.enabled is False
    assert config.transaction_cost is None
    assert config.benchmark is None
    assert config.performance is None
    assert config.artifact_subdir == "research_backtest"
    assert ResearchBacktestPipelineConfig.from_dict(config.to_dict()) == config
    json.dumps(config.to_dict(), allow_nan=False)


@pytest.mark.parametrize("missing", ["transaction_cost", "benchmark", "performance"])
def test_enabled_top_level_requires_all_explicit_business_assumptions(
    missing: str,
) -> None:
    raw = _enabled_mapping()
    raw.pop(missing)
    with pytest.raises(ResearchBacktestConfigError, match=missing):
        ResearchBacktestPipelineConfig.from_dict(raw)


def test_enabled_top_level_roundtrip_trace_and_input_immutability() -> None:
    raw = _enabled_mapping()
    before = deepcopy_mapping(raw)
    config = ResearchBacktestPipelineConfig.from_dict(raw)
    assert raw == before
    assert config.transaction_cost is not None
    assert config.transaction_cost.cost_bps == 5.25
    assert config.benchmark is not None
    assert config.benchmark.benchmark_code == "000300.SH"
    assert config.performance is not None
    assert config.performance.annual_risk_free_rate == 0.0
    assert config.performance.annualization_days == 252
    assert config.schedule.mode == "holdings_dates"
    assert config.return_alignment.effective_rule == "next_trading_day"
    assert ResearchBacktestPipelineConfig.from_dict(config.to_dict()) == config
    assert config.to_dict() == config.to_dict()
    with pytest.raises(FrozenInstanceError):
        config.enabled = False  # type: ignore[misc]


def test_enabled_files_source_is_valid() -> None:
    raw = _enabled_mapping()
    raw["source"] = {"mode": "files", "artifact_dir": "native/holdings"}
    config = ResearchBacktestPipelineConfig.from_dict(raw)
    assert config.source.artifact_dir == Path("native/holdings")


@pytest.mark.parametrize(
    "value",
    ["", " ", ".", "..", "nested/backtest", "nested\\backtest", "C:\\backtest"],
)
def test_top_level_rejects_unsafe_artifact_subdir(value: str) -> None:
    with pytest.raises(ResearchBacktestConfigError):
        ResearchBacktestPipelineConfig(artifact_subdir=value)


def test_top_level_rejects_unknown_fields_and_has_no_frequency() -> None:
    with pytest.raises(ResearchBacktestConfigError):
        ResearchBacktestPipelineConfig.from_dict({"unknown": True})
    names = {item.name for item in fields(ResearchBacktestPipelineConfig)}
    assert "frequency" not in names
    assert "rebalance_frequency" not in names


def test_public_pipeline_export_is_available_without_pipeline_integration() -> None:
    from src.pipeline import (
        BacktestScheduleConfig as ExportedSchedule,
        BacktestSourceConfig as ExportedSource,
        BenchmarkConfig as ExportedBenchmark,
        PerformanceConfig as ExportedPerformance,
        PortfolioAccountingConfig as ExportedPortfolio,
        ReturnAlignmentConfig as ExportedAlignment,
        TransactionCostConfig as ExportedCost,
    )

    assert ExportedSchedule is BacktestScheduleConfig
    assert ExportedSource is BacktestSourceConfig
    assert ExportedBenchmark is BenchmarkConfig
    assert ExportedPerformance is PerformanceConfig
    assert ExportedPortfolio is PortfolioAccountingConfig
    assert ExportedAlignment is ReturnAlignmentConfig
    assert ExportedCost is TransactionCostConfig


def deepcopy_mapping(value: dict[str, object]) -> dict[str, object]:
    """Copy the nested JSON-style fixture without sharing mutable values."""
    return json.loads(json.dumps(value))
