"""Release-level consistency checks for the completed V6 research chain."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import src.pipeline as pipeline_api
import src.research_backtest as research_api
from src.pipeline import PipelineConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT_ROOT / "config" / "research_backtest_pipeline.example.yaml"
README = PROJECT_ROOT / "README.md"
GUIDE = PROJECT_ROOT / "docs" / "10_research_backtest_pipeline.md"
READINESS = PROJECT_ROOT / "docs" / "11_v0.7.0_release_readiness.md"
STREAMLIT_APP = PROJECT_ROOT / "app" / "streamlit_app.py"


def test_release_example_roundtrip_preserves_all_frozen_owners() -> None:
    values = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(values)
    roundtrip = PipelineConfig.from_dict(config.to_dict())
    research = roundtrip.research_backtest
    assert roundtrip == config
    assert roundtrip.backtest_end == "2024-12-31"
    assert roundtrip.holdings.enabled and roundtrip.holdings.top_n == 10
    assert research.enabled and research.source.mode == "pipeline"
    assert research.schedule.mode == "holdings_dates"
    assert research.return_alignment.effective_rule == "next_trading_day"
    assert research.return_alignment.return_convention == "adjusted_close_to_close"
    assert research.transaction_cost.cost_bps == 10.0  # type: ignore[union-attr]
    assert research.benchmark.benchmark_code == "000300.SH"  # type: ignore[union-attr]
    assert research.performance.annual_risk_free_rate == 0.0  # type: ignore[union-attr]
    assert research.performance.annualization_days == 252  # type: ignore[union-attr]
    serialized = json.dumps(roundtrip.to_dict(), allow_nan=False)
    assert "frequency" not in research.to_dict()
    assert "end_date" not in research.to_dict()
    assert serialized


def test_public_api_exports_are_unique_and_release_scoped() -> None:
    for module in (pipeline_api, research_api):
        exported = module.__all__
        assert len(exported) == len(set(exported))
        assert all(hasattr(module, name) for name in exported)
        assert not any(
            name.lower().startswith("test") or name.lower().endswith("_test")
            for name in exported
        )
    assert {
        "ResearchBacktestPipelineConfig",
        "ResearchBacktestPipelineExecutor",
        "ResearchBacktestPipelineResult",
    }.issubset(pipeline_api.__all__)
    assert {
        "ResearchBacktestArtifactStore",
        "ResearchBacktestHoldingsSourceAdapter",
        "PerformanceAnalyticsEngine",
    }.issubset(research_api.__all__)


def test_release_docs_example_readme_and_ui_use_consistent_boundaries() -> None:
    readme = README.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    readiness = READINESS.read_text(encoding="utf-8")
    ui = STREAMLIT_APP.read_text(encoding="utf-8")
    assert "docs/11_v0.7.0_release_readiness.md" in readme
    assert "st.navigation" in ui
    assert "Legacy dashboard" not in ui
    for phrase in (
        "Research Backtest",
        "holdings_dates",
        "next_trading_day",
        "adjusted_close_to_close",
        "one_way_traded_notional",
        "strict_common_calendar",
        "backtest_end",
    ):
        normalized = phrase.replace("_", " ")
        assert phrase in guide or normalized in guide
        assert phrase in readiness or normalized in readiness
    assert "v0.7.0" in readiness
    assert "does not claim" in readiness
