"""Gate F tests for YAML, CLI, documentation, and artifact metadata policy."""

from pathlib import Path

import yaml

from src.pipeline import PipelineConfig


ROOT = Path(__file__).parents[1]


def test_minimum_variance_example_yaml_roundtrips_exact_nested_config():
    path = ROOT / "config" / "minimum_variance_pipeline.example.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(values)
    assert config.holdings.portfolio_construction.to_dict() == {
        "method": "minimum_variance",
        "params": {"risk_model": {
            "estimator": "ledoit_wolf",
            "params": {},
            "lookback_trading_days": 120,
            "min_observations": 80,
        }},
        "constraints": [
            {"type": "max_weight", "params": {"max_weight": 0.2}}
        ],
    }
    assert PipelineConfig.from_dict(config.to_dict()) == config


def test_cli_has_no_risk_or_solver_business_flags():
    source = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    for forbidden in (
        "--risk-model", "--covariance-estimator", "--lookback",
        "--min-observations", "--solver", "--max-weight",
    ):
        assert forbidden not in source


def test_docs_define_frozen_semantics_and_no_extra_pipeline_stage():
    docs = (ROOT / "docs" / "14_risk_model_optimizer.md").read_text(encoding="utf-8")
    for required in (
        "common complete-case", "ddof=1", "unannualized", "LedoitWolf",
        "positive semidefinite", "0.5", "SLSQP", "long-only",
        "fully invested", "max_weight", "Top-N", "risk_model",
        "historical_returns", "no post-solve clipping",
    ):
        assert required.lower() in docs.lower()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Minimum Variance" in readme
    assert "14_risk_model_optimizer.md" in readme
    assert not (ROOT / "src" / "risk_model" / "artifacts.py").exists()
