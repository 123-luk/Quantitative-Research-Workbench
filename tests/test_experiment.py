"""Tests for experiment run directory management."""

from __future__ import annotations

import json

import yaml

from src.pipeline.config import PipelineConfig
from src.pipeline.experiment import ExperimentManager


def build_config(output_dir: str) -> PipelineConfig:
    """Build a minimal PipelineConfig for experiment tests."""
    return PipelineConfig(
        backtest_start="2024-01-01",
        backtest_end="2025-03-31",
        train_years=10,
        max_lookback_months=12,
        stock_pool="hs300",
        benchmark="000300.SH",
        strategy_name="score",
        selected_factors=["pe", "pb"],
        rebalance_frequency="M",
        top_n=20,
        transaction_cost=0.001,
        data_root="data",
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        cache_dir="data/cache",
        output_dir=output_dir,
        parquet_engine="auto",
        required_datasets=["daily", "daily_basic", "adj_factor"],
    )


def test_experiment_manager_creates_unique_run_dirs(tmp_path) -> None:
    """create_run_dir should create directories without overwriting old runs."""
    manager = ExperimentManager(tmp_path)

    first = manager.create_run_dir("score", "hs300")
    second = manager.create_run_dir("score", "hs300")

    assert first.exists()
    assert second.exists()
    assert first != second


def test_experiment_manager_saves_artifacts(tmp_path) -> None:
    """ExperimentManager should save config, run info, and metrics files."""
    manager = ExperimentManager(tmp_path)
    config = build_config(str(tmp_path))
    run_dir = manager.create_run_dir("score", "hs300")

    config_path = manager.save_config_snapshot(run_dir, config)
    run_info_path = manager.save_run_info(run_dir, {"status": "missing_data"})
    metrics_path = manager.save_metrics(run_dir, {"example_metric": 1.0})

    assert config_path == run_dir / "config_snapshot.yaml"
    assert run_info_path == run_dir / "run_info.json"
    assert metrics_path == run_dir / "metrics.json"
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["strategy_name"] == "score"
    assert json.loads(run_info_path.read_text(encoding="utf-8"))["status"] == "missing_data"
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["example_metric"] == 1.0
