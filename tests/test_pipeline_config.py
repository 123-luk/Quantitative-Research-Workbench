"""Tests for pipeline configuration loading and date logic."""

from __future__ import annotations

import pytest

from src.pipeline.config import PipelineConfig


def write_config(tmp_path, content: str):
    """Write a temporary YAML config file."""
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_pipeline_config_loads_from_yaml(tmp_path) -> None:
    """PipelineConfig should load defaults from a YAML file."""
    config_path = write_config(
        tmp_path,
        """
data:
  root: data
  raw_dir: data/raw
  processed_dir: data/processed
  cache_dir: data/cache
  output_dir: data/output
  parquet_engine: auto
  required_datasets:
    - daily
pipeline:
  backtest_start: "2024-01-01"
  backtest_end: "2025-03-31"
  train_years: 10
  max_lookback_months: 12
  stock_pool: hs300
  benchmark: 000300.SH
  strategy_name: score
  rebalance_frequency: M
  top_n: 20
  transaction_cost: 0.001
factors:
  selected:
    - pe
    - pb
""",
    )

    config = PipelineConfig.from_yaml(config_path)

    assert config.backtest_start == "2024-01-01"
    assert config.backtest_end == "2025-03-31"
    assert config.selected_factors == ["pe", "pb"]
    assert config.required_datasets == ["daily"]


def test_pipeline_config_overrides_defaults(tmp_path) -> None:
    """Overrides should replace selected YAML values."""
    config_path = write_config(
        tmp_path,
        """
data:
  required_datasets:
    - daily
pipeline:
  backtest_start: "2024-01-01"
  backtest_end: "2025-03-31"
  train_years: 10
  max_lookback_months: 12
  stock_pool: hs300
  benchmark: 000300.SH
  strategy_name: score
  rebalance_frequency: M
  top_n: 20
  transaction_cost: 0.001
""",
    )

    config = PipelineConfig.from_yaml(
        config_path,
        overrides={"stock_pool": "zz500", "top_n": 30},
    )

    assert config.stock_pool == "zz500"
    assert config.top_n == 30


def test_pipeline_config_required_date_range() -> None:
    """Required start date should include training and lookback windows."""
    config = PipelineConfig(
        backtest_start="2024-01-01",
        backtest_end="2025-03-31",
        train_years=10,
        max_lookback_months=12,
        stock_pool="hs300",
        benchmark="000300.SH",
        strategy_name="score",
        selected_factors=["pe"],
        rebalance_frequency="M",
        top_n=20,
        transaction_cost=0.001,
        data_root="data",
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        cache_dir="data/cache",
        output_dir="data/output",
        parquet_engine="auto",
        required_datasets=["daily"],
    )

    assert config.required_start_date == "2013-01-01"
    assert config.required_end_date == "2025-03-31"


def test_pipeline_config_accepts_yyyymmdd_dates(tmp_path) -> None:
    """PipelineConfig should normalize YYYYMMDD dates from YAML."""
    config_path = write_config(
        tmp_path,
        """
data:
  required_datasets:
    - daily
pipeline:
  backtest_start: "20240101"
  backtest_end: "20250331"
  train_years: 10
  max_lookback_months: 12
  stock_pool: hs300
  benchmark: 000300.SH
  strategy_name: score
  rebalance_frequency: M
  top_n: 20
  transaction_cost: 0.001
""",
    )

    config = PipelineConfig.from_yaml(config_path)

    assert config.backtest_start == "2024-01-01"
    assert config.backtest_end == "2025-03-31"


def test_pipeline_config_rejects_invalid_backtest_range() -> None:
    """PipelineConfig should reject a start date after the end date."""
    with pytest.raises(ValueError):
        PipelineConfig(
            backtest_start="2025-04-01",
            backtest_end="2025-03-31",
            train_years=10,
            max_lookback_months=12,
            stock_pool="hs300",
            benchmark="000300.SH",
            strategy_name="score",
            selected_factors=["pe"],
            rebalance_frequency="M",
            top_n=20,
            transaction_cost=0.001,
            data_root="data",
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            cache_dir="data/cache",
            output_dir="data/output",
            parquet_engine="auto",
            required_datasets=["daily"],
        )
