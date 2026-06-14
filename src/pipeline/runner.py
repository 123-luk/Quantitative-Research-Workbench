"""Pipeline skeleton runner that wires data checks and experiment artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.data.data_manager import DataManager
from src.pipeline.config import PipelineConfig
from src.pipeline.experiment import ExperimentManager


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Run the V1 pipeline skeleton and return a concise run summary.

    The V1 runner only checks local data cache readiness and creates experiment
    artifacts. It does not download TuShare data, train models, or run backtests.
    """
    required_start_date = config.required_start_date
    required_end_date = config.required_end_date

    data_manager = DataManager()
    data_status = data_manager.prepare_data(
        {
            "required_start_date": required_start_date,
            "backtest_end": required_end_date,
            "required_datasets": config.required_datasets,
        }
    )

    experiment_manager = ExperimentManager(config.output_dir)
    run_dir = experiment_manager.create_run_dir(
        strategy_name=config.strategy_name,
        stock_pool=config.stock_pool,
    )

    cache_status = str(data_status["cache_status"])
    missing_ranges = data_status["missing_ranges"]
    status = "ready" if cache_status == "ready" else "missing_data"

    summary = {
        "status": status,
        "run_dir": str(run_dir),
        "required_start_date": required_start_date,
        "required_end_date": required_end_date,
        "cache_status": cache_status,
        "missing_ranges": missing_ranges,
        "strategy_name": config.strategy_name,
        "stock_pool": config.stock_pool,
    }

    experiment_manager.save_config_snapshot(run_dir, config)
    experiment_manager.save_run_info(
        run_dir,
        {
            "status": status,
            "created_at": datetime.now().replace(microsecond=0).isoformat(),
            "strategy_name": config.strategy_name,
            "stock_pool": config.stock_pool,
            "required_start_date": required_start_date,
            "required_end_date": required_end_date,
            "cache_status": cache_status,
            "missing_ranges": missing_ranges,
        },
    )
    experiment_manager.save_metrics(
        run_dir,
        {
            "status": "placeholder",
            "metrics_ready": False,
            "message": "Pipeline skeleton has not run strategy, model, or backtest metrics.",
        },
    )

    return summary
