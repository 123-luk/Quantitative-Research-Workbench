"""Command-line entry point for the V1 pipeline skeleton."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.config import PipelineConfig  # noqa: E402
from src.pipeline.runner import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline skeleton."""
    parser = argparse.ArgumentParser(description="Run the V1 pipeline skeleton.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument("--backtest-start", help="Backtest start date in YYYY-MM-DD.")
    parser.add_argument("--backtest-end", help="Backtest end date in YYYY-MM-DD.")
    parser.add_argument("--train-years", type=int, help="Training window length in years.")
    parser.add_argument(
        "--max-lookback-months",
        type=int,
        help="Maximum factor lookback window in months.",
    )
    parser.add_argument("--strategy-name", help="Strategy name used in run metadata.")
    parser.add_argument("--stock-pool", help="Stock pool name used in run metadata.")
    parser.add_argument("--top-n", type=int, help="Number of selected stocks.")
    parser.add_argument("--benchmark", help="Benchmark index code.")
    parser.add_argument(
        "--transaction-cost",
        type=float,
        help="One-way transaction cost used by later backtests.",
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build PipelineConfig overrides from provided CLI arguments."""
    mapping = {
        "backtest_start": args.backtest_start,
        "backtest_end": args.backtest_end,
        "train_years": args.train_years,
        "max_lookback_months": args.max_lookback_months,
        "strategy_name": args.strategy_name,
        "stock_pool": args.stock_pool,
        "top_n": args.top_n,
        "benchmark": args.benchmark,
        "transaction_cost": args.transaction_cost,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def main() -> int:
    """Run the pipeline skeleton and print a JSON summary."""
    args = parse_args()
    config = PipelineConfig.from_yaml(
        config_path=PROJECT_ROOT / args.config,
        overrides=build_overrides(args),
    )
    summary = run_pipeline(config)
    output = {
        "required_start_date": summary["required_start_date"],
        "required_end_date": summary["required_end_date"],
        "backtest_start": config.backtest_start,
        "backtest_end": config.backtest_end,
        "strategy_name": summary["strategy_name"],
        "stock_pool": summary["stock_pool"],
        "cache_status": summary["cache_status"],
        "missing_ranges": summary["missing_ranges"],
        "run_dir": summary["run_dir"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
