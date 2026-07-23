"""Command-line entry point for the V1 pipeline skeleton."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.config import PipelineConfig  # noqa: E402
from src.pipeline.runner import run_pipeline  # noqa: E402


MAJOR_RESEARCH_TABLES = (
    "raw_factor_panel",
    "final_factor_panel",
    "forward_returns",
    "factor_ic_results",
    "factor_quantile_results",
    "composite_scores",
    "composite_ic_results",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline skeleton."""
    parser = argparse.ArgumentParser(description="Run the V1 pipeline skeleton.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the compact run summary as JSON.",
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


def build_output(config: PipelineConfig, summary: dict[str, Any]) -> dict[str, Any]:
    """Build a compact JSON-safe summary without full tables or manifests."""
    research = summary.get("factor_research")
    enabled = bool(config.factor_research.enabled)
    research_output: dict[str, Any] = {"enabled": enabled}
    if enabled and isinstance(research, dict):
        table_shapes = research.get("table_shapes")
        if not isinstance(table_shapes, dict):
            table_shapes = {}
        research_output.update(
            {
                "artifact_dir": research.get("artifact_dir"),
                "factor_names": list(research.get("factor_names") or ()),
                "composition_method": research.get("composition_method"),
                "input_shapes": research.get("input_shapes") or {},
                "table_shapes": {
                    name: table_shapes[name]
                    for name in MAJOR_RESEARCH_TABLES
                    if name in table_shapes
                },
                "manifest_verification": (
                    "valid"
                    if config.factor_research.artifacts.verify_after_write
                    and research.get("manifest") is not None
                    else "not_checked"
                ),
            }
        )

    return {
        "status": summary.get("status"),
        "required_start_date": summary.get("required_start_date"),
        "required_end_date": summary.get("required_end_date"),
        "backtest_start": config.backtest_start,
        "backtest_end": config.backtest_end,
        "strategy_name": summary.get("strategy_name"),
        "stock_pool": summary.get("stock_pool"),
        "cache_status": summary.get("cache_status"),
        "missing_ranges": summary.get("missing_ranges") or {},
        "run_dir": summary.get("run_dir"),
        "factor_research": research_output,
    }


def print_human_summary(output: dict[str, Any]) -> None:
    """Print the compact run summary in a PowerShell-friendly form."""
    research = output["factor_research"]
    print(f"Pipeline status: {output.get('status')}")
    print(f"Run directory: {output.get('run_dir')}")
    print(f"Required start: {output.get('required_start_date')}")
    print(f"Required end: {output.get('required_end_date')}")
    print(f"Factor research enabled: {str(research['enabled']).lower()}")
    if not research["enabled"]:
        return
    print(f"Artifact directory: {research.get('artifact_dir')}")
    print(
        "Factor names: "
        + ", ".join(str(name) for name in research.get("factor_names", ()))
    )
    print(f"Composition method: {research.get('composition_method')}")
    print(
        "Input shapes: "
        + json.dumps(research.get("input_shapes", {}), ensure_ascii=False)
    )
    print(
        "Major table shapes: "
        + json.dumps(research.get("table_shapes", {}), ensure_ascii=False)
    )
    print(
        "Manifest verification status: "
        + str(research.get("manifest_verification"))
    )


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Temporarily change the process working directory and always restore it."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main() -> int:
    """Run the pipeline and print a compact human or JSON summary."""
    invocation_cwd = Path.cwd()
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = invocation_cwd / config_path
    config_path = config_path.resolve()

    with _working_directory(PROJECT_ROOT):
        config = PipelineConfig.from_yaml(
            config_path=config_path,
            overrides=build_overrides(args),
        )
        summary = run_pipeline(config)
        output = build_output(config, summary)
        if args.json:
            print(json.dumps(output, ensure_ascii=False, allow_nan=False))
        else:
            print_human_summary(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
