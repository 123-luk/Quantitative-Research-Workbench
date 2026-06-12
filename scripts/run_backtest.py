"""Command-line entry point for historical portfolio backtests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.portfolio_backtest import run_backtest  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the historical backtest script."""
    parser = argparse.ArgumentParser(
        description="Run a historical backtest for the model-selected portfolio."
    )
    parser.add_argument(
        "--input",
        default="reports/tables/selected_portfolio.csv",
        help="Input selected portfolio CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/tables",
        help="Directory for historical backtest output CSV files.",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.0005,
        help="One-way transaction cost rate used in historical backtest.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    """Resolve a CLI path relative to project root when it is not absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    """Run the historical portfolio backtest and save result tables."""
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_df = pd.read_csv(input_path)
    holdings_df, turnover_df, nav_df, metrics_df = run_backtest(
        selected_df,
        transaction_cost=args.transaction_cost,
    )

    output_paths = {
        "backtest_holdings": output_dir / "backtest_holdings.csv",
        "backtest_turnover": output_dir / "backtest_turnover.csv",
        "backtest_nav": output_dir / "backtest_nav.csv",
        "backtest_metrics": output_dir / "backtest_metrics.csv",
    }
    holdings_df.to_csv(output_paths["backtest_holdings"], index=False, encoding="utf-8-sig")
    turnover_df.to_csv(output_paths["backtest_turnover"], index=False, encoding="utf-8-sig")
    nav_df.to_csv(output_paths["backtest_nav"], index=False, encoding="utf-8-sig")
    metrics_df.to_csv(output_paths["backtest_metrics"], index=False, encoding="utf-8-sig")

    date_min = nav_df["date"].min() if "date" in nav_df and not nav_df.empty else None
    date_max = nav_df["date"].max() if "date" in nav_df and not nav_df.empty else None

    print(f"Input path: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Transaction cost: {args.transaction_cost}")
    print(f"Holdings rows: {len(holdings_df)}")
    print(f"Historical backtest periods: {len(nav_df)}")
    print(f"Date range: {date_min} to {date_max}")
    print("Historical backtest metrics:")
    print(metrics_df)
    print("Output files:")
    for name, path in output_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
