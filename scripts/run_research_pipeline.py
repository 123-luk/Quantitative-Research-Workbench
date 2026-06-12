"""One-command runner for the full quantitative research pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the research pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full historical research pipeline end to end."
    )
    parser.add_argument("--start", default="20240101", help="Start date in YYYYMMDD format.")
    parser.add_argument("--end", default="20241231", help="End date in YYYYMMDD format.")
    parser.add_argument(
        "--universe",
        default="hs300",
        choices=("hs300", "all"),
        help="Stock universe used by the TuShare fetch step.",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=50,
        help="Maximum stocks for data fetching debug runs. Use 0 for no limit.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top N model-selected stocks.")
    parser.add_argument("--n-groups", type=int, default=5, help="Factor test quantile groups.")
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.0005,
        help="Transaction cost used by the historical backtest.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to wait between TuShare API calls in fetch step.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip TuShare fetching and use existing data/raw files.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip historical backtest figure generation.",
    )
    return parser.parse_args()


def run_step(step_title: str, command: list[str]) -> int:
    """Print and run one pipeline command, returning its exit code."""
    print(f"\n{step_title}")
    print("Command:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return completed.returncode


def build_pipeline_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Build ordered pipeline steps from CLI arguments."""
    python = sys.executable
    steps: list[tuple[str, list[str]]] = []

    if not args.skip_fetch:
        steps.append(
            (
                "[1/7] Fetch TuShare data",
                [
                    python,
                    "scripts/fetch_tushare_data.py",
                    "--start",
                    args.start,
                    "--end",
                    args.end,
                    "--universe",
                    args.universe,
                    "--max-stocks",
                    str(args.max_stocks),
                    "--sleep",
                    str(args.sleep),
                ],
            )
        )
    else:
        print("[1/7] Fetch TuShare data - skipped by --skip-fetch")

    steps.extend(
        [
            (
                "[2/7] Build factor panel",
                [
                    python,
                    "scripts/build_factor_panel.py",
                    "--raw-dir",
                    "data/raw",
                    "--output",
                    "data/processed/factor_panel.csv",
                ],
            ),
            (
                "[3/7] Preprocess factor panel",
                [
                    python,
                    "scripts/preprocess_factor_panel.py",
                    "--input",
                    "data/processed/factor_panel.csv",
                    "--output",
                    "data/processed/factor_panel_clean.csv",
                ],
            ),
            (
                "[4/7] Run factor effectiveness tests",
                [
                    python,
                    "scripts/run_factor_test.py",
                    "--input",
                    "data/processed/factor_panel_clean.csv",
                    "--output-dir",
                    "reports/tables",
                    "--n-groups",
                    str(args.n_groups),
                ],
            ),
            (
                "[5/7] Run multi-factor scoring",
                [
                    python,
                    "scripts/run_scoring_model.py",
                    "--input",
                    "data/processed/factor_panel_clean.csv",
                    "--output-dir",
                    "reports/tables",
                    "--top-n",
                    str(args.top_n),
                ],
            ),
            (
                "[6/7] Run historical portfolio backtest",
                [
                    python,
                    "scripts/run_backtest.py",
                    "--input",
                    "reports/tables/selected_portfolio.csv",
                    "--output-dir",
                    "reports/tables",
                    "--transaction-cost",
                    str(args.transaction_cost),
                ],
            ),
        ]
    )

    if not args.skip_plot:
        steps.append(
            (
                "[7/7] Plot historical backtest figures",
                [
                    python,
                    "scripts/plot_backtest.py",
                    "--nav-input",
                    "reports/tables/backtest_nav.csv",
                    "--output-dir",
                    "reports/figures",
                ],
            )
        )
    else:
        print("[7/7] Plot historical backtest figures - skipped by --skip-plot")

    return steps


def print_output_locations(skip_plot: bool) -> None:
    """Print key pipeline output locations."""
    outputs = [
        "data/processed/factor_panel.csv",
        "data/processed/factor_panel_clean.csv",
        "reports/tables/ic_summary.csv",
        "reports/tables/selected_portfolio.csv",
        "reports/tables/backtest_nav.csv",
        "reports/tables/backtest_metrics.csv",
    ]
    if not skip_plot:
        outputs.extend(
            [
                "reports/figures/nav_curve.png",
                "reports/figures/monthly_return_bar.png",
                "reports/figures/drawdown_curve.png",
            ]
        )

    print("\nHistorical research pipeline output locations:")
    for output in outputs:
        print(f"- {PROJECT_ROOT / output}")


def main() -> int:
    """Run the full historical sample research pipeline."""
    args = parse_args()
    steps = build_pipeline_commands(args)
    for step_title, command in steps:
        return_code = run_step(step_title, command)
        if return_code != 0:
            print(f"\nPipeline stopped at step: {step_title}")
            print(f"Exit code: {return_code}")
            return return_code

    print_output_locations(skip_plot=args.skip_plot)
    print("\nPipeline completed. Outputs are historical sample backtest and research results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
