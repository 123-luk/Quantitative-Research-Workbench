"""Command-line entry point for the multi-factor scoring model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.scoring_model import run_scoring_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the scoring model."""
    parser = argparse.ArgumentParser(
        description="Run equal-weight multi-factor scoring on a clean factor panel."
    )
    parser.add_argument(
        "--input",
        default="data/processed/factor_panel_clean.csv",
        help="Input clean factor panel CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/tables",
        help="Directory for scoring result CSV files.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of model-selected stocks per month.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    """Resolve a CLI path relative to project root when it is not absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    """Run scoring, save output tables, and print a neutral summary."""
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(input_path)
    factor_score_df, selected_portfolio_df, factor_weights = run_scoring_pipeline(
        panel,
        top_n=args.top_n,
    )

    factor_score_path = output_dir / "factor_score.csv"
    selected_portfolio_path = output_dir / "selected_portfolio.csv"
    factor_score_df.to_csv(factor_score_path, index=False, encoding="utf-8-sig")
    selected_portfolio_df.to_csv(
        selected_portfolio_path,
        index=False,
        encoding="utf-8-sig",
    )

    date_min = panel["date"].min() if "date" in panel and not panel.empty else None
    date_max = panel["date"].max() if "date" in panel and not panel.empty else None
    stock_count = panel["ts_code"].nunique() if "ts_code" in panel else 0
    selected_counts = (
        selected_portfolio_df.groupby("date")["ts_code"].count().to_dict()
        if {"date", "ts_code"}.issubset(selected_portfolio_df.columns)
        else {}
    )

    print(f"Input path: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Rows: {len(panel)}")
    print(f"Date range: {date_min} to {date_max}")
    print(f"Stock count: {stock_count}")
    print(f"Factor weights: {factor_weights}")
    print(f"Monthly selected portfolio counts: {selected_counts}")
    print("Output files:")
    print(f"- factor_score: {factor_score_path}")
    print(f"- selected_portfolio: {selected_portfolio_path}")


if __name__ == "__main__":
    main()
