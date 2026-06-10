"""Command-line entry point for building the monthly factor panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factors.factor_engine import build_factor_panel  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for factor panel construction."""
    parser = argparse.ArgumentParser(
        description="Build monthly factor panel from raw TuShare CSV files."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing raw CSV inputs.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/factor_panel.csv",
        help="Output CSV path for the factor panel.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    """Resolve a CLI path relative to project root when it is not absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    """Build the factor panel and print a concise output summary."""
    args = parse_args()
    raw_dir = resolve_project_path(args.raw_dir)
    output_path = resolve_project_path(args.output)
    factor_panel = build_factor_panel(raw_dir=raw_dir, output_path=output_path)

    date_min = factor_panel["date"].min() if not factor_panel.empty else None
    date_max = factor_panel["date"].max() if not factor_panel.empty else None
    stock_count = factor_panel["ts_code"].nunique() if "ts_code" in factor_panel else 0
    return_next_count = (
        factor_panel["return_next"].notna().sum()
        if "return_next" in factor_panel
        else 0
    )

    print(f"Input raw_dir: {raw_dir}")
    print(f"Output path: {output_path}")
    print(f"Factor panel rows: {len(factor_panel)}")
    print(f"Columns: {list(factor_panel.columns)}")
    print(f"Date range: {date_min} to {date_max}")
    print(f"Stock count: {stock_count}")
    print(f"Non-null return_next count: {return_next_count}")


if __name__ == "__main__":
    main()
