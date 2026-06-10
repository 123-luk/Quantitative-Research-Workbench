"""Command-line entry point for preprocessing the factor panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factors.factor_preprocess import preprocess_factor_panel  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for factor panel preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess factor_panel.csv into factor_panel_clean.csv."
    )
    parser.add_argument(
        "--input",
        default="data/processed/factor_panel.csv",
        help="Input factor panel CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/factor_panel_clean.csv",
        help="Output cleaned factor panel CSV path.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    """Resolve a CLI path relative to project root when it is not absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    """Load, preprocess, save, and summarize the factor panel."""
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)

    panel = pd.read_csv(input_path)
    clean_df, factor_cols = preprocess_factor_panel(panel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    date_min = clean_df["date"].min() if "date" in clean_df and not clean_df.empty else None
    date_max = clean_df["date"].max() if "date" in clean_df and not clean_df.empty else None
    stock_count = clean_df["ts_code"].nunique() if "ts_code" in clean_df else 0
    missing_counts = clean_df[factor_cols].isna().sum().to_dict() if factor_cols else {}

    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Original rows: {len(panel)}")
    print(f"Clean rows: {len(clean_df)}")
    print(f"Factor columns: {factor_cols}")
    print(f"Factor missing counts: {missing_counts}")
    print(f"Date range: {date_min} to {date_max}")
    print(f"Stock count: {stock_count}")


if __name__ == "__main__":
    main()
