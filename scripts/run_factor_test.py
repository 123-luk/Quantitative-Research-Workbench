"""Command-line entry point for factor effectiveness tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factors.factor_test import (  # noqa: E402
    batch_calc_ic,
    batch_quantile_group_return,
    get_default_factor_cols,
    long_short_return,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for factor testing."""
    parser = argparse.ArgumentParser(
        description="Run IC and quantile return tests on a clean factor panel."
    )
    parser.add_argument(
        "--input",
        default="data/processed/factor_panel_clean.csv",
        help="Input cleaned factor panel CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/tables",
        help="Directory for factor test result CSV files.",
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=5,
        help="Number of quantile groups for group return analysis.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    """Resolve a CLI path relative to project root when it is not absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    """Run factor tests, save result tables, and print a summary."""
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(input_path)
    factor_cols = get_default_factor_cols(panel)
    ic_series_df, ic_summary_df = batch_calc_ic(panel, factor_cols)
    group_return_df = batch_quantile_group_return(
        panel,
        factor_cols,
        n_groups=args.n_groups,
    )
    long_short_df = long_short_return(
        group_return_df,
        high_group=f"Q{args.n_groups}",
        low_group="Q1",
    )

    output_paths = {
        "ic_series": output_dir / "ic_series.csv",
        "ic_summary": output_dir / "ic_summary.csv",
        "group_return": output_dir / "group_return.csv",
        "long_short_return": output_dir / "long_short_return.csv",
    }
    ic_series_df.to_csv(output_paths["ic_series"], encoding="utf-8-sig")
    ic_summary_df.to_csv(output_paths["ic_summary"], index=False, encoding="utf-8-sig")
    group_return_df.to_csv(output_paths["group_return"], index=False, encoding="utf-8-sig")
    long_short_df.to_csv(
        output_paths["long_short_return"],
        index=False,
        encoding="utf-8-sig",
    )

    date_min = panel["date"].min() if "date" in panel and not panel.empty else None
    date_max = panel["date"].max() if "date" in panel and not panel.empty else None
    stock_count = panel["ts_code"].nunique() if "ts_code" in panel else 0

    print(f"Input path: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Rows: {len(panel)}")
    print(f"Date range: {date_min} to {date_max}")
    print(f"Stock count: {stock_count}")
    print(f"Factor columns: {factor_cols}")
    print("IC summary:")
    print(ic_summary_df)
    print("Output files:")
    for name, path in output_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
