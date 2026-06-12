"""Command-line entry point for historical backtest visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtest_plot import plot_all_backtest_figures  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for historical backtest plotting."""
    parser = argparse.ArgumentParser(
        description="Create historical backtest figures from backtest_nav.csv."
    )
    parser.add_argument(
        "--nav-input",
        default="reports/tables/backtest_nav.csv",
        help="Input historical backtest NAV CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figures",
        help="Directory for historical backtest figure files.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    """Resolve a CLI path relative to project root when it is not absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    """Generate historical backtest figures and print output paths."""
    args = parse_args()
    nav_input = resolve_project_path(args.nav_input)
    output_dir = resolve_project_path(args.output_dir)
    figure_paths = plot_all_backtest_figures(nav_input, output_dir)

    print(f"Historical backtest NAV input path: {nav_input}")
    print(f"Historical backtest output dir: {output_dir}")
    print("Historical backtest figure files:")
    for name, path in figure_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
