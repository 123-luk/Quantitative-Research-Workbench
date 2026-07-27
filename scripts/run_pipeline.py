"""Command-line entry point for the V1 pipeline skeleton."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.config import PipelineConfig  # noqa: E402
from src.pipeline.ml_cli import (  # noqa: E402
    MLCLIError,
    exit_code_for_ml_error,
    format_ml_human_summary,
    merge_ml_cli_overrides,
    parse_ml_model_params,
)
from src.pipeline.ml_config import MLPipelineError  # noqa: E402
from src.pipeline.modeling_panel_config import (  # noqa: E402
    ModelingPanelPipelineConfigError,
    ModelingPanelPipelineError,
)
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


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments for the pipeline skeleton."""
    parser = argparse.ArgumentParser(description="Run the V1 pipeline skeleton.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to YAML config; direct PipelineConfig YAML supports modeling_panel.",
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
    ml_group = parser.add_mutually_exclusive_group()
    ml_group.add_argument(
        "--ml", dest="ml_enabled", action="store_true", default=None,
        help="Enable the configured ML experiment.",
    )
    ml_group.add_argument(
        "--no-ml", dest="ml_enabled", action="store_false",
        help="Disable the configured ML experiment.",
    )
    parser.add_argument(
        "--ml-panel", dest="ml_panel", default=None, metavar="PATH",
        help="Override the pre-merged ML modeling panel Parquet path.",
    )
    parser.add_argument(
        "--ml-model", dest="ml_model", default=None, metavar="NAME",
        help="Override the configured ML model name.",
    )
    parser.add_argument(
        "--ml-model-params", dest="ml_model_params", default=None,
        metavar="JSON_OBJECT",
        help="Replace all configured model parameters with a JSON object.",
    )
    importance_group = parser.add_mutually_exclusive_group()
    importance_group.add_argument(
        "--ml-permutation-importance",
        dest="ml_permutation_importance",
        action="store_true",
        default=None,
        help="Enable walk-forward permutation importance.",
    )
    importance_group.add_argument(
        "--no-ml-permutation-importance",
        dest="ml_permutation_importance",
        action="store_false",
        help="Disable walk-forward permutation importance.",
    )
    parser.add_argument(
        "--ml-importance-repeats",
        dest="ml_importance_repeats",
        type=int,
        default=None,
        metavar="INT",
        help="Override permutation-importance repetitions.",
    )
    parser.add_argument(
        "--ml-importance-scoring",
        dest="ml_importance_scoring",
        choices=("rmse", "mae"),
        default=None,
        help="Override permutation-importance scoring.",
    )
    parser.add_argument(
        "--ml-min-cross-section-size",
        dest="ml_min_cross_section_size",
        type=int,
        default=None,
        metavar="INT",
        help="Override the evaluation minimum cross-section size.",
    )
    artifact_group = parser.add_mutually_exclusive_group()
    artifact_group.add_argument(
        "--ml-save-artifacts",
        dest="ml_save_artifacts",
        action="store_true",
        default=None,
        help="Enable ML artifact persistence.",
    )
    artifact_group.add_argument(
        "--no-ml-save-artifacts",
        dest="ml_save_artifacts",
        action="store_false",
        help="Disable ML artifact persistence.",
    )
    parser.add_argument(
        "--ml-artifact-root",
        dest="ml_artifact_root",
        default=None,
        metavar="RELATIVE_PATH",
        help="Override the safe run-relative ML artifact directory.",
    )
    parser.add_argument(
        "--ml-experiment-id",
        dest="ml_experiment_id",
        default=None,
        metavar="ID",
        help="Override the ML artifact experiment identifier.",
    )
    parser.add_argument(
        "--ml-parquet-compression",
        dest="ml_parquet_compression",
        choices=("zstd", "snappy", "none"),
        default=None,
        help="Override ML artifact Parquet compression.",
    )
    return parser.parse_args(argv)


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


def build_ml_cli_overrides(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Build sparse ML overrides from explicitly supplied CLI options."""
    mapping: dict[str, object | None] = {
        "enabled": args.ml_enabled,
        "panel_path": args.ml_panel,
        "model_name": args.ml_model,
        "permutation_importance_enabled": (
            args.ml_permutation_importance
        ),
        "importance_repeats": args.ml_importance_repeats,
        "importance_scoring": args.ml_importance_scoring,
        "minimum_cross_section_size": (
            args.ml_min_cross_section_size
        ),
        "save_artifacts": args.ml_save_artifacts,
        "artifact_root": args.ml_artifact_root,
        "experiment_id": args.ml_experiment_id,
        "parquet_compression": args.ml_parquet_compression,
    }
    if args.ml_model_params is not None:
        mapping["model_params"] = parse_ml_model_params(
            args.ml_model_params
        )
    return {
        key: value for key, value in mapping.items() if value is not None
    }


def load_pipeline_config(
    config_path: Path,
    overrides: dict[str, Any],
) -> PipelineConfig:
    """Load direct Modeling Panel YAML or preserve the legacy grouped schema."""
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    if "modeling_panel" not in raw:
        return PipelineConfig.from_yaml(
            config_path=config_path,
            overrides=overrides,
        )
    values = dict(raw)
    values.update(overrides)
    return PipelineConfig.from_dict(values)

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

    output = {
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
    modeling_summary = summary.get("modeling_panel")
    if isinstance(modeling_summary, dict):
        output["modeling_panel"] = dict(modeling_summary)
    ml_summary = summary.get("ml_experiment")
    if isinstance(ml_summary, dict):
        output["ml_experiment"] = dict(ml_summary)
    return output


def print_human_summary(output: dict[str, Any]) -> None:
    """Print the compact run summary in a PowerShell-friendly form."""
    research = output["factor_research"]
    print(f"Pipeline status: {output.get('status')}")
    print(f"Run directory: {output.get('run_dir')}")
    print(f"Required start: {output.get('required_start_date')}")
    print(f"Required end: {output.get('required_end_date')}")
    print(f"Factor research enabled: {str(research['enabled']).lower()}")
    if research["enabled"]:
        print(f"Artifact directory: {research.get('artifact_dir')}")
        print(
            "Factor names: "
            + ", ".join(
                str(name) for name in research.get("factor_names", ())
            )
        )
        print(f"Composition method: {research.get('composition_method')}")
        print(
            "Input shapes: "
            + json.dumps(
                research.get("input_shapes", {}), ensure_ascii=False
            )
        )
        print(
            "Major table shapes: "
            + json.dumps(
                research.get("table_shapes", {}), ensure_ascii=False
            )
        )
        print(
            "Manifest verification status: "
            + str(research.get("manifest_verification"))
        )
    modeling = output.get("modeling_panel")
    if isinstance(modeling, dict) and modeling.get("enabled") is True:
        print("Modeling panel enabled: true")
        print(f"Modeling panel path: {modeling.get('panel_path')}")
        print(f"Modeling panel artifact: {modeling.get('artifact_dir')}")
        print(
            "Modeling panel features: "
            + ", ".join(
                str(name) for name in modeling.get("feature_names", ())
            )
        )
    for line in format_ml_human_summary(output.get("ml_experiment")):
        print(line)


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Temporarily change the process working directory and always restore it."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the pipeline and print a compact human or JSON summary."""
    invocation_cwd = Path.cwd()
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = invocation_cwd / config_path
    config_path = config_path.resolve()

    try:
        with _working_directory(PROJECT_ROOT):
            config = load_pipeline_config(
                config_path,
                build_overrides(args),
            )
            ml_overrides = build_ml_cli_overrides(args)
            if ml_overrides:
                config.ml_experiment = merge_ml_cli_overrides(
                    config.ml_experiment,
                    ml_overrides,
                )
                config = PipelineConfig.from_dict(config.to_dict())
            summary = run_pipeline(config)
            output = build_output(config, summary)
            if args.json:
                print(
                    json.dumps(
                        output,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
            else:
                print_human_summary(output)
    except ModelingPanelPipelineConfigError as exc:
        message = " ".join(str(exc).splitlines())
        print(f"Modeling Panel config error: {message}", file=sys.stderr)
        return 2
    except ModelingPanelPipelineError as exc:
        message = " ".join(str(exc).splitlines())
        print(f"Modeling Panel pipeline error: {message}", file=sys.stderr)
        return 4
    except (MLCLIError, MLPipelineError) as exc:
        message = " ".join(str(exc).splitlines())
        print(f"ML pipeline error: {message}", file=sys.stderr)
        return exit_code_for_ml_error(exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
