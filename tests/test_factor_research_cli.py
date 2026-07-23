"""End-to-end acceptance tests for the unified factor-research CLI."""

from __future__ import annotations

import importlib.util
import json
import locale
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from src.factors.registry import create_default_registry
from src.factors.research_artifacts import FactorResearchArtifactStore
from src.pipeline import FactorResearchPipelineConfig, FactorResearchPipelineExecutor
from src.pipeline.config import PipelineConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "run_pipeline.py"
EXAMPLE_CONFIG = PROJECT_ROOT / "config" / "factor_research.example.yaml"
NAMES = ("momentum_20d", "volatility_20d")


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("factor_research_cli", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_panels(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True)
    dates = pd.bdate_range("2024-01-02", periods=34)
    codes = [f"S{index:02d}" for index in range(6)]
    factor_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(dates):
        for stock_index, code in enumerate(codes):
            close = (
                100.0
                * (1.0 + 0.001 * (stock_index + 1)) ** date_index
                * (1.0 + 0.002 * np.sin(date_index + stock_index))
            )
            price_rows.append(
                {"trade_date": trade_date, "ts_code": code, "close": close}
            )
            if date_index < 31:
                factor_rows.append(
                    {"trade_date": trade_date, "ts_code": code, "close": close}
                )
    score_panel = pd.DataFrame(
        [
            {"trade_date": trade_date, "ts_code": code}
            for trade_date in dates[22:27]
            for code in codes
        ]
    )
    frames = {
        "factor_input": pd.DataFrame(factor_rows),
        "score_panel": score_panel,
        "price_panel": pd.DataFrame(price_rows),
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    return paths


def _relative_to_project(path: Path) -> str:
    """Use a project-relative path when Windows drive boundaries permit it."""
    try:
        return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()
    except ValueError:
        # pytest tmp_path may be on C: while the checked-out project is on E:.
        # Windows has no valid relative path across drives.
        return str(path.resolve())


def _config_mapping(
    tmp_path: Path,
    *,
    enabled: bool,
    paths: dict[str, Path] | None = None,
) -> dict[str, object]:
    research_paths = paths or {
        "factor_input": tmp_path / "missing-factor.parquet",
        "score_panel": tmp_path / "missing-score.parquet",
        "price_panel": tmp_path / "missing-price.parquet",
    }
    return {
        "data": {
            "root": str(tmp_path / "data"),
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "output"),
            "parquet_engine": "auto",
            "required_datasets": ["daily"],
        },
        "strategy": {"transaction_cost": 0.001},
        "pipeline": {
            "backtest_start": "2024-01-01",
            "backtest_end": "2024-03-31",
            "train_years": 1,
            "max_lookback_months": 1,
            "stock_pool": "test_pool",
            "benchmark": "000300.SH",
            "strategy_name": "factor_cli",
            "rebalance_frequency": "M",
            "top_n": 5,
            "transaction_cost": 0.001,
        },
        "factors": {"selected": list(NAMES)},
        "factor_research": {
            "enabled": enabled,
            "factor_input_path": _relative_to_project(
                research_paths["factor_input"]
            ),
            "score_panel_path": _relative_to_project(research_paths["score_panel"]),
            "price_panel_path": _relative_to_project(research_paths["price_panel"]),
            "exposure_panel_path": None,
            "artifact_subdir": "factor_research",
            "research": {
                "factor_names": list(NAMES),
                "use_neutralization": False,
                "composition_method": "equal",
                "evaluate_components": True,
                "evaluate_composite": True,
            },
            "preprocessing": {
                "missing_method": "median",
                "winsor_method": "mad",
                "standardize_method": "zscore",
                "min_cross_section_size": 3,
            },
            "evaluation": {
                "return_col": "forward_return",
                "min_cross_section_size": 5,
                "compute_ic": True,
                "compute_rank_ic": True,
            },
            "quantile": {
                "return_col": "forward_return",
                "quantiles": 3,
                "min_cross_section_size": 5,
                "min_group_size": 1,
                "compute_monotonicity": True,
            },
            "composition": {
                "method": "equal",
                "fixed_weights": [],
                "normalize_weights": True,
                "missing_policy": "renormalize",
                "min_valid_factors": 1,
                "score_col": "composite_score",
            },
            "rolling": {
                "metric": "rank_ic",
                "lookback_periods": 3,
                "min_periods": 2,
                "negative_policy": "zero",
                "fallback_method": "equal",
                "missing_policy": "renormalize",
                "min_valid_factors": 1,
                "score_col": "composite_score",
            },
            "forward_returns": {
                "price_col": "close",
                "return_col": "forward_return",
                "entry_lag_periods": 1,
                "holding_periods": 2,
                "require_positive_prices": True,
            },
            "artifacts": {
                "tables_dirname": "tables",
                "manifest_filename": "manifest.json",
                "compression": "snappy",
                "include_empty_tables": True,
                "overwrite": False,
                "schema_version": "1",
                "verify_after_write": True,
            },
        },
    }


def _write_config(path: Path, mapping: dict[str, object]) -> Path:
    path.write_text(
        yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _run_cli(
    config: Path | None = None,
    *extra: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(CLI)]
    if config is not None:
        command.extend(["--config", str(config)])
    command.extend(extra)
    return subprocess.run(
        command,
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding=locale.getpreferredencoding(False),
        errors="strict",
        check=False,
    )


def _runs(tmp_path: Path) -> list[Path]:
    root = tmp_path / "output" / "runs"
    return sorted(root.iterdir()) if root.exists() else []


def test_default_config_path_is_stable_from_unrelated_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_cli_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(CLI)])

    args = module.parse_args()

    assert Path(args.config) == PROJECT_ROOT / "config" / "config.yaml"


def test_main_resolves_explicit_relative_config_and_restores_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(
        tmp_path / "relative-disabled.yaml",
        _config_mapping(tmp_path, enabled=False),
    )
    module = _load_cli_module()
    monkeypatch.chdir(tmp_path)
    original_cwd = Path.cwd()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(CLI), "--config", config.name],
    )

    assert module.main() == 0

    assert Path.cwd() == original_cwd
    assert len(_runs(tmp_path)) == 1


def test_main_restores_cwd_when_config_loading_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_cli_module()
    monkeypatch.chdir(tmp_path)
    original_cwd = Path.cwd()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(CLI), "--config", "missing.yaml"],
    )

    with pytest.raises(FileNotFoundError):
        module.main()

    assert Path.cwd() == original_cwd


def test_help_preserves_existing_options_and_adds_no_required_argument() -> None:
    result = _run_cli(None, "--help")
    assert result.returncode == 0
    for option in (
        "--config",
        "--backtest-start",
        "--backtest-end",
        "--train-years",
        "--max-lookback-months",
        "--strategy-name",
        "--stock-pool",
        "--top-n",
        "--benchmark",
        "--transaction-cost",
        "--json",
    ):
        assert option in result.stdout
    assert "required" not in result.stderr.lower()


def test_example_config_is_safe_parseable_and_uses_registered_factors() -> None:
    text = EXAMPLE_CONFIG.read_text(encoding="utf-8")
    config = PipelineConfig.from_yaml(EXAMPLE_CONFIG)
    assert config.factor_research.enabled is False
    assert config.factor_research.research is not None
    assert config.factor_research.research.factor_names == NAMES
    lowered = text.lower()
    assert "c:\\users" not in lowered
    assert "/users/" not in lowered
    for secret_name in ("token", "secret", "password"):
        assert secret_name not in lowered
    registry = FactorResearchPipelineExecutor(
        FactorResearchPipelineConfig()
    )._build_registry()
    assert all(registry.contains(name) for name in NAMES)
    assert create_default_registry() is not registry


def test_disabled_config_runs_once_without_research_artifacts(
    tmp_path: Path,
) -> None:
    config = _write_config(
        tmp_path / "disabled.yaml",
        _config_mapping(tmp_path, enabled=False),
    )
    result = _run_cli(config)
    assert result.returncode == 0, result.stderr
    assert "Factor research enabled: false" in result.stdout
    assert len(_runs(tmp_path)) == 1
    assert not (_runs(tmp_path)[0] / "factor_research").exists()


def test_enabled_cli_runs_real_chain_from_unrelated_cwd(tmp_path: Path) -> None:
    paths = _write_panels(tmp_path / "inputs")
    config = _write_config(
        tmp_path / "enabled.yaml",
        _config_mapping(tmp_path, enabled=True, paths=paths),
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    result = _run_cli(config, cwd=unrelated)
    assert result.returncode == 0, result.stderr
    runs = _runs(tmp_path)
    assert len(runs) == 1
    artifact_dir = runs[0] / "factor_research"
    assert str(runs[0]) in result.stdout
    assert str(artifact_dir.resolve()) in result.stdout
    assert "momentum_20d, volatility_20d" in result.stdout
    assert "Composition method: equal" in result.stdout
    assert "Manifest verification status: valid" in result.stdout
    assert "DataFrame" not in result.stdout
    assert '"artifact_type"' not in result.stdout
    manifest = json.loads((artifact_dir / "manifest.json").read_text("utf-8"))
    assert manifest["artifact_type"] == "factor_research"
    assert FactorResearchArtifactStore().verify(artifact_dir)["valid"] is True
    tables = artifact_dir / "tables"
    for name in (
        "raw_factor_panel",
        "final_factor_panel",
        "forward_returns",
        "factor_ic_results",
        "factor_quantile_results",
        "composite_scores",
    ):
        assert (tables / f"{name}.parquet").is_file()
    assert not any(
        term in path.name.lower()
        for path in artifact_dir.rglob("*")
        for term in ("holding", "rebalance", "equity_curve")
    )


def test_json_output_is_compact_parseable_and_omits_manifest(
    tmp_path: Path,
) -> None:
    paths = _write_panels(tmp_path / "inputs")
    config = _write_config(
        tmp_path / "enabled.yaml",
        _config_mapping(tmp_path, enabled=True, paths=paths),
    )
    result = _run_cli(config, "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["factor_research"]["enabled"] is True
    assert payload["factor_research"]["manifest_verification"] == "valid"
    assert "manifest" not in payload["factor_research"]
    assert "DataFrame" not in result.stdout


def test_two_cli_runs_create_independent_verified_artifacts(tmp_path: Path) -> None:
    paths = _write_panels(tmp_path / "inputs")
    config = _write_config(
        tmp_path / "enabled.yaml",
        _config_mapping(tmp_path, enabled=True, paths=paths),
    )
    first = _run_cli(config)
    second = _run_cli(config)
    assert first.returncode == second.returncode == 0
    runs = _runs(tmp_path)
    assert len(runs) == 2
    assert runs[0] != runs[1]
    for run in runs:
        assert FactorResearchArtifactStore().verify(
            run / "factor_research"
        )["valid"] is True


def test_missing_config_and_invalid_yaml_fail_with_context(tmp_path: Path) -> None:
    missing = _run_cli(tmp_path / "missing.yaml")
    assert missing.returncode != 0
    assert "FileNotFoundError" in missing.stderr
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("pipeline: [", encoding="utf-8")
    invalid = _run_cli(invalid_path)
    assert invalid.returncode != 0
    assert "yaml" in invalid.stderr.lower() or "parsererror" in invalid.stderr.lower()


def test_missing_input_fails_after_one_run_without_artifact(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "missing-input.yaml",
        _config_mapping(tmp_path, enabled=True),
    )
    result = _run_cli(config)
    assert result.returncode != 0
    assert "factor_input" in result.stderr
    assert "FileNotFoundError" in result.stderr
    assert len(_runs(tmp_path)) == 1
    assert not (_runs(tmp_path)[0] / "factor_research").exists()


def test_relative_input_path_uses_project_root_from_unrelated_cwd(
    tmp_path: Path,
) -> None:
    mapping = _config_mapping(tmp_path, enabled=True)
    factor_research = mapping["factor_research"]
    assert isinstance(factor_research, dict)
    factor_research["factor_input_path"] = "data/missing-cli-factor.parquet"
    config = _write_config(tmp_path / "relative.yaml", mapping)
    unrelated = tmp_path / "unrelated-relative"
    unrelated.mkdir()

    result = _run_cli(config, cwd=unrelated)

    assert result.returncode != 0
    expected = str(
        (PROJECT_ROOT / "data" / "missing-cli-factor.parquet").resolve()
    )
    assert expected in result.stderr
    assert str(unrelated / "data" / "missing-cli-factor.parquet") not in result.stderr

def test_return_column_conflict_fails_before_creating_run(tmp_path: Path) -> None:
    mapping = _config_mapping(tmp_path, enabled=True)
    factor_research = mapping["factor_research"]
    assert isinstance(factor_research, dict)
    evaluation = factor_research["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["return_col"] = "other_return"
    config = _write_config(tmp_path / "conflict.yaml", mapping)
    result = _run_cli(config)
    assert result.returncode != 0
    assert "return_col" in result.stderr
    assert not _runs(tmp_path)
