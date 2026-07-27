"""V4-F offline end-to-end and release-smoke acceptance tests.

The files -> Modeling Panel -> ML tests execute the real public APIs.  The
Factor Research boundary test deliberately starts from a real frozen
PublishedOutputs result and is an orchestration integration test, not a full
Factor Research calculation test.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

import src.pipeline.runner as runner_module
from src.ml import MLExperimentConfig
from src.modeling_panel import (
    MODELING_PANEL_AUDIT_COLUMNS,
    MODELING_PANEL_KEY_COLUMNS,
    ModelingPanelArtifactConfig,
    ModelingPanelArtifactStore,
    ModelingPanelArtifactWriteError,
    ModelingPanelBuilder,
    ModelingPanelConfig,
)
from src.pipeline import (
    FactorResearchExecutionResult,
    FactorResearchPublishedOutputs,
    MLExperimentPipelineConfig,
    ModelingPanelPipelineConfig,
    ModelingPanelPipelineExecutionError,
    ModelingPanelPipelineExecutor,
    PipelineConfig,
    run_pipeline,
)


FEATURES = ("factor_a", "factor_b")
ARTIFACT_FILES = {
    "modeling_panel.parquet",
    "config.json",
    "audit.json",
    "manifest.json",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _frames(
    *, periods: int = 16, stocks: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factors: list[dict[str, object]] = []
    returns: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(
        pd.date_range("2024-01-01", periods=periods, freq="D")
    ):
        for stock_index in range(stocks):
            code = f"S{stock_index:02d}"
            factor_a = float(date_index + stock_index)
            entry_price = 20.0 + factor_a
            forward_return = 0.002 * (stock_index + 1) + 0.0001 * date_index
            factors.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "factor_a": factor_a,
                    "factor_b": float(stock_index - date_index / 10),
                }
            )
            returns.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "entry_trade_date": trade_date + pd.Timedelta(days=1),
                    "exit_trade_date": trade_date + pd.Timedelta(days=2),
                    "entry_price": entry_price,
                    "exit_price": entry_price * (1.0 + forward_return),
                    "forward_return": forward_return,
                }
            )
    return pd.DataFrame(factors), pd.DataFrame(returns)


def _write_inputs(
    root: Path, *, periods: int = 16, stocks: int = 4
) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    factor_path = root / "factor_panel.parquet"
    returns_path = root / "forward_returns.parquet"
    factors, returns = _frames(periods=periods, stocks=stocks)
    factors.to_parquet(factor_path, engine="pyarrow", index=False)
    returns.to_parquet(returns_path, engine="pyarrow", index=False)
    return factor_path, returns_path


def _minimal_ml_config() -> MLExperimentPipelineConfig:
    experiment = MLExperimentConfig.from_dict(
        {
            "dataset": {"label_col": "forward_return"},
            "walk_forward": {
                "train_window_periods": 2,
                "validation_periods": 2,
                "window_type": "rolling",
                "retrain_frequency": 3,
                "embargo_periods": 1,
            },
            "training": {
                "model_name": "ridge",
                "model_params": {"alpha": 1.0},
            },
            "evaluation": {"minimum_cross_section_size": 3},
            "permutation_importance": None,
        }
    )
    return MLExperimentPipelineConfig(
        enabled=True,
        panel_path=None,
        save_artifacts=False,
        experiment=experiment,
    )


def _pipeline_config(
    tmp_path: Path,
    factor_path: Path,
    returns_path: Path,
    *,
    ml: bool = False,
) -> PipelineConfig:
    return PipelineConfig(
        backtest_start="2024-01-01",
        backtest_end="2024-01-31",
        train_years=0,
        max_lookback_months=0,
        stock_pool="synthetic",
        benchmark="SYNTHETIC",
        strategy_name="v4f_e2e",
        selected_factors=list(FEATURES),
        rebalance_frequency="D",
        top_n=2,
        transaction_cost=0.0,
        data_root=str(tmp_path / "data"),
        raw_data_dir=str(tmp_path / "data" / "raw"),
        processed_data_dir=str(tmp_path / "data" / "processed"),
        cache_dir=str(tmp_path / "data" / "cache"),
        output_dir=str(tmp_path / "runs"),
        parquet_engine="pyarrow",
        required_datasets=[],
        modeling_panel=ModelingPanelPipelineConfig.from_dict(
            {
                "enabled": True,
                "source": {
                    "mode": "files",
                    "factor_panel_path": factor_path,
                    "forward_returns_path": returns_path,
                },
                "builder": {"include_features": list(FEATURES)},
            }
        ),
        ml_experiment=_minimal_ml_config() if ml else None,
    )


def _assert_artifact_layout(artifact_dir: Path) -> None:
    assert {item.name for item in artifact_dir.iterdir()} == ARTIFACT_FILES
    report = ModelingPanelArtifactStore().validate(artifact_dir)
    assert report.is_valid, report.issues


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_files_to_modeling_panel_artifact_via_runner(
    tmp_path: Path,
) -> None:
    factor_path, returns_path = _write_inputs(
        tmp_path / "inputs", periods=6, stocks=3
    )
    input_hashes = (_sha256(factor_path), _sha256(returns_path))
    config = _pipeline_config(tmp_path, factor_path, returns_path)
    config_before = deepcopy(config.to_dict())

    summary = run_pipeline(config)

    run_dir = Path(summary["run_dir"]).resolve()
    artifact_dir = Path(summary["modeling_panel"]["artifact_dir"])
    panel_path = Path(summary["modeling_panel"]["panel_path"])
    assert run_dir.is_relative_to(tmp_path.resolve())
    assert artifact_dir == run_dir / "modeling_panel"
    assert panel_path == artifact_dir / "modeling_panel.parquet"
    assert summary["modeling_panel"]["enabled"] is True
    assert summary["modeling_panel"]["source_mode"] == "files"
    assert summary["modeling_panel"]["feature_names"] == list(FEATURES)
    assert summary["modeling_panel"]["label_column"] == "forward_return"
    assert summary["modeling_panel"]["output_rows"] == 18
    assert "factor_research" not in summary
    assert "ml_experiment" not in summary
    _assert_artifact_layout(artifact_dir)
    panel = pd.read_parquet(panel_path, engine="pyarrow")
    assert tuple(panel.columns) == (
        *MODELING_PANEL_KEY_COLUMNS,
        *FEATURES,
        *MODELING_PANEL_AUDIT_COLUMNS,
        "forward_return",
    )
    assert (_sha256(factor_path), _sha256(returns_path)) == input_hashes
    assert config.to_dict() == config_before
    json.dumps(summary, allow_nan=False)
    assert (run_dir / "config_snapshot.yaml").is_file()
    assert (run_dir / "run_info.json").is_file()


def test_real_modeling_panel_to_real_ridge_ml_uses_generated_override(
    tmp_path: Path,
) -> None:
    factor_path, returns_path = _write_inputs(tmp_path / "inputs")
    config = _pipeline_config(
        tmp_path, factor_path, returns_path, ml=True
    )
    config_before = deepcopy(config.to_dict())
    ml_before = deepcopy(config.ml_experiment.to_dict())

    summary = run_pipeline(config)

    run_dir = Path(summary["run_dir"])
    panel_path = Path(summary["modeling_panel"]["panel_path"])
    ml = summary["ml_experiment"]
    assert ml["enabled"] is True
    assert ml["model_name"] == "ridge"
    assert ml["n_folds"] > 0
    assert Path(ml["panel_path"]) == panel_path
    assert panel_path.parent.parent == run_dir
    assert config.ml_experiment.panel_path is None
    assert config.ml_experiment.to_dict() == ml_before
    assert config.to_dict() == config_before
    _assert_artifact_layout(panel_path.parent)


def test_factor_research_published_outputs_orchestration_integration(
    tmp_path: Path,
) -> None:
    upstream = tmp_path / "current-factor-research"
    factor_path, returns_path = _write_inputs(upstream)
    stale = tmp_path / "newer-but-unrelated"
    _write_inputs(stale)
    published = FactorResearchPublishedOutputs(
        upstream,
        factor_path,
        returns_path,
        FEATURES,
        "forward_return",
    )
    research_result = FactorResearchExecutionResult(
        enabled=True,
        artifact_dir=str(upstream),
        published_outputs=published,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = ModelingPanelPipelineConfig.from_dict(
        {
            "enabled": True,
            "source": {"mode": "factor_research"},
            "builder": {"include_features": list(FEATURES)},
        }
    )

    result = ModelingPanelPipelineExecutor(
        config, project_root=tmp_path
    ).execute(run_dir, factor_research_result=research_result)

    assert result.source_mode == "factor_research"
    assert result.feature_names == FEATURES
    assert result.output_rows == 64
    assert result.artifact_dir == (run_dir / "modeling_panel").resolve()
    assert not any(
        isinstance(value, pd.DataFrame) for value in vars(result).values()
    )
    assert not (stale / "modeling_panel").exists()
    _assert_artifact_layout(result.artifact_dir)


def test_two_real_runs_are_isolated_without_staging_residue(
    tmp_path: Path,
) -> None:
    factor_path, returns_path = _write_inputs(
        tmp_path / "inputs", periods=6, stocks=3
    )
    config = _pipeline_config(tmp_path, factor_path, returns_path)

    first = run_pipeline(config)
    second = run_pipeline(config)

    first_dir = Path(first["run_dir"])
    second_dir = Path(second["run_dir"])
    assert first_dir != second_dir
    for summary, run_dir in ((first, first_dir), (second, second_dir)):
        artifact = Path(summary["modeling_panel"]["artifact_dir"])
        assert artifact.parent == run_dir
        _assert_artifact_layout(artifact)
        manifest = ModelingPanelArtifactStore().read_manifest(artifact)
        assert manifest.row_count == 18
    assert not list(tmp_path.rglob(".tmp-*"))
    assert not [path for path in tmp_path.rglob("*") if "backup" in path.name.lower()]


def test_artifact_store_no_overwrite_preserves_first_publish(
    tmp_path: Path,
) -> None:
    factors, returns = _frames(periods=4, stocks=3)
    result = ModelingPanelBuilder(
        ModelingPanelConfig(include_features=FEATURES)
    ).build(factors, returns)
    artifact_dir = tmp_path / "artifact"
    store = ModelingPanelArtifactStore()
    store.write(result, ModelingPanelArtifactConfig(artifact_dir=artifact_dir))
    hashes = {name: _sha256(artifact_dir / name) for name in ARTIFACT_FILES}

    with pytest.raises(
        ModelingPanelArtifactWriteError, match="already exists"
    ):
        store.write(
            result,
            ModelingPanelArtifactConfig(artifact_dir=artifact_dir),
        )

    assert {name: _sha256(artifact_dir / name) for name in ARTIFACT_FILES} == hashes
    _assert_artifact_layout(artifact_dir)


def test_builder_failure_stops_runner_before_ml_and_leaves_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factor_path, returns_path = _write_inputs(tmp_path / "inputs")
    duplicate = pd.read_parquet(factor_path, engine="pyarrow")
    duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
    duplicate.to_parquet(factor_path, engine="pyarrow", index=False)
    config = _pipeline_config(tmp_path, factor_path, returns_path, ml=True)
    ml_called = False

    def forbidden_ml(*args: object, **kwargs: object) -> object:
        nonlocal ml_called
        ml_called = True
        raise AssertionError("ML must not execute after Modeling Panel failure")

    monkeypatch.setattr(
        runner_module.MLExperimentPipelineExecutor, "execute", forbidden_ml
    )
    with pytest.raises(ModelingPanelPipelineExecutionError, match="build failed"):
        run_pipeline(config)

    assert ml_called is False
    assert not list((tmp_path / "runs").rglob("modeling_panel"))
    assert not list((tmp_path / "runs").rglob(".tmp-*"))


def test_artifact_collision_is_propagated_and_existing_target_preserved(
    tmp_path: Path,
) -> None:
    factor_path, returns_path = _write_inputs(tmp_path / "inputs")
    run_dir = tmp_path / "run"
    target = run_dir / "modeling_panel"
    target.mkdir(parents=True)
    marker = target / "owner.txt"
    marker.write_text("pre-existing", encoding="utf-8")
    config = _pipeline_config(tmp_path, factor_path, returns_path).modeling_panel

    with pytest.raises(
        ModelingPanelPipelineExecutionError, match="Artifact write failed"
    ):
        ModelingPanelPipelineExecutor(
            config, project_root=tmp_path
        ).execute(run_dir)

    assert marker.read_text(encoding="utf-8") == "pre-existing"
    assert set(target.iterdir()) == {marker}
    assert not list(run_dir.glob(".tmp-*"))


def test_orchestrated_ml_failure_keeps_real_modeling_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factor_path, returns_path = _write_inputs(tmp_path / "inputs")
    config = _pipeline_config(tmp_path, factor_path, returns_path, ml=True)
    observed: dict[str, Path] = {}

    def fail_ml(
        self: object,
        run_dir: str | Path,
        *,
        panel_path_override: str | Path | None = None,
    ) -> object:
        observed["run_dir"] = Path(run_dir)
        assert panel_path_override is not None
        observed["panel"] = Path(panel_path_override)
        raise RuntimeError("controlled ML boundary failure")

    monkeypatch.setattr(
        runner_module.MLExperimentPipelineExecutor, "execute", fail_ml
    )
    with pytest.raises(RuntimeError, match="controlled ML"):
        run_pipeline(config)

    assert observed["panel"].parent.parent == observed["run_dir"]
    _assert_artifact_layout(observed["panel"].parent)


def test_builder_is_deterministic_and_defensive() -> None:
    factors, returns = _frames(periods=5, stocks=3)
    factor_before = factors.copy(deep=True)
    returns_before = returns.copy(deep=True)
    builder = ModelingPanelBuilder(
        ModelingPanelConfig(include_features=FEATURES)
    )

    first = builder.build(factors, returns)
    second = builder.build(factors, returns)

    pdt.assert_frame_equal(first.panel, second.panel)
    assert first.feature_names == second.feature_names == FEATURES
    assert first.audit.warnings == second.audit.warnings
    assert first.audit.factor_only == second.audit.factor_only
    assert first.audit.return_only == second.audit.return_only
    pdt.assert_frame_equal(factors, factor_before)
    pdt.assert_frame_equal(returns, returns_before)

    json.dumps(first.audit.as_dict(), allow_nan=False)


def test_cli_help_and_example_config_release_smoke(tmp_path: Path) -> None:
    example = REPO_ROOT / "config" / "modeling_panel_pipeline.example.yaml"
    raw = yaml.safe_load(example.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(raw)
    assert config.modeling_panel.enabled is True
    assert config.factor_research.enabled is False
    assert config.ml_experiment.enabled is False
    assert config.required_datasets == []
    before = {path.resolve() for path in tmp_path.rglob("*")}

    completed = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert {path.resolve() for path in tmp_path.rglob("*")} == before
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    guide = (
        REPO_ROOT / "docs" / "05_modeling_panel_pipeline.md"
    ).read_text(encoding="utf-8")
    assert "config/modeling_panel_pipeline.example.yaml" in readme
    assert "docs/05_modeling_panel_pipeline.md" in readme
    assert "--config" in guide

