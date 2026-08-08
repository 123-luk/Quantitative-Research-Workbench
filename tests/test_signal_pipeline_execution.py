"""Execution-boundary tests for the optional V5 Signal pipeline stage."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

import src.pipeline.signal_execution as execution_module
from src.ml import (
    MLArtifactConfig,
    MLExperimentArtifactStore,
    MLExperimentConfig,
    MLExperimentRunner,
    MLDatasetConfig,
    ModelEvaluationConfig,
    WalkForwardConfig,
    WalkForwardTrainingConfig,
)
from src.pipeline import (
    MLExperimentPipelineResult,
    PredictionSourceConfig,
    SignalPipelineConfig,
    SignalPipelineExecutionError,
    SignalPipelineExecutor,
)
from src.signals import SignalArtifactStore


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, trade_date in enumerate(pd.date_range("2024-01-01", periods=12)):
        for stock_number in range(24):
            value = float(date_number + stock_number / 10)
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"S{stock_number:03d}",
                    "factor_a": value,
                    "factor_b": float(stock_number - date_number / 10),
                    "entry_trade_date": trade_date + pd.Timedelta(days=1),
                    "exit_trade_date": trade_date + pd.Timedelta(days=2),
                    "forward_return": value / 100,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def native_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("signal-execution-native")
    config = MLExperimentConfig(
        dataset_config=MLDatasetConfig(),
        walk_forward_config=WalkForwardConfig(
            train_window_periods=2,
            validation_periods=2,
            window_type="rolling",
            retrain_frequency=3,
            embargo_periods=1,
        ),
        training_config=WalkForwardTrainingConfig("ridge"),
        evaluation_config=ModelEvaluationConfig(minimum_cross_section_size=3),
    )
    result = MLExperimentRunner().run(_panel(), config)
    return MLExperimentArtifactStore().write(
        result, MLArtifactConfig(root, "source")
    ).experiment_dir


def _config(source: Path, **changes: object) -> SignalPipelineConfig:
    values: dict[str, object] = {
        "enabled": True,
        "source": PredictionSourceConfig("files", source),
        "prediction_column": "prediction",
        "signal_direction": "descending",
        "artifact_subdir": "signal-output",
    }
    values.update(changes)
    return SignalPipelineConfig(**values)  # type: ignore[arg-type]


def _ml_result(native: Path, panel_path: Path) -> MLExperimentPipelineResult:
    return MLExperimentPipelineResult(
        enabled=True,
        model_name="ridge",
        n_folds=1,
        n_prediction_rows=1,
        n_prediction_dates=1,
        mae=0.1,
        rmse=0.1,
        r2=0.0,
        r2_valid=True,
        r2_invalid_reason=None,
        pearson_ic_mean=0.0,
        rank_ic_mean=0.0,
        permutation_importance_enabled=False,
        permutation_importance_completed=False,
        artifacts_saved=True,
        artifact_dir=str(native.resolve()),
        panel_path=str(panel_path.resolve()),
    )


@pytest.mark.parametrize("direction", ["ascending", "descending"])
def test_files_mode_builds_valid_artifact_from_exact_source(
    native_artifact: Path, tmp_path: Path, direction: str
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "newer-decoy").mkdir()
    config = _config(native_artifact, signal_direction=direction)
    before = asdict(config)
    result = SignalPipelineExecutor(config).execute(run_dir)
    assert result.enabled and result.source_mode == "files"
    assert result.source_artifact_dir == native_artifact.resolve()
    assert result.artifact_dir == (run_dir / "signal-output").resolve()
    assert result.signal_path == result.artifact_dir / "signals.parquet"
    assert result.prediction_column == "prediction"
    assert result.signal_direction == direction
    assert result.rows > 0 and result.trade_date_count > 0
    assert not hasattr(result, "signals")
    assert SignalArtifactStore().validate(result.artifact_dir).is_valid
    assert asdict(config) == before


def test_ml_mode_uses_exact_runtime_result_artifact(
    native_artifact: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(panel_path)
    config = SignalPipelineConfig(
        enabled=True,
        source=PredictionSourceConfig("ml"),
        artifact_subdir="signal",
    )
    result = SignalPipelineExecutor(config).execute(
        run_dir, ml_result=_ml_result(native_artifact, panel_path)
    )
    assert result.source_mode == "ml"
    assert result.source_artifact_dir == native_artifact.resolve()
    assert SignalArtifactStore().validate(result.artifact_dir).is_valid


def test_source_handoff_errors_are_explicit(
    native_artifact: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ml_config = SignalPipelineConfig(enabled=True, source=PredictionSourceConfig("ml"))
    with pytest.raises(SignalPipelineExecutionError, match="requires ML"):
        SignalPipelineExecutor(ml_config).execute(run_dir)
    with pytest.raises(SignalPipelineExecutionError, match="does not accept"):
        SignalPipelineExecutor(_config(native_artifact)).execute(
            run_dir, ml_result=MLExperimentPipelineResult.disabled()
        )
    missing = tmp_path / "missing"
    with pytest.raises(SignalPipelineExecutionError, match="source validation"):
        SignalPipelineExecutor(_config(missing)).execute(run_dir)


def test_existing_target_is_not_overwritten(
    native_artifact: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    executor = SignalPipelineExecutor(_config(native_artifact))
    first = executor.execute(run_dir)
    before = first.signal_path.read_bytes()
    with pytest.raises(SignalPipelineExecutionError, match="Artifact write"):
        executor.execute(run_dir)
    assert first.signal_path.read_bytes() == before


def test_disabled_stage_short_circuits_without_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("disabled Signal stage performed I/O")

    monkeypatch.setattr(execution_module.PredictionSourceAdapter, "load_native_ml_artifact", forbidden)
    monkeypatch.setattr(execution_module.SignalArtifactStore, "write", forbidden)
    result = SignalPipelineExecutor(SignalPipelineConfig()).execute(
        tmp_path / "need-not-exist"
    )
    assert result == execution_module.SignalPipelineResult.disabled()
    assert result.as_dict()["enabled"] is False
