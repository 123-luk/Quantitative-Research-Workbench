"""Execution-boundary tests for the optional V5 Holdings pipeline stage."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import src.pipeline.holdings_execution as execution_module
from src.holdings import HoldingsArtifactStore
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
    HoldingsPipelineConfig,
    HoldingsPipelineExecutionError,
    HoldingsPipelineExecutor,
    PredictionSourceConfig,
    SignalPipelineConfig,
    SignalPipelineExecutor,
    SignalPipelineResult,
)
from src.signals import (
    PredictionSourceProvenance,
    SignalArtifactConfig,
    SignalArtifactStore,
    SignalBuilder,
)


def _predictions(counts: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, count in enumerate(counts, start=1):
        for stock_number in range(count):
            rows.append(
                {
                    "trade_date": pd.Timestamp(f"2024-01-{date_number:02d}"),
                    "ts_code": f"S{stock_number:03d}",
                    "prediction": float(count - stock_number),
                }
            )
    return pd.DataFrame(rows)


def _signal_result(
    tmp_path: Path, counts: tuple[int, ...] = (24, 24), name: str = "signal"
) -> SignalPipelineResult:
    native = tmp_path / f"{name}-native"
    native.mkdir()
    prediction_path = native / "predictions.parquet"
    prediction_path.write_bytes(b"explicit-native-source")
    provenance = PredictionSourceProvenance(
        native,
        prediction_path,
        "1.0",
        f"{name}-experiment",
        "ridge",
        hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
    )
    built = SignalBuilder().build(
        _predictions(counts),
        prediction_column="prediction",
        signal_direction="descending",
    )
    written = SignalArtifactStore().write(
        built, provenance, SignalArtifactConfig(tmp_path / name)
    )
    return SignalPipelineResult(
        enabled=True,
        source_mode="files",
        source_artifact_dir=native,
        artifact_dir=written.artifact_dir,
        signal_path=written.signal_path,
        manifest_path=written.manifest_path,
        rows=built.audit.output_rows,
        trade_date_count=built.audit.trade_date_count,
        prediction_column="prediction",
        signal_direction="descending",
        schema_version=written.schema_version,
    )


def _holdings_config(top_n: int, **changes: object) -> HoldingsPipelineConfig:
    values: dict[str, object] = {
        "enabled": True,
        "top_n": top_n,
        "insufficient_universe_policy": "error",
        "weighting": "equal_weight",
        "artifact_subdir": f"holdings-{top_n}",
    }
    values.update(changes)
    return HoldingsPipelineConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_n", [1, 10, 20])
def test_exact_top_n_builds_and_records_valid_holdings(
    tmp_path: Path, top_n: int
) -> None:
    signal = _signal_result(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _holdings_config(top_n)
    before = asdict(config)
    result = HoldingsPipelineExecutor(config).execute(
        run_dir, signal_result=signal
    )
    assert result.enabled and result.requested_top_n == top_n
    assert result.rows == 2 * top_n and result.trade_date_count == 2
    assert result.source_signal_artifact_dir == signal.artifact_dir
    assert result.holdings_path == result.artifact_dir / "holdings.parquet"
    assert not hasattr(result, "holdings")
    report = HoldingsArtifactStore().validate(result.artifact_dir)
    assert report.is_valid and report.manifest is not None
    assert report.manifest.top_n == top_n
    assert asdict(config) == before


def test_same_signal_changes_only_selection_and_weights_by_n(tmp_path: Path) -> None:
    signal = _signal_result(tmp_path)
    source_manifest = SignalArtifactStore().read_manifest(signal.artifact_dir)
    source_sha = next(
        item.sha256 for item in source_manifest.files
        if item.relative_path == "signals.parquet"
    )
    outputs: dict[int, pd.DataFrame] = {}
    recorded_sha: dict[int, str] = {}
    for top_n in (5, 10):
        run_dir = tmp_path / f"run-{top_n}"
        run_dir.mkdir()
        result = HoldingsPipelineExecutor(_holdings_config(top_n)).execute(
            run_dir, signal_result=signal
        )
        outputs[top_n] = pd.read_parquet(result.holdings_path)
        manifest = HoldingsArtifactStore().read_manifest(result.artifact_dir)
        recorded_sha[top_n] = manifest.source_signal_provenance["signal_sha256"]
        assert json.loads((result.artifact_dir / "config.json").read_text())["top_n"] == top_n
        assert np.allclose(
            outputs[top_n].groupby("trade_date")["target_weight"].first(),
            1 / top_n,
        )
    assert recorded_sha == {5: source_sha, 10: source_sha}
    pdt.assert_frame_equal(
        outputs[5].drop(columns="target_weight"),
        outputs[10].loc[outputs[10]["rank"] <= 5].reset_index(drop=True).drop(
            columns="target_weight"
        ),
    )


def test_partial_and_error_policies_are_passed_unchanged(tmp_path: Path) -> None:
    partial_signal = _signal_result(tmp_path, (7, 7))
    error_run = tmp_path / "error-run"
    error_run.mkdir()
    with pytest.raises(HoldingsPipelineExecutionError, match="Holdings build"):
        HoldingsPipelineExecutor(_holdings_config(10)).execute(
            error_run, signal_result=partial_signal
        )
    partial_run = tmp_path / "partial-run"
    partial_run.mkdir()
    result = HoldingsPipelineExecutor(
        _holdings_config(
            10,
            insufficient_universe_policy="allow_partial",
            artifact_subdir="partial",
        )
    ).execute(partial_run, signal_result=partial_signal)
    frame = pd.read_parquet(result.holdings_path)
    assert result.insufficient_universe_policy == "allow_partial"
    assert result.weighting == "equal_weight"
    assert result.rows == 14
    assert np.allclose(frame.groupby("trade_date")["target_weight"].first(), 1 / 7)


def test_explicit_handoff_and_validated_source_are_required(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    executor = HoldingsPipelineExecutor(_holdings_config(5))
    with pytest.raises(HoldingsPipelineExecutionError, match="requires Signal"):
        executor.execute(run_dir)
    with pytest.raises(HoldingsPipelineExecutionError, match="enabled Signal"):
        executor.execute(run_dir, signal_result=SignalPipelineResult.disabled())
    signal = _signal_result(tmp_path)
    signal.signal_path.write_bytes(signal.signal_path.read_bytes() + b"tamper")
    with pytest.raises(HoldingsPipelineExecutionError, match="validation is invalid"):
        executor.execute(run_dir, signal_result=signal)


def test_existing_target_is_not_overwritten(tmp_path: Path) -> None:
    signal = _signal_result(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    executor = HoldingsPipelineExecutor(_holdings_config(5))
    first = executor.execute(run_dir, signal_result=signal)
    before = first.holdings_path.read_bytes()
    with pytest.raises(HoldingsPipelineExecutionError, match="Artifact write"):
        executor.execute(run_dir, signal_result=signal)
    assert first.holdings_path.read_bytes() == before


def test_disabled_stage_short_circuits_without_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("disabled Holdings stage performed I/O")

    monkeypatch.setattr(execution_module.SignalArtifactStore, "validate", forbidden)
    monkeypatch.setattr(execution_module.HoldingsArtifactStore, "write", forbidden)
    result = HoldingsPipelineExecutor(HoldingsPipelineConfig()).execute(
        tmp_path / "need-not-exist"
    )
    assert result == execution_module.HoldingsPipelineResult.disabled()


def _native_ml_artifact(tmp_path: Path) -> Path:
    rows: list[dict[str, object]] = []
    for date_number, trade_date in enumerate(pd.date_range("2024-03-01", periods=12)):
        for stock_number in range(12):
            value = float(date_number + stock_number / 10)
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"I{stock_number:03d}",
                    "factor_a": value,
                    "entry_trade_date": trade_date + pd.Timedelta(days=1),
                    "exit_trade_date": trade_date + pd.Timedelta(days=2),
                    "forward_return": value / 100,
                }
            )
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
    result = MLExperimentRunner().run(pd.DataFrame(rows), config)
    return MLExperimentArtifactStore().write(
        result, MLArtifactConfig(tmp_path / "ml", "integration")
    ).experiment_dir


def test_real_executor_to_executor_integration(tmp_path: Path) -> None:
    native = _native_ml_artifact(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    signal = SignalPipelineExecutor(
        SignalPipelineConfig(
            enabled=True,
            source=PredictionSourceConfig("files", native),
            artifact_subdir="signal",
        )
    ).execute(run_dir)
    holdings = HoldingsPipelineExecutor(
        HoldingsPipelineConfig(enabled=True, top_n=5, artifact_subdir="holdings")
    ).execute(run_dir, signal_result=signal)
    assert SignalArtifactStore().validate(signal.artifact_dir).is_valid
    assert HoldingsArtifactStore().validate(holdings.artifact_dir).is_valid
    assert holdings.source_signal_artifact_dir == signal.artifact_dir
    assert holdings.requested_top_n == 5
