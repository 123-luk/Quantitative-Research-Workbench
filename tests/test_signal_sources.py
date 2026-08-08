"""Tests for loading predictions from explicit native ML Artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pandas.testing as pdt
import pytest

import src.signals.sources as source_module
from src.ml import (
    MLArtifactConfig,
    MLArtifactValidationError,
    MLExperimentArtifactStore,
    MLExperimentConfig,
    MLExperimentRunner,
    MLDatasetConfig,
    ModelEvaluationConfig,
    WalkForwardConfig,
    WalkForwardTrainingConfig,
)
from src.signals import (
    NATIVE_ML_PREDICTIONS_FILENAME,
    PredictionSourceAdapter,
    PredictionSourceError,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, trade_date in enumerate(
        pd.date_range("2024-01-01", periods=12, freq="D")
    ):
        for stock_number in range(3):
            value = float(date_number + stock_number)
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"S{stock_number:02d}",
                    "factor_a": value,
                    "factor_b": float(stock_number - date_number / 10),
                    "entry_trade_date": trade_date + pd.Timedelta(days=1),
                    "exit_trade_date": trade_date + pd.Timedelta(days=2),
                    "forward_return": value / 100.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def native_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("native-ml-artifact")
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
    result = MLExperimentRunner().run(_frame(), config)
    written = MLExperimentArtifactStore().write(
        result, MLArtifactConfig(root, "signal-source")
    )
    return written.experiment_dir


def _copy_artifact(source: Path, destination: Path) -> Path:
    target = destination / source.name
    shutil.copytree(source, target)
    return target


def _hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_valid_native_artifact_loads_exact_predictions_and_provenance(
    native_artifact: Path,
) -> None:
    before = _hashes(native_artifact)
    result = PredictionSourceAdapter().load_native_ml_artifact(native_artifact)
    expected = pd.read_parquet(
        native_artifact / NATIVE_ML_PREDICTIONS_FILENAME, engine="pyarrow"
    )
    pdt.assert_frame_equal(result.predictions, expected)
    assert result.provenance.artifact_dir == native_artifact.resolve()
    assert result.provenance.prediction_path == (
        native_artifact / NATIVE_ML_PREDICTIONS_FILENAME
    ).resolve()
    assert result.provenance.artifact_schema_version == "1.0"
    assert result.provenance.experiment_id == native_artifact.name
    assert result.provenance.model_name == "ridge"
    assert len(result.provenance.prediction_sha256) == 64
    assert _hashes(native_artifact) == before


def test_source_result_predictions_are_defensive(native_artifact: Path) -> None:
    result = PredictionSourceAdapter().load_native_ml_artifact(native_artifact)
    original = result.predictions
    changed = result.predictions
    changed.iloc[0, changed.columns.get_loc("prediction")] = 999.0
    pdt.assert_frame_equal(result.predictions, original)


@pytest.mark.parametrize("value", [None, "", " missing ", 1])
def test_invalid_explicit_artifact_inputs_are_rejected(value: object) -> None:
    with pytest.raises(PredictionSourceError):
        PredictionSourceAdapter().load_native_ml_artifact(value)  # type: ignore[arg-type]


def test_bare_predictions_and_ordinary_files_are_rejected(
    native_artifact: Path, tmp_path: Path
) -> None:
    adapter = PredictionSourceAdapter()
    with pytest.raises(PredictionSourceError, match="bare predictions"):
        adapter.load_native_ml_artifact(
            native_artifact / NATIVE_ML_PREDICTIONS_FILENAME
        )
    ordinary = tmp_path / "ordinary.txt"
    ordinary.write_text("not an artifact", encoding="utf-8")
    with pytest.raises(PredictionSourceError, match="directory"):
        adapter.load_native_ml_artifact(ordinary)


def test_missing_required_artifact_file_is_rejected(
    native_artifact: Path, tmp_path: Path
) -> None:
    damaged = _copy_artifact(native_artifact, tmp_path)
    (damaged / "training_audit.json").unlink()
    with pytest.raises(PredictionSourceError, match="validation failed"):
        PredictionSourceAdapter().load_native_ml_artifact(damaged)


def test_manifest_tamper_is_rejected(native_artifact: Path, tmp_path: Path) -> None:
    damaged = _copy_artifact(native_artifact, tmp_path)
    path = damaged / "experiment_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["model_name"] = "tampered"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PredictionSourceError, match="validation failed"):
        PredictionSourceAdapter().load_native_ml_artifact(damaged)


def test_prediction_checksum_tamper_is_rejected(
    native_artifact: Path, tmp_path: Path
) -> None:
    damaged = _copy_artifact(native_artifact, tmp_path)
    path = damaged / NATIVE_ML_PREDICTIONS_FILENAME
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(PredictionSourceError, match="validation failed"):
        PredictionSourceAdapter().load_native_ml_artifact(damaged)


def test_validator_failure_has_no_parquet_fallback(
    native_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[Path] = []

    def fail_validation(self: object, directory: object) -> object:
        raise MLArtifactValidationError("injected failure")

    def record_read(path: object, **kwargs: object) -> pd.DataFrame:
        reads.append(Path(path))
        raise AssertionError("fallback read must not occur")

    monkeypatch.setattr(MLExperimentArtifactStore, "validate", fail_validation)
    monkeypatch.setattr(source_module.pd, "read_parquet", record_read)
    with pytest.raises(PredictionSourceError, match="validation failed"):
        PredictionSourceAdapter().load_native_ml_artifact(native_artifact)
    assert reads == []


def test_explicit_source_ignores_sibling_artifacts(
    native_artifact: Path, tmp_path: Path
) -> None:
    selected = _copy_artifact(native_artifact, tmp_path)
    sibling = tmp_path / "newer-latest-run"
    sibling.mkdir()
    (sibling / NATIVE_ML_PREDICTIONS_FILENAME).write_bytes(b"decoy")
    first = PredictionSourceAdapter().load_native_ml_artifact(selected)
    second = PredictionSourceAdapter().load_native_ml_artifact(selected)
    pdt.assert_frame_equal(first.predictions, second.predictions)
    assert first.provenance == second.provenance
    assert first.provenance.artifact_dir == selected.resolve()