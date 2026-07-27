"""Contract tests for V3-H ML experiment artifact persistence."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import src.ml.artifacts as artifact_module
from src.ml import (
    ArtifactFileRecord,
    ML_ARTIFACT_SCHEMA_VERSION,
    MLArtifactConfig,
    MLArtifactConfigError,
    MLArtifactDataError,
    MLArtifactExistsError,
    MLArtifactIntegrityError,
    MLArtifactManifest,
    MLArtifactValidationError,
    MLArtifactWriteError,
    MLExperimentArtifactStore,
    MLExperimentConfig,
    MLExperimentRunner,
    MLDatasetConfig,
    ModelEvaluationConfig,
    PermutationImportanceOptionsConfig,
    WalkForwardConfig,
    WalkForwardTrainingConfig,
)


def _frame(periods: int = 16, stocks: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, date in enumerate(
        pd.date_range("2024-01-01", periods=periods, freq="D")
    ):
        for stock_number in range(stocks):
            factor_a = float(date_number + stock_number)
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": f"S{stock_number:02d}",
                    "factor_a": factor_a,
                    "factor_b": float(stock_number - date_number / 10),
                    "entry_trade_date": date + pd.Timedelta(days=1),
                    "exit_trade_date": date + pd.Timedelta(days=2),
                    "forward_return": factor_a / 100.0
                    + (stock_number % 2) / 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _result(*, importance: bool = False):
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
        evaluation_config=ModelEvaluationConfig(
            minimum_cross_section_size=3
        ),
        permutation_importance=(
            PermutationImportanceOptionsConfig(n_repeats=2, random_state=7)
            if importance
            else None
        ),
    )
    return MLExperimentRunner().run(_frame(), config)


@pytest.fixture(scope="module")
def result_without_importance():
    return _result()


@pytest.fixture(scope="module")
def result_with_importance():
    return _result(importance=True)


@pytest.mark.parametrize("root", ["artifacts", Path("artifacts")])
def test_config_accepts_root_types_and_is_json_safe(root: object) -> None:
    config = MLArtifactConfig(root, "Ridge.Demo-1", " ZSTD ")  # type: ignore[arg-type]
    assert config.artifact_root == Path("artifacts")
    assert config.experiment_id == "Ridge.Demo-1"
    assert config.parquet_compression == "zstd"
    json.dumps(config.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "experiment_id",
    ["", ".", "..", "/bad", "bad/name", r"bad\name", "bad:name", "_bad",
     " bad", "bad ", "a" * 129],
)
def test_config_rejects_invalid_experiment_ids(experiment_id: str) -> None:
    with pytest.raises(MLArtifactConfigError):
        MLArtifactConfig("artifacts", experiment_id)


@pytest.mark.parametrize("compression", ["zstd", "snappy", "none"])
def test_config_accepts_supported_compression(compression: str) -> None:
    assert (
        MLArtifactConfig("artifacts", "demo", compression).parquet_compression
        == compression
    )


def test_config_rejects_invalid_values_unknown_and_overwrite() -> None:
    with pytest.raises(MLArtifactConfigError):
        MLArtifactConfig("", "demo")
    with pytest.raises(MLArtifactConfigError):
        MLArtifactConfig("artifacts", "demo", "gzip")
    with pytest.raises(MLArtifactConfigError):
        MLArtifactConfig.from_dict(
            {"artifact_root": "a", "experiment_id": "b", "extra": 1}
        )
    with pytest.raises(TypeError):
        MLArtifactConfig("artifacts", "demo", overwrite=True)  # type: ignore[call-arg]


def test_config_is_frozen_and_defensive() -> None:
    source = {"artifact_root": "a", "experiment_id": "Demo"}
    config = MLArtifactConfig.from_dict(source)
    source["experiment_id"] = "Changed"
    returned = config.as_dict()
    returned["experiment_id"] = "Changed"
    assert config.experiment_id == "Demo"
    with pytest.raises(FrozenInstanceError):
        config.experiment_id = "x"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (np.int64(2), 2),
        (np.float64(1.5), 1.5),
        (pd.Timestamp("2024-01-01"), "2024-01-01T00:00:00"),
        (Path("a/b"), str(Path("a/b"))),
        ((1, "x"), [1, "x"]),
    ],
)
def test_strict_json_conversion_accepts_supported_values(
    value: object, expected: object
) -> None:
    assert artifact_module._to_json_safe(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        pd.DataFrame({"x": [1]}),
        pd.Series([1]),
        np.array([1]),
        lambda: None,
        {1: "bad"},
        object(),
    ],
)
def test_strict_json_conversion_rejects_unsafe_values(value: object) -> None:
    with pytest.raises(MLArtifactDataError):
        artifact_module._to_json_safe(value)


def test_strict_json_conversion_rejects_cycle() -> None:
    value: list[object] = []
    value.append(value)
    with pytest.raises(MLArtifactDataError, match="circular"):
        artifact_module._to_json_safe(value)


def _json_record(path: str = "audit.json") -> ArtifactFileRecord:
    return ArtifactFileRecord(
        relative_path=path,
        artifact_type="json",
        media_type="application/json",
        size_bytes=1,
        sha256="a" * 64,
    )


def _parquet_record() -> ArtifactFileRecord:
    return ArtifactFileRecord(
        relative_path="table.parquet",
        artifact_type="parquet",
        media_type="application/vnd.apache.parquet",
        size_bytes=10,
        sha256="b" * 64,
        row_count=2,
        columns=("x",),
        dtypes=(("x", "int64"),),
    )


def test_file_records_roundtrip_json_and_parquet() -> None:
    for record in (_json_record(), _parquet_record()):
        assert ArtifactFileRecord.from_dict(record.as_dict()) == record
        json.dumps(record.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relative_path": "/absolute.json"},
        {"relative_path": "../escape.json"},
        {"relative_path": r"bad\path.json"},
        {"relative_path": "experiment_manifest.json"},
        {"sha256": "BAD"},
        {"size_bytes": 0},
    ],
)
def test_file_record_rejects_invalid_fields(kwargs: dict[str, object]) -> None:
    values = _json_record().as_dict()
    values.update(kwargs)
    with pytest.raises(MLArtifactValidationError):
        ArtifactFileRecord.from_dict(values)


def test_manifest_roundtrip_and_validation() -> None:
    records = (_json_record(), _parquet_record())
    manifest = MLArtifactManifest(
        ML_ARTIFACT_SCHEMA_VERSION, "Demo", "ridge", False, 2, records
    )
    assert MLArtifactManifest.from_dict(manifest.as_dict()) == manifest
    json.dumps(manifest.as_dict(), allow_nan=False)
    with pytest.raises(MLArtifactValidationError):
        MLArtifactManifest("2.0", "Demo", "ridge", False, 2, records)
    with pytest.raises(MLArtifactValidationError):
        MLArtifactManifest("1.0", "Demo", "ridge", False, 1, records)
    with pytest.raises(MLArtifactValidationError):
        MLArtifactManifest(
            "1.0", "Demo", "ridge", False, 2, (_json_record(), _json_record())
        )


@pytest.mark.parametrize(
    ("importance", "count"),
    [(False, 13), (True, 17)],
)
def test_real_write_fixed_files_and_validation(
    tmp_path: Path,
    importance: bool,
    count: int,
    result_without_importance,
    result_with_importance,
) -> None:
    result = (
        result_with_importance if importance else result_without_importance
    )
    predictions = result.training_result.predictions
    store = MLExperimentArtifactStore()
    written = store.write(
        result, MLArtifactConfig(tmp_path, f"Demo-{importance}")
    )
    assert written.experiment_dir.is_absolute()
    assert written.manifest.artifact_count == count
    assert written.validation_report.artifact_count == count
    assert all(
        value is True
        for key, value in written.validation_report.as_dict().items()
        if key.endswith("_verified")
    )
    assert store.read_manifest(written.experiment_dir) == written.manifest
    assert store.validate(written.experiment_dir) == written.validation_report
    assert not any(tmp_path.glob(".*.tmp-*"))
    assert (
        (written.experiment_dir / "permutation_importance").exists()
        is importance
    )
    restored = pd.read_parquet(
        written.experiment_dir / "predictions.parquet", engine="pyarrow"
    )
    expected = predictions.copy(deep=True)
    expected.index.name = "dataset_index"
    pdt.assert_frame_equal(restored, expected)
    pdt.assert_frame_equal(result.training_result.predictions, predictions)
    assert "experiment_manifest.json" not in {
        item.relative_path for item in written.manifest.artifacts
    }
    json.dumps(written.as_dict(), allow_nan=False)


@pytest.mark.parametrize("compression", ["zstd", "snappy", "none"])
def test_all_compressions_write_and_read(
    tmp_path: Path, result_without_importance, compression: str
) -> None:
    written = MLExperimentArtifactStore().write(
        result_without_importance,
        MLArtifactConfig(tmp_path, compression, compression),
    )
    assert written.manifest.artifact_count == 13


def test_json_files_are_utf8_strict_and_newline(
    tmp_path: Path, result_without_importance
) -> None:
    directory = MLExperimentArtifactStore().write(
        result_without_importance, MLArtifactConfig(tmp_path, "unicode")
    ).experiment_dir
    for path in directory.rglob("*.json"):
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert raw.endswith(b"\n")
        text = raw.decode("utf-8")
        assert "NaN" not in text and "Infinity" not in text
        assert isinstance(json.loads(text), dict)


def test_plan_contains_public_indices_and_no_model_data(
    tmp_path: Path, result_without_importance
) -> None:
    directory = MLExperimentArtifactStore().write(
        result_without_importance, MLArtifactConfig(tmp_path, "plan")
    ).experiment_dir
    plan = json.loads((directory / "walk_forward_plan.json").read_text("utf-8"))
    assert plan["splits"][0]["train_indices"]
    assert plan["splits"][0]["prediction_indices"]
    text = json.dumps(plan).lower()
    assert "estimator" not in text and "feature_matrix" not in text


def test_environment_excludes_sensitive_machine_values(
    tmp_path: Path, result_without_importance, monkeypatch
) -> None:
    monkeypatch.setattr(artifact_module.platform, "node", lambda: "SECRET-NODE")
    directory = MLExperimentArtifactStore().write(
        result_without_importance, MLArtifactConfig(tmp_path, "environment")
    ).experiment_dir
    environment = json.loads((directory / "environment.json").read_text("utf-8"))
    assert set(environment) == {
        "python_version",
        "python_implementation",
        "platform_system",
        "platform_release",
        "platform_machine",
        "numpy_version",
        "pandas_version",
        "scipy_version",
        "sklearn_version",
        "pyarrow_version",
        "artifact_schema_version",
    }
    assert "SECRET-NODE" not in json.dumps(environment)
    assert "executable" not in environment


def test_existing_target_is_never_overwritten(
    tmp_path: Path, result_without_importance
) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(MLArtifactExistsError):
        MLExperimentArtifactStore().write(
            result_without_importance, MLArtifactConfig(tmp_path, "existing")
        )
    assert marker.read_text("utf-8") == "keep"
    assert not any(tmp_path.glob(".*.tmp-*"))


def test_existing_target_file_and_second_write_fail(
    tmp_path: Path, result_without_importance
) -> None:
    (tmp_path / "file").write_text("keep", encoding="utf-8")
    store = MLExperimentArtifactStore()
    with pytest.raises(MLArtifactExistsError):
        store.write(
            result_without_importance, MLArtifactConfig(tmp_path, "file")
        )
    store.write(
        result_without_importance, MLArtifactConfig(tmp_path, "twice")
    )
    with pytest.raises(MLArtifactExistsError):
        store.write(
            result_without_importance, MLArtifactConfig(tmp_path, "twice")
        )


def test_write_failure_cleans_only_current_staging(
    tmp_path: Path, result_without_importance, monkeypatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()

    def fail(*args: object, **kwargs: object) -> None:
        raise MLArtifactWriteError("experiment_config.json: injected")

    monkeypatch.setattr(artifact_module, "_write_json", fail)
    with pytest.raises(MLArtifactWriteError):
        MLExperimentArtifactStore().write(
            result_without_importance, MLArtifactConfig(tmp_path, "fail")
        )
    assert other.exists()
    assert tmp_path.exists()
    assert not (tmp_path / "fail").exists()
    assert not any(tmp_path.glob(".fail.tmp-*"))


def test_rename_failure_cleans_staging_and_leaves_no_target(
    tmp_path: Path, result_without_importance, monkeypatch
) -> None:
    def fail(source: object, destination: object) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(artifact_module.os, "replace", fail)
    with pytest.raises(MLArtifactWriteError):
        MLExperimentArtifactStore().write(
            result_without_importance, MLArtifactConfig(tmp_path, "rename")
        )
    assert not (tmp_path / "rename").exists()
    assert not any(tmp_path.glob(".rename.tmp-*"))


def test_manifest_is_written_last_and_rename_once(
    tmp_path: Path, result_without_importance, monkeypatch
) -> None:
    writes: list[str] = []
    original_write = artifact_module._write_json
    original_replace = artifact_module.os.replace
    replaces: list[tuple[object, object]] = []

    def tracked_write(
        path: Path, value: object, relative_path: str
    ) -> None:
        writes.append(relative_path)
        original_write(path, value, relative_path)

    def tracked_replace(source: object, destination: object) -> None:
        replaces.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(artifact_module, "_write_json", tracked_write)
    monkeypatch.setattr(artifact_module.os, "replace", tracked_replace)
    MLExperimentArtifactStore().write(
        result_without_importance, MLArtifactConfig(tmp_path, "atomic")
    )
    assert writes[-1] == "experiment_manifest.json"
    assert len(replaces) == 1
    assert ".atomic.tmp-" in str(replaces[0][0])
    assert str(replaces[0][1]).endswith("atomic")


def _write_result(tmp_path: Path, result_without_importance) -> Path:
    return MLExperimentArtifactStore().write(
        result_without_importance, MLArtifactConfig(tmp_path, "corrupt")
    ).experiment_dir


def test_validate_rejects_missing_manifest(tmp_path: Path) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()
    with pytest.raises(MLArtifactValidationError):
        MLExperimentArtifactStore().validate(directory)


def test_validate_rejects_corrupt_manifest(
    tmp_path: Path, result_without_importance
) -> None:
    directory = _write_result(tmp_path, result_without_importance)
    (directory / "experiment_manifest.json").write_text(
        '{"schema_version": NaN}', encoding="utf-8"
    )
    with pytest.raises(MLArtifactValidationError):
        MLExperimentArtifactStore().validate(directory)


def test_validate_rejects_missing_extra_and_forbidden_files(
    tmp_path: Path, result_without_importance
) -> None:
    directory = _write_result(tmp_path, result_without_importance)
    (directory / "extra.pkl").write_bytes(b"model")
    with pytest.raises(MLArtifactValidationError):
        MLExperimentArtifactStore().validate(directory)
    (directory / "extra.pkl").unlink()
    (directory / "dataset_audit.json").unlink()
    with pytest.raises(MLArtifactValidationError):
        MLExperimentArtifactStore().validate(directory)


def test_validate_rejects_checksum_and_size_damage(
    tmp_path: Path, result_without_importance
) -> None:
    directory = _write_result(tmp_path, result_without_importance)
    path = directory / "dataset_audit.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MLArtifactValidationError, match="size|SHA"):
        MLExperimentArtifactStore().validate(directory)


def _refresh_record(directory: Path, relative_path: str) -> None:
    manifest_path = directory / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    path = directory / relative_path
    for record in manifest["artifacts"]:
        if record["relative_path"] == relative_path:
            record["size_bytes"] = path.stat().st_size
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if relative_path.endswith(".parquet"):
                frame = pd.read_parquet(path, engine="pyarrow")
                record["row_count"] = len(frame)
                record["columns"] = list(frame.columns)
                record["dtypes"] = [
                    [name, str(dtype)] for name, dtype in frame.dtypes.items()
                ]
            break
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_cross_file_json_tamper_is_detected(
    tmp_path: Path, result_without_importance
) -> None:
    directory = _write_result(tmp_path, result_without_importance)
    path = directory / "training_audit.json"
    value = json.loads(path.read_text("utf-8"))
    value["n_prediction_rows"] += 1
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_record(directory, "training_audit.json")
    with pytest.raises(MLArtifactIntegrityError):
        MLExperimentArtifactStore().validate(directory)


def test_cross_file_parquet_tamper_is_detected(
    tmp_path: Path, result_without_importance
) -> None:
    directory = _write_result(tmp_path, result_without_importance)
    path = directory / "predictions.parquet"
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.loc[frame.index[0], "target"] = np.inf
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=True)
    _refresh_record(directory, "predictions.parquet")
    with pytest.raises(MLArtifactIntegrityError):
        MLExperimentArtifactStore().validate(directory)


def test_store_does_not_cache_results_or_paths(
    tmp_path: Path, result_without_importance
) -> None:
    store = MLExperimentArtifactStore()
    first = store.write(
        result_without_importance, MLArtifactConfig(tmp_path, "first")
    )
    second = store.write(
        result_without_importance, MLArtifactConfig(tmp_path, "second")
    )
    assert first.experiment_dir != second.experiment_dir
    assert vars(store) == {}
    suffixes = {path.suffix for path in first.experiment_dir.rglob("*") if path.is_file()}
    assert suffixes == {".json", ".parquet"}


def test_parquet_write_failure_is_chained_and_cleans_staging(
    tmp_path: Path, result_without_importance, monkeypatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("injected parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(MLArtifactWriteError) as caught:
        MLExperimentArtifactStore().write(
            result_without_importance, MLArtifactConfig(tmp_path, "parquet-fail")
        )
    assert caught.value.__cause__ is not None
    assert not (tmp_path / "parquet-fail").exists()
    assert not any(tmp_path.glob(".parquet-fail.tmp-*"))


def test_staging_validation_failure_cleans_staging(
    tmp_path: Path, result_without_importance, monkeypatch
) -> None:
    store = MLExperimentArtifactStore()

    def fail(*args: object, **kwargs: object) -> None:
        raise MLArtifactValidationError(
            "experiment_manifest.json: injected validation failure"
        )

    monkeypatch.setattr(store, "_validate", fail)
    with pytest.raises(MLArtifactValidationError):
        store.write(
            result_without_importance, MLArtifactConfig(tmp_path, "validate-fail")
        )
    assert not (tmp_path / "validate-fail").exists()
    assert not any(tmp_path.glob(".validate-fail.tmp-*"))


def test_post_rename_validation_failure_keeps_formal_directory(
    tmp_path: Path, result_without_importance, monkeypatch
) -> None:
    store = MLExperimentArtifactStore()
    original_validate = store.validate
    calls = 0

    def fail_formal(path: str | Path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MLArtifactValidationError(
                "experiment_manifest.json: formal validation failure"
            )
        return original_validate(path)

    monkeypatch.setattr(store, "validate", fail_formal)
    with pytest.raises(MLArtifactValidationError):
        store.write(
            result_without_importance, MLArtifactConfig(tmp_path, "formal-fail")
        )
    formal = tmp_path / "formal-fail"
    assert formal.is_dir()
    assert (formal / "experiment_manifest.json").is_file()
    assert not any(tmp_path.glob(".formal-fail.tmp-*"))
