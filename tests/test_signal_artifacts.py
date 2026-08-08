"""Tests for safe Signal Artifact persistence and independent validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import src.signals.artifacts as artifact_module
from src.signals import (
    SIGNAL_ARTIFACT_FILENAMES,
    SIGNAL_ARTIFACT_SCHEMA_VERSION,
    SIGNAL_ARTIFACT_TYPE,
    SIGNAL_OUTPUT_COLUMNS,
    PredictionSourceProvenance,
    SignalArtifactConfig,
    SignalArtifactExistsError,
    SignalArtifactStore,
    SignalArtifactValidationIssue,
    SignalArtifactValidationReport,
    SignalArtifactWriteError,
    SignalBuilder,
)


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": [
            "2024-01-01", "2024-01-01", "2024-01-01",
            "2024-01-02", "2024-01-02",
        ],
        "ts_code": ["B", "A", "C", "B", "A"],
        "prediction": [0.4, 0.4, -0.2, 0.1, 0.9],
        "target": [1.0, 2.0, 3.0, 4.0, 5.0],
        "fold_id": [0, 0, 0, 1, 1],
    })


def _result(direction: str = "descending"):
    return SignalBuilder().build(
        _prediction_frame(),
        prediction_column="prediction",
        signal_direction=direction,
    )


def _provenance(tmp_path: Path) -> PredictionSourceProvenance:
    source = tmp_path / "native-ml"
    source.mkdir(exist_ok=True)
    prediction = source / "predictions.parquet"
    prediction.write_bytes(b"validated-native-predictions")
    return PredictionSourceProvenance(
        artifact_dir=source,
        prediction_path=prediction,
        artifact_schema_version="1.0",
        experiment_id="experiment-001",
        model_name="ridge",
        prediction_sha256=hashlib.sha256(prediction.read_bytes()).hexdigest(),
    )


def _write(tmp_path: Path, name: str = "signal", direction: str = "descending"):
    result = _result(direction)
    provenance = _provenance(tmp_path)
    written = SignalArtifactStore().write(
        result, provenance, SignalArtifactConfig(tmp_path / name)
    )
    return result, provenance, written


def _hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _refresh_record(directory: Path, filename: str) -> None:
    manifest_path = directory / "manifest.json"
    manifest = _json(manifest_path)
    path = directory / filename
    for record in manifest["files"]:  # type: ignore[index]
        if record["relative_path"] == filename:
            record["size_bytes"] = path.stat().st_size
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            break
    _write_json(manifest_path, manifest)


def _replace_parquet(directory: Path, frame: pd.DataFrame) -> None:
    path = directory / "signals.parquet"
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    manifest = _json(directory / "manifest.json")
    manifest["row_count"] = len(frame)
    manifest["column_count"] = len(frame.columns)
    manifest["columns"] = list(frame.columns)
    manifest["pandas_dtypes"] = [
        [str(name), str(dtype)] for name, dtype in frame.dtypes.items()
    ]
    _write_json(directory / "manifest.json", manifest)
    _refresh_record(directory, "signals.parquet")


def test_public_constants_and_config_contract(tmp_path: Path) -> None:
    assert SIGNAL_ARTIFACT_SCHEMA_VERSION == "1.0"
    assert SIGNAL_ARTIFACT_TYPE == "signal"
    assert SIGNAL_ARTIFACT_FILENAMES == (
        "signals.parquet", "config.json", "audit.json", "manifest.json"
    )
    config = SignalArtifactConfig(tmp_path / "artifact", " ZSTD ", False)
    assert config.parquet_compression == "zstd"
    assert not config.artifact_dir.exists()


@pytest.mark.parametrize("path", ["", ".", "..", " spaced "])
def test_config_rejects_ambiguous_paths(path: str) -> None:
    with pytest.raises(Exception):
        SignalArtifactConfig(path)


@pytest.mark.parametrize("compression", ["gzip", "", None, 1])
def test_config_rejects_bad_compression(
    compression: object, tmp_path: Path
) -> None:
    with pytest.raises(Exception):
        SignalArtifactConfig(tmp_path / "a", compression)  # type: ignore[arg-type]


@pytest.mark.parametrize("compression", ["zstd", "snappy"])
def test_write_validate_layout_and_metadata(
    tmp_path: Path, compression: str
) -> None:
    result = _result()
    before = result.signals
    provenance = _provenance(tmp_path)
    provenance_before = provenance.as_dict()
    target = tmp_path / f"signal-{compression}"
    written = SignalArtifactStore().write(
        result, provenance, SignalArtifactConfig(target, compression)
    )
    assert {path.name for path in target.iterdir()} == set(SIGNAL_ARTIFACT_FILENAMES)
    assert written.artifact_dir == target.resolve()
    assert written.signal_path == (target / "signals.parquet").resolve()
    assert written.rows == len(before)
    assert written.schema_version == "1.0"
    assert not hasattr(written, "signals")
    assert written.validation.is_valid
    assert SignalArtifactStore().validate(target).is_valid
    assert SignalArtifactStore().read_manifest(target) == written.manifest
    with pytest.raises(TypeError):
        written.manifest.source_provenance["model_name"] = "changed"  # type: ignore[index]
    persisted = pd.read_parquet(target / "signals.parquet", engine="pyarrow")
    pdt.assert_frame_equal(persisted, before)
    assert tuple(persisted.columns) == SIGNAL_OUTPUT_COLUMNS
    assert persisted.equals(
        persisted.sort_values(
            ["trade_date", "rank", "ts_code"], kind="mergesort"
        ).reset_index(drop=True)
    )
    config = _json(target / "config.json")
    assert config == {
        "prediction_column": "prediction",
        "signal_direction": "descending",
    }
    assert not {"top_n", "weighting", "selected", "target_weight"} & set(config)
    audit = _json(target / "audit.json")
    assert audit["source_provenance"] == provenance_before
    assert audit["duplicate_key_count"] == 0
    assert audit["score_finite"] is True
    assert audit["rank_integrity"] is True
    manifest = _json(target / "manifest.json")
    assert manifest["artifact_schema_version"] == "1.0"
    assert [item["relative_path"] for item in manifest["files"]] == list(
        SIGNAL_ARTIFACT_FILENAMES[:3]
    )
    for record in manifest["files"]:
        path = target / record["relative_path"]
        assert record["size_bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    pdt.assert_frame_equal(result.signals, before)
    assert provenance.as_dict() == provenance_before


def test_business_and_audit_snapshots_are_detached(tmp_path: Path) -> None:
    result, provenance, written = _write(tmp_path)
    config = _json(written.config_path)
    audit = _json(written.audit_path)
    config["prediction_column"] = "changed"
    audit["source_provenance"]["model_name"] = "changed"  # type: ignore[index]
    assert _json(written.config_path)["prediction_column"] == result.audit.prediction_column
    assert _json(written.audit_path)["source_provenance"] == provenance.as_dict()


def test_same_input_has_deterministic_semantic_metadata(tmp_path: Path) -> None:
    _, _, first = _write(tmp_path, "first")
    _, _, second = _write(tmp_path, "second")
    assert first.signal_path.read_bytes() == second.signal_path.read_bytes()
    assert first.config_path.read_bytes() == second.config_path.read_bytes()
    assert first.audit_path.read_bytes() == second.audit_path.read_bytes()
    left = _json(first.manifest_path)
    right = _json(second.manifest_path)
    left.pop("created_at_utc")
    right.pop("created_at_utc")
    assert left == right


def test_second_write_rejects_and_preserves_first_artifact(tmp_path: Path) -> None:
    result, provenance, written = _write(tmp_path)
    before = _hashes(written.artifact_dir)
    with pytest.raises(SignalArtifactExistsError):
        SignalArtifactStore().write(
            result, provenance, SignalArtifactConfig(written.artifact_dir)
        )
    assert _hashes(written.artifact_dir) == before
    assert not any(tmp_path.glob(".tmp-signal-*"))
    assert not any(tmp_path.glob("*backup*"))


def test_existing_user_directory_is_not_modified(tmp_path: Path) -> None:
    target = tmp_path / "owned"
    target.mkdir()
    marker = target / "owned.txt"
    marker.write_text("user", encoding="utf-8")
    with pytest.raises(SignalArtifactExistsError):
        SignalArtifactStore().write(
            _result(), _provenance(tmp_path), SignalArtifactConfig(target)
        )
    assert marker.read_text(encoding="utf-8") == "user"


def test_manifest_written_last_and_publish_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes: list[str] = []
    replaces: list[tuple[object, object]] = []
    original_write = artifact_module._write_json
    original_replace = artifact_module.os.replace

    def tracked_write(path: Path, value: object) -> None:
        writes.append(path.name)
        original_write(path, value)  # type: ignore[arg-type]

    def tracked_replace(source: object, target: object) -> None:
        replaces.append((source, target))
        original_replace(source, target)

    monkeypatch.setattr(artifact_module, "_write_json", tracked_write)
    monkeypatch.setattr(artifact_module.os, "replace", tracked_replace)
    _write(tmp_path)
    assert writes == ["config.json", "audit.json", "manifest.json"]
    assert len(replaces) == 1
    assert ".tmp-signal-" in str(replaces[0][0])
    assert Path(replaces[0][1]).name == "signal"


def test_parquet_failure_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("injected parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(SignalArtifactWriteError) as caught:
        SignalArtifactStore().write(
            _result(), _provenance(tmp_path), SignalArtifactConfig(tmp_path / "bad")
        )
    assert caught.value.__cause__ is not None
    assert not (tmp_path / "bad").exists()
    assert not any(tmp_path.glob(".tmp-bad-*"))


@pytest.mark.parametrize("failure_name", ["config.json", "audit.json", "manifest.json"])
def test_json_failure_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_name: str
) -> None:
    original = artifact_module._write_json

    def fail(path: Path, value: object) -> None:
        if path.name == failure_name:
            raise SignalArtifactWriteError("injected JSON failure")
        original(path, value)  # type: ignore[arg-type]

    monkeypatch.setattr(artifact_module, "_write_json", fail)
    with pytest.raises(SignalArtifactWriteError):
        SignalArtifactStore().write(
            _result(), _provenance(tmp_path), SignalArtifactConfig(tmp_path / "bad")
        )
    assert not (tmp_path / "bad").exists()
    assert not any(tmp_path.glob(".tmp-bad-*"))


def test_pre_publish_validation_failure_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SignalArtifactStore()
    report = SignalArtifactValidationReport(
        tmp_path / "placeholder",
        False,
        (SignalArtifactValidationIssue("injected", "Injected failure."),),
        None,
    )
    monkeypatch.setattr(store, "validate", lambda path: report)
    with pytest.raises(SignalArtifactWriteError, match="pre-publish"):
        store.write(
            _result(), _provenance(tmp_path), SignalArtifactConfig(tmp_path / "bad")
        )
    assert not (tmp_path / "bad").exists()
    assert not any(tmp_path.glob(".tmp-bad-*"))


def test_rename_failure_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        artifact_module.os, "replace",
        lambda source, target: (_ for _ in ()).throw(OSError("rename failed")),
    )
    with pytest.raises(SignalArtifactWriteError):
        SignalArtifactStore().write(
            _result(), _provenance(tmp_path), SignalArtifactConfig(tmp_path / "bad")
        )
    assert not (tmp_path / "bad").exists()
    assert not any(tmp_path.glob(".tmp-bad-*"))


@pytest.mark.parametrize("filename", SIGNAL_ARTIFACT_FILENAMES)
def test_validator_rejects_each_missing_file(tmp_path: Path, filename: str) -> None:
    _, _, written = _write(tmp_path)
    (written.artifact_dir / filename).unlink()
    report = SignalArtifactStore().validate(written.artifact_dir)
    assert not report.is_valid
    assert any(issue.code == "missing_file" for issue in report.issues)


def test_validator_rejects_extra_entry(tmp_path: Path) -> None:
    _, _, written = _write(tmp_path)
    (written.artifact_dir / "extra.pkl").write_bytes(b"forbidden")
    report = SignalArtifactStore().validate(written.artifact_dir)
    assert not report.is_valid
    assert any(issue.code == "unexpected_entry" for issue in report.issues)


@pytest.mark.parametrize("filename", ["signals.parquet", "config.json", "audit.json"])
def test_validator_detects_payload_checksum_and_size_tamper(
    tmp_path: Path, filename: str
) -> None:
    _, _, written = _write(tmp_path)
    path = written.artifact_dir / filename
    path.write_bytes(path.read_bytes() + b"tamper")
    codes = {issue.code for issue in SignalArtifactStore().validate(written.artifact_dir).issues}
    assert {"file_size_mismatch", "checksum_mismatch"} <= codes


@pytest.mark.parametrize("filename", ["config.json", "audit.json", "manifest.json"])
def test_validator_rejects_malformed_json(tmp_path: Path, filename: str) -> None:
    _, _, written = _write(tmp_path)
    (written.artifact_dir / filename).write_text("{bad", encoding="utf-8")
    report = SignalArtifactStore().validate(written.artifact_dir)
    assert not report.is_valid
    expected = {
        "config.json": "invalid_config_json",
        "audit.json": "invalid_audit_json",
        "manifest.json": "invalid_manifest_json",
    }[filename]
    assert any(issue.code == expected for issue in report.issues)


def test_validator_rejects_wrong_schema_and_file_record(tmp_path: Path) -> None:
    _, _, written = _write(tmp_path)
    manifest = _json(written.manifest_path)
    manifest["artifact_schema_version"] = "9.9"
    _write_json(written.manifest_path, manifest)
    assert not SignalArtifactStore().validate(written.artifact_dir).is_valid


@pytest.mark.parametrize(
    "field,value",
    [
        ("signal_direction", "ascending"),
        ("prediction_column", "other_prediction"),
    ],
)
def test_cross_file_config_tamper_is_detected(
    tmp_path: Path, field: str, value: str
) -> None:
    _, _, written = _write(tmp_path)
    config = _json(written.config_path)
    config[field] = value
    _write_json(written.config_path, config)
    _refresh_record(written.artifact_dir, "config.json")
    report = SignalArtifactStore().validate(written.artifact_dir)
    assert not report.is_valid
    assert any(issue.code == "config_manifest_mismatch" for issue in report.issues)


def test_cross_file_provenance_tamper_is_detected(tmp_path: Path) -> None:
    _, _, written = _write(tmp_path)
    audit = _json(written.audit_path)
    audit["source_provenance"]["model_name"] = "lasso"  # type: ignore[index]
    _write_json(written.audit_path, audit)
    _refresh_record(written.artifact_dir, "audit.json")
    report = SignalArtifactStore().validate(written.artifact_dir)
    assert not report.is_valid
    assert any(issue.code == "audit_manifest_mismatch" for issue in report.issues)


def test_cross_file_row_count_tamper_is_detected(tmp_path: Path) -> None:
    _, _, written = _write(tmp_path)
    audit = _json(written.audit_path)
    audit["output_rows"] = audit["output_rows"] + 1  # type: ignore[operator]
    audit["input_rows"] = audit["output_rows"]
    _write_json(written.audit_path, audit)
    _refresh_record(written.artifact_dir, "audit.json")
    report = SignalArtifactStore().validate(written.artifact_dir)
    assert not report.is_valid
    assert any(issue.code == "audit_manifest_mismatch" for issue in report.issues)


@pytest.mark.parametrize(
    "mutation",
    [
        "reordered_columns",
        "duplicate_key",
        "nan_score",
        "inf_score",
        "zero_rank",
        "duplicate_rank",
        "skipped_rank",
        "wrong_order",
    ],
)
def test_validator_rejects_invalid_signal_semantics(
    tmp_path: Path, mutation: str
) -> None:
    _, _, written = _write(tmp_path)
    frame = pd.read_parquet(written.signal_path, engine="pyarrow")
    if mutation == "reordered_columns":
        frame = frame[["ts_code", "trade_date", "score", "rank"]]
    elif mutation == "duplicate_key":
        frame.loc[1, ["trade_date", "ts_code"]] = frame.loc[0, ["trade_date", "ts_code"]]
    elif mutation == "nan_score":
        frame.loc[0, "score"] = np.nan
    elif mutation == "inf_score":
        frame.loc[0, "score"] = np.inf
    elif mutation == "zero_rank":
        frame.loc[0, "rank"] = 0
    elif mutation == "duplicate_rank":
        frame.loc[1, "rank"] = frame.loc[0, "rank"]
    elif mutation == "skipped_rank":
        frame.loc[1, "rank"] = 9
    elif mutation == "wrong_order":
        frame = frame.iloc[::-1].reset_index(drop=True)
    _replace_parquet(written.artifact_dir, frame)
    report = SignalArtifactStore().validate(written.artifact_dir)
    assert not report.is_valid


@pytest.mark.parametrize(
    "forbidden",
    [
        "target", "y_true", "fold_id", "entry_trade_date", "exit_trade_date",
        "top_n", "selected", "target_weight", "feature_a",
    ],
)
def test_store_and_validator_reject_extra_or_forbidden_columns(
    tmp_path: Path, forbidden: str
) -> None:
    _, _, written = _write(tmp_path)
    frame = pd.read_parquet(written.signal_path, engine="pyarrow")
    frame[forbidden] = 1
    _replace_parquet(written.artifact_dir, frame)
    report = SignalArtifactStore().validate(written.artifact_dir)
    assert not report.is_valid


def test_validator_does_not_require_source_artifact_to_remain(tmp_path: Path) -> None:
    _, provenance, written = _write(tmp_path)
    provenance.prediction_path.unlink()
    provenance.artifact_dir.rmdir()
    assert SignalArtifactStore().validate(written.artifact_dir).is_valid


def test_store_rejects_noncanonical_build_result_payload(tmp_path: Path) -> None:
    result = _result()
    result._signals = result.signals.iloc[::-1].reset_index(drop=True)  # type: ignore[attr-defined]
    with pytest.raises(SignalArtifactWriteError, match="not canonical"):
        SignalArtifactStore().write(
            result, _provenance(tmp_path), SignalArtifactConfig(tmp_path / "bad")
        )
    assert not (tmp_path / "bad").exists()
