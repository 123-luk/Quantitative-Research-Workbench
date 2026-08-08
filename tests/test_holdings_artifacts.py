"""Tests for safe Holdings Artifact persistence and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import src.holdings.artifacts as artifact_module
from src.holdings.artifacts import (
    HOLDINGS_ARTIFACT_FILENAMES,
    HOLDINGS_ARTIFACT_SCHEMA_VERSION,
    HOLDINGS_ARTIFACT_TYPE,
    HoldingsArtifactConfig,
    HoldingsArtifactExistsError,
    HoldingsArtifactStore,
    HoldingsArtifactValidationIssue,
    HoldingsArtifactValidationReport,
    HoldingsArtifactWriteError,
    SignalArtifactProvenance,
)
from src.holdings.builder import HoldingsBuilder
from src.holdings.contracts import HOLDINGS_OUTPUT_COLUMNS
from src.signals import (
    PredictionSourceProvenance,
    SignalArtifactConfig,
    SignalArtifactStore,
    SignalBuilder,
)


def _signal(counts: tuple[int, ...] = (15, 15)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, count in enumerate(counts, start=1):
        for rank in range(1, count + 1):
            rows.append({
                "trade_date": pd.Timestamp(f"2024-01-{date_number:02d}"),
                "ts_code": f"S{rank:03d}",
                "score": float(count - rank),
                "rank": rank,
            })
    frame = pd.DataFrame(rows)
    frame["trade_date"] = frame["trade_date"].astype("datetime64[ns]")
    frame["ts_code"] = frame["ts_code"].astype("string")
    frame["rank"] = frame["rank"].astype(np.int64)
    return frame


def _result(
    counts: tuple[int, ...] = (15, 15),
    top_n: int = 10,
    policy: str = "error",
):
    return HoldingsBuilder().build(
        _signal(counts),
        top_n=top_n,
        insufficient_universe_policy=policy,
        weighting="equal_weight",
    )


def _provenance(tmp_path: Path) -> SignalArtifactProvenance:
    directory = tmp_path / "source-signal"
    directory.mkdir(exist_ok=True)
    signal_path = directory / "signals.parquet"
    signal_path.write_bytes(b"validated-signal-artifact-payload")
    return SignalArtifactProvenance(
        signal_artifact_dir=directory,
        signal_path=signal_path,
        signal_schema_version="1.0",
        signal_sha256=hashlib.sha256(signal_path.read_bytes()).hexdigest(),
    )


def _write(
    tmp_path: Path,
    name: str = "holdings",
    counts: tuple[int, ...] = (15, 15),
    top_n: int = 10,
    policy: str = "error",
):
    result = _result(counts, top_n, policy)
    provenance = _provenance(tmp_path)
    written = HoldingsArtifactStore().write(
        result, provenance, HoldingsArtifactConfig(tmp_path / name)
    )
    return result, provenance, written


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }


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
    path = directory / "holdings.parquet"
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    manifest = _json(directory / "manifest.json")
    manifest["row_count"] = len(frame)
    manifest["column_count"] = len(frame.columns)
    manifest["columns"] = list(frame.columns)
    manifest["pandas_dtypes"] = [
        [str(name), str(dtype)] for name, dtype in frame.dtypes.items()
    ]
    _write_json(directory / "manifest.json", manifest)
    _refresh_record(directory, "holdings.parquet")


def test_constants_and_artifact_config(tmp_path: Path) -> None:
    assert HOLDINGS_ARTIFACT_SCHEMA_VERSION == "1.0"
    assert HOLDINGS_ARTIFACT_TYPE == "holdings"
    assert HOLDINGS_ARTIFACT_FILENAMES == (
        "holdings.parquet", "config.json", "audit.json", "manifest.json"
    )
    config = HoldingsArtifactConfig(tmp_path / "artifact", " SNAPPY ", False)
    assert config.parquet_compression == "snappy"
    assert not config.artifact_dir.exists()


@pytest.mark.parametrize("value", ["", ".", "..", " spaced "])
def test_artifact_config_rejects_ambiguous_path(value: str) -> None:
    with pytest.raises(Exception):
        HoldingsArtifactConfig(value)


@pytest.mark.parametrize("value", ["gzip", "", None, 1])
def test_artifact_config_rejects_bad_compression(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(Exception):
        HoldingsArtifactConfig(tmp_path / "a", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_n,expected_weight", [(1, 1.0), (10, 0.1), (20, 0.05)])
def test_write_validate_and_trace_configurable_top_n(
    tmp_path: Path, top_n: int, expected_weight: float
) -> None:
    counts = (30, 30)
    result = _result(counts, top_n)
    before = result.holdings
    provenance = _provenance(tmp_path)
    provenance_before = provenance.as_dict()
    target = tmp_path / f"holdings-{top_n}"
    written = HoldingsArtifactStore().write(
        result, provenance, HoldingsArtifactConfig(target)
    )
    assert {path.name for path in target.iterdir()} == set(HOLDINGS_ARTIFACT_FILENAMES)
    assert written.artifact_dir == target.resolve()
    assert written.holdings_path == (target / "holdings.parquet").resolve()
    assert written.rows == 2 * top_n
    assert written.schema_version == "1.0"
    assert not hasattr(written, "holdings")
    assert written.validation.is_valid
    assert HoldingsArtifactStore().validate(target).is_valid
    assert HoldingsArtifactStore().read_manifest(target) == written.manifest
    persisted = pd.read_parquet(written.holdings_path, engine="pyarrow")
    pdt.assert_frame_equal(persisted, before)
    assert tuple(persisted.columns) == HOLDINGS_OUTPUT_COLUMNS
    assert np.allclose(persisted["target_weight"], expected_weight, rtol=0, atol=0)
    config = _json(written.config_path)
    assert config == {
        "top_n": top_n,
        "insufficient_universe_policy": "error",
        "weighting": "equal_weight",
    }
    audit = _json(written.audit_path)
    assert audit["requested_top_n"] == top_n
    assert audit["source_signal_provenance"] == provenance_before
    assert [item["selected_count"] for item in audit["per_date_counts"]] == [top_n, top_n]
    manifest = _json(written.manifest_path)
    assert manifest["artifact_schema_version"] == "1.0"
    assert manifest["top_n"] == top_n
    assert manifest["source_signal_provenance"] == provenance_before
    assert [item["relative_path"] for item in manifest["files"]] == list(
        HOLDINGS_ARTIFACT_FILENAMES[:3]
    )
    for record in manifest["files"]:
        path = target / record["relative_path"]
        assert record["size_bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    pdt.assert_frame_equal(result.holdings, before)
    assert provenance.as_dict() == provenance_before
    with pytest.raises(TypeError):
        written.manifest.source_signal_provenance["signal_schema_version"] = "x"  # type: ignore[index]


def test_allow_partial_artifact_records_actual_counts(tmp_path: Path) -> None:
    _, _, written = _write(
        tmp_path, counts=(12, 7), top_n=10, policy="allow_partial"
    )
    config = _json(written.config_path)
    audit = _json(written.audit_path)
    assert config["top_n"] == 10
    assert config["insufficient_universe_policy"] == "allow_partial"
    assert [item["available_count"] for item in audit["per_date_counts"]] == [12, 7]
    assert [item["selected_count"] for item in audit["per_date_counts"]] == [10, 7]
    assert audit["partial_dates"] == ["2024-01-02"]
    frame = pd.read_parquet(written.holdings_path)
    assert np.allclose(frame.tail(7)["target_weight"], 1 / 7)
    assert HoldingsArtifactStore().validate(written.artifact_dir).is_valid


def test_signal_provenance_from_real_signal_write_result(tmp_path: Path) -> None:
    predictions = pd.DataFrame({
        "trade_date": ["2024-01-01", "2024-01-01"],
        "ts_code": ["A", "B"],
        "prediction": [0.2, 0.1],
    })
    signal_result = SignalBuilder().build(
        predictions,
        prediction_column="prediction",
        signal_direction="descending",
    )
    native = tmp_path / "native"
    native.mkdir()
    prediction_path = native / "predictions.parquet"
    prediction_path.write_bytes(b"native")
    native_provenance = PredictionSourceProvenance(
        native, prediction_path, "1.0", "experiment", "ridge",
        hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
    )
    signal_written = SignalArtifactStore().write(
        signal_result,
        native_provenance,
        SignalArtifactConfig(tmp_path / "signal-artifact"),
    )
    provenance = SignalArtifactProvenance.from_signal_write_result(signal_written)
    assert provenance.signal_artifact_dir == signal_written.artifact_dir
    assert provenance.signal_path == signal_written.signal_path
    assert provenance.signal_schema_version == "1.0"
    assert provenance.signal_sha256 == next(
        item.sha256 for item in signal_written.manifest.files
        if item.relative_path == "signals.parquet"
    )


def test_source_signal_need_not_remain_after_publication(tmp_path: Path) -> None:
    _, provenance, written = _write(tmp_path)
    provenance.signal_path.unlink()
    provenance.signal_artifact_dir.rmdir()
    assert HoldingsArtifactStore().validate(written.artifact_dir).is_valid


def test_second_write_preserves_first_artifact(tmp_path: Path) -> None:
    result, provenance, written = _write(tmp_path)
    before = _hashes(written.artifact_dir)
    with pytest.raises(HoldingsArtifactExistsError):
        HoldingsArtifactStore().write(
            result, provenance, HoldingsArtifactConfig(written.artifact_dir)
        )
    assert _hashes(written.artifact_dir) == before
    assert not any(tmp_path.glob(".tmp-holdings-*"))
    assert not any(tmp_path.glob("*backup*"))


def test_existing_user_directory_is_not_modified(tmp_path: Path) -> None:
    target = tmp_path / "owned"
    target.mkdir()
    marker = target / "owned.txt"
    marker.write_text("user", encoding="utf-8")
    with pytest.raises(HoldingsArtifactExistsError):
        HoldingsArtifactStore().write(
            _result(), _provenance(tmp_path), HoldingsArtifactConfig(target)
        )
    assert marker.read_text(encoding="utf-8") == "user"


def test_manifest_is_last_and_publish_occurs_once(
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
    assert ".tmp-holdings-" in str(replaces[0][0])


def test_parquet_failure_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pd.DataFrame, "to_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("parquet failed")),
    )
    with pytest.raises(HoldingsArtifactWriteError) as caught:
        HoldingsArtifactStore().write(
            _result(), _provenance(tmp_path), HoldingsArtifactConfig(tmp_path / "bad")
        )
    assert caught.value.__cause__ is not None
    assert not (tmp_path / "bad").exists()
    assert not any(tmp_path.glob(".tmp-bad-*"))


@pytest.mark.parametrize("filename", ["config.json", "audit.json", "manifest.json"])
def test_json_failure_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    original = artifact_module._write_json

    def fail(path: Path, value: object) -> None:
        if path.name == filename:
            raise HoldingsArtifactWriteError("injected JSON failure")
        original(path, value)  # type: ignore[arg-type]

    monkeypatch.setattr(artifact_module, "_write_json", fail)
    with pytest.raises(HoldingsArtifactWriteError):
        HoldingsArtifactStore().write(
            _result(), _provenance(tmp_path), HoldingsArtifactConfig(tmp_path / "bad")
        )
    assert not (tmp_path / "bad").exists()
    assert not any(tmp_path.glob(".tmp-bad-*"))


def test_pre_publish_validation_failure_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = HoldingsArtifactStore()
    report = HoldingsArtifactValidationReport(
        tmp_path / "placeholder",
        False,
        (HoldingsArtifactValidationIssue("injected", "Injected failure."),),
        None,
    )
    monkeypatch.setattr(store, "validate", lambda path: report)
    with pytest.raises(HoldingsArtifactWriteError, match="pre-publish"):
        store.write(
            _result(), _provenance(tmp_path), HoldingsArtifactConfig(tmp_path / "bad")
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
    with pytest.raises(HoldingsArtifactWriteError):
        HoldingsArtifactStore().write(
            _result(), _provenance(tmp_path), HoldingsArtifactConfig(tmp_path / "bad")
        )
    assert not (tmp_path / "bad").exists()
    assert not any(tmp_path.glob(".tmp-bad-*"))


@pytest.mark.parametrize("filename", HOLDINGS_ARTIFACT_FILENAMES)
def test_validator_rejects_each_missing_file(tmp_path: Path, filename: str) -> None:
    _, _, written = _write(tmp_path)
    (written.artifact_dir / filename).unlink()
    report = HoldingsArtifactStore().validate(written.artifact_dir)
    assert not report.is_valid
    assert any(issue.code == "missing_file" for issue in report.issues)


def test_validator_rejects_extra_file(tmp_path: Path) -> None:
    _, _, written = _write(tmp_path)
    (written.artifact_dir / "extra.pkl").write_bytes(b"bad")
    assert not HoldingsArtifactStore().validate(written.artifact_dir).is_valid


@pytest.mark.parametrize("filename", ["holdings.parquet", "config.json", "audit.json"])
def test_validator_detects_checksum_and_size_tamper(
    tmp_path: Path, filename: str
) -> None:
    _, _, written = _write(tmp_path)
    path = written.artifact_dir / filename
    path.write_bytes(path.read_bytes() + b"tamper")
    codes = {issue.code for issue in HoldingsArtifactStore().validate(written.artifact_dir).issues}
    assert {"file_size_mismatch", "checksum_mismatch"} <= codes


@pytest.mark.parametrize("filename", ["config.json", "audit.json", "manifest.json"])
def test_validator_rejects_malformed_json(tmp_path: Path, filename: str) -> None:
    _, _, written = _write(tmp_path)
    (written.artifact_dir / filename).write_text("{bad", encoding="utf-8")
    assert not HoldingsArtifactStore().validate(written.artifact_dir).is_valid


def test_validator_rejects_wrong_schema(tmp_path: Path) -> None:
    _, _, written = _write(tmp_path)
    manifest = _json(written.manifest_path)
    manifest["artifact_schema_version"] = "9.9"
    _write_json(written.manifest_path, manifest)
    assert not HoldingsArtifactStore().validate(written.artifact_dir).is_valid


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_n", 11),
        ("insufficient_universe_policy", "allow_partial"),
        ("weighting", "score_weight"),
    ],
)
def test_config_manifest_tamper_is_detected(
    tmp_path: Path, field: str, value: object
) -> None:
    _, _, written = _write(tmp_path)
    config = _json(written.config_path)
    config[field] = value
    _write_json(written.config_path, config)
    _refresh_record(written.artifact_dir, "config.json")
    assert not HoldingsArtifactStore().validate(written.artifact_dir).is_valid


def test_provenance_tamper_is_detected(tmp_path: Path) -> None:
    _, _, written = _write(tmp_path)
    audit = _json(written.audit_path)
    audit["source_signal_provenance"]["signal_schema_version"] = "9.9"  # type: ignore[index]
    _write_json(written.audit_path, audit)
    _refresh_record(written.artifact_dir, "audit.json")
    report = HoldingsArtifactStore().validate(written.artifact_dir)
    assert not report.is_valid
    assert any(issue.code == "audit_manifest_mismatch" for issue in report.issues)


def test_available_count_and_input_rows_must_agree(tmp_path: Path) -> None:
    _, _, written = _write(tmp_path)
    audit = _json(written.audit_path)
    audit["per_date_counts"][0]["available_count"] += 1  # type: ignore[index,operator]
    _write_json(written.audit_path, audit)
    _refresh_record(written.artifact_dir, "audit.json")
    report = HoldingsArtifactStore().validate(written.artifact_dir)
    assert not report.is_valid
    assert any(issue.code == "invalid_audit_json" for issue in report.issues)


def test_per_date_date_tamper_returns_invalid_report(tmp_path: Path) -> None:
    _, _, written = _write(tmp_path)
    audit = _json(written.audit_path)
    audit["per_date_counts"][1]["trade_date"] = "2024-01-03"  # type: ignore[index]
    audit["max_trade_date"] = "2024-01-03"
    _write_json(written.audit_path, audit)
    _refresh_record(written.artifact_dir, "audit.json")
    report = HoldingsArtifactStore().validate(written.artifact_dir)
    assert not report.is_valid
    assert any(issue.code == "audit_parquet_mismatch" for issue in report.issues)

@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_columns", "zero_weight", "negative_weight", "nan_weight",
        "weight_sum", "too_many", "skipped_rank", "wrong_order",
        "duplicate_key", "nan_score",
    ],
)
def test_validator_rejects_tampered_holdings_semantics(
    tmp_path: Path, mutation: str
) -> None:
    _, _, written = _write(tmp_path)
    frame = pd.read_parquet(written.holdings_path)
    if mutation == "wrong_columns":
        frame = frame[["ts_code", "trade_date", "target_weight", "score", "rank"]]
    elif mutation == "zero_weight":
        frame.loc[0, "target_weight"] = 0.0
    elif mutation == "negative_weight":
        frame.loc[0, "target_weight"] = -0.1
    elif mutation == "nan_weight":
        frame.loc[0, "target_weight"] = np.nan
    elif mutation == "weight_sum":
        frame.loc[0, "target_weight"] = 0.2
    elif mutation == "too_many":
        extra = frame.iloc[[0]].copy()
        extra["ts_code"] = "EXTRA"
        extra["rank"] = 11
        frame = pd.concat([frame.iloc[:10], extra, frame.iloc[10:]], ignore_index=True)
    elif mutation == "skipped_rank":
        frame.loc[1, "rank"] = 9
    elif mutation == "wrong_order":
        frame = frame.iloc[::-1].reset_index(drop=True)
    elif mutation == "duplicate_key":
        frame.loc[1, ["trade_date", "ts_code"]] = frame.loc[0, ["trade_date", "ts_code"]]
    elif mutation == "nan_score":
        frame.loc[0, "score"] = np.nan
    _replace_parquet(written.artifact_dir, frame)
    assert not HoldingsArtifactStore().validate(written.artifact_dir).is_valid


def test_store_rejects_tampered_build_result(tmp_path: Path) -> None:
    result = _result()
    changed = result.holdings
    changed.loc[0, "target_weight"] = -1.0
    result._holdings = changed  # type: ignore[attr-defined]
    with pytest.raises(HoldingsArtifactWriteError, match="not canonical"):
        HoldingsArtifactStore().write(
            result, _provenance(tmp_path), HoldingsArtifactConfig(tmp_path / "bad")
        )
    assert not (tmp_path / "bad").exists()
