"""Tests for safe Modeling Panel artifact persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling_panel import (
    MODELING_PANEL_ARTIFACT_FILENAMES,
    MODELING_PANEL_ARTIFACT_SCHEMA_VERSION,
    MODELING_PANEL_ARTIFACT_TYPE,
    ModelingPanelArtifactConfig,
    ModelingPanelArtifactConfigError,
    ModelingPanelArtifactFileRecord,
    ModelingPanelArtifactManifest,
    ModelingPanelArtifactStore,
    ModelingPanelArtifactValidationError,
    ModelingPanelArtifactWriteError,
    ModelingPanelBuilder,
)


def _result():
    factors = pd.DataFrame({
        "trade_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "ts_code": ["A", "B", "A", "B"],
        "factor_a": [1.0, 2.0, 3.0, 4.0],
        "factor_b": [10.0, np.nan, 30.0, 40.0],
    })
    returns = pd.DataFrame({
        "trade_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "ts_code": ["A", "B", "A", "B"],
        "entry_trade_date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
        "exit_trade_date": ["2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04"],
        "entry_price": [10.0, 20.0, 30.0, 40.0],
        "exit_price": [11.0, 18.0, 33.0, 44.0],
        "forward_return": [0.1, -0.1, 0.1, 0.1],
    })
    return ModelingPanelBuilder().build(factors, returns)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_constants_and_config_contract(tmp_path: Path) -> None:
    assert MODELING_PANEL_ARTIFACT_SCHEMA_VERSION == "1.0"
    assert MODELING_PANEL_ARTIFACT_TYPE == "modeling_panel"
    assert MODELING_PANEL_ARTIFACT_FILENAMES == (
        "modeling_panel.parquet", "config.json", "audit.json", "manifest.json"
    )
    config = ModelingPanelArtifactConfig(tmp_path / "artifact", " ZSTD ", False)
    assert config.parquet_compression == "zstd"
    assert ModelingPanelArtifactConfig.from_dict(config.as_dict()) == config
    assert not (tmp_path / "artifact").exists()


@pytest.mark.parametrize("value", ["", ".", ".."])
def test_config_rejects_ambiguous_paths(value: str) -> None:
    with pytest.raises(ModelingPanelArtifactConfigError):
        ModelingPanelArtifactConfig(value)


@pytest.mark.parametrize("value", ["gzip", None, 1])
def test_config_rejects_compression(value: object, tmp_path: Path) -> None:
    with pytest.raises(ModelingPanelArtifactConfigError):
        ModelingPanelArtifactConfig(tmp_path / "a", value)  # type: ignore[arg-type]


def test_file_record_is_strict() -> None:
    record = ModelingPanelArtifactFileRecord("config.json", 1, "0" * 64)
    assert ModelingPanelArtifactFileRecord.from_dict(record.as_dict()) == record
    with pytest.raises(ModelingPanelArtifactValidationError):
        ModelingPanelArtifactFileRecord("../config.json", 1, "0" * 64)
    with pytest.raises(ModelingPanelArtifactValidationError):
        ModelingPanelArtifactFileRecord("config.json", True, "0" * 64)
    with pytest.raises(ModelingPanelArtifactValidationError):
        ModelingPanelArtifactFileRecord("config.json", 1, "A" * 64)


@pytest.mark.parametrize("compression", ["zstd", "snappy"])
def test_write_validate_and_read_manifest(tmp_path: Path, compression: str) -> None:
    result = _result()
    before = result.panel
    target = tmp_path / f"artifact-{compression}"
    written = ModelingPanelArtifactStore().write(
        result, ModelingPanelArtifactConfig(target, compression)
    )
    assert tuple(path.name for path in target.iterdir()) != ()
    assert set(path.name for path in target.iterdir()) == set(MODELING_PANEL_ARTIFACT_FILENAMES)
    assert written.artifact_dir.is_absolute()
    assert written.validation.is_valid
    assert ModelingPanelArtifactStore().validate(target).is_valid
    assert ModelingPanelArtifactStore().read_manifest(target) == written.manifest
    assert result.panel.equals(before)
    persisted = pd.read_parquet(target / "modeling_panel.parquet")
    assert list(persisted.columns) == list(before.columns)
    raw = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert [item["relative_path"] for item in raw["files"]] == list(MODELING_PANEL_ARTIFACT_FILENAMES[:3])
    for item in raw["files"]:
        path = target / item["relative_path"]
        assert item["size_bytes"] == path.stat().st_size
        assert item["sha256"] == _digest(path)


def test_existing_target_is_not_modified(tmp_path: Path) -> None:
    target = tmp_path / "artifact"
    target.mkdir()
    marker = target / "owned.txt"
    marker.write_text("user", encoding="utf-8")
    with pytest.raises(ModelingPanelArtifactWriteError):
        ModelingPanelArtifactStore().write(_result(), ModelingPanelArtifactConfig(target))
    assert marker.read_text(encoding="utf-8") == "user"


def test_validate_reports_checksum_tamper(tmp_path: Path) -> None:
    target = tmp_path / "artifact"
    ModelingPanelArtifactStore().write(_result(), ModelingPanelArtifactConfig(target))
    path = target / "config.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    report = ModelingPanelArtifactStore().validate(target)
    assert not report.is_valid
    assert {item.code for item in report.issues} >= {"file_size_mismatch", "checksum_mismatch"}


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    target = tmp_path / "artifact"
    ModelingPanelArtifactStore().write(_result(), ModelingPanelArtifactConfig(target))
    (target / "manifest.json").write_text('{"artifact_type":"x","artifact_type":"y"}', encoding="utf-8")
    report = ModelingPanelArtifactStore().validate(target)
    assert not report.is_valid
    assert report.issues[0].code == "invalid_manifest_json"


def test_validator_rejects_unexpected_entry(tmp_path: Path) -> None:
    target = tmp_path / "artifact"
    ModelingPanelArtifactStore().write(_result(), ModelingPanelArtifactConfig(target))
    (target / "extra.txt").write_text("x", encoding="utf-8")
    report = ModelingPanelArtifactStore().validate(target)
    assert not report.is_valid
    assert any(item.code == "unexpected_entry" for item in report.issues)
