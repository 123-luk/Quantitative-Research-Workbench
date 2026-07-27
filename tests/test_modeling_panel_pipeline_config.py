"""Tests for V4-E1 Modeling Panel Pipeline configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from src.modeling_panel import ModelingPanelConfig
from src.pipeline import (
    FactorResearchPipelineConfig,
    MLExperimentPipelineConfig,
    ModelingPanelOutputConfig,
    ModelingPanelPipelineConfig,
    ModelingPanelPipelineConfigError,
    ModelingPanelSourceConfig,
    PipelineConfig,
)


def _pipeline_values(modeling_panel: object | None = None) -> dict[str, object]:
    values: dict[str, object] = {
        "backtest_start": "2024-01-01",
        "backtest_end": "2025-03-31",
        "train_years": 10,
        "max_lookback_months": 12,
        "stock_pool": "hs300",
        "benchmark": "000300.SH",
        "strategy_name": "score",
        "selected_factors": ["pe"],
        "rebalance_frequency": "M",
        "top_n": 20,
        "transaction_cost": 0.001,
        "data_root": "data",
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "cache_dir": "data/cache",
        "output_dir": "data/output",
        "parquet_engine": "auto",
        "required_datasets": ["daily"],
    }
    if modeling_panel is not None:
        values["modeling_panel"] = modeling_panel
    return values


def _files_source() -> dict[str, object]:
    return {
        "mode": "files",
        "factor_panel_path": "inputs/factors.parquet",
        "forward_returns_path": "inputs/returns.parquet",
    }


def test_source_defaults_normalization_roundtrip_and_frozen() -> None:
    default = ModelingPanelSourceConfig.from_dict(None)
    assert default == ModelingPanelSourceConfig()
    assert default.mode == "files"
    assert default.factor_panel_path is None
    normalized = ModelingPanelSourceConfig(
        mode=" FACTOR_RESEARCH ",
    )
    assert normalized.mode == "factor_research"
    assert ModelingPanelSourceConfig.from_dict(normalized.as_dict()) == normalized
    json.dumps(normalized.as_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        normalized.mode = "files"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"mode": 1}, "string"),
        ({"mode": "csv"}, "files"),
        ({"mode": "files", "factor_panel_path": "a.parquet"}, "both"),
        ({"mode": "files", "forward_returns_path": "b.parquet"}, "both"),
        (
            {
                "mode": "files",
                "factor_panel_path": "same.parquet",
                "forward_returns_path": Path("same.parquet"),
            },
            "different",
        ),
        (
            {
                "mode": "files",
                "factor_panel_path": "a.csv",
                "forward_returns_path": "b.parquet",
            },
            "parquet",
        ),
        (
            {
                "mode": "factor_research",
                "factor_panel_path": "a.parquet",
                "forward_returns_path": "b.parquet",
            },
            "must not",
        ),
        ({"unknown": True}, "unknown"),
    ],
)
def test_source_rejects_invalid_values(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(ModelingPanelPipelineConfigError, match=message):
        ModelingPanelSourceConfig.from_dict(values)


def test_files_source_is_json_safe_detached_and_does_no_io(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    source = ModelingPanelSourceConfig.from_dict(
        {
            "factor_panel_path": missing / "factors.parquet",
            "forward_returns_path": missing / "returns.parquet",
        }
    )
    payload = source.as_dict()
    payload["mode"] = "factor_research"
    assert source.mode == "files"
    assert source.factor_panel_path == missing / "factors.parquet"
    assert not missing.exists()
    json.dumps(source.as_dict(), allow_nan=False)


def test_output_defaults_normalization_roundtrip_and_frozen() -> None:
    default = ModelingPanelOutputConfig.from_dict(None)
    assert default == ModelingPanelOutputConfig()
    assert default.as_dict() == {
        "save_artifact": True,
        "artifact_subdir": "modeling_panel",
        "parquet_compression": "zstd",
        "verify_after_write": True,
    }
    normalized = ModelingPanelOutputConfig(parquet_compression=" SNAPPY ")
    assert normalized.parquet_compression == "snappy"
    assert ModelingPanelOutputConfig.from_dict(normalized.as_dict()) == normalized
    json.dumps(normalized.as_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        normalized.save_artifact = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"save_artifact": False}, "True"),
        ({"artifact_subdir": ""}, "non-empty"),
        ({"artifact_subdir": "."}, "safe"),
        ({"artifact_subdir": ".."}, "safe"),
        ({"artifact_subdir": "a/b"}, "safe"),
        ({"artifact_subdir": "a\\b"}, "safe"),
        ({"artifact_subdir": " C:\\output "}, "trimmed"),
        ({"parquet_compression": "gzip"}, "zstd"),
        ({"verify_after_write": 1}, "bool"),
        ({"unknown": True}, "unknown"),
    ],
)
def test_output_rejects_invalid_values(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(ModelingPanelPipelineConfigError, match=message):
        ModelingPanelOutputConfig.from_dict(values)


def test_pipeline_defaults_enabled_modes_builder_and_roundtrip() -> None:
    default = ModelingPanelPipelineConfig.from_dict(None)
    assert default.enabled is False
    assert default.source == ModelingPanelSourceConfig()
    assert default.output == ModelingPanelOutputConfig()
    assert default.builder == ModelingPanelConfig()

    files = ModelingPanelPipelineConfig.from_dict(
        {
            "enabled": True,
            "source": _files_source(),
            "builder": {
                "label_column": "target",
                "include_features": ["factor_b", "factor_a"],
            },
            "output": {
                "artifact_subdir": "panel_v1",
                "parquet_compression": "snappy",
            },
        }
    )
    research = ModelingPanelPipelineConfig.from_dict(
        {"enabled": True, "source": {"mode": "factor_research"}}
    )
    assert files.builder.include_features == ("factor_b", "factor_a")
    assert research.source.mode == "factor_research"
    assert ModelingPanelPipelineConfig.from_dict(files.as_dict()) == files
    payload = files.as_dict()
    payload["enabled"] = False
    cast_builder = payload["builder"]
    assert isinstance(cast_builder, dict)
    cast_builder["label_column"] = "changed"
    assert files.enabled is True
    assert files.builder.label_column == "target"
    json.dumps(files.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "values",
    [
        {"enabled": 1},
        {"enabled": True},
        {"enabled": True, "source": {"factor_panel_path": "a.parquet"}},
        {"unknown": True},
        {"builder": {"unknown": True}},
    ],
)
def test_pipeline_rejects_invalid_or_unknown_values(
    values: dict[str, object],
) -> None:
    with pytest.raises(ModelingPanelPipelineConfigError):
        ModelingPanelPipelineConfig.from_dict(values)


def test_pipeline_config_integration_backward_compatibility_and_roundtrip() -> None:
    legacy = PipelineConfig.from_dict(_pipeline_values())
    assert legacy.modeling_panel == ModelingPanelPipelineConfig()
    assert isinstance(legacy.factor_research, FactorResearchPipelineConfig)
    assert isinstance(legacy.ml_experiment, MLExperimentPipelineConfig)

    values = _pipeline_values(
        {
            "enabled": True,
            "source": _files_source(),
            "builder": {"include_features": ["factor_a"]},
        }
    )
    configured = PipelineConfig.from_dict(values)
    assert configured.modeling_panel.enabled is True
    assert configured.modeling_panel.builder.include_features == ("factor_a",)
    serialized = configured.to_dict()
    json.dumps(serialized, allow_nan=False)
    assert PipelineConfig.from_dict(serialized) == configured
    assert serialized["modeling_panel"]["enabled"] is True  # type: ignore[index]


def test_pipeline_config_preserves_strict_unknown_policy() -> None:
    values = _pipeline_values()
    values["unknown_stage"] = {}
    with pytest.raises(ValueError, match="Unknown PipelineConfig"):
        PipelineConfig.from_dict(values)
