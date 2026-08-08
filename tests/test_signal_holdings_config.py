"""Tests for canonical V5 Signal and Holdings static configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.holdings_config import (
    HoldingsConfigError,
    HoldingsPipelineConfig,
)
from src.pipeline.signal_config import (
    PredictionSourceConfig,
    SignalConfigError,
    SignalPipelineConfig,
)


def test_prediction_source_defaults_and_ml_semantics() -> None:
    source = PredictionSourceConfig()
    assert source.mode == "ml"
    assert source.artifact_dir is None
    assert source.to_dict() == {"mode": "ml", "artifact_dir": None}
    assert PredictionSourceConfig.from_dict(source.to_dict()) == source


def test_files_source_accepts_explicit_native_artifact_directory_statically() -> None:
    source = PredictionSourceConfig.from_dict(
        {"mode": " FILES ", "artifact_dir": "artifacts/ml/run-1"}
    )
    assert source.mode == "files"
    assert source.artifact_dir == Path("artifacts/ml/run-1")
    assert source.to_dict()["artifact_dir"] == str(Path("artifacts/ml/run-1"))


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "ml", "artifact_dir": "artifact"},
        {"mode": "files"},
        {"mode": "files", "artifact_dir": ""},
        {"mode": "files", "artifact_dir": " artifact "},
        {"mode": "unknown"},
        {"prediction_path": "predictions.parquet"},
        {"manifest_path": "experiment_manifest.json"},
        {"mode": "files", "artifact_dir": "artifact", "extra": True},
        {1: "ml"},
    ],
)
def test_prediction_source_rejects_invalid_or_unknown_fields(value: object) -> None:
    with pytest.raises(SignalConfigError):
        PredictionSourceConfig.from_dict(value)  # type: ignore[arg-type]


def test_prediction_source_does_not_mutate_input_and_roundtrips() -> None:
    raw = {"mode": "files", "artifact_dir": Path("native/ml-artifact")}
    before = dict(raw)
    config = PredictionSourceConfig.from_dict(raw)
    assert raw == before
    assert PredictionSourceConfig.from_dict(config.to_dict()) == config


def test_signal_pipeline_defaults_and_roundtrip() -> None:
    config = SignalPipelineConfig()
    assert config.enabled is False
    assert config.source == PredictionSourceConfig()
    assert config.prediction_column == "prediction"
    assert config.signal_direction == "descending"
    assert config.artifact_subdir == "signal"
    assert SignalPipelineConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("direction", ["descending", "ascending", " ASCENDING "])
def test_signal_pipeline_accepts_enabled_and_direction_values(
    enabled: bool, direction: str
) -> None:
    config = SignalPipelineConfig.from_dict(
        {"enabled": enabled, "signal_direction": direction}
    )
    assert config.enabled is enabled
    assert config.signal_direction == direction.strip().lower()


@pytest.mark.parametrize(
    "value",
    [
        {"prediction_column": ""},
        {"prediction_column": "   "},
        {"prediction_column": 1},
        {"signal_direction": "sideways"},
        {"signal_direction": ""},
        {"enabled": 1},
        {"unknown": True},
        {"artifact_subdir": "../signal"},
        {"artifact_subdir": "nested/signal"},
        {"artifact_subdir": ""},
    ],
)
def test_signal_pipeline_rejects_invalid_values(value: object) -> None:
    with pytest.raises(SignalConfigError):
        SignalPipelineConfig.from_dict(value)  # type: ignore[arg-type]


def test_signal_nested_source_input_is_detached_and_serialized() -> None:
    raw = {
        "enabled": True,
        "source": {"mode": "files", "artifact_dir": "native/artifact"},
        "prediction_column": " prediction ",
        "signal_direction": "ASCENDING",
        "artifact_subdir": "signals_v5",
    }
    before = {**raw, "source": dict(raw["source"])}  # type: ignore[arg-type]
    config = SignalPipelineConfig.from_dict(raw)
    assert raw == before
    assert config.prediction_column == "prediction"
    assert config.source.mode == "files"
    assert SignalPipelineConfig.from_dict(config.to_dict()) == config
    with pytest.raises(FrozenInstanceError):
        config.enabled = False  # type: ignore[misc]


def test_holdings_pipeline_canonical_defaults_and_roundtrip() -> None:
    config = HoldingsPipelineConfig()
    assert config.enabled is False
    assert config.top_n == 20
    assert config.insufficient_universe_policy == "error"
    assert config.weighting == "equal_weight"
    assert config.artifact_subdir == "holdings"
    assert HoldingsPipelineConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("top_n", [1, 10, 20, 1000])
def test_holdings_pipeline_accepts_unbounded_positive_builtin_int(top_n: int) -> None:
    config = HoldingsPipelineConfig.from_dict({"top_n": top_n})
    assert config.top_n == top_n
    assert config.to_dict()["top_n"] == top_n


@pytest.mark.parametrize("top_n", [0, -1, True, False, 1.0, "10", np.int64(10)])
def test_holdings_pipeline_rejects_nonpositive_or_nonstrict_int(top_n: object) -> None:
    with pytest.raises(HoldingsConfigError, match="top_n"):
        HoldingsPipelineConfig(top_n=top_n)  # type: ignore[arg-type]


@pytest.mark.parametrize("policy", ["error", "allow_partial", " ALLOW_PARTIAL "])
def test_holdings_pipeline_accepts_policy_allowlist(policy: str) -> None:
    config = HoldingsPipelineConfig(insufficient_universe_policy=policy)
    assert config.insufficient_universe_policy == policy.strip().lower()


@pytest.mark.parametrize("policy", ["", "partial", "ignore", 1])
def test_holdings_pipeline_rejects_unknown_policy(policy: object) -> None:
    with pytest.raises(HoldingsConfigError):
        HoldingsPipelineConfig(insufficient_universe_policy=policy)  # type: ignore[arg-type]


def test_holdings_pipeline_allows_only_equal_weight() -> None:
    assert HoldingsPipelineConfig(weighting=" EQUAL_WEIGHT ").weighting == "equal_weight"
    for value in ("score_weight", "risk_parity", "", 1):
        with pytest.raises(HoldingsConfigError):
            HoldingsPipelineConfig(weighting=value)  # type: ignore[arg-type]


def test_holdings_pipeline_rejects_unknown_fields_and_unsafe_subdirs() -> None:
    with pytest.raises(HoldingsConfigError):
        HoldingsPipelineConfig.from_dict({"selected": True})
    for value in ("", ".", "..", "nested/holdings", "C:\\holdings"):
        with pytest.raises(HoldingsConfigError):
            HoldingsPipelineConfig(artifact_subdir=value)


def test_holdings_input_is_not_mutated_and_ui_value_has_canonical_semantics() -> None:
    raw = {
        "enabled": True,
        "top_n": 10,
        "insufficient_universe_policy": "allow_partial",
        "weighting": "equal_weight",
        "artifact_subdir": "holdings_v5",
    }
    before = dict(raw)
    config = HoldingsPipelineConfig.from_dict(raw)
    assert raw == before
    assert config.top_n == 10
    assert HoldingsPipelineConfig.from_dict(config.to_dict()).top_n == 10
    with pytest.raises(FrozenInstanceError):
        config.top_n = 20  # type: ignore[misc]
