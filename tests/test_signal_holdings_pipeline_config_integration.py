"""Top-level PipelineConfig integration tests for V5 Signal and Holdings."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from src.ml import MLExperimentConfig
from src.pipeline import (
    HoldingsPipelineConfig,
    MLExperimentPipelineConfig,
    PipelineConfig,
    SignalPipelineConfig,
)


def _values(**updates: object) -> dict[str, object]:
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
        "required_datasets": [],
    }
    values.update(updates)
    return values


def _ml(*, save_artifacts: bool = True) -> MLExperimentPipelineConfig:
    return MLExperimentPipelineConfig.from_dict({
        "enabled": True,
        "panel_path": "data/panel.parquet",
        "save_artifacts": save_artifacts,
        "experiment_id": "signal-source",
        "experiment": {
            "dataset": {"label_col": "forward_return"},
            "walk_forward": {
                "train_window_periods": 2,
                "validation_periods": 2,
                "window_type": "rolling",
                "retrain_frequency": 3,
                "embargo_periods": 1,
            },
            "training": {"model_name": "ridge", "model_params": {}},
            "evaluation": {"minimum_cross_section_size": 3},
            "permutation_importance": None,
        },
    })


def test_old_direct_config_gets_disabled_v5_defaults() -> None:
    config = PipelineConfig.from_dict(_values())
    assert config.signal == SignalPipelineConfig()
    assert config.holdings == HoldingsPipelineConfig()
    snapshot = config.to_dict()
    assert snapshot["signal"]["enabled"] is False  # type: ignore[index]
    assert snapshot["holdings"]["enabled"] is False  # type: ignore[index]
    json.dumps(snapshot, allow_nan=False)


def test_v5_nested_config_roundtrip_and_input_immutability() -> None:
    raw = _values(
        signal={
            "enabled": True,
            "source": {"mode": "files", "artifact_dir": "native/ml"},
            "signal_direction": "ascending",
        },
        holdings={
            "enabled": True,
            "top_n": 20,
            "insufficient_universe_policy": "allow_partial",
        },
    )
    before = deepcopy(raw)
    config = PipelineConfig.from_dict(raw)
    assert raw == before
    assert config.signal.enabled is True
    assert config.signal.signal_direction == "ascending"
    assert config.holdings.top_n == 20
    assert PipelineConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("direction", ["ascending", "descending"])
def test_signal_files_mode_is_independent_of_ml(direction: str) -> None:
    config = PipelineConfig.from_dict(_values(signal={
        "enabled": True,
        "source": {"mode": "files", "artifact_dir": "native/ml"},
        "signal_direction": direction,
    }))
    assert config.ml_experiment.enabled is False
    assert config.signal.signal_direction == direction


def test_signal_ml_mode_requires_enabled_artifact_saving_ml() -> None:
    with pytest.raises(ValueError, match="ml_experiment.enabled"):
        PipelineConfig.from_dict(_values(signal={"enabled": True}))
    valid = PipelineConfig.from_dict(_values(
        ml_experiment=_ml(save_artifacts=False),
        signal={"enabled": True},
    ))
    assert valid.signal.source.mode == "ml"
    assert valid.ml_experiment.enabled is True


def test_holdings_requires_signal_without_auto_enabling() -> None:
    raw = _values(holdings={"enabled": True, "top_n": 20})
    with pytest.raises(ValueError, match="signal.enabled"):
        PipelineConfig.from_dict(raw)
    assert "signal" not in raw


@pytest.mark.parametrize("top_n", [1, 10, 20])
def test_holdings_top_n_is_nested_canonical_value(top_n: int) -> None:
    config = PipelineConfig.from_dict(_values(
        top_n=top_n,
        signal={
            "enabled": True,
            "source": {"mode": "files", "artifact_dir": "native/ml"},
        },
        holdings={"enabled": True, "top_n": top_n},
    ))
    assert config.holdings.top_n == top_n
    assert config.to_dict()["holdings"]["top_n"] == top_n  # type: ignore[index]


def test_legacy_root_top_n_is_unchanged_when_holdings_disabled() -> None:
    config = PipelineConfig.from_dict(_values(top_n=7))
    assert config.top_n == 7
    assert config.holdings.enabled is False
    assert config.holdings.top_n == 20


def test_enabled_legacy_nested_top_n_same_value_is_allowed() -> None:
    config = PipelineConfig.from_dict(_values(
        top_n=10,
        signal={
            "enabled": True,
            "source": {"mode": "files", "artifact_dir": "native/ml"},
        },
        holdings={"enabled": True, "top_n": 10},
    ))
    assert config.top_n == config.holdings.top_n == 10


def test_enabled_legacy_nested_top_n_conflict_is_rejected() -> None:
    with pytest.raises(ValueError, match="legacy root top_n conflicts"):
        PipelineConfig.from_dict(_values(
            top_n=5,
            signal={
                "enabled": True,
                "source": {"mode": "files", "artifact_dir": "native/ml"},
            },
            holdings={"enabled": True, "top_n": 10},
        ))


def test_unknown_top_level_field_remains_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown PipelineConfig keys"):
        PipelineConfig.from_dict({**_values(), "unknown": True})


def test_grouped_yaml_loads_v5_nested_sections(tmp_path) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        """
data:
  required_datasets: []
pipeline:
  backtest_start: '2024-01-01'
  backtest_end: '2025-01-01'
  top_n: 10
signal:
  enabled: true
  source:
    mode: files
    artifact_dir: native/ml
holdings:
  enabled: true
  top_n: 10
""",
        encoding="utf-8",
    )
    config = PipelineConfig.from_yaml(path)
    assert config.signal.enabled is True
    assert config.holdings.enabled is True
    assert config.holdings.top_n == 10
