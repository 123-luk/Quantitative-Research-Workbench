"""Tests for optional ML Pipeline configuration and PipelineConfig wiring."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from src.ml import MLExperimentConfig
from src.pipeline import (
    FactorResearchPipelineConfig,
    MLExperimentPipelineConfig,
    MLPipelineConfigError,
    ModelingPanelPipelineConfig,
    PipelineConfig,
)


def _experiment_mapping(
    model_name: str = "ridge",
) -> dict[str, object]:
    return {
        "dataset": {"label_col": "forward_return"},
        "walk_forward": {
            "train_window_periods": 2,
            "validation_periods": 2,
            "window_type": "rolling",
            "retrain_frequency": 3,
            "embargo_periods": 1,
        },
        "training": {
            "model_name": model_name,
            "model_params": {"alpha": 2.0}
            if model_name == "ridge"
            else {},
        },
        "evaluation": {"minimum_cross_section_size": 3},
        "permutation_importance": None,
    }


def _pipeline_values(
    ml: object = None,
    *,
    factor_research: object | None = None,
    modeling_panel: object | None = None,
) -> dict[str, object]:
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
    if ml is not None:
        values["ml_experiment"] = ml
    if factor_research is not None:
        values["factor_research"] = factor_research
    if modeling_panel is not None:
        values["modeling_panel"] = modeling_panel
    return values


def _enabled(**overrides: object) -> MLExperimentPipelineConfig:
    values: dict[str, object] = {
        "enabled": True,
        "panel_path": "data/panel.parquet",
        "experiment": _experiment_mapping(),
    }
    values.update(overrides)
    return MLExperimentPipelineConfig.from_dict(values)


def test_defaults_and_none_are_disabled_json_safe_and_frozen() -> None:
    direct = MLExperimentPipelineConfig()
    parsed = MLExperimentPipelineConfig.from_dict(None)
    assert direct == parsed
    assert direct.enabled is False
    assert direct.panel_path is None
    assert direct.save_artifacts is False
    assert direct.artifact_root == "ml_artifacts"
    assert direct.experiment is None
    json.dumps(direct.to_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        direct.enabled = True  # type: ignore[misc]


def test_disabled_does_not_require_panel_experiment_or_id() -> None:
    config = MLExperimentPipelineConfig(
        experiment_id=" retained-id ",
    )
    assert config.experiment_id == "retained-id"


@pytest.mark.parametrize(
    ("values", "message"),
    [

        (
            {"enabled": True, "panel_path": "panel.parquet"},
            "experiment",
        ),
        ({"save_artifacts": True}, "enabled"),
        (
            {
                "enabled": True,
                "panel_path": "panel.parquet",
                "experiment": _experiment_mapping(),
                "save_artifacts": True,
            },
            "experiment_id",
        ),
    ],
)
def test_enabled_and_artifact_dependencies(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(MLPipelineConfigError, match=message):
        MLExperimentPipelineConfig.from_dict(values)


def test_enabled_without_panel_is_valid_at_ml_stage_level() -> None:
    config = MLExperimentPipelineConfig.from_dict(
        {"enabled": True, "experiment": _experiment_mapping()}
    )
    assert config.enabled is True
    assert config.panel_path is None
    assert MLExperimentPipelineConfig.from_dict(config.to_dict()) == config


def test_panel_path_requires_parquet_suffix_without_filesystem_io(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing" / "panel.csv"
    with pytest.raises(MLPipelineConfigError, match="parquet"):
        _enabled(panel_path=str(missing))
    assert not missing.parent.exists()

def test_text_fields_strip_and_nested_artifact_root_normalizes() -> None:
    config = _enabled(
        panel_path=" data/panel.parquet ",
        artifact_root=r" nested\ml ",
        experiment_id=" Demo-ID ",
    )
    assert config.panel_path == "data/panel.parquet"
    assert config.artifact_root == "nested/ml"
    assert config.experiment_id == "Demo-ID"


@pytest.mark.parametrize("field_name", ["panel_path", "experiment_id"])
def test_optional_text_rejects_empty_values(field_name: str) -> None:
    values: dict[str, object] = {field_name: " "}
    with pytest.raises(MLPipelineConfigError, match=field_name):
        MLExperimentPipelineConfig.from_dict(values)


@pytest.mark.parametrize(
    "artifact_root",
    [
        "",
        " ",
        ".",
        "..",
        "../escape",
        "nested/../escape",
        r"C:\absolute",
        "C:/absolute",
        "/absolute",
    ],
)
def test_artifact_root_rejects_unsafe_paths(artifact_root: str) -> None:
    with pytest.raises(MLPipelineConfigError, match="artifact_root"):
        MLExperimentPipelineConfig(artifact_root=artifact_root)


@pytest.mark.parametrize("compression", ["zstd", "snappy", "none"])
def test_compressions_are_supported(compression: str) -> None:
    config = MLExperimentPipelineConfig(
        parquet_compression=f" {compression.upper()} "
    )
    assert config.parquet_compression == compression


def test_invalid_compression_and_boolean_types_raise() -> None:
    with pytest.raises(MLPipelineConfigError, match="parquet_compression"):
        MLExperimentPipelineConfig(parquet_compression="gzip")
    with pytest.raises(MLPipelineConfigError, match="enabled"):
        MLExperimentPipelineConfig(enabled=1)  # type: ignore[arg-type]
    with pytest.raises(MLPipelineConfigError, match="save_artifacts"):
        MLExperimentPipelineConfig(save_artifacts=1)  # type: ignore[arg-type]


def test_experiment_mapping_and_object_are_supported() -> None:
    mapping_config = _enabled()
    assert isinstance(mapping_config.experiment, MLExperimentConfig)
    assert mapping_config.experiment.training_config.model_name == "ridge"
    object_config = MLExperimentPipelineConfig(
        enabled=True,
        panel_path="panel.parquet",
        experiment=mapping_config.experiment,
    )
    assert object_config.experiment is mapping_config.experiment


def test_experiment_invalid_type_unknown_field_and_non_mapping_raise() -> None:
    with pytest.raises(MLPipelineConfigError, match="experiment"):
        MLExperimentPipelineConfig(experiment=[])  # type: ignore[arg-type]
    with pytest.raises(MLPipelineConfigError, match="unknown"):
        MLExperimentPipelineConfig.from_dict({"unknown": True})
    with pytest.raises(MLPipelineConfigError, match="Mapping"):
        MLExperimentPipelineConfig.from_dict([])  # type: ignore[arg-type]


def test_config_input_and_output_are_defensive() -> None:
    source = {
        "enabled": True,
        "panel_path": "panel.parquet",
        "experiment": _experiment_mapping(),
    }
    config = MLExperimentPipelineConfig.from_dict(source)
    source["experiment"]["training"]["model_params"]["alpha"] = 99.0  # type: ignore[index]
    returned = config.to_dict()
    returned["experiment"]["training"]["model_params"]["alpha"] = 88.0  # type: ignore[index]
    assert config.experiment is not None
    assert config.experiment.training_config.model_params["alpha"] == 2.0


def test_config_construction_has_no_filesystem_side_effect(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing" / "panel.parquet"
    config = _enabled(panel_path=str(missing))
    assert config.panel_path == str(missing)
    assert list(tmp_path.iterdir()) == []


def test_pipeline_default_contains_disabled_ml_and_dates_unchanged() -> None:
    config = PipelineConfig(**_pipeline_values())
    assert config.ml_experiment == MLExperimentPipelineConfig()
    assert config.required_start_date == "2013-01-01"
    assert config.required_end_date == "2025-03-31"


def test_pipeline_from_dict_accepts_mapping_object_none_and_roundtrips() -> None:
    mapping = _enabled().to_dict()
    from_mapping = PipelineConfig.from_dict(_pipeline_values(mapping))
    from_object = PipelineConfig.from_dict(
        _pipeline_values(from_mapping.ml_experiment)
    )
    from_none = PipelineConfig.from_dict(
        {**_pipeline_values(), "ml_experiment": None}
    )
    assert from_mapping.ml_experiment.enabled is True
    assert from_object.ml_experiment == from_mapping.ml_experiment
    assert from_none.ml_experiment.enabled is False
    assert PipelineConfig.from_dict(from_mapping.to_dict()) == from_mapping
    assert "ml_experiment" in from_mapping.to_dict()


def test_pipeline_factor_research_and_ml_are_independent() -> None:
    config = PipelineConfig.from_dict(
        _pipeline_values(
            _enabled(),
            factor_research=FactorResearchPipelineConfig(),
        )
    )
    assert config.factor_research.enabled is False
    assert config.ml_experiment.enabled is True


def _modeling_files() -> ModelingPanelPipelineConfig:
    return ModelingPanelPipelineConfig.from_dict(
        {
            "enabled": True,
            "source": {
                "mode": "files",
                "factor_panel_path": "factors.parquet",
                "forward_returns_path": "returns.parquet",
            },
        }
    )


def _modeling_research() -> ModelingPanelPipelineConfig:
    return ModelingPanelPipelineConfig.from_dict(
        {"enabled": True, "source": {"mode": "factor_research"}}
    )


def _enabled_research() -> FactorResearchPipelineConfig:
    from src.factors.research_pipeline import FactorResearchConfig

    return FactorResearchPipelineConfig(
        enabled=True,
        factor_input_path="factor.parquet",
        score_panel_path="score.parquet",
        price_panel_path="price.parquet",
        research=FactorResearchConfig(
            factor_names=("factor_a",),
            composition_method="equal",
        ),
    )


def test_pipeline_ml_panel_source_rules() -> None:
    direct = PipelineConfig.from_dict(_pipeline_values(_enabled()))
    generated = PipelineConfig.from_dict(
        _pipeline_values(
            _enabled(panel_path=None),
            modeling_panel=_modeling_files(),
        )
    )
    assert direct.ml_experiment.panel_path == "data/panel.parquet"
    assert generated.ml_experiment.panel_path is None
    with pytest.raises(ValueError, match="exactly one"):
        PipelineConfig.from_dict(_pipeline_values(_enabled(panel_path=None)))
    with pytest.raises(ValueError, match="conflict"):
        PipelineConfig.from_dict(
            _pipeline_values(_enabled(), modeling_panel=_modeling_files())
        )


def test_pipeline_modeling_source_dependency_and_independence() -> None:
    with pytest.raises(ValueError, match="factor_research.enabled"):
        PipelineConfig.from_dict(
            _pipeline_values(modeling_panel=_modeling_research())
        )
    linked = PipelineConfig.from_dict(
        _pipeline_values(
            factor_research=_enabled_research(),
            modeling_panel=_modeling_research(),
        )
    )
    files_without_research = PipelineConfig.from_dict(
        _pipeline_values(modeling_panel=_modeling_files())
    )
    files_with_research = PipelineConfig.from_dict(
        _pipeline_values(
            factor_research=_enabled_research(),
            modeling_panel=_modeling_files(),
        )
    )
    assert linked.factor_research.enabled is True
    assert files_without_research.factor_research.enabled is False
    assert files_with_research.factor_research.enabled is True


def test_pipeline_disabled_stages_and_cross_stage_roundtrip() -> None:
    default = PipelineConfig.from_dict(_pipeline_values())
    modeling_only = PipelineConfig.from_dict(
        _pipeline_values(modeling_panel=_modeling_files())
    )
    assert default.ml_experiment.enabled is False
    assert modeling_only.ml_experiment.enabled is False
    assert PipelineConfig.from_dict(modeling_only.to_dict()) == modeling_only

def _write_yaml(path: Path, ml_text: str = "") -> Path:
    path.write_text(
        f"""
data:
  output_dir: data/output
  required_datasets: [daily]
pipeline:
  backtest_start: "2024-01-01"
  backtest_end: "2025-03-31"
  train_years: 10
  max_lookback_months: 12
  stock_pool: hs300
  benchmark: 000300.SH
  strategy_name: score
  rebalance_frequency: M
  top_n: 20
  transaction_cost: 0.001
{ml_text}
""",
        encoding="utf-8",
    )
    return path


def test_yaml_without_ml_remains_disabled(tmp_path: Path) -> None:
    config = PipelineConfig.from_yaml(_write_yaml(tmp_path / "old.yaml"))
    assert config.ml_experiment.enabled is False


def test_yaml_full_ml_section_parses_and_preserves_model_params(
    tmp_path: Path,
) -> None:
    ml_text = """
ml_experiment:
  enabled: true
  panel_path: data/panel.parquet
  save_artifacts: false
  artifact_root: nested/ml
  parquet_compression: snappy
  experiment:
    dataset:
      label_col: forward_return
    walk_forward:
      train_window_periods: 2
      validation_periods: 2
      retrain_frequency: 3
      embargo_periods: 1
    training:
      model_name: ridge
      model_params:
        alpha: 3.0
    evaluation:
      minimum_cross_section_size: 3
    permutation_importance: null
"""
    config = PipelineConfig.from_yaml(
        _write_yaml(tmp_path / "ml.yaml", ml_text)
    )
    assert config.ml_experiment.enabled is True
    assert config.ml_experiment.experiment is not None
    assert (
        config.ml_experiment.experiment.training_config.model_params["alpha"]
        == 3.0
    )
    assert config.ml_experiment.parquet_compression == "snappy"


def test_yaml_unknown_ml_field_is_strict(tmp_path: Path) -> None:
    with pytest.raises(MLPipelineConfigError, match="unknown"):
        PipelineConfig.from_yaml(
            _write_yaml(
                tmp_path / "bad.yaml",
                """
ml_experiment:
  unexpected: true
""",
            )
        )
