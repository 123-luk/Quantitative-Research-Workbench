"""Tests for V2-G4A factor-research pipeline configuration."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.factors.composition import FactorCompositionConfig
from src.factors.dynamic_composition import RollingICWeightConfig
from src.factors.evaluation import FactorEvaluationConfig
from src.factors.forward_returns import ForwardReturnConfig
from src.factors.research_pipeline import FactorResearchConfig
from src.pipeline import FactorResearchPipelineConfig, PipelineConfig


NAMES = ("momentum_20d", "volatility_20d")


def _research(
    method: str = "equal",
    *,
    use_neutralization: bool = False,
    evaluate_components: bool = True,
    evaluate_composite: bool | None = None,
) -> FactorResearchConfig:
    if evaluate_composite is None:
        evaluate_composite = method != "none"
    return FactorResearchConfig(
        factor_names=NAMES,
        use_neutralization=use_neutralization,
        composition_method=method,
        evaluate_components=evaluate_components,
        evaluate_composite=evaluate_composite,
    )


def _enabled(**overrides) -> FactorResearchPipelineConfig:
    values = {
        "enabled": True,
        "factor_input_path": "missing/factor_input.parquet",
        "score_panel_path": "missing/score_panel.parquet",
        "price_panel_path": "missing/price_panel.parquet",
        "research": _research(),
    }
    values.update(overrides)
    return FactorResearchPipelineConfig(**values)


def _pipeline_values() -> dict:
    return {
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


def _full_mapping(method: str = "equal") -> dict:
    composition = {
        "method": "equal",
        "fixed_weights": [],
        "normalize_weights": True,
        "missing_policy": "renormalize",
        "min_valid_factors": 1,
        "score_col": "composite_score",
    }
    if method == "fixed":
        composition["method"] = "fixed"
        composition["fixed_weights"] = [
            ["momentum_20d", 0.4],
            ["volatility_20d", 0.6],
        ]
    return {
        "enabled": True,
        "factor_input_path": "data/processed/factor_input.parquet",
        "score_panel_path": "data/processed/score_panel.parquet",
        "price_panel_path": "data/processed/price_panel.parquet",
        "exposure_panel_path": None,
        "artifact_subdir": "factor_research",
        "research": {
            "factor_names": ["momentum_20d", "volatility_20d"],
            "use_neutralization": False,
            "composition_method": method,
            "evaluate_components": True,
            "evaluate_composite": True,
        },
        "preprocessing": {
            "missing_method": "median",
            "winsor_method": "mad",
            "mad_limit": 3.0,
            "standardize_method": "zscore",
            "min_cross_section_size": 3,
        },
        "neutralization": {
            "neutralize_industry": True,
            "neutralize_size": True,
            "industry_col": "industry",
            "size_col": "log_total_mv",
            "min_cross_section_size": 10,
            "min_industry_size": 2,
            "standardize_residuals": True,
            "size_exempt_factors": ["log_total_mv", "log_circ_mv"],
        },
        "evaluation": {
            "return_col": "forward_return",
            "min_cross_section_size": 20,
            "compute_ic": True,
            "compute_rank_ic": True,
        },
        "quantile": {
            "return_col": "forward_return",
            "quantiles": 5,
            "min_cross_section_size": 20,
            "min_group_size": 1,
            "compute_monotonicity": True,
        },
        "composition": composition,
        "rolling": {
            "metric": "rank_ic",
            "lookback_periods": 12,
            "min_periods": 6,
            "negative_policy": "zero",
            "fallback_method": "equal",
            "missing_policy": "renormalize",
            "min_valid_factors": 1,
            "score_col": "composite_score",
        },
        "forward_returns": {
            "price_col": "close",
            "return_col": "forward_return",
            "entry_lag_periods": 1,
            "holding_periods": 20,
            "require_positive_prices": True,
        },
        "artifacts": {
            "tables_dirname": "tables",
            "manifest_filename": "manifest.json",
            "compression": "snappy",
            "include_empty_tables": True,
            "overwrite": False,
            "schema_version": "1",
            "verify_after_write": True,
        },
    }


def test_disabled_defaults_are_safe_and_serializable() -> None:
    config = FactorResearchPipelineConfig()
    assert config.enabled is False
    assert config.research is None
    assert config.factor_input_path is None
    assert config.score_panel_path is None
    assert config.price_panel_path is None
    assert config.artifact_subdir == "factor_research"
    payload = config.to_dict()
    assert payload["research"] is None
    assert "DataFrame" not in repr(config)
    json.dumps(payload)


@pytest.mark.parametrize("value", [1, "false", None])
def test_enabled_must_be_bool(value: object) -> None:
    with pytest.raises(TypeError, match="enabled"):
        FactorResearchPipelineConfig(enabled=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        ".",
        "..",
        "../factor_research",
        r"..\factor_research",
        "/factor_research",
        r"\factor_research",
        "C:/factor_research",
        "C:\\factor_research",
        "nested//factor_research",
    ],
)
def test_unsafe_artifact_subdir_raises(value: str) -> None:
    with pytest.raises(ValueError, match="artifact_subdir"):
        FactorResearchPipelineConfig(artifact_subdir=value)


def test_safe_nested_artifact_subdir_is_allowed() -> None:
    config = FactorResearchPipelineConfig(artifact_subdir="research/factors")
    assert config.artifact_subdir == "research/factors"


@pytest.mark.parametrize(
    "field_name",
    [
        "factor_input_path",
        "score_panel_path",
        "price_panel_path",
        "exposure_panel_path",
    ],
)
def test_input_paths_require_nonempty_strings_or_none(field_name: str) -> None:
    with pytest.raises(TypeError, match=field_name):
        FactorResearchPipelineConfig(**{field_name: 1})
    with pytest.raises(ValueError, match=field_name):
        FactorResearchPipelineConfig(**{field_name: " "})


def test_config_creation_does_not_check_or_create_paths(tmp_path: Path) -> None:
    missing = tmp_path / "not-created" / "factor.parquet"
    config = _enabled(factor_input_path=str(missing))
    assert config.factor_input_path == str(missing)
    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("missing_field", "message"),
    [
        ("research", "research"),
        ("factor_input_path", "factor_input_path"),
        ("score_panel_path", "score_panel_path"),
        ("price_panel_path", "price_panel_path"),
    ],
)
def test_enabled_requires_research_and_three_paths(
    missing_field: str, message: str
) -> None:
    values = {
        "enabled": True,
        "factor_input_path": "factor.parquet",
        "score_panel_path": "scores.parquet",
        "price_panel_path": "prices.parquet",
        "research": _research(),
    }
    values[missing_field] = None
    with pytest.raises(ValueError, match=message):
        FactorResearchPipelineConfig(**values)


def test_neutralization_requires_exposure_path_only_when_enabled() -> None:
    with pytest.raises(ValueError, match="exposure_panel_path"):
        _enabled(research=_research(use_neutralization=True))
    config = _enabled(research=_research(use_neutralization=False))
    assert config.exposure_panel_path is None
    enabled = _enabled(
        research=_research(use_neutralization=True),
        exposure_panel_path="exposures.parquet",
    )
    assert enabled.exposure_panel_path == "exposures.parquet"


def test_full_nested_mapping_parses_without_mutating_input() -> None:
    data = _full_mapping()
    before = deepcopy(data)
    config = FactorResearchPipelineConfig.from_dict(data)
    assert data == before
    assert config.enabled is True
    assert config.research.factor_names == NAMES  # type: ignore[union-attr]
    assert config.neutralization.size_exempt_factors == (
        "log_total_mv",
        "log_circ_mv",
    )
    assert config.composition.fixed_weights == ()
    assert config.score_col == "composite_score"


def test_fixed_weight_lists_become_immutable_pairs() -> None:
    config = FactorResearchPipelineConfig.from_dict(_full_mapping("fixed"))
    assert config.composition.fixed_weights == (
        ("momentum_20d", 0.4),
        ("volatility_20d", 0.6),
    )
    assert isinstance(config.composition.fixed_weights, tuple)


def test_from_dict_rejects_non_mapping_and_unknown_top_level() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        FactorResearchPipelineConfig.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown_key"):
        FactorResearchPipelineConfig.from_dict({"unknown_key": True})


@pytest.mark.parametrize(
    "field_name",
    [
        "research",
        "preprocessing",
        "neutralization",
        "evaluation",
        "quantile",
        "composition",
        "rolling",
        "forward_returns",
        "artifacts",
    ],
)
def test_nested_sections_must_be_mappings(field_name: str) -> None:
    with pytest.raises(TypeError, match=field_name):
        FactorResearchPipelineConfig.from_dict({field_name: []})


@pytest.mark.parametrize("field_name", ["research", "preprocessing"])
def test_unknown_nested_keys_raise_clear_error(field_name: str) -> None:
    with pytest.raises(TypeError, match=field_name):
        FactorResearchPipelineConfig.from_dict(
            {field_name: {"unknown_setting": True}}
        )


@pytest.mark.parametrize(
    ("field_name", "different"),
    [
        ("evaluation", FactorEvaluationConfig(return_col="evaluation_return")),
        ("forward_returns", ForwardReturnConfig(return_col="future_return")),
    ],
)
def test_return_columns_must_match(field_name: str, different: object) -> None:
    with pytest.raises(ValueError, match="return_col"):
        FactorResearchPipelineConfig(**{field_name: different})


def test_quantile_return_column_must_match() -> None:
    data = {"quantile": {"return_col": "quantile_return"}}
    with pytest.raises(ValueError, match="return_col"):
        FactorResearchPipelineConfig.from_dict(data)


@pytest.mark.parametrize(
    ("research", "composition"),
    [
        (
            _research("equal"),
            FactorCompositionConfig(
                method="fixed", fixed_weights=(("momentum_20d", 1.0),)
            ),
        ),
        (_research("fixed"), FactorCompositionConfig(method="equal")),
    ],
)
def test_static_composition_method_conflicts_raise(
    research: FactorResearchConfig, composition: FactorCompositionConfig
) -> None:
    with pytest.raises(ValueError, match="composition"):
        FactorResearchPipelineConfig(
            research=research, composition=composition
        )


@pytest.mark.parametrize(
    ("method", "metric"),
    [
        ("rolling_ic", "rank_ic"),
        ("rolling_rank_ic", "ic"),
    ],
)
def test_rolling_metric_conflicts_raise(method: str, metric: str) -> None:
    with pytest.raises(ValueError, match="rolling.metric"):
        FactorResearchPipelineConfig(
            research=_research(method),
            rolling=RollingICWeightConfig(metric=metric),
        )


def test_invalid_none_and_rolling_dependencies_raise_during_parse() -> None:
    with pytest.raises(ValueError, match="evaluate_composite"):
        FactorResearchPipelineConfig.from_dict(
            {
                "research": {
                    "factor_names": list(NAMES),
                    "composition_method": "none",
                    "evaluate_composite": True,
                }
            }
        )
    with pytest.raises(ValueError, match="Rolling"):
        FactorResearchPipelineConfig.from_dict(
            {
                "research": {
                    "factor_names": list(NAMES),
                    "composition_method": "rolling_ic",
                    "evaluate_components": False,
                }
            }
        )


@pytest.mark.parametrize("method", ["equal", "fixed"])
def test_to_dict_round_trip_is_detached_and_json_safe(method: str) -> None:
    config = FactorResearchPipelineConfig.from_dict(_full_mapping(method))
    payload = config.to_dict()
    assert isinstance(payload, dict)
    assert all(
        isinstance(payload[name], dict)
        for name in (
            "preprocessing",
            "neutralization",
            "evaluation",
            "quantile",
            "composition",
            "rolling",
            "forward_returns",
            "artifacts",
        )
    )
    assert isinstance(payload["research"]["factor_names"], list)
    assert isinstance(payload["composition"]["fixed_weights"], list)
    json.dumps(payload)
    assert FactorResearchPipelineConfig.from_dict(payload) == config
    payload["preprocessing"]["missing_method"] = "none"
    payload["research"]["factor_names"].append("extra")
    assert config.preprocessing.missing_method == "median"
    assert config.research.factor_names == NAMES  # type: ignore[union-attr]


def test_disabled_round_trip_and_custom_score_columns() -> None:
    disabled = FactorResearchPipelineConfig()
    assert FactorResearchPipelineConfig.from_dict(disabled.to_dict()) == disabled
    static = _enabled(
        composition=FactorCompositionConfig(score_col="static_score")
    )
    assert static.score_col == "static_score"
    assert FactorResearchPipelineConfig.from_dict(static.to_dict()) == static
    rolling = _enabled(
        research=_research("rolling_ic"),
        rolling=RollingICWeightConfig(metric="ic", score_col="rolling_score"),
    )
    assert rolling.score_col == "rolling_score"
    assert FactorResearchPipelineConfig.from_dict(rolling.to_dict()) == rolling


def test_old_pipeline_constructor_defaults_research_to_disabled() -> None:
    config = PipelineConfig(**_pipeline_values())
    assert isinstance(config.factor_research, FactorResearchPipelineConfig)
    assert config.factor_research.enabled is False
    assert config.required_start_date == "2013-01-01"


def test_pipeline_from_dict_and_to_dict_round_trip_factor_research() -> None:
    values = _pipeline_values()
    values["factor_research"] = _full_mapping()
    config = PipelineConfig.from_dict(values)
    assert config.factor_research.enabled is True
    payload = config.to_dict()
    assert isinstance(payload["factor_research"], dict)
    assert PipelineConfig.from_dict(payload) == config
    with pytest.raises(ValueError, match="unknown"):
        PipelineConfig.from_dict({**values, "unknown": True})


def test_factor_research_does_not_change_pipeline_dates() -> None:
    old = PipelineConfig(**_pipeline_values())
    new = PipelineConfig(
        **_pipeline_values(),
        factor_research=_enabled(),
    )
    assert new.backtest_start == old.backtest_start
    assert new.backtest_end == old.backtest_end
    assert new.required_start_date == old.required_start_date
    assert new.required_end_date == old.required_end_date


def test_old_yaml_defaults_disabled_and_new_yaml_parses(tmp_path: Path) -> None:
    base_yaml = """
data:
  required_datasets: [daily]
pipeline:
  backtest_start: "2024-01-01"
  backtest_end: "2025-03-31"
  train_years: 10
  max_lookback_months: 12
"""
    old_path = tmp_path / "old.yaml"
    old_path.write_text(base_yaml, encoding="utf-8")
    old = PipelineConfig.from_yaml(old_path)
    assert old.factor_research.enabled is False

    new_path = tmp_path / "new.yaml"
    new_path.write_text(
        base_yaml
        + """
factor_research:
  enabled: true
  factor_input_path: missing/factors.parquet
  score_panel_path: missing/scores.parquet
  price_panel_path: missing/prices.parquet
  artifact_subdir: factor_research
  research:
    factor_names: [momentum_20d, volatility_20d]
    use_neutralization: false
    composition_method: equal
    evaluate_components: true
    evaluate_composite: true
""",
        encoding="utf-8",
    )
    new = PipelineConfig.from_yaml(new_path)
    assert new.factor_research.enabled is True
    assert new.factor_research.research.factor_names == NAMES  # type: ignore[union-attr]
    assert not Path(new.factor_research.factor_input_path).exists()  # type: ignore[arg-type]


def test_pipeline_public_imports_remain_available() -> None:
    from src.pipeline import ExperimentManager, run_pipeline

    assert ExperimentManager is not None
    assert PipelineConfig is not None
    assert FactorResearchPipelineConfig is not None
    assert callable(run_pipeline)
