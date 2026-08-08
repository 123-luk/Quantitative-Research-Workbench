"""Tests for the UI-independent canonical V5 configuration bridge."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from app.services.pipeline_config_service import (
    ERROR_IF_INSUFFICIENT,
    HIGH_SCORE_FIRST,
    LOW_SCORE_FIRST,
    USE_ALL_VALID,
    build_effective_pipeline_config,
    build_selection_summary,
    get_default_holdings_top_n,
    load_canonical_base_config,
)
from src.ml import (
    MLExperimentConfig,
    MLDatasetConfig,
    ModelEvaluationConfig,
    WalkForwardConfig,
    WalkForwardTrainingConfig,
)
from src.pipeline import (
    HoldingsPipelineConfig,
    MLExperimentPipelineConfig,
    PipelineConfig,
    PredictionSourceConfig,
    SignalPipelineConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = PROJECT_ROOT / "app" / "streamlit_app.py"


def _experiment() -> MLExperimentConfig:
    return MLExperimentConfig(
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


def _base(*, source_mode: str = "ml") -> PipelineConfig:
    ml = MLExperimentPipelineConfig()
    source = PredictionSourceConfig("files", Path("native-ml-artifact"))
    if source_mode == "ml":
        ml = MLExperimentPipelineConfig(
            enabled=True,
            panel_path="panel.parquet",
            save_artifacts=True,
            experiment_id="ui-bridge",
            experiment=_experiment(),
        )
        source = PredictionSourceConfig("ml")
    return PipelineConfig(
        backtest_start="2024-01-01",
        backtest_end="2024-12-31",
        train_years=1,
        max_lookback_months=1,
        stock_pool="hs300",
        benchmark="000300.SH",
        strategy_name="preserved_strategy",
        selected_factors=["factor_a"],
        rebalance_frequency="M",
        top_n=7,
        transaction_cost=0.001,
        data_root="data",
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        cache_dir="data/cache",
        output_dir="output",
        parquet_engine="auto",
        required_datasets=[],
        ml_experiment=ml,
        signal=SignalPipelineConfig(source=source),
        holdings=HoldingsPipelineConfig(top_n=33),
    )


def test_ui_default_is_the_backend_default() -> None:
    assert HoldingsPipelineConfig().top_n == 20
    assert get_default_holdings_top_n() == HoldingsPipelineConfig().top_n
    effective = build_effective_pipeline_config(_base())
    assert effective.holdings.top_n == 20
    assert effective.top_n == 20


@pytest.mark.parametrize("top_n", [1, 10, 20, 1000])
def test_top_n_mapping_accepts_the_backend_range(top_n: int) -> None:
    effective = build_effective_pipeline_config(_base(), top_n=top_n)
    assert effective.holdings.top_n == top_n
    assert effective.top_n == top_n


@pytest.mark.parametrize("top_n", [0, -1, True, 1.0, "10"])
def test_top_n_mapping_reuses_strict_backend_validation(top_n: object) -> None:
    with pytest.raises((TypeError, ValueError), match="top_n"):
        build_effective_pipeline_config(_base(), top_n=top_n)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("direction_label", "backend"),
    [(HIGH_SCORE_FIRST, "descending"), (LOW_SCORE_FIRST, "ascending")],
)
def test_direction_display_mapping(direction_label: str, backend: str) -> None:
    effective = build_effective_pipeline_config(
        _base(), signal_direction_label=direction_label
    )
    assert effective.signal.signal_direction == backend


@pytest.mark.parametrize(
    ("policy_label", "backend"),
    [(ERROR_IF_INSUFFICIENT, "error"), (USE_ALL_VALID, "allow_partial")],
)
def test_insufficient_universe_display_mapping(
    policy_label: str, backend: str
) -> None:
    effective = build_effective_pipeline_config(
        _base(), insufficient_policy_label=policy_label
    )
    assert effective.holdings.insufficient_universe_policy == backend
    assert effective.holdings.weighting == "equal_weight"


def test_bridge_is_detached_preserves_unrelated_config_and_source() -> None:
    base = _base(source_mode="files")
    before = deepcopy(base.to_dict())
    effective = build_effective_pipeline_config(
        base,
        top_n=10,
        signal_direction_label=LOW_SCORE_FIRST,
        insufficient_policy_label=USE_ALL_VALID,
    )
    assert base.to_dict() == before
    assert effective is not base
    assert effective.strategy_name == "preserved_strategy"
    assert effective.selected_factors == ["factor_a"]
    assert effective.signal.enabled and effective.holdings.enabled
    assert effective.signal.source.mode == "files"
    assert effective.signal.source.artifact_dir == Path("native-ml-artifact")
    assert PipelineConfig.from_dict(effective.to_dict()) == effective


def test_ml_source_dependency_and_display_option_validation() -> None:
    effective = build_effective_pipeline_config(_base(source_mode="ml"))
    assert effective.ml_experiment.enabled
    assert effective.signal.source.mode == "ml"
    with pytest.raises(ValueError, match="direction option"):
        build_effective_pipeline_config(_base(), signal_direction_label="unknown")
    with pytest.raises(ValueError, match="insufficient-universe option"):
        build_effective_pipeline_config(_base(), insufficient_policy_label="unknown")


def test_loader_uses_direct_pipeline_schema_without_mutating_yaml(tmp_path: Path) -> None:
    values = _base().to_dict()
    before = deepcopy(values)
    path = tmp_path / "direct.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    loaded = load_canonical_base_config(path)
    assert loaded == PipelineConfig.from_dict(values)
    assert values == before
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == before


def test_pre_run_summary_is_derived_from_effective_config() -> None:
    config = build_effective_pipeline_config(
        _base(),
        top_n=10,
        signal_direction_label=LOW_SCORE_FIRST,
        insufficient_policy_label=USE_ALL_VALID,
    )
    assert build_selection_summary(config) == {
        "Top N": 10,
        "Signal 排序": LOW_SCORE_FIRST,
        "股票不足 N": USE_ALL_VALID,
        "权重方式": "等权",
        "source mode": "ml",
    }


def test_streamlit_v5_surface_is_separate_from_legacy_top_n() -> None:
    source = STREAMLIT_APP.read_text(encoding="utf-8")
    canonical = source.split("def render_legacy_pipeline_controls", 1)[0]
    assert "value=get_default_holdings_top_n()" in canonical
    assert "max_value=100" not in canonical
    assert "run_canonical_pipeline(effective_config)" in canonical
    assert "run_research_pipeline_from_app(" not in canonical
    assert '"--top-n"' not in canonical
    for forbidden in ("sort_values(", "nlargest(", "nsmallest(", "1.0 /"):
        assert forbidden not in canonical
