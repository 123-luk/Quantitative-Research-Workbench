"""Tests for the UI-independent canonical V5 configuration bridge."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from app.services.pipeline_config_service import (
    ERROR_IF_INSUFFICIENT,
    HIGH_SCORE_FIRST,
    INVERSE_VOLATILITY_LABEL,
    LOW_SCORE_FIRST,
    RANK_WEIGHT_LABEL,
    USE_ALL_VALID,
    build_effective_pipeline_config,
    build_portfolio_construction_ui_config,
    build_research_backtest_ui_config,
    build_selection_summary,
    get_default_holdings_top_n,
    get_default_research_backtest_enabled,
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


def test_research_backtest_disabled_default_has_no_business_assumptions() -> None:
    assert get_default_research_backtest_enabled() is False
    research = build_research_backtest_ui_config(enabled=False)
    assert research.enabled is False
    assert research.transaction_cost is None
    assert research.benchmark is None
    assert research.performance is None


@pytest.mark.parametrize("cost_bps", [0.0, 10.0, 2.5])
@pytest.mark.parametrize("risk_free", [0.0, 0.02, -0.005])
def test_research_backtest_enabled_maps_canonical_invariants(
    cost_bps: float, risk_free: float
) -> None:
    effective = build_effective_pipeline_config(
        _base(),
        top_n=5,
        research_backtest_enabled=True,
        research_backtest_cost_bps=cost_bps,
        research_backtest_benchmark="000905.SH",
        annual_risk_free_rate=risk_free,
    )
    research = effective.research_backtest
    assert research.enabled
    assert research.source.mode == "pipeline"
    assert research.source.artifact_dir is None
    assert research.schedule.mode == "holdings_dates"
    assert research.return_alignment.effective_rule == "next_trading_day"
    assert research.return_alignment.return_convention == "adjusted_close_to_close"
    assert research.portfolio.initial_nav == 1.0
    assert research.portfolio.turnover_definition == "half_l1_pre_to_target"
    assert research.transaction_cost is not None
    assert research.transaction_cost.cost_bps == cost_bps
    assert research.transaction_cost.rate_basis == "one_way_traded_notional"
    assert research.benchmark is not None
    assert research.benchmark.benchmark_code == "000905.SH"
    assert research.benchmark.alignment_policy == "strict_common_calendar"
    assert research.performance is not None
    assert research.performance.annual_risk_free_rate == risk_free
    assert research.performance.annualization_days == 252
    assert research.artifact_subdir == "research_backtest"
    assert effective.backtest_end == _base().backtest_end
    assert effective.holdings.top_n == 5
    assert "top_n" not in research.to_dict()
    assert effective.signal.enabled and effective.holdings.enabled
    assert PipelineConfig.from_dict(effective.to_dict()) == effective


def test_top_n_changes_only_holdings_not_research_backtest() -> None:
    configs = [
        build_effective_pipeline_config(
            _base(), top_n=top_n, research_backtest_enabled=True
        )
        for top_n in (5, 10)
    ]
    assert [config.holdings.top_n for config in configs] == [5, 10]
    assert configs[0].research_backtest == configs[1].research_backtest


@pytest.mark.parametrize(
    ("label", "method"),
    [
        ("等权", "equal_weight"),
        (RANK_WEIGHT_LABEL, "rank_weight"),
        (INVERSE_VOLATILITY_LABEL, "inverse_volatility"),
    ],
)
def test_portfolio_method_display_mapping(label: str, method: str) -> None:
    config = build_portfolio_construction_ui_config(method_label=label)
    assert config.method == method
    expected = (
        {"lookback_trading_days": 60, "min_observations": 40}
        if method == "inverse_volatility"
        else {}
    )
    assert dict(config.params) == expected


def test_portfolio_cap_percent_maps_once_to_decimal_constraint() -> None:
    effective = build_effective_pipeline_config(
        _base(),
        portfolio_method_label=RANK_WEIGHT_LABEL,
        max_weight_enabled=True,
        max_weight_percent=12.5,
    )
    portfolio = effective.holdings.portfolio_construction
    assert portfolio.method == "rank_weight"
    assert portfolio.constraints[0].to_dict() == {
        "type": "max_weight",
        "params": {"max_weight": 0.125},
    }


def test_inverse_volatility_ui_parameters_roundtrip_canonically() -> None:
    effective = build_effective_pipeline_config(
        _base(),
        portfolio_method_label=INVERSE_VOLATILITY_LABEL,
        inverse_volatility_lookback=80,
        inverse_volatility_min_observations=55,
    )
    assert effective.holdings.portfolio_construction.to_dict() == {
        "method": "inverse_volatility",
        "params": {
            "lookback_trading_days": 80,
            "min_observations": 55,
        },
        "constraints": [],
    }
    assert PipelineConfig.from_dict(effective.to_dict()) == effective


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
        "组合构建": "等权",
        "单股权重上限": None,
        "source mode": "ml",
    }


def test_streamlit_v5_surface_is_separate_from_legacy_top_n() -> None:
    root = STREAMLIT_APP.parent
    canonical = "\n".join((root / path).read_text(encoding="utf-8") for path in ("views/new_run.py", "services/pipeline_config_service.py", "i18n/catalog.py"))
    assert "get_default_holdings_top_n" in canonical
    assert 't("new.top_n"' in canonical
    assert "service.run(draft" in canonical
    assert "run_research_pipeline_from_app(" not in canonical
    assert '"--top-n"' not in canonical
    for forbidden in ("sort_values(", "nlargest(", "nsmallest(", "1.0 /"):
        assert forbidden not in canonical
    assert '"new.portfolio"' in canonical
    assert "build_pipeline_config(" in canonical
    assert "PortfolioConstructionEngine" not in canonical


def test_streamlit_v6_surface_has_no_second_owner_or_unsupported_controls() -> None:
    root = STREAMLIT_APP.parent
    canonical = "\n".join((root / path).read_text(encoding="utf-8") for path in ("views/new_run.py", "views/results.py", "i18n/catalog.py"))
    assert '"Enable Research Backtest"' in canonical
    assert '"Transaction Cost (bps)"' in canonical
    assert '"Benchmark Code"' in canonical
    assert '"Annual Risk-Free Rate"' in canonical
    assert "Portfolio Net NAV vs Benchmark NAV" in canonical
    assert "Research Backtest End Date" not in canonical
    assert "Backtest frequency" not in canonical
    assert "source dropdown" not in canonical.lower()
    assert "ResearchBacktestPipelineExecutor" not in canonical
    assert "cumprod(" not in canonical
