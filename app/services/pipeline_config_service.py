"""UI-independent configuration bridge for the canonical V5 pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Final

import yaml

from src.pipeline.config import PipelineConfig
from src.pipeline.holdings_config import HoldingsPipelineConfig
from src.pipeline.research_backtest_config import (
    BacktestScheduleConfig,
    BacktestSourceConfig,
    BenchmarkConfig,
    PerformanceConfig,
    PortfolioAccountingConfig,
    ResearchBacktestPipelineConfig,
    ReturnAlignmentConfig,
    TransactionCostConfig,
)
from src.portfolio_construction import ConstraintSpec, PortfolioConstructionConfig

from app.services.capability_catalog_service import CapabilityCatalogService


HIGH_SCORE_FIRST: Final = "分数越高越优"
LOW_SCORE_FIRST: Final = "分数越低越优"
ERROR_IF_INSUFFICIENT: Final = "报错"
USE_ALL_VALID: Final = "使用全部有效股票"
EQUAL_WEIGHT_LABEL: Final = "等权"
RANK_WEIGHT_LABEL: Final = "排名加权"
INVERSE_VOLATILITY_LABEL: Final = "逆波动率"
MINIMUM_VARIANCE_LABEL: Final = "Minimum Variance"
SAMPLE_COVARIANCE_LABEL: Final = "Sample Covariance"
LEDOIT_WOLF_LABEL: Final = "Ledoit-Wolf"
SUGGESTED_INVERSE_VOLATILITY_LOOKBACK: Final = 60
SUGGESTED_INVERSE_VOLATILITY_MIN_OBSERVATIONS: Final = 40
SUGGESTED_RISK_MODEL_LOOKBACK: Final = 120
SUGGESTED_RISK_MODEL_MIN_OBSERVATIONS: Final = 80
SUGGESTED_MAX_WEIGHT_PERCENT: Final = 20.0
SUGGESTED_RESEARCH_BACKTEST_COST_BPS: Final = 10.0
SUGGESTED_RESEARCH_BACKTEST_BENCHMARK: Final = "000300.SH"
SUGGESTED_ANNUAL_RISK_FREE_RATE: Final = 0.0

SIGNAL_DIRECTION_BY_LABEL: Final = {
    HIGH_SCORE_FIRST: "descending",
    LOW_SCORE_FIRST: "ascending",
}
INSUFFICIENT_POLICY_BY_LABEL: Final = {
    ERROR_IF_INSUFFICIENT: "error",
    USE_ALL_VALID: "allow_partial",
}
PORTFOLIO_METHOD_BY_LABEL: Final = {
    EQUAL_WEIGHT_LABEL: "equal_weight",
    RANK_WEIGHT_LABEL: "rank_weight",
    INVERSE_VOLATILITY_LABEL: "inverse_volatility",
    MINIMUM_VARIANCE_LABEL: "minimum_variance",
}
RISK_ESTIMATOR_BY_LABEL: Final = {
    SAMPLE_COVARIANCE_LABEL: "sample_covariance",
    LEDOIT_WOLF_LABEL: "ledoit_wolf",
}


def build_portfolio_construction_ui_config(
    *,
    method_label: str = EQUAL_WEIGHT_LABEL,
    lookback_trading_days: int = SUGGESTED_INVERSE_VOLATILITY_LOOKBACK,
    min_observations: int = SUGGESTED_INVERSE_VOLATILITY_MIN_OBSERVATIONS,
    risk_model_estimator_label: str = LEDOIT_WOLF_LABEL,
    risk_model_lookback: int = SUGGESTED_RISK_MODEL_LOOKBACK,
    risk_model_min_observations: int = SUGGESTED_RISK_MODEL_MIN_OBSERVATIONS,
    max_weight_enabled: bool = False,
    max_weight_percent: float = SUGGESTED_MAX_WEIGHT_PERCENT,
) -> PortfolioConstructionConfig:
    """Map display values onto the canonical V7 portfolio config."""
    try:
        method = PORTFOLIO_METHOD_BY_LABEL[method_label]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported portfolio construction option: {method_label!r}."
        ) from exc
    params: dict[str, object] = {}
    if method == "inverse_volatility":
        params = {
            "lookback_trading_days": lookback_trading_days,
            "min_observations": min_observations,
        }
    if method == "minimum_variance":
        try:
            estimator = RISK_ESTIMATOR_BY_LABEL[risk_model_estimator_label]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported risk estimator option: {risk_model_estimator_label!r}."
            ) from exc
        params = {
            "risk_model": {
                "estimator": estimator,
                "params": {},
                "lookback_trading_days": risk_model_lookback,
                "min_observations": risk_model_min_observations,
            }
        }
    constraints = ()
    if max_weight_enabled:
        constraints = (
            ConstraintSpec(
                type="max_weight",
                params={"max_weight": max_weight_percent / 100.0},
            ),
        )
    return PortfolioConstructionConfig(
        method=method,
        params=params,
        constraints=constraints,
    )


def get_default_holdings_top_n() -> int:
    """Return the canonical backend default used by the V5 UI widget."""
    return HoldingsPipelineConfig().top_n


def get_default_research_backtest_enabled() -> bool:
    """Return the canonical backend default for the V6 enable control."""
    return ResearchBacktestPipelineConfig().enabled


def build_research_backtest_ui_config(
    *,
    enabled: bool,
    cost_bps: float = SUGGESTED_RESEARCH_BACKTEST_COST_BPS,
    benchmark_code: str = SUGGESTED_RESEARCH_BACKTEST_BENCHMARK,
    annual_risk_free_rate: float = SUGGESTED_ANNUAL_RISK_FREE_RATE,
    annualization_days: int = 252,
    initial_nav: float = 1.0,
) -> ResearchBacktestPipelineConfig:
    """Map the narrow ordinary-UI surface onto canonical V6 config classes."""
    if not enabled:
        return ResearchBacktestPipelineConfig()
    return ResearchBacktestPipelineConfig(
        enabled=True,
        source=BacktestSourceConfig(),
        schedule=BacktestScheduleConfig(),
        return_alignment=ReturnAlignmentConfig(),
        portfolio=PortfolioAccountingConfig(initial_nav=initial_nav),
        transaction_cost=TransactionCostConfig(cost_bps=cost_bps),
        benchmark=BenchmarkConfig(benchmark_code=benchmark_code),
        performance=PerformanceConfig(
            annual_risk_free_rate=annual_risk_free_rate,
            annualization_days=annualization_days,
        ),
    )


def load_canonical_base_config(config_path: str | Path) -> PipelineConfig:
    """Load one direct-schema canonical PipelineConfig YAML file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError(f"Canonical UI config must be a YAML mapping: {path}")
    return PipelineConfig.from_dict(values)


def build_effective_pipeline_config(
    base_config: PipelineConfig,
    *,
    top_n: int | None = None,
    signal_direction_label: str = HIGH_SCORE_FIRST,
    insufficient_policy_label: str = ERROR_IF_INSUFFICIENT,
    portfolio_method_label: str = EQUAL_WEIGHT_LABEL,
    inverse_volatility_lookback: int = SUGGESTED_INVERSE_VOLATILITY_LOOKBACK,
    inverse_volatility_min_observations: int = (
        SUGGESTED_INVERSE_VOLATILITY_MIN_OBSERVATIONS
    ),
    risk_model_estimator_label: str = LEDOIT_WOLF_LABEL,
    risk_model_lookback: int = SUGGESTED_RISK_MODEL_LOOKBACK,
    risk_model_min_observations: int = SUGGESTED_RISK_MODEL_MIN_OBSERVATIONS,
    max_weight_enabled: bool = False,
    max_weight_percent: float = SUGGESTED_MAX_WEIGHT_PERCENT,
    research_backtest_enabled: bool = False,
    research_backtest_cost_bps: float = SUGGESTED_RESEARCH_BACKTEST_COST_BPS,
    research_backtest_benchmark: str = SUGGESTED_RESEARCH_BACKTEST_BENCHMARK,
    annual_risk_free_rate: float = SUGGESTED_ANNUAL_RISK_FREE_RATE,
) -> PipelineConfig:
    """Apply the small V5 UI surface to a detached canonical config."""
    if not isinstance(base_config, PipelineConfig):
        raise TypeError("base_config must be a PipelineConfig.")
    try:
        signal_direction = SIGNAL_DIRECTION_BY_LABEL[signal_direction_label]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Signal direction option: {signal_direction_label!r}."
        ) from exc
    try:
        insufficient_policy = INSUFFICIENT_POLICY_BY_LABEL[
            insufficient_policy_label
        ]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported insufficient-universe option: {insufficient_policy_label!r}."
        ) from exc

    effective_top_n = get_default_holdings_top_n() if top_n is None else top_n
    values = base_config.to_dict()
    signal = dict(values["signal"])
    signal.update(
        {
            "enabled": True,
            "signal_direction": signal_direction,
        }
    )
    holdings = dict(values["holdings"])
    holdings.update(
        {
            "enabled": True,
            "top_n": effective_top_n,
            "insufficient_universe_policy": insufficient_policy,
            "weighting": "equal_weight",
            "portfolio_construction": build_portfolio_construction_ui_config(
                method_label=portfolio_method_label,
                lookback_trading_days=inverse_volatility_lookback,
                min_observations=inverse_volatility_min_observations,
                risk_model_estimator_label=risk_model_estimator_label,
                risk_model_lookback=risk_model_lookback,
                risk_model_min_observations=risk_model_min_observations,
                max_weight_enabled=max_weight_enabled,
                max_weight_percent=max_weight_percent,
            ).to_dict(),
        }
    )
    values["signal"] = signal
    values["holdings"] = holdings
    values["research_backtest"] = build_research_backtest_ui_config(
        enabled=research_backtest_enabled,
        cost_bps=research_backtest_cost_bps,
        benchmark_code=research_backtest_benchmark,
        annual_risk_free_rate=annual_risk_free_rate,
    ).to_dict()

    # PipelineConfig currently requires the legacy root field to equal enabled
    # holdings.top_n. This is a one-way compatibility mirror: UI input and V5
    # execution read only holdings.top_n, so the root never becomes a second truth.
    values["top_n"] = effective_top_n
    return PipelineConfig.from_dict(values)


def build_selection_summary(config: PipelineConfig) -> dict[str, object]:
    """Build the pre-run display strictly from the effective config."""
    if not config.signal.enabled or not config.holdings.enabled:
        raise ValueError("Effective V5 config must enable Signal and Holdings.")
    direction_labels = {
        backend: label for label, backend in SIGNAL_DIRECTION_BY_LABEL.items()
    }
    policy_labels = {
        backend: label for label, backend in INSUFFICIENT_POLICY_BY_LABEL.items()
    }
    method_labels = {
        backend: label for label, backend in PORTFOLIO_METHOD_BY_LABEL.items()
    }
    portfolio = config.holdings.portfolio_construction
    cap = next(
        (
            float(spec.params["max_weight"])
            for spec in portfolio.constraints
            if spec.type == "max_weight"
        ),
        None,
    )
    return {
        "Top N": config.holdings.top_n,
        "Signal 排序": direction_labels[config.signal.signal_direction],
        "股票不足 N": policy_labels[
            config.holdings.insufficient_universe_policy
        ],
        "组合构建": method_labels[portfolio.method],
        "单股权重上限": None if cap is None else f"{cap:.2%}",
        "source mode": config.signal.source.mode,
    }


def _required(state: Mapping[str, object], name: str) -> object:
    if name not in state:
        raise ValueError(f"Missing New Run field: {name}.")
    return state[name]


def _canonical_portfolio_config(
    state: Mapping[str, object], catalog: CapabilityCatalogService
) -> dict[str, object]:
    method = str(_required(state, "portfolio_method"))
    if method not in catalog.list_portfolio_methods():
        raise ValueError(f"Unknown portfolio method: {method!r}.")
    params: dict[str, object] = {}
    if method == "inverse_volatility":
        params = {
            "lookback_trading_days": state.get("lookback_trading_days", 60),
            "min_observations": state.get("min_observations", 40),
        }
    elif method == "minimum_variance":
        estimator = str(state.get("risk_estimator", "ledoit_wolf"))
        if estimator not in catalog.list_risk_estimators():
            raise ValueError(f"Unknown risk estimator: {estimator!r}.")
        params = {
            "risk_model": {
                "estimator": estimator,
                "params": {},
                "lookback_trading_days": state.get(
                    "risk_lookback_trading_days", 120
                ),
                "min_observations": state.get("risk_min_observations", 80),
            }
        }
    constraints: list[dict[str, object]] = []
    if bool(state.get("max_weight_enabled", False)):
        if "max_weight" not in catalog.list_constraints():
            raise ValueError("The backend does not register max_weight.")
        percent = state.get("max_weight_percent", 20.0)
        if isinstance(percent, bool) or not isinstance(percent, (int, float)):
            raise ValueError("max_weight_percent must be numeric.")
        constraints.append(
            {"type": "max_weight", "params": {"max_weight": float(percent) / 100.0}}
        )
    return PortfolioConstructionConfig.from_dict(
        {"method": method, "params": params, "constraints": constraints}
    ).to_dict()


def build_pipeline_config(
    form_state: Mapping[str, object],
    *,
    catalog: CapabilityCatalogService | None = None,
    base_config: PipelineConfig | None = None,
) -> PipelineConfig:
    """Build one deterministic canonical config from detached New Run state."""
    if not isinstance(form_state, Mapping):
        raise TypeError("form_state must be a Mapping.")
    state = deepcopy(dict(form_state))
    capabilities = catalog or CapabilityCatalogService()
    factors = tuple(_required(state, "selected_factors"))  # type: ignore[arg-type]
    unknown_factors = sorted(set(factors) - set(capabilities.list_factor_names()))
    if unknown_factors:
        raise ValueError(f"Unknown registered factor(s): {unknown_factors!r}.")
    if not factors:
        raise ValueError("Select at least one registered factor.")

    model_name = str(_required(state, "model_name"))
    if model_name not in capabilities.list_model_names():
        raise ValueError(f"Unknown registered model: {model_name!r}.")
    raw_model_params = state.get("model_params", {})
    if not isinstance(raw_model_params, dict):
        raise ValueError("model_params must be a mapping.")
    model_params = capabilities.validate_model_parameters(
        model_name, deepcopy(raw_model_params)
    )

    source = base_config or PipelineConfig.from_yaml(
        Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    )
    values = source.to_dict()
    values.update(
        {
            "backtest_start": str(_required(state, "backtest_start")),
            "backtest_end": str(_required(state, "backtest_end")),
            "train_years": state.get("train_years", source.train_years),
            "max_lookback_months": state.get(
                "max_lookback_months", source.max_lookback_months
            ),
            "stock_pool": str(_required(state, "stock_pool")),
            "benchmark": str(_required(state, "benchmark")),
            "strategy_name": str(state.get("strategy_name", "research_workbench")),
            "selected_factors": list(factors),
            "top_n": _required(state, "top_n"),
        }
    )

    factor_research_enabled = bool(state.get("factor_research_enabled", False))
    composition_method = str(state.get("composition_method", "equal"))
    evaluate_composite = bool(state.get("evaluate_composite", True))
    if composition_method == "none":
        evaluate_composite = False
    factor_research: dict[str, object] = {
        "enabled": factor_research_enabled,
        "research": {
            "factor_names": list(factors),
            "use_neutralization": bool(state.get("use_neutralization", False)),
            "composition_method": composition_method,
            "evaluate_components": bool(state.get("evaluate_components", True)),
            "evaluate_composite": evaluate_composite,
        },
    }
    if factor_research_enabled:
        factor_research.update(
            {
                "factor_input_path": str(
                    state.get("factor_input_path", "data/processed/factor_input.parquet")
                ),
                "score_panel_path": str(
                    state.get("score_panel_path", "data/processed/score_panel.parquet")
                ),
                "price_panel_path": str(
                    state.get("price_panel_path", "data/processed/price_panel.parquet")
                ),
                "exposure_panel_path": (
                    str(state.get("exposure_panel_path"))
                    if state.get("exposure_panel_path")
                    else None
                ),
            }
        )
        if composition_method in {"rolling_ic", "rolling_rank_ic"}:
            factor_research["rolling"] = {
                "metric": "ic" if composition_method == "rolling_ic" else "rank_ic"
            }
    values["factor_research"] = factor_research

    modeling_source = (
        {"mode": "factor_research"}
        if factor_research_enabled
        else {
            "mode": "files",
            "factor_panel_path": str(
                state.get(
                    "factor_panel_path", "data/processed/modeling_factor_panel.parquet"
                )
            ),
            "forward_returns_path": str(
                state.get(
                    "forward_returns_path",
                    "data/processed/modeling_forward_returns.parquet",
                )
            ),
        }
    )
    values["modeling_panel"] = {
        "enabled": True,
        "source": modeling_source,
        "builder": {
            "label_column": "forward_return",
            "include_features": list(factors),
            "exclude_features": [],
            "unmatched_policy": "audit_and_drop",
            "require_entry_after_signal": True,
            "allow_missing_labels": True,
        },
    }
    values["ml_experiment"] = {
        "enabled": True,
        "panel_path": None,
        "save_artifacts": True,
        "artifact_root": "ml_artifacts",
        "experiment_id": str(state.get("experiment_id", "research_workbench")),
        "parquet_compression": "zstd",
        "experiment": {
            "dataset": {"label_col": "forward_return"},
            "walk_forward": {
                "train_window_periods": state.get("train_window_periods", 252),
                "validation_periods": state.get("validation_periods", 20),
                "window_type": state.get("window_type", "rolling"),
                "retrain_frequency": state.get("retrain_frequency", 20),
                "embargo_periods": state.get("embargo_periods", 1),
            },
            "training": {
                "model_name": model_name,
                "model_params": model_params,
            },
            "evaluation": {
                "minimum_cross_section_size": state.get(
                    "minimum_cross_section_size", 3
                )
            },
            "permutation_importance": None,
        },
    }
    values["signal"] = {
        "enabled": True,
        "source": {"mode": "ml", "artifact_dir": None},
        "prediction_column": "prediction",
        "signal_direction": state.get("signal_direction", "descending"),
        "artifact_subdir": "signal",
    }
    values["holdings"] = {
        "enabled": True,
        "top_n": _required(state, "top_n"),
        "insufficient_universe_policy": state.get(
            "insufficient_universe_policy", "error"
        ),
        "weighting": "equal_weight",
        "artifact_subdir": "holdings",
        "portfolio_construction": _canonical_portfolio_config(state, capabilities),
    }
    values["research_backtest"] = build_research_backtest_ui_config(
        enabled=bool(state.get("research_backtest_enabled", False)),
        cost_bps=float(state.get("transaction_cost_bps", 10.0)),
        benchmark_code=str(state.get("research_backtest_benchmark", values["benchmark"])),
        annual_risk_free_rate=float(state.get("annual_risk_free_rate", 0.0)),
        annualization_days=state.get("annualization_days", 252),  # type: ignore[arg-type]
        initial_nav=float(state.get("initial_nav", 1.0)),
    ).to_dict()
    return PipelineConfig.from_dict(values)
