"""UI-independent configuration bridge for the canonical V5 pipeline."""

from __future__ import annotations

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


HIGH_SCORE_FIRST: Final = "分数越高越优"
LOW_SCORE_FIRST: Final = "分数越低越优"
ERROR_IF_INSUFFICIENT: Final = "报错"
USE_ALL_VALID: Final = "使用全部有效股票"
EQUAL_WEIGHT_LABEL: Final = "等权"
RANK_WEIGHT_LABEL: Final = "排名加权"
INVERSE_VOLATILITY_LABEL: Final = "逆波动率"
SUGGESTED_INVERSE_VOLATILITY_LOOKBACK: Final = 60
SUGGESTED_INVERSE_VOLATILITY_MIN_OBSERVATIONS: Final = 40
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
}


def build_portfolio_construction_ui_config(
    *,
    method_label: str = EQUAL_WEIGHT_LABEL,
    lookback_trading_days: int = SUGGESTED_INVERSE_VOLATILITY_LOOKBACK,
    min_observations: int = SUGGESTED_INVERSE_VOLATILITY_MIN_OBSERVATIONS,
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
) -> ResearchBacktestPipelineConfig:
    """Map the narrow ordinary-UI surface onto canonical V6 config classes."""
    if not enabled:
        return ResearchBacktestPipelineConfig()
    return ResearchBacktestPipelineConfig(
        enabled=True,
        source=BacktestSourceConfig(),
        schedule=BacktestScheduleConfig(),
        return_alignment=ReturnAlignmentConfig(),
        portfolio=PortfolioAccountingConfig(),
        transaction_cost=TransactionCostConfig(cost_bps=cost_bps),
        benchmark=BenchmarkConfig(benchmark_code=benchmark_code),
        performance=PerformanceConfig(
            annual_risk_free_rate=annual_risk_free_rate
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
