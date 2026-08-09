"""Read-only presentation helpers for the V6 Research Backtest dashboard."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.pipeline.config import PipelineConfig
from src.research_backtest import (
    BENCHMARK_FILENAME,
    DAILY_PORTFOLIO_FILENAME,
    ResearchBacktestArtifactStore,
)


class ResearchBacktestDashboardError(ValueError):
    """Raised when an exact V6 result cannot be displayed safely."""


@dataclass(frozen=True)
class ResearchBacktestDashboardPayload:
    """Validated Artifact NAV values plus the unchanged pipeline metrics."""

    artifact_dir: Path
    metrics: dict[str, int | float | None]
    nav: pd.DataFrame


PERCENT_METRICS = frozenset(
    {
        "net_total_return",
        "net_annualized_return",
        "net_max_drawdown",
        "benchmark_total_return",
        "excess_total_return",
        "tracking_error",
        "average_turnover",
        "transaction_cost_return_drag",
    }
)


def format_research_backtest_metric(
    value: object, *, percent: bool = False, decimals: int = 2
) -> str:
    """Apply presentation-only formatting and suppress non-finite values."""
    if value is None or isinstance(value, bool):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    if percent:
        return f"{number * 100:.{decimals}f}%"
    return f"{number:.{decimals}f}"


def load_research_backtest_dashboard(
    result: Mapping[str, object],
    *,
    store: ResearchBacktestArtifactStore | None = None,
) -> ResearchBacktestDashboardPayload:
    """Validate and read NAV columns from one exact pipeline result Artifact."""
    if not isinstance(result, Mapping) or result.get("enabled") is not True:
        raise ResearchBacktestDashboardError(
            "An enabled Research Backtest pipeline result is required."
        )
    raw_dir = result.get("artifact_dir")
    if not isinstance(raw_dir, (str, Path)) or not str(raw_dir).strip():
        raise ResearchBacktestDashboardError(
            "Research Backtest result has no exact artifact_dir."
        )
    artifact_dir = Path(raw_dir)
    validator = ResearchBacktestArtifactStore() if store is None else store
    try:
        validation = validator.validate(artifact_dir)
    except Exception as exc:
        raise ResearchBacktestDashboardError(
            f"Research Backtest Artifact validation failed: {exc}"
        ) from exc
    if not validation.is_valid:
        issue_codes = ", ".join(issue.code for issue in validation.issues)
        raise ResearchBacktestDashboardError(
            f"Research Backtest Artifact validation failed: {issue_codes or 'invalid'}."
        )

    try:
        daily = pd.read_parquet(
            artifact_dir / DAILY_PORTFOLIO_FILENAME, engine="pyarrow"
        )
        benchmark = pd.read_parquet(
            artifact_dir / BENCHMARK_FILENAME, engine="pyarrow"
        )
    except Exception as exc:
        raise ResearchBacktestDashboardError(
            f"Validated Research Backtest payload could not be read: {exc}"
        ) from exc
    daily_columns = {"trade_date", "gross_nav", "net_nav"}
    benchmark_columns = {"trade_date", "benchmark_nav"}
    if not daily_columns.issubset(daily.columns) or not benchmark_columns.issubset(
        benchmark.columns
    ):
        raise ResearchBacktestDashboardError("Artifact NAV columns are invalid.")
    if tuple(daily["trade_date"]) != tuple(benchmark["trade_date"]):
        raise ResearchBacktestDashboardError(
            "Daily portfolio and benchmark dates must match exactly."
        )
    nav = pd.DataFrame(
        {
            "trade_date": daily["trade_date"].to_numpy(copy=True),
            "gross_nav": daily["gross_nav"].to_numpy(copy=True),
            "net_nav": daily["net_nav"].to_numpy(copy=True),
            "benchmark_nav": benchmark["benchmark_nav"].to_numpy(copy=True),
        }
    )
    raw_metrics = result.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise ResearchBacktestDashboardError(
            "Research Backtest result metrics are invalid."
        )
    metrics = deepcopy(dict(raw_metrics))
    return ResearchBacktestDashboardPayload(artifact_dir, metrics, nav)


def build_research_backtest_details(
    result: Mapping[str, object], config: PipelineConfig
) -> dict[str, object]:
    """Build audit details without deriving any portfolio performance values."""
    research = config.research_backtest
    if not research.enabled:
        raise ResearchBacktestDashboardError(
            "Research Backtest details require an enabled config."
        )
    assert research.transaction_cost is not None
    assert research.benchmark is not None
    return {
        "Artifact path": result.get("artifact_dir"),
        "Benchmark code": result.get("benchmark_code"),
        "Start date": result.get("start_date"),
        "End date": result.get("end_date"),
        "Observation count": result.get("observation_count"),
        "Rebalance count": result.get("rebalance_count"),
        "Cost bps": research.transaction_cost.cost_bps,
        "Top-N (Holdings)": config.holdings.top_n,
        "Schedule": research.schedule.mode,
        "Effective rule": research.return_alignment.effective_rule,
        "Return convention": research.return_alignment.return_convention,
    }
