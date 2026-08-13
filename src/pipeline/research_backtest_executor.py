"""Standalone orchestration for the native V6 research backtest."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime
import json
import math
import os
from pathlib import Path

import pandas as pd

from src.pipeline.holdings_execution import HoldingsPipelineResult
from src.pipeline.research_backtest_config import ResearchBacktestPipelineConfig
from src.research_backtest import (
    RESEARCH_BACKTEST_ARTIFACT_SCHEMA_VERSION,
    PERFORMANCE_METRIC_KEYS,
    PerformanceAnalyticsEngine,
    PortfolioDailyAccountingEngine,
    RebalanceAccountingEngine,
    ResearchBacktestArtifactError,
    ResearchBacktestArtifactStore,
    TushareBenchmarkDailyReturnAdapter,
    TushareSecurityDailyReturnAdapter,
    TushareSecurityLifecycleAdapter,
    TushareSecuritySuspensionAdapter,
    TushareTradingCalendarAdapter,
    build_security_status,
)
from src.research_backtest.sources import ResearchBacktestHoldingsSourceAdapter


_CALENDAR_COVERAGE_BUFFER_DAYS = 31


class ResearchBacktestPipelineExecutionError(Exception):
    """Raised when the standalone V6 orchestration cannot complete safely."""


def _date(value: object, *, field_name: str) -> pd.Timestamp:
    if isinstance(value, bool) or not isinstance(
        value, (str, date, datetime, pd.Timestamp)
    ):
        raise ResearchBacktestPipelineExecutionError(
            f"{field_name} must be a canonical date value."
        )
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ResearchBacktestPipelineExecutionError(
            f"{field_name} must be a canonical date value."
        ) from exc
    if pd.isna(result) or result.tz is not None or result != result.normalize():
        raise ResearchBacktestPipelineExecutionError(
            f"{field_name} must be a timezone-naive normalized date."
        )
    return result


def _artifact_dir(value: object) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ResearchBacktestPipelineExecutionError(
            "artifact_dir must be a str or os.PathLike."
        )
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
        raise ResearchBacktestPipelineExecutionError(
            "artifact_dir must be a non-empty trimmed path."
        )
    return Path(os.path.abspath(raw))


@dataclass(frozen=True)
class ResearchBacktestPipelineResult:
    """Compact JSON-safe stage result without runtime DataFrames."""

    enabled: bool
    artifact_dir: Path | None = None
    schema_version: str | None = None
    observation_count: int = 0
    rebalance_count: int = 0
    start_date: str | None = None
    end_date: str | None = None
    benchmark_code: str | None = None
    metrics: dict[str, int | float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ResearchBacktestPipelineExecutionError("enabled must be a bool.")
        if not self.enabled:
            if (
                self.artifact_dir is not None
                or self.schema_version is not None
                or self.observation_count
                or self.rebalance_count
                or self.start_date is not None
                or self.end_date is not None
                or self.benchmark_code is not None
                or self.metrics
            ):
                raise ResearchBacktestPipelineExecutionError(
                    "disabled result fields must use empty defaults."
                )
        else:
            if (
                self.artifact_dir is None
                or self.schema_version != RESEARCH_BACKTEST_ARTIFACT_SCHEMA_VERSION
                or type(self.observation_count) is not int
                or self.observation_count < 1
                or type(self.rebalance_count) is not int
                or self.rebalance_count < 1
                or not isinstance(self.start_date, str)
                or not isinstance(self.end_date, str)
                or not isinstance(self.benchmark_code, str)
                or not self.benchmark_code
            ):
                raise ResearchBacktestPipelineExecutionError(
                    "enabled result metadata is invalid."
                )
            path = _artifact_dir(self.artifact_dir)
            if not path.is_dir() or path.is_symlink():
                raise ResearchBacktestPipelineExecutionError(
                    "enabled result artifact_dir is invalid."
                )
            object.__setattr__(self, "artifact_dir", path)
            if (
                not isinstance(self.metrics, dict)
                or set(self.metrics) != set(PERFORMANCE_METRIC_KEYS)
            ):
                raise ResearchBacktestPipelineExecutionError(
                    "enabled result metrics are invalid."
                )
            if self.rebalance_count > self.observation_count:
                raise ResearchBacktestPipelineExecutionError(
                    "rebalance_count cannot exceed observation_count."
                )
            for name in ("start_date", "end_date"):
                value = getattr(self, name)
                if _date(value, field_name=name).date().isoformat() != value:
                    raise ResearchBacktestPipelineExecutionError(
                        f"{name} must use YYYY-MM-DD format."
                    )
            if self.start_date > self.end_date:
                raise ResearchBacktestPipelineExecutionError(
                    "result date range is invalid."
                )
        for value in self.metrics.values():
            if value is not None and type(value) not in (int, float):
                raise ResearchBacktestPipelineExecutionError(
                    "result metrics must contain built-in numeric values or None."
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ResearchBacktestPipelineExecutionError(
                    "result metrics must be finite or None."
                )
        snapshot = deepcopy(self.metrics)
        json.dumps(snapshot, allow_nan=False)
        object.__setattr__(self, "metrics", snapshot)

    @classmethod
    def disabled(cls) -> ResearchBacktestPipelineResult:
        return cls(enabled=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "artifact_dir": (
                None if self.artifact_dir is None else str(self.artifact_dir)
            ),
            "schema_version": self.schema_version,
            "observation_count": self.observation_count,
            "rebalance_count": self.rebalance_count,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "benchmark_code": self.benchmark_code,
            "metrics": deepcopy(self.metrics),
        }


class ResearchBacktestPipelineExecutor:
    """Chain the existing B1/B2/B3/C/D/E/F components exactly once."""

    def __init__(
        self,
        config: ResearchBacktestPipelineConfig,
        client: object | None = None,
    ) -> None:
        if not isinstance(config, ResearchBacktestPipelineConfig):
            raise ResearchBacktestPipelineExecutionError(
                "config must be a ResearchBacktestPipelineConfig."
            )
        if config.enabled and client is None:
            raise ResearchBacktestPipelineExecutionError(
                "an injected market-data client is required."
            )
        self.config = config
        self.client = client

    def execute(
        self,
        *,
        artifact_dir: object,
        end_date: object,
        holdings_result: HoldingsPipelineResult | None = None,
    ) -> ResearchBacktestPipelineResult:
        if not self.config.enabled:
            return ResearchBacktestPipelineResult.disabled()
        target = _artifact_dir(artifact_dir)
        end = _date(end_date, field_name="end_date")
        assert self.config.transaction_cost is not None
        assert self.config.benchmark is not None
        assert self.config.performance is not None
        try:
            source = ResearchBacktestHoldingsSourceAdapter().load(
                self.config.source,
                holdings_result=holdings_result,
            )
            holdings = source.holdings
            holdings_start = holdings["trade_date"].min()
            if holdings_start > end or holdings["trade_date"].max() >= end:
                raise ResearchBacktestPipelineExecutionError(
                    "every Holdings snapshot must precede explicit end_date."
                )
            coverage_end = end + pd.Timedelta(
                days=_CALENDAR_COVERAGE_BUFFER_DAYS
            )
            calendar = TushareTradingCalendarAdapter(self.client).load(
                start_date=holdings_start,
                end_date=coverage_end,
            )
            if not calendar.is_trading_day(end):
                raise ResearchBacktestPipelineExecutionError(
                    "end_date must be an open trading date."
                )
            effective_dates = tuple(
                calendar.next_trading_day(item)
                for item in holdings["trade_date"].drop_duplicates()
            )
            if any(item > end for item in effective_dates):
                raise ResearchBacktestPipelineExecutionError(
                    "a Holdings effective date falls after explicit end_date."
                )
            evaluation_start = effective_dates[0]
            evaluation_dates = tuple(
                item
                for item in calendar.open_dates
                if evaluation_start <= item <= end
            )
            codes = tuple(sorted(set(holdings["ts_code"])))
            security_returns = TushareSecurityDailyReturnAdapter(
                self.client
            ).load(
                ts_codes=codes,
                start_date=evaluation_start,
                end_date=end,
            )
            lifecycle = TushareSecurityLifecycleAdapter(self.client).load(
                list_statuses=("L", "D", "P")
            )
            if self.config.suspension_mode == "STRICT_EVENT":
                suspensions = TushareSecuritySuspensionAdapter(self.client).load(
                    ts_codes=codes,
                    start_date=evaluation_start,
                    end_date=end,
                )
            else:
                try:
                    suspensions = TushareSecuritySuspensionAdapter(self.client).load(
                        ts_codes=codes,
                        start_date=evaluation_start,
                        end_date=end,
                    )
                except (TypeError, ValueError, OSError):
                    suspensions = pd.DataFrame(columns=("trade_date", "ts_code", "suspend_type", "suspend_timing"))
            security_status = build_security_status(
                ts_codes=codes,
                evaluation_dates=evaluation_dates,
                lifecycle=lifecycle,
                suspensions=suspensions,
                security_returns=security_returns,
                trading_calendar=calendar,
            )
            if self.config.suspension_mode == "STANDARD_ROBUST":
                missing = security_status.loc[security_status["status"].eq("UNKNOWN_MISSING")]
                fractions = missing.groupby("trade_date").size().div(max(1, len(codes)))
                if not fractions.empty and float(fractions.max()) > self.config.max_unexplained_missing_fraction:
                    raise ResearchBacktestPipelineExecutionError(
                        "unexplained missing daily rows exceed the configured market-wide quality threshold."
                    )
            rebalances = RebalanceAccountingEngine(calendar, self.config.suspension_mode).run(
                holdings=holdings,
                security_returns=security_returns,
                security_status=security_status,
            )
            portfolio = PortfolioDailyAccountingEngine(
                calendar,
                self.config.portfolio,
                self.config.transaction_cost,
                self.config.suspension_mode,
            ).run(
                rebalances=rebalances,
                security_returns=security_returns,
                security_status=security_status,
                end_date=end,
            )
            benchmark_returns = TushareBenchmarkDailyReturnAdapter(
                self.client
            ).load(
                benchmark_code=self.config.benchmark.benchmark_code,
                start_date=evaluation_start,
                end_date=end,
            )
            analytics = PerformanceAnalyticsEngine(
                self.config.benchmark,
                self.config.performance,
            ).run(
                portfolio=portfolio,
                benchmark_returns=benchmark_returns,
            )
            written = ResearchBacktestArtifactStore().publish(
                artifact_dir=target,
                rebalances=rebalances,
                portfolio=portfolio,
                analytics=analytics,
                config=self.config,
                holdings_artifact_dir=source.artifact_dir,
            )
        except ResearchBacktestPipelineExecutionError:
            raise
        except (ResearchBacktestArtifactError, TypeError, ValueError, OSError) as exc:
            raise ResearchBacktestPipelineExecutionError(
                f"research backtest execution failed: {type(exc).__name__}."
            ) from exc
        daily = portfolio.daily_portfolio
        return ResearchBacktestPipelineResult(
            enabled=True,
            artifact_dir=written.artifact_dir,
            schema_version=written.schema_version,
            observation_count=written.observation_count,
            rebalance_count=written.rebalance_count,
            start_date=daily["trade_date"].iloc[0].date().isoformat(),
            end_date=daily["trade_date"].iloc[-1].date().isoformat(),
            benchmark_code=written.benchmark_code,
            metrics=analytics.metrics,
        )
