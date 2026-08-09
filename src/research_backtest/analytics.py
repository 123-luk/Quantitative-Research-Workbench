"""Benchmark alignment and performance analytics for canonical V6 results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.research_backtest.benchmark_calendar import validate_strict_common_calendar
from src.research_backtest.portfolio import (
    DAILY_PORTFOLIO_COLUMNS,
    PortfolioDailyAccountingResult,
    PortfolioDailyInputError,
    _portfolio_code,
    _portfolio_date,
    _portfolio_real,
)
from src.research_backtest.rebalance import WEIGHT_TOLERANCE
from src.research_backtest.returns import BENCHMARK_DAILY_RETURN_COLUMNS

if TYPE_CHECKING:
    from src.pipeline.research_backtest_config import BenchmarkConfig, PerformanceConfig


BENCHMARK_DAILY_COLUMNS = (
    "trade_date",
    "benchmark_code",
    "benchmark_return",
    "benchmark_nav",
)
PERFORMANCE_METRIC_KEYS = (
    "observation_count",
    "rebalance_count",
    "gross_total_return",
    "net_total_return",
    "gross_annualized_return",
    "net_annualized_return",
    "net_annualized_volatility",
    "net_sharpe_ratio",
    "net_max_drawdown",
    "benchmark_total_return",
    "benchmark_annualized_return",
    "excess_total_return",
    "annualized_excess_return",
    "tracking_error",
    "information_ratio",
    "average_turnover",
    "total_turnover",
    "total_traded_notional",
    "total_transaction_cost",
    "transaction_cost_return_drag",
)


class PerformanceAnalyticsError(ValueError):
    """Base error for benchmark and performance analytics."""


class PerformanceAnalyticsInputError(PerformanceAnalyticsError):
    """Raised when a D result or canonical benchmark panel is invalid."""


class BenchmarkSeriesError(PerformanceAnalyticsError):
    """Raised when benchmark identity, returns, or NAV are invalid."""


class PerformanceMetricError(PerformanceAnalyticsError):
    """Raised when a metric cannot be represented as finite JSON-safe data."""


def _analytics_date(value: object, *, field_name: str) -> pd.Timestamp:
    try:
        return _portfolio_date(value, field_name=field_name)
    except PortfolioDailyInputError as exc:
        raise PerformanceAnalyticsInputError(str(exc)) from exc


def _analytics_real(value: object, *, field_name: str) -> float:
    try:
        return _portfolio_real(value, field_name=field_name)
    except PortfolioDailyInputError as exc:
        raise PerformanceAnalyticsInputError(str(exc)) from exc


def _canonical_portfolio(
    value: object,
) -> tuple[pd.DataFrame, PortfolioDailyAccountingResult]:
    if not isinstance(value, PortfolioDailyAccountingResult):
        raise PerformanceAnalyticsInputError(
            "portfolio must be a PortfolioDailyAccountingResult."
        )
    rows = value.daily_portfolio
    if rows.empty or tuple(rows.columns) != DAILY_PORTFOLIO_COLUMNS:
        raise PerformanceAnalyticsInputError(
            f"daily_portfolio columns must be {DAILY_PORTFOLIO_COLUMNS!r}."
        )
    rows["trade_date"] = [
        _analytics_date(item, field_name=f"trade_date[{index!r}]")
        for index, item in rows["trade_date"].items()
    ]
    if rows["trade_date"].duplicated().any():
        raise PerformanceAnalyticsInputError("daily_portfolio dates must be unique.")
    rows = rows.sort_values("trade_date", kind="mergesort", ignore_index=True)
    numeric_columns = (
        "gross_return",
        "transaction_cost",
        "net_return",
        "gross_nav",
        "net_nav",
        "turnover",
        "traded_notional",
    )
    for column in numeric_columns:
        rows[column] = [
            _analytics_real(item, field_name=f"{column}[{index!r}]")
            for index, item in rows[column].items()
        ]
    if any(type(item) is not bool for item in rows["is_rebalance"]):
        raise PerformanceAnalyticsInputError("is_rebalance must contain strict bools.")
    if (rows[["gross_nav", "net_nav"]] <= 0.0).any().any():
        raise PerformanceAnalyticsInputError("gross_nav and net_nav must be positive.")
    if (rows[["transaction_cost", "turnover", "traded_notional"]] < 0.0).any().any():
        raise PerformanceAnalyticsInputError(
            "cost, turnover, and traded_notional must be non-negative."
        )
    quiet = rows.loc[~rows["is_rebalance"]]
    for column in ("transaction_cost", "turnover", "traded_notional"):
        if not np.isclose(
            quiet[column].to_numpy(dtype=float),
            0.0,
            rtol=0.0,
            atol=WEIGHT_TOLERANCE,
        ).all():
            raise PerformanceAnalyticsInputError(
                f"non-rebalance {column} must be zero."
            )
    if len(rows) != value.row_count:
        raise PerformanceAnalyticsInputError("daily row count disagrees with metadata.")
    if rows["trade_date"].iloc[0] != value.start_date:
        raise PerformanceAnalyticsInputError("start_date disagrees with daily rows.")
    if rows["trade_date"].iloc[-1] != value.end_date:
        raise PerformanceAnalyticsInputError("end_date disagrees with daily rows.")
    rebalance_count = int(rows["is_rebalance"].sum())
    if rebalance_count != value.rebalance_count:
        raise PerformanceAnalyticsInputError(
            "rebalance_count disagrees with daily rows."
        )
    initial_nav = _analytics_real(value.initial_nav, field_name="initial_nav")
    if initial_nav <= 0.0:
        raise PerformanceAnalyticsInputError("initial_nav must be positive.")
    return rows, value


def _canonical_benchmark(
    value: object,
    *,
    code: str,
    strategy_dates: pd.Series,
    initial_nav: float,
) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise PerformanceAnalyticsInputError(
            "benchmark_returns must be a pandas DataFrame."
        )
    if tuple(value.columns) != BENCHMARK_DAILY_RETURN_COLUMNS:
        raise PerformanceAnalyticsInputError(
            f"benchmark_returns columns must be {BENCHMARK_DAILY_RETURN_COLUMNS!r}."
        )
    rows = value.copy(deep=True)
    rows["trade_date"] = [
        _analytics_date(item, field_name=f"benchmark.trade_date[{index!r}]")
        for index, item in rows["trade_date"].items()
    ]
    rows["benchmark_code"] = [
        _portfolio_code(item, field_name=f"benchmark_code[{index!r}]")
        for index, item in rows["benchmark_code"].items()
    ]
    rows["return"] = [
        _analytics_real(item, field_name=f"benchmark.return[{index!r}]")
        for index, item in rows["return"].items()
    ]
    if rows.duplicated(["trade_date", "benchmark_code"]).any():
        raise PerformanceAnalyticsInputError("benchmark return keys must be unique.")
    if tuple(sorted(set(rows["benchmark_code"]))) != (code,):
        raise BenchmarkSeriesError(
            f"benchmark panel must contain only explicit code {code!r}."
        )
    if rows["return"].lt(-1.0).any():
        raise BenchmarkSeriesError("benchmark return cannot be below -1.")
    start = strategy_dates.iloc[0]
    end = strategy_dates.iloc[-1]
    rows = rows.loc[rows["trade_date"].between(start, end)].copy(deep=True)
    rows = rows.sort_values("trade_date", kind="mergesort", ignore_index=True)
    validate_strict_common_calendar(strategy_dates, rows["trade_date"])

    output: list[dict[str, object]] = []
    benchmark_nav = initial_nav
    for index, row in enumerate(rows.to_dict("records")):
        benchmark_return = 0.0 if index == 0 else float(row["return"])
        factor = 1.0 + benchmark_return
        if not np.isfinite(factor) or factor <= 0.0:
            raise BenchmarkSeriesError(
                "benchmark return factor and NAV must remain positive."
            )
        benchmark_nav *= factor
        if not np.isfinite(benchmark_nav) or benchmark_nav <= 0.0:
            raise BenchmarkSeriesError("benchmark NAV must remain finite and positive.")
        output.append(
            {
                "trade_date": row["trade_date"],
                "benchmark_code": code,
                "benchmark_return": benchmark_return,
                "benchmark_nav": benchmark_nav,
            }
        )
    return pd.DataFrame(output, columns=list(BENCHMARK_DAILY_COLUMNS))


def _annualized(total_return: float, observations: int, periods: int) -> float:
    base = 1.0 + total_return
    if base <= 0.0:
        raise PerformanceMetricError("annualized return base must be positive.")
    try:
        value = base ** (periods / observations) - 1.0
    except OverflowError as exc:
        raise PerformanceMetricError("annualized return must be finite.") from exc
    if not np.isfinite(value):
        raise PerformanceMetricError("annualized return must be finite.")
    return float(value)


def _optional_sample_std(values: np.ndarray, periods: int) -> float | None:
    if len(values) < 2:
        return None
    result = float(np.std(values, ddof=1) * sqrt(periods))
    if not np.isfinite(result):
        raise PerformanceMetricError("annualized sample deviation must be finite.")
    return result


def _metrics(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    result: PortfolioDailyAccountingResult,
    *,
    annualization_days: int,
    annual_risk_free_rate: float,
) -> dict[str, int | float | None]:
    observations = len(daily)
    initial_nav = float(result.initial_nav)
    gross_total = float(daily["gross_nav"].iloc[-1] / initial_nav - 1.0)
    net_total = float(daily["net_nav"].iloc[-1] / initial_nav - 1.0)
    benchmark_total = float(
        benchmark["benchmark_nav"].iloc[-1] / initial_nav - 1.0
    )
    gross_annualized = _annualized(
        gross_total, observations, annualization_days
    )
    net_annualized = _annualized(net_total, observations, annualization_days)
    benchmark_annualized = _annualized(
        benchmark_total, observations, annualization_days
    )
    net_returns = daily["net_return"].to_numpy(dtype=float)
    benchmark_returns = benchmark["benchmark_return"].to_numpy(dtype=float)
    volatility = _optional_sample_std(net_returns, annualization_days)
    if volatility is not None and np.isclose(
        volatility, 0.0, rtol=0.0, atol=WEIGHT_TOLERANCE
    ):
        volatility = 0.0
    if volatility is None or volatility == 0.0:
        sharpe: float | None = None
    else:
        daily_rf = annual_risk_free_rate / annualization_days
        sharpe = float(
            np.mean(net_returns - daily_rf)
            / np.std(net_returns, ddof=1)
            * sqrt(annualization_days)
        )

    nav_path = np.concatenate(([initial_nav], daily["net_nav"].to_numpy(dtype=float)))
    running_high = np.maximum.accumulate(nav_path)
    max_drawdown = float(np.min(nav_path / running_high - 1.0))
    active = net_returns - benchmark_returns
    tracking_error = _optional_sample_std(active, annualization_days)
    if tracking_error is not None and np.isclose(
        tracking_error, 0.0, rtol=0.0, atol=WEIGHT_TOLERANCE
    ):
        tracking_error = 0.0
    if tracking_error is None or tracking_error == 0.0:
        information_ratio: float | None = None
    else:
        information_ratio = float(
            np.mean(active) * annualization_days / tracking_error
        )
    event_rows = daily.loc[daily["is_rebalance"]]
    rebalance_count = len(event_rows)
    metrics: dict[str, int | float | None] = {
        "observation_count": int(observations),
        "rebalance_count": int(rebalance_count),
        "gross_total_return": gross_total,
        "net_total_return": net_total,
        "gross_annualized_return": gross_annualized,
        "net_annualized_return": net_annualized,
        "net_annualized_volatility": volatility,
        "net_sharpe_ratio": sharpe,
        "net_max_drawdown": max_drawdown,
        "benchmark_total_return": benchmark_total,
        "benchmark_annualized_return": benchmark_annualized,
        "excess_total_return": float(net_total - benchmark_total),
        "annualized_excess_return": float(
            net_annualized - benchmark_annualized
        ),
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "average_turnover": (
            None
            if rebalance_count == 0
            else float(event_rows["turnover"].mean())
        ),
        "total_turnover": float(event_rows["turnover"].sum()),
        "total_traded_notional": float(event_rows["traded_notional"].sum()),
        "total_transaction_cost": float(event_rows["transaction_cost"].sum()),
        "transaction_cost_return_drag": float(gross_total - net_total),
    }
    if tuple(metrics) != PERFORMANCE_METRIC_KEYS:
        raise PerformanceMetricError("performance metric keys are inconsistent.")
    for name, value in metrics.items():
        if value is not None and not isinstance(value, (int, float)):
            raise PerformanceMetricError(f"metric {name!r} is not JSON-safe.")
        if isinstance(value, float) and not np.isfinite(value):
            raise PerformanceMetricError(f"metric {name!r} must be finite or None.")
    json.dumps(metrics, allow_nan=False)
    return metrics


class PerformanceAnalyticsResult:
    """Defensively expose benchmark accounting and JSON-safe metrics."""

    __slots__ = (
        "_benchmark_daily",
        "_metrics",
        "_start_date",
        "_end_date",
        "_benchmark_code",
    )

    def __init__(
        self,
        benchmark_daily: pd.DataFrame,
        metrics: dict[str, int | float | None],
        *,
        benchmark_code: str,
    ) -> None:
        if (
            not isinstance(benchmark_daily, pd.DataFrame)
            or benchmark_daily.empty
            or tuple(benchmark_daily.columns) != BENCHMARK_DAILY_COLUMNS
        ):
            raise PerformanceAnalyticsInputError("benchmark_daily has invalid schema.")
        if tuple(metrics) != PERFORMANCE_METRIC_KEYS:
            raise PerformanceAnalyticsInputError("metrics have invalid keys.")
        self._benchmark_daily = benchmark_daily.copy(deep=True)
        self._metrics = deepcopy(metrics)
        self._start_date = benchmark_daily["trade_date"].iloc[0]
        self._end_date = benchmark_daily["trade_date"].iloc[-1]
        self._benchmark_code = benchmark_code

    @property
    def benchmark_daily(self) -> pd.DataFrame:
        return self._benchmark_daily.copy(deep=True)

    @property
    def metrics(self) -> dict[str, int | float | None]:
        return deepcopy(self._metrics)

    @property
    def start_date(self) -> pd.Timestamp:
        return self._start_date

    @property
    def end_date(self) -> pd.Timestamp:
        return self._end_date

    @property
    def observation_count(self) -> int:
        return int(self._metrics["observation_count"])

    @property
    def benchmark_code(self) -> str:
        return self._benchmark_code


@dataclass(frozen=True)
class PerformanceAnalyticsEngine:
    """Align one explicit benchmark and compute canonical daily metrics."""

    benchmark_config: BenchmarkConfig
    performance_config: PerformanceConfig

    def __post_init__(self) -> None:
        from src.pipeline.research_backtest_config import (
            BenchmarkConfig,
            PerformanceConfig,
        )

        object.__setattr__(
            self,
            "benchmark_config",
            BenchmarkConfig.from_dict(self.benchmark_config),
        )
        object.__setattr__(
            self,
            "performance_config",
            PerformanceConfig.from_dict(self.performance_config),
        )

    def run(
        self,
        *,
        portfolio: PortfolioDailyAccountingResult,
        benchmark_returns: pd.DataFrame,
    ) -> PerformanceAnalyticsResult:
        daily, result = _canonical_portfolio(portfolio)
        benchmark = _canonical_benchmark(
            benchmark_returns,
            code=self.benchmark_config.benchmark_code,
            strategy_dates=daily["trade_date"],
            initial_nav=result.initial_nav,
        )
        metrics = _metrics(
            daily,
            benchmark,
            result,
            annualization_days=self.performance_config.annualization_days,
            annual_risk_free_rate=self.performance_config.annual_risk_free_rate,
        )
        return PerformanceAnalyticsResult(
            benchmark,
            metrics,
            benchmark_code=self.benchmark_config.benchmark_code,
        )
