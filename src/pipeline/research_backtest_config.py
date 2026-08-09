"""Strict standalone configuration for the future V6 research backtest stage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from numbers import Real
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeVar

from src.research_backtest import (
    ResearchBacktestContractError,
    validate_benchmark_alignment_policy,
    validate_cost_rate_basis,
    validate_effective_date_rule,
    validate_return_convention,
    validate_schedule_mode,
    validate_source_mode,
    validate_turnover_definition,
)


class ResearchBacktestConfigError(ValueError):
    """Raised when standalone research-backtest configuration is invalid."""


_ConfigT = TypeVar("_ConfigT")


def _strict_mapping(
    value: object, allowed: frozenset[str], context: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ResearchBacktestConfigError(f"{context} must be a Mapping.")
    if any(not isinstance(key, str) for key in value):
        raise ResearchBacktestConfigError(f"{context} field names must be strings.")
    values = deepcopy(dict(value))
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ResearchBacktestConfigError(
            f"{context} contains unknown fields: {unknown!r}."
        )
    return values


def _construct(
    cls: Callable[..., _ConfigT], values: dict[str, object], context: str
) -> _ConfigT:
    try:
        return cls(**values)
    except ResearchBacktestConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ResearchBacktestConfigError(f"{context} configuration is invalid.") from exc


def _choice(validator: Callable[[object], str], value: object, context: str) -> str:
    try:
        return validator(value)
    except ResearchBacktestContractError as exc:
        raise ResearchBacktestConfigError(f"{context}: {exc}") from exc


def _optional_artifact_dir(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise ResearchBacktestConfigError(
            "artifact_dir must be a str, os.PathLike, or None."
        )
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ResearchBacktestConfigError("artifact_dir must be path-like.") from exc
    if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
        raise ResearchBacktestConfigError(
            "artifact_dir must be a non-empty trimmed path."
        )
    return Path(raw)


def _safe_artifact_subdir(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ResearchBacktestConfigError(
            "artifact_subdir must be a non-empty trimmed string."
        )
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
        or "://" in value
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or bool(PureWindowsPath(value).drive)
    ):
        raise ResearchBacktestConfigError(
            "artifact_subdir must be one safe relative directory name."
        )
    return value


def _finite_real(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ResearchBacktestConfigError(f"{field_name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ResearchBacktestConfigError(f"{field_name} must be finite.")
    return normalized


@dataclass(frozen=True)
class BacktestSourceConfig:
    """Select a current-run Holdings handoff or an explicit native Artifact."""

    mode: str = "pipeline"
    artifact_dir: Path | None = None

    def __post_init__(self) -> None:
        mode = _choice(validate_source_mode, self.mode, "research_backtest.source")
        artifact_dir = _optional_artifact_dir(self.artifact_dir)
        if mode == "pipeline" and artifact_dir is not None:
            raise ResearchBacktestConfigError(
                "pipeline source must not configure artifact_dir."
            )
        if mode == "files" and artifact_dir is None:
            raise ResearchBacktestConfigError(
                "files source requires an explicit native Holdings artifact_dir."
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "artifact_dir", artifact_dir)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | BacktestSourceConfig | None
    ) -> BacktestSourceConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value, frozenset({"mode", "artifact_dir"}), "research_backtest.source"
        )
        return _construct(cls, values, "research_backtest.source")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "artifact_dir": None if self.artifact_dir is None else str(self.artifact_dir),
        }


@dataclass(frozen=True)
class BacktestScheduleConfig:
    """Declare that ordered Holdings snapshots own rebalance events."""

    mode: str = "holdings_dates"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _choice(validate_schedule_mode, self.mode, "research_backtest.schedule"),
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | BacktestScheduleConfig | None
    ) -> BacktestScheduleConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value, frozenset({"mode"}), "research_backtest.schedule"
        )
        return _construct(cls, values, "research_backtest.schedule")

    def to_dict(self) -> dict[str, object]:
        return {"mode": self.mode}


@dataclass(frozen=True)
class ReturnAlignmentConfig:
    """Freeze research timing and return intent, not provider implementation."""

    effective_rule: str = "next_trading_day"
    return_convention: str = "adjusted_close_to_close"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "effective_rule",
            _choice(
                validate_effective_date_rule,
                self.effective_rule,
                "research_backtest.return_alignment",
            ),
        )
        object.__setattr__(
            self,
            "return_convention",
            _choice(
                validate_return_convention,
                self.return_convention,
                "research_backtest.return_alignment",
            ),
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | ReturnAlignmentConfig | None
    ) -> ReturnAlignmentConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value,
            frozenset({"effective_rule", "return_convention"}),
            "research_backtest.return_alignment",
        )
        return _construct(cls, values, "research_backtest.return_alignment")

    def to_dict(self) -> dict[str, object]:
        return {
            "effective_rule": self.effective_rule,
            "return_convention": self.return_convention,
        }


@dataclass(frozen=True)
class PortfolioAccountingConfig:
    """Configure initial value and complete-state turnover vocabulary."""

    initial_nav: float = 1.0
    turnover_definition: str = "half_l1_pre_to_target"

    def __post_init__(self) -> None:
        initial_nav = _finite_real(self.initial_nav, "initial_nav")
        if initial_nav <= 0:
            raise ResearchBacktestConfigError("initial_nav must be > 0.")
        object.__setattr__(self, "initial_nav", initial_nav)
        object.__setattr__(
            self,
            "turnover_definition",
            _choice(
                validate_turnover_definition,
                self.turnover_definition,
                "research_backtest.portfolio",
            ),
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | PortfolioAccountingConfig | None
    ) -> PortfolioAccountingConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value,
            frozenset({"initial_nav", "turnover_definition"}),
            "research_backtest.portfolio",
        )
        return _construct(cls, values, "research_backtest.portfolio")

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_nav": self.initial_nav,
            "turnover_definition": self.turnover_definition,
        }


@dataclass(frozen=True)
class TransactionCostConfig:
    """Configure proportional research friction on one-way traded notional.

    A future engine applies ``traded_notional * cost_bps / 10000``. Zero bps
    is the explicit no-cost assumption; no separate switch is needed.
    """

    cost_bps: float
    rate_basis: str = "one_way_traded_notional"

    def __post_init__(self) -> None:
        cost_bps = _finite_real(self.cost_bps, "cost_bps")
        if cost_bps < 0:
            raise ResearchBacktestConfigError("cost_bps must be >= 0.")
        object.__setattr__(self, "cost_bps", cost_bps)
        object.__setattr__(
            self,
            "rate_basis",
            _choice(
                validate_cost_rate_basis,
                self.rate_basis,
                "research_backtest.transaction_cost",
            ),
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | TransactionCostConfig
    ) -> TransactionCostConfig:
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value,
            frozenset({"cost_bps", "rate_basis"}),
            "research_backtest.transaction_cost",
        )
        if "cost_bps" not in values:
            raise ResearchBacktestConfigError(
                "research_backtest.transaction_cost requires cost_bps."
            )
        return _construct(cls, values, "research_backtest.transaction_cost")

    def to_dict(self) -> dict[str, object]:
        return {"cost_bps": self.cost_bps, "rate_basis": self.rate_basis}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configure an explicit benchmark identity and strict calendar alignment."""

    benchmark_code: str
    alignment_policy: str = "strict_common_calendar"

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_code, str):
            raise ResearchBacktestConfigError("benchmark_code must be a string.")
        code = self.benchmark_code.strip()
        if not code:
            raise ResearchBacktestConfigError("benchmark_code must be non-empty.")
        object.__setattr__(self, "benchmark_code", code)
        object.__setattr__(
            self,
            "alignment_policy",
            _choice(
                validate_benchmark_alignment_policy,
                self.alignment_policy,
                "research_backtest.benchmark",
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object] | BenchmarkConfig) -> BenchmarkConfig:
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value,
            frozenset({"benchmark_code", "alignment_policy"}),
            "research_backtest.benchmark",
        )
        if "benchmark_code" not in values:
            raise ResearchBacktestConfigError(
                "research_backtest.benchmark requires benchmark_code."
            )
        return _construct(cls, values, "research_backtest.benchmark")

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_code": self.benchmark_code,
            "alignment_policy": self.alignment_policy,
        }


@dataclass(frozen=True)
class PerformanceConfig:
    """Configure annual assumptions for analytics over daily returns."""

    annual_risk_free_rate: float
    annualization_days: int = 252

    def __post_init__(self) -> None:
        if type(self.annualization_days) is not int:
            raise ResearchBacktestConfigError(
                "annualization_days must be a strict int."
            )
        if self.annualization_days < 1:
            raise ResearchBacktestConfigError("annualization_days must be >= 1.")
        object.__setattr__(
            self,
            "annual_risk_free_rate",
            _finite_real(self.annual_risk_free_rate, "annual_risk_free_rate"),
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | PerformanceConfig
    ) -> PerformanceConfig:
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value,
            frozenset({"annualization_days", "annual_risk_free_rate"}),
            "research_backtest.performance",
        )
        if "annual_risk_free_rate" not in values:
            raise ResearchBacktestConfigError(
                "research_backtest.performance requires annual_risk_free_rate."
            )
        return _construct(cls, values, "research_backtest.performance")

    def to_dict(self) -> dict[str, object]:
        return {
            "annualization_days": self.annualization_days,
            "annual_risk_free_rate": self.annual_risk_free_rate,
        }


@dataclass(frozen=True)
class ResearchBacktestPipelineConfig:
    """Standalone V6 config; top-level PipelineConfig integration comes later."""

    enabled: bool = False
    source: BacktestSourceConfig = field(default_factory=BacktestSourceConfig)
    schedule: BacktestScheduleConfig = field(default_factory=BacktestScheduleConfig)
    return_alignment: ReturnAlignmentConfig = field(
        default_factory=ReturnAlignmentConfig
    )
    portfolio: PortfolioAccountingConfig = field(
        default_factory=PortfolioAccountingConfig
    )
    transaction_cost: TransactionCostConfig | None = None
    benchmark: BenchmarkConfig | None = None
    performance: PerformanceConfig | None = None
    artifact_subdir: str = "research_backtest"

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ResearchBacktestConfigError("enabled must be a bool.")
        source = BacktestSourceConfig.from_dict(self.source)
        schedule = BacktestScheduleConfig.from_dict(self.schedule)
        return_alignment = ReturnAlignmentConfig.from_dict(self.return_alignment)
        portfolio = PortfolioAccountingConfig.from_dict(self.portfolio)
        transaction_cost = (
            None
            if self.transaction_cost is None
            else TransactionCostConfig.from_dict(self.transaction_cost)
        )
        benchmark = (
            None if self.benchmark is None else BenchmarkConfig.from_dict(self.benchmark)
        )
        performance = (
            None
            if self.performance is None
            else PerformanceConfig.from_dict(self.performance)
        )
        if self.enabled:
            missing = [
                name
                for name, item in (
                    ("transaction_cost", transaction_cost),
                    ("benchmark", benchmark),
                    ("performance", performance),
                )
                if item is None
            ]
            if missing:
                raise ResearchBacktestConfigError(
                    "enabled research_backtest requires explicit business assumptions: "
                    f"{missing!r}."
                )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "return_alignment", return_alignment)
        object.__setattr__(self, "portfolio", portfolio)
        object.__setattr__(self, "transaction_cost", transaction_cost)
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "performance", performance)
        object.__setattr__(
            self, "artifact_subdir", _safe_artifact_subdir(self.artifact_subdir)
        )
        json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object] | ResearchBacktestPipelineConfig | None,
    ) -> ResearchBacktestPipelineConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        values = _strict_mapping(
            value,
            frozenset(
                {
                    "enabled",
                    "source",
                    "schedule",
                    "return_alignment",
                    "portfolio",
                    "transaction_cost",
                    "benchmark",
                    "performance",
                    "artifact_subdir",
                }
            ),
            "research_backtest",
        )
        nested: dict[str, Callable[[Any], object]] = {
            "source": BacktestSourceConfig.from_dict,
            "schedule": BacktestScheduleConfig.from_dict,
            "return_alignment": ReturnAlignmentConfig.from_dict,
            "portfolio": PortfolioAccountingConfig.from_dict,
            "transaction_cost": TransactionCostConfig.from_dict,
            "benchmark": BenchmarkConfig.from_dict,
            "performance": PerformanceConfig.from_dict,
        }
        for name, parser in nested.items():
            if name in values and values[name] is not None:
                values[name] = parser(values[name])
        return _construct(cls, values, "research_backtest")

    def to_dict(self) -> dict[str, object]:
        result = {
            "enabled": self.enabled,
            "source": self.source.to_dict(),
            "schedule": self.schedule.to_dict(),
            "return_alignment": self.return_alignment.to_dict(),
            "portfolio": self.portfolio.to_dict(),
            "transaction_cost": (
                None if self.transaction_cost is None else self.transaction_cost.to_dict()
            ),
            "benchmark": None if self.benchmark is None else self.benchmark.to_dict(),
            "performance": (
                None if self.performance is None else self.performance.to_dict()
            ),
            "artifact_subdir": self.artifact_subdir,
        }
        return deepcopy(result)
