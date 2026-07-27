"""Pipeline configuration dataclass for research runs."""

from __future__ import annotations

import calendar
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.research_config import FactorResearchPipelineConfig
from src.pipeline.ml_config import MLExperimentPipelineConfig


@dataclass
class PipelineConfig:
    """Unified configuration for one pipeline run."""

    backtest_start: str
    backtest_end: str
    train_years: int
    max_lookback_months: int
    stock_pool: str
    benchmark: str
    strategy_name: str
    selected_factors: list[str]
    rebalance_frequency: str
    top_n: int
    transaction_cost: float
    data_root: str
    raw_data_dir: str
    processed_data_dir: str
    cache_dir: str
    output_dir: str
    parquet_engine: str
    required_datasets: list[str]

    factor_research: FactorResearchPipelineConfig = field(
        default_factory=FactorResearchPipelineConfig
    )
    ml_experiment: MLExperimentPipelineConfig = field(
        default_factory=MLExperimentPipelineConfig
    )

    def __post_init__(self) -> None:
        """Normalize dates and validate the backtest range."""
        self.backtest_start = normalize_date(self.backtest_start)
        self.backtest_end = normalize_date(self.backtest_end)
        if isinstance(self.factor_research, Mapping):
            self.factor_research = FactorResearchPipelineConfig.from_dict(
                self.factor_research
            )
        elif not isinstance(self.factor_research, FactorResearchPipelineConfig):
            raise TypeError(
                "factor_research must be a FactorResearchPipelineConfig or Mapping."
            )
        self.ml_experiment = MLExperimentPipelineConfig.from_dict(
            self.ml_experiment
        )
        if parse_date(self.backtest_start) > parse_date(self.backtest_end):
            raise ValueError("backtest_start must be earlier than or equal to backtest_end.")

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path = "config/config.yaml",
        overrides: dict[str, Any] | None = None,
    ) -> "PipelineConfig":
        """Build a PipelineConfig from YAML defaults and optional overrides."""
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Config must be a YAML mapping: {path}")

        merged = merge_overrides(config, overrides or {})
        pipeline = merged.get("pipeline", {}) or {}
        data = merged.get("data", {}) or {}
        strategy = merged.get("strategy", {}) or {}
        factors = merged.get("factors", {}) or {}

        values: dict[str, Any] = {
            "backtest_start": pipeline.get("backtest_start", data.get("start_date")),
            "backtest_end": pipeline.get("backtest_end", data.get("end_date")),
            "train_years": pipeline.get("train_years", 10),
            "max_lookback_months": pipeline.get("max_lookback_months", 12),
            "stock_pool": pipeline.get("stock_pool", data.get("universe", "hs300")),
            "benchmark": pipeline.get("benchmark", data.get("index_code", "000300.SH")),
            "strategy_name": pipeline.get("strategy_name", "score"),
            "selected_factors": factors.get("selected", []),
            "rebalance_frequency": pipeline.get(
                "rebalance_frequency",
                strategy.get("rebalance_freq", "M"),
            ),
            "top_n": pipeline.get("top_n", 20),
            "transaction_cost": pipeline.get(
                "transaction_cost",
                strategy.get("transaction_cost", 0.001),
            ),
            "data_root": data.get("root", "data"),
            "raw_data_dir": data.get("raw_dir", "data/raw"),
            "processed_data_dir": data.get("processed_dir", "data/processed"),
            "cache_dir": data.get("cache_dir", "data/cache"),
            "output_dir": data.get("output_dir", "data/output"),
            "parquet_engine": data.get("parquet_engine", "auto"),
            "required_datasets": data.get(
                "required_datasets",
                ["daily", "daily_basic", "adj_factor"],
            ),
            "factor_research": merged.get("factor_research", {}),
            "ml_experiment": merged.get("ml_experiment"),
        }
        direct_overrides = {
            key: value for key, value in (overrides or {}).items() if key in values
        }
        values.update(direct_overrides)
        return cls(**values)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineConfig":
        """Build from direct dataclass fields, including factor_research."""
        if not isinstance(data, Mapping):
            raise TypeError("PipelineConfig data must be a Mapping.")
        values = dict(data)
        values.pop("required_start_date", None)
        values.pop("required_end_date", None)
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Unknown PipelineConfig keys: " + ", ".join(unknown) + ".")
        raw_research = values.get("factor_research")
        if isinstance(raw_research, Mapping):
            values["factor_research"] = FactorResearchPipelineConfig.from_dict(raw_research)
        values["ml_experiment"] = MLExperimentPipelineConfig.from_dict(
            values.get("ml_experiment")
        )
        return cls(**values)

    @property
    def required_start_date(self) -> str:
        """Return the earliest data date required by the pipeline."""
        start = parse_date(self.backtest_start)
        start = add_years(start, -self.train_years)
        start = add_months(start, -self.max_lookback_months)
        return format_date(start)

    @property
    def required_end_date(self) -> str:
        """Return the latest data date required by the pipeline."""
        return self.backtest_end

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation."""
        result = {
            name: deepcopy(getattr(self, name))
            for name in self.__dataclass_fields__
            if name not in {"factor_research", "ml_experiment"}
        }
        result["factor_research"] = self.factor_research.to_dict()
        result["ml_experiment"] = self.ml_experiment.to_dict()
        result["required_start_date"] = self.required_start_date
        result["required_end_date"] = self.required_end_date
        return result


def normalize_date(value: str) -> str:
    """Normalize YYYY-MM-DD or YYYYMMDD text to YYYY-MM-DD."""
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")
    return format_date(parse_date(text))


def parse_date(value: str) -> date:
    """Parse a YYYY-MM-DD date string."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Date must use YYYY-MM-DD or YYYYMMDD format: {value}") from exc


def format_date(value: date) -> str:
    """Format a date as YYYY-MM-DD."""
    return value.strftime("%Y-%m-%d")


def add_years(value: date, years: int) -> date:
    """Add years to a date, clamping leap-day dates when needed."""
    target_year = value.year + years
    target_day = min(value.day, calendar.monthrange(target_year, value.month)[1])
    return value.replace(year=target_year, day=target_day)


def add_months(value: date, months: int) -> date:
    """Add months to a date, clamping the day to the target month length."""
    month_index = value.year * 12 + value.month - 1 + months
    target_year = month_index // 12
    target_month = month_index % 12 + 1
    target_day = min(value.day, calendar.monthrange(target_year, target_month)[1])
    return date(target_year, target_month, target_day)


def merge_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge nested override dictionaries into a shallow copy of config."""
    merged = dict(config)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        elif key not in PipelineConfig.__dataclass_fields__:
            merged[key] = value
    return merged
