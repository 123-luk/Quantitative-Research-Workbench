"""Core daily valuation and size factors for the V2-C2A library.

``availability_lag_days=0`` means the metadata adds no extra calendar-day
delay. It does not authorize using data obtained after market close to trade at
that same closing price; execution timing belongs to the pipeline and backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import FactorMetadata, FunctionFactor
from src.factors.frequency import point_in_time_frequency_specs
from src.factors.registry import FactorRegistry


def _positive_numeric(data: pd.DataFrame, field_name: str) -> pd.Series:
    """Return numeric values with zero and negative observations masked."""
    values = pd.to_numeric(data[field_name], errors="coerce")
    return values.where(values > 0)


def _compute_ep_ttm(data: pd.DataFrame) -> pd.Series:
    return 1.0 / _positive_numeric(data, "pe_ttm")


def _compute_bp(data: pd.DataFrame) -> pd.Series:
    return 1.0 / _positive_numeric(data, "pb")


def _compute_sp_ttm(data: pd.DataFrame) -> pd.Series:
    return 1.0 / _positive_numeric(data, "ps_ttm")


def _compute_dividend_yield_ttm(data: pd.DataFrame) -> pd.Series:
    """Return dv_ttm in its original input unit, masking negative values."""
    values = pd.to_numeric(data["dv_ttm"], errors="coerce")
    return values.where(values >= 0)


def _compute_log_total_mv(data: pd.DataFrame) -> pd.Series:
    return np.log(_positive_numeric(data, "total_mv"))


def _compute_log_circ_mv(data: pd.DataFrame) -> pd.Series:
    return np.log(_positive_numeric(data, "circ_mv"))


EP_TTM = FunctionFactor(
    FactorMetadata(
        name="ep_ttm",
        category="valuation",
        direction=1,
        required_datasets=("daily_basic",),
        source_fields=("pe_ttm",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description="Earnings yield from positive pe_ttm values: 1 / pe_ttm.",
        version="1.0",
        frequency_specs=point_in_time_frequency_specs(dataset="daily_basic", fields=("pe_ttm",), calculator_id="ep_ttm"),
    ),
    _compute_ep_ttm,
)

BP = FunctionFactor(
    FactorMetadata(
        name="bp",
        category="valuation",
        direction=1,
        required_datasets=("daily_basic",),
        source_fields=("pb",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description="Book-to-price ratio from positive pb values: 1 / pb.",
        version="1.0",
        frequency_specs=point_in_time_frequency_specs(dataset="daily_basic", fields=("pb",), calculator_id="bp"),
    ),
    _compute_bp,
)

SP_TTM = FunctionFactor(
    FactorMetadata(
        name="sp_ttm",
        category="valuation",
        direction=1,
        required_datasets=("daily_basic",),
        source_fields=("ps_ttm",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description="Sales yield from positive ps_ttm values: 1 / ps_ttm.",
        version="1.0",
        frequency_specs=point_in_time_frequency_specs(dataset="daily_basic", fields=("ps_ttm",), calculator_id="sp_ttm"),
    ),
    _compute_sp_ttm,
)

DIVIDEND_YIELD_TTM = FunctionFactor(
    FactorMetadata(
        name="dividend_yield_ttm",
        category="valuation",
        direction=1,
        required_datasets=("daily_basic",),
        source_fields=("dv_ttm",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description="Non-negative dv_ttm retained in the input's original unit.",
        version="1.0",
    ),
    _compute_dividend_yield_ttm,
)

LOG_TOTAL_MV = FunctionFactor(
    FactorMetadata(
        name="log_total_mv",
        category="size",
        direction=-1,
        required_datasets=("daily_basic",),
        source_fields=("total_mv",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description=(
            "Natural log of positive total_mv in its original unit, used only "
            "for relative size comparison."
        ),
        version="1.0",
        frequency_specs=point_in_time_frequency_specs(dataset="daily_basic", fields=("total_mv",), calculator_id="log_total_mv"),
    ),
    _compute_log_total_mv,
)

LOG_CIRC_MV = FunctionFactor(
    FactorMetadata(
        name="log_circ_mv",
        category="size",
        direction=-1,
        required_datasets=("daily_basic",),
        source_fields=("circ_mv",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description=(
            "Natural log of positive circ_mv in its original unit, used only "
            "for relative circulating-market-value comparison."
        ),
        version="1.0",
        frequency_specs=point_in_time_frequency_specs(dataset="daily_basic", fields=("circ_mv",), calculator_id="log_circ_mv"),
    ),
    _compute_log_circ_mv,
)


VALUATION_FACTORS = (
    EP_TTM,
    BP,
    SP_TTM,
    DIVIDEND_YIELD_TTM,
    LOG_TOTAL_MV,
    LOG_CIRC_MV,
)


def register_valuation_factors(registry: FactorRegistry) -> FactorRegistry:
    """Register the six V2-C2A valuation and size factors explicitly."""
    for factor in VALUATION_FACTORS:
        registry.register(factor)
    return registry
