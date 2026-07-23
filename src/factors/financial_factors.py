"""Core factors from standardized point-in-time financial fields.

This module does not determine whether an announcement was available. Upstream
code must first run ``FinancialPointInTimeAligner`` or an equivalent PIT process
and retain ``source_ann_date`` and ``source_end_date`` for audit. Each ``fin_*``
input must represent information available on the current ``trade_date``.

The factor functions never fill values, inspect future announcements, or infer
availability from ``end_date``. ``availability_lag_days=0`` means announcement
lag was already handled upstream; it does not authorize future information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import FactorMetadata, FunctionFactor
from src.factors.registry import FactorRegistry


def _standardized_financial_field(
    data: pd.DataFrame,
    field_name: str,
) -> pd.Series:
    """Return finite numeric values and NaN without changing units or signs."""
    values = pd.to_numeric(data[field_name], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def _compute_roe_ttm(data: pd.DataFrame) -> pd.Series:
    return _standardized_financial_field(data, "fin_roe_ttm")


def _compute_roa_ttm(data: pd.DataFrame) -> pd.Series:
    return _standardized_financial_field(data, "fin_roa_ttm")


def _compute_gross_margin_ttm(data: pd.DataFrame) -> pd.Series:
    return _standardized_financial_field(data, "fin_gross_margin_ttm")


def _compute_net_margin_ttm(data: pd.DataFrame) -> pd.Series:
    return _standardized_financial_field(data, "fin_net_margin_ttm")


def _compute_revenue_yoy(data: pd.DataFrame) -> pd.Series:
    return _standardized_financial_field(data, "fin_revenue_yoy")


def _compute_net_profit_yoy(data: pd.DataFrame) -> pd.Series:
    return _standardized_financial_field(data, "fin_net_profit_yoy")


def _compute_debt_to_assets(data: pd.DataFrame) -> pd.Series:
    return _standardized_financial_field(data, "fin_debt_to_assets")


def _compute_operating_cf_to_assets(data: pd.DataFrame) -> pd.Series:
    return _standardized_financial_field(data, "fin_operating_cf_to_assets")


ROE_TTM = FunctionFactor(
    FactorMetadata(
        name="roe_ttm",
        category="profitability",
        direction=1,
        required_datasets=("financial_pit",),
        source_fields=("fin_roe_ttm",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description="Standardized PIT-aligned ROE TTM in its original unit.",
        version="1.0",
    ),
    _compute_roe_ttm,
)

ROA_TTM = FunctionFactor(
    FactorMetadata(
        name="roa_ttm",
        category="profitability",
        direction=1,
        required_datasets=("financial_pit",),
        source_fields=("fin_roa_ttm",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description="Standardized PIT-aligned ROA TTM in its original unit.",
        version="1.0",
    ),
    _compute_roa_ttm,
)

GROSS_MARGIN_TTM = FunctionFactor(
    FactorMetadata(
        name="gross_margin_ttm",
        category="profitability",
        direction=1,
        required_datasets=("financial_pit",),
        source_fields=("fin_gross_margin_ttm",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description="Standardized PIT-aligned gross margin TTM in its original unit.",
        version="1.0",
    ),
    _compute_gross_margin_ttm,
)

NET_MARGIN_TTM = FunctionFactor(
    FactorMetadata(
        name="net_margin_ttm",
        category="profitability",
        direction=1,
        required_datasets=("financial_pit",),
        source_fields=("fin_net_margin_ttm",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description="Standardized PIT-aligned net margin TTM in its original unit.",
        version="1.0",
    ),
    _compute_net_margin_ttm,
)

REVENUE_YOY = FunctionFactor(
    FactorMetadata(
        name="revenue_yoy",
        category="growth",
        direction=1,
        required_datasets=("financial_pit",),
        source_fields=("fin_revenue_yoy",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description="Standardized PIT-aligned revenue growth in its original unit.",
        version="1.0",
    ),
    _compute_revenue_yoy,
)

NET_PROFIT_YOY = FunctionFactor(
    FactorMetadata(
        name="net_profit_yoy",
        category="growth",
        direction=1,
        required_datasets=("financial_pit",),
        source_fields=("fin_net_profit_yoy",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description="Standardized PIT-aligned net-profit growth in its original unit.",
        version="1.0",
    ),
    _compute_net_profit_yoy,
)

DEBT_TO_ASSETS = FunctionFactor(
    FactorMetadata(
        name="debt_to_assets",
        category="leverage",
        direction=-1,
        required_datasets=("financial_pit",),
        source_fields=("fin_debt_to_assets",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description="Standardized PIT-aligned leverage in its original unit.",
        version="1.0",
    ),
    _compute_debt_to_assets,
)

OPERATING_CF_TO_ASSETS = FunctionFactor(
    FactorMetadata(
        name="operating_cf_to_assets",
        category="quality",
        direction=1,
        required_datasets=("financial_pit",),
        source_fields=("fin_operating_cf_to_assets",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description=(
            "Standardized PIT-aligned operating cash flow to assets in its "
            "original unit."
        ),
        version="1.0",
    ),
    _compute_operating_cf_to_assets,
)


FINANCIAL_FACTORS = (
    ROE_TTM,
    ROA_TTM,
    GROSS_MARGIN_TTM,
    NET_MARGIN_TTM,
    REVENUE_YOY,
    NET_PROFIT_YOY,
    DEBT_TO_ASSETS,
    OPERATING_CF_TO_ASSETS,
)


def register_financial_factors(registry: FactorRegistry) -> FactorRegistry:
    """Register the eight V2-C2C financial factors explicitly."""
    for factor in FINANCIAL_FACTORS:
        registry.register(factor)
    return registry
