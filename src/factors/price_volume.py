"""Core daily price-volume factors for the V2-C1 research library."""

from __future__ import annotations

import pandas as pd

from src.factors.base import FactorMetadata, FunctionFactor
from src.factors.registry import FactorRegistry


def _close(data: pd.DataFrame) -> pd.Series:
    """Return close values as numeric data without changing the input."""
    return pd.to_numeric(data["close"], errors="coerce")


def _compute_momentum_60d(data: pd.DataFrame) -> pd.Series:
    close = _close(data)
    return close / close.shift(60) - 1.0


def _compute_momentum_120d(data: pd.DataFrame) -> pd.Series:
    close = _close(data)
    return close / close.shift(120) - 1.0


def _compute_momentum_252_20d(data: pd.DataFrame) -> pd.Series:
    close = _close(data)
    return close.shift(20) / close.shift(252) - 1.0


def _compute_short_term_reversal_5d(data: pd.DataFrame) -> pd.Series:
    close = _close(data)
    return -(close / close.shift(5) - 1.0)


def _compute_price_52w_high(data: pd.DataFrame) -> pd.Series:
    close = _close(data)
    trailing_high = close.rolling(window=252, min_periods=252, center=False).max()
    return close / trailing_high


def _compute_volatility_60d(data: pd.DataFrame) -> pd.Series:
    close = _close(data)
    returns = close.pct_change(fill_method=None)
    return returns.rolling(window=60, min_periods=60, center=False).std()


def _compute_turnover_mean_20d(data: pd.DataFrame) -> pd.Series:
    """Average turnover in the input's original percentage/unit convention."""
    turnover = pd.to_numeric(data["turnover_rate"], errors="coerce")
    return turnover.rolling(window=20, min_periods=20, center=False).mean()


def _compute_amihud_20d(data: pd.DataFrame) -> pd.Series:
    """Calculate Amihud illiquidity; scale follows the input amount unit."""
    close = _close(data)
    amount = pd.to_numeric(data["amount"], errors="coerce")
    if (amount < 0).any():
        raise ValueError("Factor 'amihud_20d' requires non-negative amount values.")
    safe_amount = amount.mask(amount == 0)
    daily_illiquidity = close.pct_change(fill_method=None).abs() / safe_amount
    return daily_illiquidity.rolling(
        window=20,
        min_periods=20,
        center=False,
    ).mean()


MOMENTUM_60D = FunctionFactor(
    FactorMetadata(
        name="momentum_60d",
        category="momentum",
        direction=1,
        required_datasets=("daily",),
        source_fields=("close",),
        lookback_days=60,
        frequency="daily",
        availability_lag_days=0,
        description="Trailing 60-day close-price momentum.",
        version="1.0",
    ),
    _compute_momentum_60d,
)

MOMENTUM_120D = FunctionFactor(
    FactorMetadata(
        name="momentum_120d",
        category="momentum",
        direction=1,
        required_datasets=("daily",),
        source_fields=("close",),
        lookback_days=120,
        frequency="daily",
        availability_lag_days=0,
        description="Trailing 120-day close-price momentum.",
        version="1.0",
    ),
    _compute_momentum_120d,
)

MOMENTUM_252_20D = FunctionFactor(
    FactorMetadata(
        name="momentum_252_20d",
        category="momentum",
        direction=1,
        required_datasets=("daily",),
        source_fields=("close",),
        lookback_days=252,
        frequency="daily",
        availability_lag_days=0,
        description="Approximate 12-month momentum excluding the latest 20 days.",
        version="1.0",
    ),
    _compute_momentum_252_20d,
)

SHORT_TERM_REVERSAL_5D = FunctionFactor(
    FactorMetadata(
        name="short_term_reversal_5d",
        category="reversal",
        direction=1,
        required_datasets=("daily",),
        source_fields=("close",),
        lookback_days=5,
        frequency="daily",
        availability_lag_days=0,
        description="Negative of the trailing 5-day close return.",
        version="1.0",
    ),
    _compute_short_term_reversal_5d,
)

PRICE_52W_HIGH = FunctionFactor(
    FactorMetadata(
        name="price_52w_high",
        category="momentum",
        direction=1,
        required_datasets=("daily",),
        source_fields=("close",),
        lookback_days=252,
        frequency="daily",
        availability_lag_days=0,
        description="Close divided by the trailing 252-day high including today.",
        version="1.0",
    ),
    _compute_price_52w_high,
)

VOLATILITY_60D = FunctionFactor(
    FactorMetadata(
        name="volatility_60d",
        category="volatility",
        direction=-1,
        required_datasets=("daily",),
        source_fields=("close",),
        lookback_days=60,
        frequency="daily",
        availability_lag_days=0,
        description="Trailing 60-day volatility of close returns.",
        version="1.0",
    ),
    _compute_volatility_60d,
)

TURNOVER_MEAN_20D = FunctionFactor(
    FactorMetadata(
        name="turnover_mean_20d",
        category="liquidity",
        direction=1,
        required_datasets=("daily_basic",),
        source_fields=("turnover_rate",),
        lookback_days=20,
        frequency="daily",
        availability_lag_days=0,
        description="Trailing 20-day mean turnover in the input's original unit.",
        version="1.0",
    ),
    _compute_turnover_mean_20d,
)

AMIHUD_20D = FunctionFactor(
    FactorMetadata(
        name="amihud_20d",
        category="liquidity",
        direction=-1,
        required_datasets=("daily",),
        source_fields=("close", "amount"),
        lookback_days=20,
        frequency="daily",
        availability_lag_days=0,
        description=(
            "Trailing 20-day mean absolute return divided by amount; result scale "
            "depends on the input amount unit."
        ),
        version="1.0",
    ),
    _compute_amihud_20d,
)


PRICE_VOLUME_FACTORS = (
    MOMENTUM_60D,
    MOMENTUM_120D,
    MOMENTUM_252_20D,
    SHORT_TERM_REVERSAL_5D,
    PRICE_52W_HIGH,
    VOLATILITY_60D,
    TURNOVER_MEAN_20D,
    AMIHUD_20D,
)


def register_price_volume_factors(registry: FactorRegistry) -> FactorRegistry:
    """Register the eight V2-C1 price-volume factors in the supplied registry."""
    for factor in PRICE_VOLUME_FACTORS:
        registry.register(factor)
    return registry
