"""Minimal example factors demonstrating the registry calculation contract."""

from __future__ import annotations

import pandas as pd

from src.factors.base import FactorMetadata, FunctionFactor
from src.factors.registry import FactorRegistry


def _compute_momentum_20d(data: pd.DataFrame) -> pd.Series:
    """Calculate trailing 20-row close-price momentum without future data."""
    close = pd.to_numeric(data["close"], errors="coerce")
    return close / close.shift(20) - 1.0


def _compute_volatility_20d(data: pd.DataFrame) -> pd.Series:
    """Calculate trailing 20-row standard deviation of close returns."""
    close = pd.to_numeric(data["close"], errors="coerce")
    returns = close.pct_change(fill_method=None)
    return returns.rolling(window=20, min_periods=20, center=False).std()


MOMENTUM_20D = FunctionFactor(
    metadata=FactorMetadata(
        name="momentum_20d",
        category="momentum",
        direction=1,
        required_datasets=("daily",),
        source_fields=("close",),
        lookback_days=20,
        frequency="daily",
        availability_lag_days=0,
        description="Trailing 20-day close-price momentum.",
        version="1.0",
    ),
    function=_compute_momentum_20d,
)


VOLATILITY_20D = FunctionFactor(
    metadata=FactorMetadata(
        name="volatility_20d",
        category="volatility",
        direction=-1,
        required_datasets=("daily",),
        source_fields=("close",),
        lookback_days=20,
        frequency="daily",
        availability_lag_days=0,
        description="Trailing 20-day volatility of close returns.",
        version="1.0",
    ),
    function=_compute_volatility_20d,
)


def register_example_factors(registry: FactorRegistry) -> FactorRegistry:
    """Register only the V2-A momentum and volatility examples."""
    registry.register(MOMENTUM_20D)
    registry.register(VOLATILITY_20D)
    return registry
