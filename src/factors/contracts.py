"""Data contracts for in-memory factor calculation inputs."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


FACTOR_KEY_FIELDS: Tuple[str, str] = ("trade_date", "ts_code")
DAILY_MARKET_FIELDS: Tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "vol",
    "amount",
)


def validate_required_fields(
    data: pd.DataFrame,
    required_fields: Iterable[str],
) -> None:
    """Raise a clear error when required DataFrame columns are absent."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Factor input must be a pandas DataFrame.")

    missing_fields = sorted(
        field_name for field_name in set(required_fields) if field_name not in data.columns
    )
    if missing_fields:
        raise ValueError(
            "Factor input is missing required fields: " + ", ".join(missing_fields) + "."
        )


def normalize_factor_input(
    data: pd.DataFrame,
    required_fields: Iterable[str] = (),
) -> pd.DataFrame:
    """Validate and copy factor data into the standard key representation.

    The returned frame has datetime64 ``trade_date`` values and stripped
    pandas string ``ts_code`` values. The input frame is never modified.
    """
    validate_required_fields(data, FACTOR_KEY_FIELDS)
    validate_required_fields(data, required_fields)
    normalized = data.copy(deep=True)

    if normalized["trade_date"].isna().any():
        raise ValueError("Factor input trade_date cannot be empty.")
    trade_dates = pd.to_datetime(normalized["trade_date"], errors="coerce")
    if trade_dates.isna().any():
        raise ValueError("Factor input contains invalid or empty trade_date values.")
    normalized["trade_date"] = trade_dates

    ts_codes = normalized["ts_code"].astype("string").str.strip()
    if ts_codes.isna().any() or ts_codes.eq("").any():
        raise ValueError("Factor input ts_code cannot be empty.")
    normalized["ts_code"] = ts_codes

    duplicate_keys = normalized.duplicated(list(FACTOR_KEY_FIELDS), keep=False)
    if duplicate_keys.any():
        raise ValueError("Factor input contains duplicate trade_date + ts_code rows.")

    return normalized


def validate_factor_input(
    data: pd.DataFrame,
    required_fields: Iterable[str] = (),
) -> None:
    """Validate factor data without exposing the normalized copy."""
    normalize_factor_input(data, required_fields=required_fields)
