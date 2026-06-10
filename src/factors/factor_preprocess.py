"""Preprocess monthly factor panels for factor testing and scoring."""

from __future__ import annotations

import pandas as pd


DEFAULT_FACTOR_COLS = [
    "ep",
    "bp",
    "ps_inverse",
    "size_factor",
    "turnover_factor",
    "amount_factor",
    "momentum_1m",
    "momentum_3m",
    "volatility_6m",
]


def get_default_factor_cols(df: pd.DataFrame) -> list[str]:
    """Return default V2-A factor columns that exist in the DataFrame."""
    return [col for col in DEFAULT_FACTOR_COLS if col in df.columns]


def fill_missing_by_date_industry(
    df: pd.DataFrame,
    factor_cols: list[str],
    date_col: str = "date",
    industry_col: str = "industry",
) -> pd.DataFrame:
    """Fill factor missing values by industry-date median, then date median."""
    result = df.copy()
    for col in factor_cols:
        if col not in result.columns:
            continue

        result[col] = pd.to_numeric(result[col], errors="coerce")
        if industry_col in result.columns:
            industry_median = result.groupby([date_col, industry_col])[col].transform("median")
            result[col] = result[col].fillna(industry_median)

        date_median = result.groupby(date_col)[col].transform("median")
        result[col] = result[col].fillna(date_median)
    return result


def winsorize_by_date(
    df: pd.DataFrame,
    factor_cols: list[str],
    date_col: str = "date",
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Winsorize factor columns by monthly cross-sectional quantiles."""
    result = df.copy()
    for col in factor_cols:
        if col not in result.columns:
            continue

        result[col] = pd.to_numeric(result[col], errors="coerce")
        for _, index in result.groupby(date_col).groups.items():
            values = result.loc[index, col]
            valid = values.dropna()
            if len(valid) < 2:
                continue

            lower_bound = valid.quantile(lower)
            upper_bound = valid.quantile(upper)
            if pd.isna(lower_bound) or pd.isna(upper_bound):
                continue

            result.loc[index, col] = values.clip(lower=lower_bound, upper=upper_bound)
    return result


def standardize_by_date(
    df: pd.DataFrame,
    factor_cols: list[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """Standardize factor columns by monthly cross-sectional z-score."""
    result = df.copy()
    for col in factor_cols:
        if col not in result.columns:
            continue

        result[col] = pd.to_numeric(result[col], errors="coerce")
        grouped = result.groupby(date_col)[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        result[col] = ((result[col] - mean) / std).where(std.notna() & (std != 0))
    return result


def preprocess_factor_panel(
    df: pd.DataFrame,
    factor_cols: list[str] | None = None,
    drop_missing_target: bool = True,
    target_col: str = "return_next",
) -> tuple[pd.DataFrame, list[str]]:
    """Run missing-value fill, winsorization, and standardization on factors."""
    used_factor_cols = get_default_factor_cols(df) if factor_cols is None else [
        col for col in factor_cols if col in df.columns
    ]
    clean_df = df.copy()

    if drop_missing_target and target_col in clean_df.columns:
        clean_df = clean_df.dropna(subset=[target_col]).copy()

    clean_df = fill_missing_by_date_industry(clean_df, used_factor_cols)
    clean_df = winsorize_by_date(clean_df, used_factor_cols)
    clean_df = standardize_by_date(clean_df, used_factor_cols)
    return clean_df, used_factor_cols
