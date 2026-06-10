"""Factor effectiveness tests for IC and quantile return analysis."""

from __future__ import annotations

import math

import numpy as np
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


def calc_rank_ic(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str = "return_next",
    date_col: str = "date",
) -> pd.Series:
    """Calculate monthly Spearman Rank IC between a factor and next return."""
    ic_values: dict[object, float] = {}
    for date, group in df.groupby(date_col):
        valid = group[[factor_col, ret_col]].dropna()
        if len(valid) < 3:
            ic_values[date] = np.nan
            continue
        if valid[factor_col].nunique() <= 1 or valid[ret_col].nunique() <= 1:
            ic_values[date] = np.nan
            continue
        ic_values[date] = valid[factor_col].corr(valid[ret_col], method="spearman")

    return pd.Series(ic_values, name=factor_col).sort_index()


def summarize_ic(ic_series: pd.Series) -> dict[str, float]:
    """Summarize an IC series with mean, volatility, ICIR, t-stat, and hit ratio."""
    valid = ic_series.dropna()
    n_periods = len(valid)
    if n_periods == 0:
        return {
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "icir": np.nan,
            "t_stat": np.nan,
            "positive_ratio": np.nan,
            "n_periods": 0.0,
        }

    mean_ic = valid.mean()
    std_ic = valid.std()
    positive_ratio = (valid > 0).mean()
    if pd.isna(std_ic) or std_ic == 0:
        icir = np.nan
        t_stat = np.nan
    else:
        icir = mean_ic / std_ic
        t_stat = mean_ic / (std_ic / math.sqrt(n_periods))

    return {
        "mean_ic": float(mean_ic),
        "std_ic": float(std_ic),
        "icir": float(icir) if not pd.isna(icir) else np.nan,
        "t_stat": float(t_stat) if not pd.isna(t_stat) else np.nan,
        "positive_ratio": float(positive_ratio),
        "n_periods": float(n_periods),
    }


def batch_calc_ic(
    df: pd.DataFrame,
    factor_cols: list[str],
    ret_col: str = "return_next",
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate IC series and summary statistics for multiple factors."""
    ic_map = {
        factor: calc_rank_ic(df, factor, ret_col=ret_col, date_col=date_col)
        for factor in factor_cols
    }
    ic_series_df = pd.DataFrame(ic_map)
    summary_rows = []
    for factor, series in ic_map.items():
        summary = summarize_ic(series)
        summary["factor"] = factor
        summary_rows.append(summary)

    summary_cols = [
        "factor",
        "mean_ic",
        "std_ic",
        "icir",
        "t_stat",
        "positive_ratio",
        "n_periods",
    ]
    ic_summary_df = pd.DataFrame(summary_rows, columns=summary_cols)
    return ic_series_df, ic_summary_df


def quantile_group_return(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str = "return_next",
    date_col: str = "date",
    n_groups: int = 5,
) -> pd.DataFrame:
    """Calculate monthly mean forward returns by factor quantile group."""
    rows: list[dict[str, object]] = []
    labels = [f"Q{i}" for i in range(1, n_groups + 1)]
    for date, group in df.groupby(date_col):
        valid = group[[factor_col, ret_col]].dropna().copy()
        if len(valid) < n_groups:
            continue

        try:
            valid["group"] = pd.qcut(valid[factor_col], q=n_groups, labels=labels)
        except ValueError:
            ranked = valid[factor_col].rank(method="first")
            try:
                valid["group"] = pd.qcut(ranked, q=n_groups, labels=labels)
            except ValueError:
                continue

        grouped = valid.groupby("group", observed=False)[ret_col]
        for group_name, returns in grouped:
            rows.append(
                {
                    "date": date,
                    "factor": factor_col,
                    "group": str(group_name),
                    "mean_return": returns.mean(),
                    "n_stocks": int(returns.count()),
                }
            )

    return pd.DataFrame(rows, columns=["date", "factor", "group", "mean_return", "n_stocks"])


def batch_quantile_group_return(
    df: pd.DataFrame,
    factor_cols: list[str],
    ret_col: str = "return_next",
    date_col: str = "date",
    n_groups: int = 5,
) -> pd.DataFrame:
    """Calculate quantile group returns for multiple factors."""
    frames = [
        quantile_group_return(
            df,
            factor,
            ret_col=ret_col,
            date_col=date_col,
            n_groups=n_groups,
        )
        for factor in factor_cols
    ]
    if not frames:
        return pd.DataFrame(columns=["date", "factor", "group", "mean_return", "n_stocks"])
    return pd.concat(frames, ignore_index=True)


def long_short_return(
    group_return_df: pd.DataFrame,
    high_group: str = "Q5",
    low_group: str = "Q1",
) -> pd.DataFrame:
    """Calculate high-minus-low quantile returns for each factor and date."""
    rows: list[dict[str, object]] = []
    for (factor, date), group in group_return_df.groupby(["factor", "date"]):
        returns = group.set_index("group")["mean_return"]
        if high_group not in returns.index or low_group not in returns.index:
            continue
        rows.append(
            {
                "date": date,
                "factor": factor,
                "long_short_return": returns.loc[high_group] - returns.loc[low_group],
            }
        )
    return pd.DataFrame(rows, columns=["date", "factor", "long_short_return"])


def get_default_factor_cols(df: pd.DataFrame) -> list[str]:
    """Return default factor columns that exist in the DataFrame."""
    return [col for col in DEFAULT_FACTOR_COLS if col in df.columns]
