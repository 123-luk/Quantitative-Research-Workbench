"""Single-stock chart data preparation helpers for dashboard pages."""

from __future__ import annotations

from typing import Any

import pandas as pd


FACTOR_DIRECTION_HINTS = {
    "ep": "越高通常代表估值越低",
    "bp": "越高通常代表账面市值比越高",
    "ps_inverse": "越高通常代表市销率越低",
    "size_factor": "越高通常代表市值越小",
    "turnover_factor": "越高代表换手率越高",
    "amount_factor": "越高代表成交额越高",
    "momentum_1m": "越高代表1个月动量越强",
    "momentum_3m": "越高代表3个月动量越强",
    "volatility_6m": "越高代表6个月波动率越高",
}


def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric_value):
        return None
    return numeric_value


def _exposure_label(value: float | None) -> str:
    """Convert a standardized factor value into an exposure label."""
    if value is None:
        return "N/A"
    if value >= 0.5:
        return "显著高于样本均值"
    if value >= 0:
        return "略高于样本均值"
    if value > -0.5:
        return "略低于样本均值"
    return "显著低于样本均值"


def prepare_time_series(
    history_df: pd.DataFrame,
    columns: list[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """Prepare a sorted time-series DataFrame for selected columns."""
    if history_df.empty or date_col not in history_df.columns:
        return pd.DataFrame()

    existing_cols = [col for col in columns if col in history_df.columns]
    if not existing_cols:
        return pd.DataFrame()

    result = history_df.loc[:, [date_col, *existing_cols]].copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    result = result.dropna(subset=[date_col]).sort_values(date_col)
    result = result.dropna(subset=existing_cols, how="all")
    return result.reset_index(drop=True)


def prepare_score_trend(history_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare composite score trend data for a single stock."""
    return prepare_time_series(history_df, ["composite_score"])


def prepare_rank_trend(history_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare score rank trend data for a single stock."""
    return prepare_time_series(history_df, ["score_rank"])


def prepare_percentile_trend(history_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare score percentile trend data for a single stock."""
    return prepare_time_series(history_df, ["score_pct_rank"])


def prepare_momentum_risk_trend(history_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare momentum and volatility trend data for a single stock."""
    return prepare_time_series(
        history_df,
        ["momentum_1m", "momentum_3m", "volatility_6m"],
    )


def prepare_factor_exposure_table(snapshot: dict[str, object]) -> pd.DataFrame:
    """Prepare a factor exposure table from the latest standardized snapshot."""
    rows: list[dict[str, object]] = []
    for factor, direction_hint in FACTOR_DIRECTION_HINTS.items():
        value = _safe_float(snapshot.get(factor))
        if value is None:
            continue
        rows.append(
            {
                "factor": factor,
                "value": value,
                "direction_hint": direction_hint,
                "exposure_label": _exposure_label(value),
            }
        )
    return pd.DataFrame(rows, columns=["factor", "value", "direction_hint", "exposure_label"])


def prepare_single_stock_chart_data(
    history_df: pd.DataFrame,
    snapshot: dict[str, object],
) -> dict[str, pd.DataFrame]:
    """Prepare all single-stock chart and exposure datasets."""
    return {
        "score_trend": prepare_score_trend(history_df),
        "rank_trend": prepare_rank_trend(history_df),
        "percentile_trend": prepare_percentile_trend(history_df),
        "momentum_risk_trend": prepare_momentum_risk_trend(history_df),
        "factor_exposure": prepare_factor_exposure_table(snapshot),
    }
