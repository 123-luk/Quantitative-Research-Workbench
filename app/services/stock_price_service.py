"""Single-stock historical price and return data preparation helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


PRICE_COLUMNS = [
    "date",
    "ts_code",
    "name",
    "close",
    "monthly_return",
    "return_next",
]


def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric_value):
        return None
    return numeric_value


def _compound_return(returns: pd.Series) -> float | None:
    """Calculate compounded return from a monthly return series."""
    clean_returns = pd.to_numeric(returns, errors="coerce").dropna()
    if clean_returns.empty:
        return None
    return float((1.0 + clean_returns).prod() - 1.0)


def prepare_price_history(
    history_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Prepare local historical price and return data for one stock.

    return_next is kept only as a historical backtest label, not as a future
    return estimate.
    """
    if history_df.empty or date_col not in history_df.columns:
        return pd.DataFrame()

    existing_cols = [col for col in PRICE_COLUMNS if col in history_df.columns]
    if date_col not in existing_cols:
        existing_cols.insert(0, date_col)

    result = history_df.loc[:, existing_cols].copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    result = result.dropna(subset=[date_col]).sort_values(date_col)
    return result.reset_index(drop=True)


def prepare_close_trend(history_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare historical close price trend data for one stock."""
    price_history = prepare_price_history(history_df)
    if price_history.empty or "close" not in price_history.columns:
        return pd.DataFrame()
    return price_history.loc[:, ["date", "close"]].dropna(subset=["close"], how="all")


def prepare_monthly_return_trend(history_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare historical monthly return trend data for one stock."""
    price_history = prepare_price_history(history_df)
    if price_history.empty or "monthly_return" not in price_history.columns:
        return pd.DataFrame()
    return price_history.loc[:, ["date", "monthly_return"]].dropna(
        subset=["monthly_return"],
        how="all",
    )


def calculate_recent_return_stats(history_df: pd.DataFrame) -> dict[str, object]:
    """Calculate recent historical return and volatility statistics."""
    price_history = prepare_price_history(history_df)
    if price_history.empty:
        return {
            "latest_date": None,
            "latest_close": None,
            "latest_monthly_return": None,
            "recent_3m_return": None,
            "recent_6m_return": None,
            "recent_6m_volatility": None,
            "available_months": 0,
        }

    returns = (
        pd.to_numeric(price_history.get("monthly_return"), errors="coerce")
        if "monthly_return" in price_history.columns
        else pd.Series(dtype=float)
    )
    available_returns = returns.dropna()
    latest_row = price_history.iloc[-1]
    latest_close = _safe_float(latest_row.get("close")) if "close" in price_history.columns else None
    latest_monthly_return = (
        _safe_float(returns.iloc[-1]) if len(returns) > 0 else None
    )
    recent_3m_return = (
        _compound_return(available_returns.tail(3)) if len(available_returns) >= 3 else None
    )
    recent_6m_tail = available_returns.tail(6)
    recent_6m_return = (
        _compound_return(recent_6m_tail) if len(recent_6m_tail) >= 6 else None
    )
    recent_6m_volatility = (
        float(recent_6m_tail.std()) if len(recent_6m_tail) >= 6 else None
    )

    return {
        "latest_date": latest_row.get("date"),
        "latest_close": latest_close,
        "latest_monthly_return": latest_monthly_return,
        "recent_3m_return": recent_3m_return,
        "recent_6m_return": recent_6m_return,
        "recent_6m_volatility": recent_6m_volatility,
        "available_months": int(len(available_returns)),
    }


def classify_recent_return(recent_return: float | None) -> str:
    """Classify a recent historical compounded return."""
    value = _safe_float(recent_return)
    if value is None:
        return "N/A"
    if value >= 0.20:
        return "历史阶段涨幅较高"
    if value >= 0.05:
        return "历史阶段表现偏强"
    if value > -0.05:
        return "历史阶段表现中性"
    if value > -0.20:
        return "历史阶段表现偏弱"
    return "历史阶段跌幅较大"


def classify_recent_volatility(volatility: float | None) -> str:
    """Classify recent historical monthly-return volatility."""
    value = _safe_float(volatility)
    if value is None:
        return "N/A"
    if value <= 0.05:
        return "历史波动较低"
    if value <= 0.10:
        return "历史波动中等"
    return "历史波动较高"


def build_price_summary(history_df: pd.DataFrame) -> dict[str, object]:
    """Build recent historical price and return summary labels."""
    summary = calculate_recent_return_stats(history_df)
    summary["recent_3m_return_label"] = classify_recent_return(
        summary.get("recent_3m_return")
    )
    summary["recent_6m_return_label"] = classify_recent_return(
        summary.get("recent_6m_return")
    )
    summary["recent_6m_volatility_label"] = classify_recent_volatility(
        summary.get("recent_6m_volatility")
    )
    return summary


def prepare_single_stock_price_data(history_df: pd.DataFrame) -> dict[str, object]:
    """Prepare all single-stock historical price and return datasets."""
    return {
        "price_history": prepare_price_history(history_df),
        "close_trend": prepare_close_trend(history_df),
        "monthly_return_trend": prepare_monthly_return_trend(history_df),
        "price_summary": build_price_summary(history_df),
    }
