"""Stock name and code query helpers for dashboard research pages."""

from __future__ import annotations

import re

import pandas as pd


DEFAULT_HISTORY_COLUMNS = [
    "date",
    "ts_code",
    "name",
    "industry",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "monthly_return",
    "composite_score",
    "score_rank",
    "score_pct_rank",
    "return_next",
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


def normalize_query(query: str) -> str:
    """Normalize user search text by trimming whitespace and uppercasing."""
    if not query:
        return ""
    return str(query).replace("\u3000", " ").strip().upper()


def normalize_ts_code(code: str) -> str:
    """Normalize common A-share code inputs into TuShare ts_code format."""
    cleaned = normalize_query(code)
    if cleaned.endswith(".SH") or cleaned.endswith(".SZ"):
        return cleaned

    if re.fullmatch(r"\d{6}", cleaned):
        if cleaned.startswith("6"):
            return f"{cleaned}.SH"
        if cleaned.startswith(("0", "3")):
            return f"{cleaned}.SZ"
    return cleaned


def build_stock_lookup(
    factor_score: pd.DataFrame,
    selected_portfolio: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a de-duplicated stock lookup table from local research outputs."""
    frames: list[pd.DataFrame] = []
    required_cols = ["ts_code", "name", "industry"]

    for source in [factor_score, selected_portfolio]:
        if source is None or source.empty or "ts_code" not in source.columns:
            continue

        frame = source.copy()
        for col in required_cols:
            if col not in frame.columns:
                frame[col] = pd.NA
        frames.append(frame.loc[:, required_cols])

    if not frames:
        return pd.DataFrame(columns=required_cols)

    lookup = pd.concat(frames, ignore_index=True)
    lookup["ts_code"] = lookup["ts_code"].astype(str).map(normalize_ts_code)
    lookup = lookup.dropna(subset=["ts_code"]).drop_duplicates("ts_code")
    return lookup.loc[:, required_cols].sort_values("ts_code").reset_index(drop=True)


def search_stock(query: str, lookup_df: pd.DataFrame) -> pd.DataFrame:
    """Search stocks by exact ts_code, six-digit code, or fuzzy name match."""
    if lookup_df.empty or "ts_code" not in lookup_df.columns:
        return pd.DataFrame(columns=["ts_code", "name", "industry"])

    normalized = normalize_query(query)
    if not normalized:
        return pd.DataFrame(columns=["ts_code", "name", "industry"])

    lookup = lookup_df.copy()
    for col in ["ts_code", "name", "industry"]:
        if col not in lookup.columns:
            lookup[col] = pd.NA

    normalized_code = normalize_ts_code(normalized)
    ts_code_series = lookup["ts_code"].astype(str).str.upper()
    raw_code_series = ts_code_series.str.slice(0, 6)
    name_series = lookup["name"].fillna("").astype(str).str.upper()

    mask = (
        (ts_code_series == normalized_code)
        | (raw_code_series == normalized)
        | name_series.str.contains(re.escape(normalized), na=False)
    )
    return lookup.loc[mask, ["ts_code", "name", "industry"]].reset_index(drop=True)


def get_stock_factor_history(
    ts_code: str,
    factor_score: pd.DataFrame,
) -> pd.DataFrame:
    """Return one stock's historical factor score records."""
    if factor_score.empty or "ts_code" not in factor_score.columns:
        return pd.DataFrame()

    normalized_code = normalize_ts_code(ts_code)
    history = factor_score[factor_score["ts_code"].astype(str).str.upper() == normalized_code].copy()
    if history.empty:
        return pd.DataFrame()

    if "date" in history.columns:
        history = history.sort_values("date")
    output_cols = [col for col in DEFAULT_HISTORY_COLUMNS if col in history.columns]
    return history.loc[:, output_cols].reset_index(drop=True)


def get_stock_selection_history(
    ts_code: str,
    selected_portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """Return one stock's historical model-selected portfolio records."""
    if selected_portfolio.empty or "ts_code" not in selected_portfolio.columns:
        return pd.DataFrame()

    normalized_code = normalize_ts_code(ts_code)
    history = selected_portfolio[
        selected_portfolio["ts_code"].astype(str).str.upper() == normalized_code
    ].copy()
    if history.empty:
        return pd.DataFrame()

    if "date" in history.columns:
        history = history.sort_values("date")
    return history.reset_index(drop=True)


def get_latest_stock_snapshot(
    ts_code: str,
    factor_score: pd.DataFrame,
    selected_portfolio: pd.DataFrame,
) -> dict[str, object]:
    """Return the latest research snapshot for one stock.

    return_next is a historical backtest label, not a future return forecast.
    """
    history = get_stock_factor_history(ts_code, factor_score)
    if history.empty:
        return {}

    latest = history.iloc[-1]
    latest_date = latest.get("date")
    normalized_code = normalize_ts_code(ts_code)
    is_selected_latest = False
    if not selected_portfolio.empty and {"date", "ts_code"}.issubset(selected_portfolio.columns):
        selected_mask = (
            (selected_portfolio["ts_code"].astype(str).str.upper() == normalized_code)
            & (selected_portfolio["date"] == latest_date)
        )
        is_selected_latest = bool(selected_mask.any())

    snapshot = {
        "ts_code": latest.get("ts_code"),
        "name": latest.get("name"),
        "industry": latest.get("industry"),
        "latest_date": latest_date,
        "composite_score": latest.get("composite_score"),
        "score_rank": latest.get("score_rank"),
        "score_pct_rank": latest.get("score_pct_rank"),
        "return_next": latest.get("return_next"),
        "is_selected_latest": is_selected_latest,
    }
    factor_cols = [
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
    for col in factor_cols:
        if col in latest.index:
            snapshot[col] = latest.get(col)
    return snapshot


def calculate_selection_frequency(
    ts_code: str,
    selected_portfolio: pd.DataFrame,
    factor_score: pd.DataFrame,
) -> dict[str, object]:
    """Calculate one stock's model-selection frequency over the sample period."""
    normalized_code = normalize_ts_code(ts_code)
    if factor_score.empty or "ts_code" not in factor_score.columns:
        total_periods = 0
    else:
        stock_factor = factor_score[
            factor_score["ts_code"].astype(str).str.upper() == normalized_code
        ]
        total_periods = stock_factor["date"].nunique() if "date" in stock_factor.columns else len(stock_factor)

    if selected_portfolio.empty or "ts_code" not in selected_portfolio.columns:
        selected_periods = 0
    else:
        stock_selected = selected_portfolio[
            selected_portfolio["ts_code"].astype(str).str.upper() == normalized_code
        ]
        selected_periods = (
            stock_selected["date"].nunique()
            if "date" in stock_selected.columns
            else len(stock_selected)
        )

    selection_frequency = (
        selected_periods / total_periods if total_periods > 0 else None
    )
    return {
        "total_periods": total_periods,
        "selected_periods": selected_periods,
        "selection_frequency": selection_frequency,
    }
