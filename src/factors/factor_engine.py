"""Build monthly factor panels from raw market data files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_RAW_FILES = {
    "stock_basic": "stock_basic.csv",
    "hs300_components": "hs300_components.csv",
    "monthly": "monthly.csv",
    "daily_basic": "daily_basic.csv",
}


def load_raw_data(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load required raw CSV files for factor panel construction.

    Args:
        raw_dir: Directory containing raw TuShare CSV files.

    Returns:
        Mapping from dataset name to loaded DataFrame.

    Raises:
        FileNotFoundError: If any required raw CSV file is missing.
    """
    data: dict[str, pd.DataFrame] = {}
    for name, filename in REQUIRED_RAW_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw data file: {path}")
        frame = pd.read_csv(path, dtype={"ts_code": "string", "trade_date": "string"})
        if "trade_date" in frame.columns:
            frame["trade_date"] = frame["trade_date"].astype("string")
        data[name] = frame
    return data


def prepare_monthly_data(monthly: pd.DataFrame) -> pd.DataFrame:
    """Prepare monthly market data and derive monthly returns.

    monthly_return uses close / pre_close - 1 to avoid pct_chg unit differences
    causing incorrect return scaling.
    """
    fields = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "vol",
        "amount",
    ]
    prepared = monthly.loc[:, fields].copy()
    trade_dates = pd.to_datetime(prepared["trade_date"].astype(str), format="%Y%m%d")
    prepared["date"] = trade_dates.dt.to_period("M").dt.to_timestamp("M")
    close = pd.to_numeric(prepared["close"], errors="coerce")
    pre_close = pd.to_numeric(prepared["pre_close"], errors="coerce")
    prepared["monthly_return"] = (close / pre_close - 1.0).where(pre_close > 0)
    prepared = prepared.sort_values(["ts_code", "date"]).reset_index(drop=True)
    return prepared


def prepare_daily_basic_month_end(daily_basic: pd.DataFrame) -> pd.DataFrame:
    """Keep each stock's last daily_basic record for every month."""
    prepared = daily_basic.copy()
    prepared["trade_date_dt"] = pd.to_datetime(
        prepared["trade_date"].astype(str),
        format="%Y%m%d",
    )
    prepared["month"] = prepared["trade_date_dt"].dt.to_period("M")
    prepared = prepared.sort_values(["ts_code", "trade_date_dt"])
    month_end = prepared.groupby(["ts_code", "month"], as_index=False).tail(1).copy()
    month_end["date"] = month_end["month"].dt.to_timestamp("M")

    fields = [
        "ts_code",
        "date",
        "turnover_rate",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "total_mv",
        "circ_mv",
    ]
    return month_end.loc[:, fields].sort_values(["ts_code", "date"]).reset_index(drop=True)


def _positive_numeric(series: pd.Series) -> pd.Series:
    """Return numeric values and mask non-positive entries as NaN."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric > 0)


def _rolling_compound_return(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling compounded returns excluding the current row."""
    shifted = returns.shift(1)
    return shifted.rolling(window=window, min_periods=window).apply(
        lambda values: np.prod(1.0 + values) - 1.0,
        raw=True,
    )


def build_base_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Build first-version valuation, size, turnover, momentum, and risk factors."""
    result = panel.copy()
    result["ep"] = 1.0 / _positive_numeric(result["pe_ttm"])
    result["bp"] = 1.0 / _positive_numeric(result["pb"])
    result["ps_inverse"] = 1.0 / _positive_numeric(result["ps_ttm"])
    result["size_factor"] = -np.log(_positive_numeric(result["total_mv"]))
    result["turnover_factor"] = pd.to_numeric(result["turnover_rate"], errors="coerce")
    result["amount_factor"] = np.log(_positive_numeric(result["amount"]))

    grouped_returns = result.groupby("ts_code", group_keys=False)["monthly_return"]
    result["momentum_1m"] = grouped_returns.shift(1)
    result["momentum_3m"] = grouped_returns.apply(
        lambda returns: _rolling_compound_return(returns, window=3)
    )
    result["volatility_6m"] = grouped_returns.apply(
        lambda returns: returns.shift(1).rolling(window=6, min_periods=6).std()
    )
    return result


def add_forward_return(panel: pd.DataFrame) -> pd.DataFrame:
    """Add next-month return labels for later factor testing."""
    result = panel.sort_values(["ts_code", "date"]).copy()
    result["return_next"] = result.groupby("ts_code")["monthly_return"].shift(-1)
    return result


def build_factor_panel(raw_dir: Path, output_path: Path) -> pd.DataFrame:
    """Build and save the monthly factor panel from raw CSV inputs."""
    raw_data = load_raw_data(raw_dir)
    stock_basic = raw_data["stock_basic"]
    hs300_components = raw_data["hs300_components"]
    monthly = raw_data["monthly"]
    daily_basic = raw_data["daily_basic"]

    hs300_codes = hs300_components["ts_code"].dropna().astype(str).unique()
    monthly_prepared = prepare_monthly_data(monthly)
    monthly_prepared = monthly_prepared[monthly_prepared["ts_code"].isin(hs300_codes)]

    daily_month_end = prepare_daily_basic_month_end(daily_basic)
    daily_month_end = daily_month_end[daily_month_end["ts_code"].isin(hs300_codes)]

    stock_info = stock_basic.loc[:, ["ts_code", "name", "industry"]].drop_duplicates("ts_code")
    panel = monthly_prepared.merge(stock_info, on="ts_code", how="left")
    panel = panel.merge(daily_month_end, on=["ts_code", "date"], how="left")
    panel = build_base_factors(panel)
    panel = add_forward_return(panel)
    panel = panel.sort_values(["date", "ts_code"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False, encoding="utf-8-sig")
    return panel
