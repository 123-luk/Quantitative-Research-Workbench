"""Core portfolio backtest utilities for historical sample analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_holdings(
    selected_df: pd.DataFrame,
    date_col: str = "date",
    code_col: str = "ts_code",
    ret_col: str = "return_next",
) -> pd.DataFrame:
    """Prepare equal-weight holdings from model-selected stocks."""
    holdings = selected_df.copy()
    holdings = holdings.dropna(subset=[date_col, code_col, ret_col])
    holdings = holdings.drop_duplicates(subset=[date_col, code_col])
    counts = holdings.groupby(date_col)[code_col].transform("count")
    holdings["weight"] = 1.0 / counts

    required_cols = [date_col, code_col, "weight", ret_col]
    extra_cols = [col for col in holdings.columns if col not in required_cols]
    output_cols = required_cols + extra_cols
    return holdings.loc[:, output_cols].sort_values([date_col, code_col]).reset_index(drop=True)


def calc_turnover(
    holdings: pd.DataFrame,
    date_col: str = "date",
    code_col: str = "ts_code",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Calculate portfolio turnover between adjacent rebalance periods."""
    weights = holdings.pivot_table(
        index=date_col,
        columns=code_col,
        values=weight_col,
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    rows: list[dict[str, object]] = []
    previous_weights: pd.Series | None = None
    for date, current_weights in weights.iterrows():
        if previous_weights is None:
            turnover = current_weights.abs().sum()
        else:
            aligned_current, aligned_previous = current_weights.align(
                previous_weights,
                fill_value=0.0,
            )
            turnover = 0.5 * (aligned_current - aligned_previous).abs().sum()
        rows.append({date_col: date, "turnover": float(turnover)})
        previous_weights = current_weights

    return pd.DataFrame(rows, columns=[date_col, "turnover"]).sort_values(date_col)


def calc_period_returns(
    holdings: pd.DataFrame,
    turnover_df: pd.DataFrame,
    transaction_cost: float = 0.0005,
    date_col: str = "date",
    ret_col: str = "return_next",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Calculate gross and transaction-cost-adjusted period returns."""
    prepared = holdings.copy()
    prepared[ret_col] = pd.to_numeric(prepared[ret_col], errors="coerce")
    prepared[weight_col] = pd.to_numeric(prepared[weight_col], errors="coerce")
    prepared["weighted_return"] = prepared[weight_col] * prepared[ret_col]

    period_returns = (
        prepared.groupby(date_col, as_index=False)["weighted_return"]
        .sum()
        .rename(columns={"weighted_return": "gross_return"})
    )
    period_returns = period_returns.merge(turnover_df, on=date_col, how="left")
    period_returns["turnover"] = period_returns["turnover"].fillna(0.0)
    period_returns["cost"] = period_returns["turnover"] * transaction_cost
    period_returns["net_return"] = period_returns["gross_return"] - period_returns["cost"]
    return period_returns.loc[
        :,
        [date_col, "gross_return", "turnover", "cost", "net_return"],
    ].sort_values(date_col)


def calc_nav(
    period_returns: pd.DataFrame,
    date_col: str = "date",
    ret_col: str = "net_return",
) -> pd.DataFrame:
    """Calculate net asset value from period returns with initial NAV of 1.0."""
    nav_df = period_returns.copy().sort_values(date_col)
    nav_df[ret_col] = pd.to_numeric(nav_df[ret_col], errors="coerce").fillna(0.0)
    nav_df["nav"] = (1.0 + nav_df[ret_col]).cumprod()
    return nav_df.loc[
        :,
        [date_col, "gross_return", "turnover", "cost", "net_return", "nav"],
    ]


def calc_backtest_metrics(
    nav_df: pd.DataFrame,
    date_col: str = "date",
    ret_col: str = "net_return",
    nav_col: str = "nav",
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Calculate historical sample backtest performance metrics."""
    metric_cols = [
        "start_date",
        "end_date",
        "n_periods",
        "cumulative_return",
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "mean_monthly_return",
        "win_rate",
        "average_turnover",
    ]
    if nav_df.empty:
        return pd.DataFrame([{col: np.nan for col in metric_cols}])

    data = nav_df.copy().sort_values(date_col)
    returns = pd.to_numeric(data[ret_col], errors="coerce").fillna(0.0)
    nav = pd.to_numeric(data[nav_col], errors="coerce")
    n_periods = len(data)
    final_nav = nav.iloc[-1]
    cumulative_return = final_nav - 1.0
    annual_return = final_nav ** (periods_per_year / n_periods) - 1.0
    annual_volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = (
        annual_return / annual_volatility
        if pd.notna(annual_volatility) and annual_volatility != 0
        else np.nan
    )
    drawdown = nav / nav.cummax() - 1.0

    metrics = {
        "start_date": data[date_col].iloc[0],
        "end_date": data[date_col].iloc[-1],
        "n_periods": n_periods,
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": drawdown.min(),
        "mean_monthly_return": returns.mean(),
        "win_rate": (returns > 0).mean(),
        "average_turnover": data["turnover"].mean() if "turnover" in data.columns else np.nan,
    }
    return pd.DataFrame([metrics], columns=metric_cols)


def run_backtest(
    selected_df: pd.DataFrame,
    transaction_cost: float = 0.0005,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full equal-weight historical sample portfolio backtest flow."""
    holdings_df = prepare_holdings(selected_df)
    turnover_df = calc_turnover(holdings_df)
    period_returns = calc_period_returns(
        holdings_df,
        turnover_df,
        transaction_cost=transaction_cost,
    )
    nav_df = calc_nav(period_returns)
    metrics_df = calc_backtest_metrics(nav_df)
    return holdings_df, turnover_df, nav_df, metrics_df
