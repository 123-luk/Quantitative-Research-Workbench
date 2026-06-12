"""Multi-factor scoring model for research stock selection signals."""

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


def get_default_factor_directions() -> dict[str, int]:
    """Return default empirical factor directions for the first scoring model.

    A direction of 1 means higher factor values are preferred; -1 means lower
    factor values are preferred. The current version uses empirical directions,
    which can later be updated dynamically from IC results.
    """
    return {
        "ep": 1,
        "bp": 1,
        "ps_inverse": 1,
        "size_factor": 1,
        "turnover_factor": 1,
        "amount_factor": 1,
        "momentum_1m": 1,
        "momentum_3m": 1,
        "volatility_6m": -1,
    }


def get_default_factor_weights(factor_cols: list[str]) -> dict[str, float]:
    """Return equal weights for the provided factor columns."""
    if not factor_cols:
        return {}

    weight = 1.0 / len(factor_cols)
    return {factor: weight for factor in factor_cols}


def calc_composite_score(
    df: pd.DataFrame,
    factor_cols: list[str],
    factor_directions: dict[str, int] | None = None,
    factor_weights: dict[str, float] | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Calculate direction-adjusted weighted composite factor scores."""
    result = df.copy()
    directions = factor_directions or get_default_factor_directions()
    existing_factors = [factor for factor in factor_cols if factor in result.columns]
    weights = factor_weights or get_default_factor_weights(existing_factors)

    contribution = pd.Series(0.0, index=result.index)
    valid_count = pd.Series(0, index=result.index)
    for factor in existing_factors:
        if factor not in weights:
            continue

        values = pd.to_numeric(result[factor], errors="coerce")
        direction = directions.get(factor, 1)
        adjusted = values * direction
        contribution = contribution.add(adjusted.fillna(0.0) * weights[factor], fill_value=0.0)
        valid_count = valid_count + adjusted.notna().astype(int)

    result["composite_score"] = contribution.where(valid_count > 0)
    result["score_rank"] = result.groupby(date_col)["composite_score"].rank(
        method="first",
        ascending=False,
    )
    result["score_pct_rank"] = result.groupby(date_col)["composite_score"].rank(
        method="average",
        ascending=True,
        pct=True,
    )
    return result


def select_top_n(
    score_df: pd.DataFrame,
    top_n: int = 10,
    date_col: str = "date",
    score_col: str = "composite_score",
) -> pd.DataFrame:
    """Select top-scored stocks for each month as model research signals."""
    required_cols = [date_col, "ts_code", score_col]
    missing_required = [col for col in required_cols if col not in score_df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns for selection: {missing_required}")

    valid = score_df.dropna(subset=[score_col]).copy()
    selected = (
        valid.sort_values([date_col, score_col], ascending=[True, False])
        .groupby(date_col, group_keys=False)
        .head(top_n)
        .sort_values([date_col, "score_rank" if "score_rank" in valid.columns else score_col])
        .reset_index(drop=True)
    )

    preferred_cols = [
        date_col,
        "ts_code",
        "name",
        "industry",
        "composite_score",
        "score_rank",
        "score_pct_rank",
        "return_next",
    ]
    output_cols = [col for col in preferred_cols if col in selected.columns]
    return selected.loc[:, output_cols]


def run_scoring_pipeline(
    df: pd.DataFrame,
    factor_cols: list[str] | None = None,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run the full equal-weight multi-factor scoring pipeline."""
    candidate_factors = DEFAULT_FACTOR_COLS if factor_cols is None else factor_cols
    used_factor_cols = [factor for factor in candidate_factors if factor in df.columns]
    factor_directions = get_default_factor_directions()
    factor_weights = get_default_factor_weights(used_factor_cols)
    factor_score_df = calc_composite_score(
        df,
        factor_cols=used_factor_cols,
        factor_directions=factor_directions,
        factor_weights=factor_weights,
    )
    selected_portfolio_df = select_top_n(factor_score_df, top_n=top_n)
    return factor_score_df, selected_portfolio_df, factor_weights
