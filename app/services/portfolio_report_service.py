"""Portfolio report helpers for model-selected research portfolios."""

from __future__ import annotations

from typing import Any

import pandas as pd


UNKNOWN_INDUSTRY = "未知行业"
PORTFOLIO_DISCLAIMER = (
    "以上结果仅为历史样本中的模型筛选结果和量化研究参考，"
    "不代表未来表现，不构成投资建议。"
)
PREFERRED_PORTFOLIO_COLUMNS = [
    "date",
    "ts_code",
    "name",
    "industry",
    "weight",
    "composite_score",
    "score_rank",
    "score_pct_rank",
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


def safe_percent(value: Any, default: str = "N/A") -> str:
    """Format a decimal value as a percentage string with two decimals."""
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return default
    return f"{numeric_value:.2%}"


def safe_number(value: Any, digits: int = 2, default: str = "N/A") -> str:
    """Format a numeric value with a fixed number of decimal places."""
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return default
    return f"{numeric_value:.{digits}f}"


def get_latest_portfolio_date(selected_portfolio: pd.DataFrame) -> object | None:
    """Return the latest portfolio date, or None when unavailable."""
    if selected_portfolio.empty or "date" not in selected_portfolio.columns:
        return None
    dates = pd.to_datetime(selected_portfolio["date"], errors="coerce")
    if dates.dropna().empty:
        return None
    return dates.max()


def prepare_latest_portfolio(selected_portfolio: pd.DataFrame) -> pd.DataFrame:
    """Prepare the latest model-selected portfolio table."""
    if selected_portfolio.empty or "date" not in selected_portfolio.columns:
        return pd.DataFrame()

    latest_date = get_latest_portfolio_date(selected_portfolio)
    if latest_date is None:
        return pd.DataFrame()

    portfolio = selected_portfolio.copy()
    portfolio["_date_dt"] = pd.to_datetime(portfolio["date"], errors="coerce")
    latest = portfolio[portfolio["_date_dt"] == latest_date].copy()
    if latest.empty:
        return pd.DataFrame()

    if "industry" in latest.columns:
        latest["industry"] = latest["industry"].fillna(UNKNOWN_INDUSTRY)
    if "score_rank" in latest.columns:
        latest = latest.sort_values("score_rank", ascending=True)
    elif "composite_score" in latest.columns:
        latest = latest.sort_values("composite_score", ascending=False)

    output_cols = [col for col in PREFERRED_PORTFOLIO_COLUMNS if col in latest.columns]
    return latest.loc[:, output_cols].reset_index(drop=True)


def prepare_portfolio_industry_distribution(
    latest_portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare industry count and weight distribution for the latest portfolio."""
    if latest_portfolio.empty or "industry" not in latest_portfolio.columns:
        return pd.DataFrame(columns=["industry", "count", "weight"])

    portfolio = latest_portfolio.copy()
    portfolio["industry"] = portfolio["industry"].fillna(UNKNOWN_INDUSTRY)
    count_col = "ts_code" if "ts_code" in portfolio.columns else "industry"
    if "weight" in portfolio.columns:
        portfolio["weight"] = pd.to_numeric(portfolio["weight"], errors="coerce")
        grouped = portfolio.groupby("industry", dropna=False).agg(
            count=(count_col, "count"),
            weight=("weight", "sum"),
        )
    else:
        grouped = portfolio.groupby("industry", dropna=False).agg(count=(count_col, "count"))
        total = grouped["count"].sum()
        grouped["weight"] = grouped["count"] / total if total else 0.0

    return (
        grouped.reset_index()
        .loc[:, ["industry", "count", "weight"]]
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )


def prepare_portfolio_weight_distribution(
    latest_portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare stock-level portfolio weight distribution."""
    if latest_portfolio.empty:
        return pd.DataFrame()

    portfolio = latest_portfolio.copy()
    if "industry" in portfolio.columns:
        portfolio["industry"] = portfolio["industry"].fillna(UNKNOWN_INDUSTRY)
    if "weight" not in portfolio.columns:
        portfolio["weight"] = 1.0 / len(portfolio)
    else:
        portfolio["weight"] = pd.to_numeric(portfolio["weight"], errors="coerce")

    preferred_cols = [
        "ts_code",
        "name",
        "industry",
        "weight",
        "composite_score",
        "score_rank",
    ]
    output_cols = [col for col in preferred_cols if col in portfolio.columns]
    sort_cols = ["weight"]
    ascending = [False]
    if "score_rank" in portfolio.columns:
        sort_cols.append("score_rank")
        ascending.append(True)
    return portfolio.sort_values(sort_cols, ascending=ascending).loc[:, output_cols].reset_index(drop=True)


def _stock_label(row: pd.Series) -> object | None:
    """Return a display label using name when available, otherwise ts_code."""
    if "name" in row.index and pd.notna(row.get("name")):
        return row.get("name")
    if "ts_code" in row.index and pd.notna(row.get("ts_code")):
        return row.get("ts_code")
    return None


def build_portfolio_summary(latest_portfolio: pd.DataFrame) -> dict[str, object]:
    """Build summary metrics for the latest model-selected portfolio."""
    if latest_portfolio.empty:
        return {
            "latest_date": None,
            "holding_count": 0,
            "industry_count": None,
            "top_weight_stock": None,
            "top_score_stock": None,
            "average_score": None,
            "average_score_pct_rank": None,
        }

    portfolio = latest_portfolio.copy()
    if "industry" in portfolio.columns:
        portfolio["industry"] = portfolio["industry"].fillna(UNKNOWN_INDUSTRY)

    latest_date = portfolio["date"].iloc[0] if "date" in portfolio.columns else None
    industry_count = portfolio["industry"].nunique() if "industry" in portfolio.columns else None
    weight_table = prepare_portfolio_weight_distribution(portfolio)
    top_weight_stock = _stock_label(weight_table.iloc[0]) if not weight_table.empty else None

    top_score_stock = None
    if "composite_score" in portfolio.columns:
        score_values = pd.to_numeric(portfolio["composite_score"], errors="coerce")
        if not score_values.dropna().empty:
            top_score_stock = _stock_label(portfolio.loc[score_values.idxmax()])

    average_score = (
        pd.to_numeric(portfolio["composite_score"], errors="coerce").mean()
        if "composite_score" in portfolio.columns
        else None
    )
    average_score_pct_rank = (
        pd.to_numeric(portfolio["score_pct_rank"], errors="coerce").mean()
        if "score_pct_rank" in portfolio.columns
        else None
    )

    return {
        "latest_date": latest_date,
        "holding_count": len(portfolio),
        "industry_count": industry_count,
        "top_weight_stock": top_weight_stock,
        "top_score_stock": top_score_stock,
        "average_score": average_score,
        "average_score_pct_rank": average_score_pct_rank,
    }


def build_portfolio_research_comment(
    latest_portfolio: pd.DataFrame,
    industry_distribution: pd.DataFrame | None = None,
) -> str:
    """Build a neutral Chinese research comment for the latest portfolio."""
    summary = build_portfolio_summary(latest_portfolio)
    industry_distribution = (
        industry_distribution
        if industry_distribution is not None
        else prepare_portfolio_industry_distribution(latest_portfolio)
    )
    holding_count = summary.get("holding_count", 0)
    industry_count = summary.get("industry_count")
    top_weight_stock = summary.get("top_weight_stock") or "N/A"
    top_score_stock = summary.get("top_score_stock") or "N/A"

    concentration = "行业分布数据不足，暂无法判断集中度"
    if industry_distribution is not None and not industry_distribution.empty:
        top_industry = industry_distribution.iloc[0].get("industry") or UNKNOWN_INDUSTRY
        top_weight = _safe_float(industry_distribution.iloc[0].get("weight"))
        if top_weight is not None and top_weight >= 0.4:
            concentration = f"组合在{top_industry}等行业上暴露相对较高，存在一定行业集中度"
        else:
            concentration = "组合行业分布相对分散"

    return (
        f"最新一期模型组合共包含 {holding_count} 只股票，覆盖 {industry_count or 'N/A'} 个行业。"
        f"组合采用等权配置或回测权重字段，其中权重最高的股票为{top_weight_stock}，"
        f"模型评分最高的股票为{top_score_stock}。从行业分布看，{concentration}。"
        f"{PORTFOLIO_DISCLAIMER}"
    )


def prepare_portfolio_report_data(
    selected_portfolio: pd.DataFrame,
) -> dict[str, object]:
    """Prepare all report data for the latest model-selected portfolio."""
    latest_portfolio = prepare_latest_portfolio(selected_portfolio)
    industry_distribution = prepare_portfolio_industry_distribution(latest_portfolio)
    weight_distribution = prepare_portfolio_weight_distribution(latest_portfolio)
    summary = build_portfolio_summary(latest_portfolio)
    research_comment = build_portfolio_research_comment(
        latest_portfolio,
        industry_distribution,
    )
    return {
        "latest_portfolio": latest_portfolio,
        "industry_distribution": industry_distribution,
        "weight_distribution": weight_distribution,
        "summary": summary,
        "research_comment": research_comment,
    }
