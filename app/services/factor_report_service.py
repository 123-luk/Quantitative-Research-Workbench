"""Factor IC report helpers for historical factor research output."""

from __future__ import annotations

from typing import Any

import pandas as pd


FACTOR_IC_DISCLAIMER = "以上结果仅为历史样本因子有效性研究和量化研究参考，不代表未来表现，不构成投资建议。"
IC_SUMMARY_COLUMNS = [
    "factor",
    "mean_ic",
    "std_ic",
    "icir",
    "t_stat",
    "positive_ratio",
    "n_periods",
]


def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible, returning None for missing values."""
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


def safe_number(value: Any, digits: int = 4, default: str = "N/A") -> str:
    """Format a numeric value with a fixed number of decimal places."""
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return default
    return f"{numeric_value:.{digits}f}"


def prepare_ic_summary(ic_summary: pd.DataFrame) -> pd.DataFrame:
    """Prepare and sort the IC summary table for factor research reporting."""
    if ic_summary.empty:
        return pd.DataFrame()

    summary = ic_summary.copy()
    output_cols = [column for column in IC_SUMMARY_COLUMNS if column in summary.columns]
    if not output_cols:
        return pd.DataFrame()

    summary = summary.loc[:, output_cols].copy()
    if "mean_ic" not in summary.columns:
        return summary.reset_index(drop=True)

    summary["mean_ic"] = pd.to_numeric(summary["mean_ic"], errors="coerce")
    if "factor" in summary.columns:
        summary["abs_mean_ic"] = summary["mean_ic"].abs()
        summary = summary.sort_values("abs_mean_ic", ascending=False)
    return summary.reset_index(drop=True)


def prepare_factor_rank_table(ic_summary: pd.DataFrame) -> pd.DataFrame:
    """Prepare a factor IC ranking table with historical effectiveness labels."""
    rank_table = prepare_ic_summary(ic_summary)
    if rank_table.empty:
        return pd.DataFrame()

    labels: list[str] = []
    for _, row in rank_table.iterrows():
        mean_ic = _safe_float(row.get("mean_ic"))
        positive_ratio = _safe_float(row.get("positive_ratio"))
        if mean_ic is None or positive_ratio is None:
            labels.append("N/A")
        elif abs(mean_ic) >= 0.05 and positive_ratio >= 0.60:
            labels.append("历史 IC 表现较强")
        elif abs(mean_ic) >= 0.03 and positive_ratio >= 0.50:
            labels.append("历史 IC 表现中等")
        else:
            labels.append("历史 IC 表现较弱")

    rank_table = rank_table.copy()
    rank_table["effectiveness_label"] = labels
    if "abs_mean_ic" in rank_table.columns:
        rank_table = rank_table.sort_values("abs_mean_ic", ascending=False)
    return rank_table.reset_index(drop=True)


def get_top_factors(factor_rank_table: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return the top-ranked factors from a factor IC ranking table."""
    if factor_rank_table.empty:
        return pd.DataFrame()
    return factor_rank_table.head(top_n).reset_index(drop=True)


def build_factor_ic_comment(
    factor_rank_table: pd.DataFrame,
    top_n: int = 5,
) -> str:
    """Build a neutral Chinese research summary for historical RankIC results."""
    if factor_rank_table.empty:
        return f"暂无可用因子 IC 摘要数据。{FACTOR_IC_DISCLAIMER}"

    factor_count = len(factor_rank_table)
    top_factors = get_top_factors(factor_rank_table, top_n=top_n)
    if "factor" in top_factors.columns:
        factor_names = [str(value) for value in top_factors["factor"].dropna().tolist()]
    else:
        factor_names = []
    top_factor_text = "、".join(factor_names) if factor_names else "N/A"

    strong_count = 0
    if "effectiveness_label" in factor_rank_table.columns:
        strong_count = int((factor_rank_table["effectiveness_label"] == "历史 IC 表现较强").sum())
    if strong_count > 0:
        strength_text = f"其中 {strong_count} 个因子在样本期内被标记为历史 IC 表现较强。"
    else:
        strength_text = "当前样本中未识别出历史 IC 表现较强的因子。"

    return (
        f"本次因子 IC 研究为历史样本 RankIC 因子有效性研究，共覆盖 {factor_count} 个因子。"
        f"从 RankIC 结果看，IC 排名前 {top_n} 的因子包括 {top_factor_text}。"
        f"{strength_text}"
        f"需要注意的是，RankIC 仅反映历史样本中的因子排序相关性，不代表未来表现。"
        f"{FACTOR_IC_DISCLAIMER}"
    )


def prepare_factor_ic_report_data(ic_summary: pd.DataFrame) -> dict[str, object]:
    """Prepare IC summary, ranking, top factors, and research comment data."""
    ic_table = prepare_ic_summary(ic_summary)
    factor_rank_table = prepare_factor_rank_table(ic_summary)
    top_factors = get_top_factors(factor_rank_table, top_n=5)
    research_comment = build_factor_ic_comment(factor_rank_table, top_n=5)
    return {
        "ic_table": ic_table,
        "factor_rank_table": factor_rank_table,
        "top_factors": top_factors,
        "research_comment": research_comment,
    }


def prepare_long_short_summary(long_short_return: pd.DataFrame) -> pd.DataFrame:
    """Prepare historical long-short return summary by factor."""
    if long_short_return.empty:
        return pd.DataFrame()

    summary = long_short_return.copy()
    if "date" in summary.columns:
        summary["date"] = pd.to_datetime(summary["date"], errors="coerce")

    required_cols = {"factor", "long_short_return"}
    if not required_cols.issubset(summary.columns):
        return summary.reset_index(drop=True)

    summary["long_short_return"] = pd.to_numeric(summary["long_short_return"], errors="coerce")
    grouped = summary.groupby("factor", dropna=False)["long_short_return"].agg(
        mean_long_short_return="mean",
        std_long_short_return="std",
        positive_ratio=lambda values: (values > 0).mean(),
        n_periods="count",
    )
    return grouped.reset_index().sort_values("mean_long_short_return", ascending=False).reset_index(drop=True)


def prepare_group_return_summary(group_return: pd.DataFrame) -> pd.DataFrame:
    """Prepare historical quantile group return summary by factor and group."""
    if group_return.empty:
        return pd.DataFrame()

    summary = group_return.copy()
    if "date" in summary.columns:
        summary["date"] = pd.to_datetime(summary["date"], errors="coerce")

    return_col = "group_return" if "group_return" in summary.columns else None
    if return_col is None and "mean_return" in summary.columns:
        return_col = "mean_return"

    required_cols = {"factor", "group"}
    if not required_cols.issubset(summary.columns) or return_col is None:
        return summary.reset_index(drop=True)

    summary[return_col] = pd.to_numeric(summary[return_col], errors="coerce")
    grouped = (
        summary.groupby(["factor", "group"], dropna=False)[return_col]
        .mean()
        .reset_index(name="mean_group_return")
    )
    return grouped.sort_values(["factor", "group"]).reset_index(drop=True)


def build_factor_return_comment(
    long_short_summary: pd.DataFrame,
    top_n: int = 5,
) -> str:
    """Build a neutral Chinese summary for historical group and long-short returns."""
    if long_short_summary.empty:
        return "暂无可用的历史多空收益摘要数据。相关结果仅供量化研究参考，不代表未来表现，不构成投资建议。"

    factor_count = len(long_short_summary)
    top_table = long_short_summary.copy()
    if "mean_long_short_return" in top_table.columns:
        top_table = top_table.sort_values("mean_long_short_return", ascending=False)
    top_table = top_table.head(top_n)
    if "factor" in top_table.columns:
        factor_names = [str(value) for value in top_table["factor"].dropna().tolist()]
    else:
        factor_names = []
    top_factor_text = "、".join(factor_names) if factor_names else "N/A"

    return (
        f"本次因子收益研究为历史样本分组收益和多空收益研究，共覆盖 {factor_count} 个因子。"
        f"按历史平均多空收益排序，排名前 {top_n} 的因子包括 {top_factor_text}。"
        "相关多空收益和分组收益仅反映历史样本中的统计结果，不代表未来表现。"
        "以上结果仅为历史样本因子研究和量化研究参考，不构成投资建议。"
    )


def prepare_factor_report_data(
    ic_summary: pd.DataFrame,
    group_return: pd.DataFrame | None = None,
    long_short_return: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Prepare combined IC, group return, and long-short return report data."""
    ic_data = prepare_factor_ic_report_data(ic_summary)
    group_return_summary = (
        prepare_group_return_summary(group_return)
        if group_return is not None
        else pd.DataFrame()
    )
    long_short_summary = (
        prepare_long_short_summary(long_short_return)
        if long_short_return is not None
        else pd.DataFrame()
    )
    return_comment = build_factor_return_comment(long_short_summary)
    return {
        "ic_table": ic_data["ic_table"],
        "factor_rank_table": ic_data["factor_rank_table"],
        "top_factors": ic_data["top_factors"],
        "ic_research_comment": ic_data["research_comment"],
        "group_return_summary": group_return_summary,
        "long_short_summary": long_short_summary,
        "return_research_comment": return_comment,
    }
