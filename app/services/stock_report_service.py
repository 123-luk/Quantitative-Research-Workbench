"""Natural-language stock research summaries for dashboard pages."""

from __future__ import annotations

import math
from typing import Any


FACTOR_COMMENTS = {
    "ep": ("盈利收益率因子", "高于样本均值", "低于样本均值"),
    "bp": ("账面市值比因子", "高于样本均值", "低于样本均值"),
    "ps_inverse": ("市销率倒数因子", "高于样本均值", "低于样本均值"),
    "size_factor": ("规模因子", "高于样本均值", "低于样本均值"),
    "turnover_factor": ("换手率因子", "高于样本均值", "低于样本均值"),
    "amount_factor": ("成交额因子", "高于样本均值", "低于样本均值"),
    "momentum_1m": ("1个月动量", "高于样本均值", "低于样本均值"),
    "momentum_3m": ("3个月动量", "高于样本均值", "低于样本均值"),
    "volatility_6m": ("6个月波动率", "高于样本均值，风险暴露相对更高", "低于样本均值，风险暴露相对更低"),
}


def _to_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric_value):
        return None
    return numeric_value


def safe_text(value: Any, default: str = "N/A") -> str:
    """Return a safe text representation for missing-aware report output."""
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value)
    if not text or text.lower() == "nan":
        return default
    return text


def safe_percent(value: Any, default: str = "N/A") -> str:
    """Format a decimal value as a percentage string with two decimals."""
    numeric_value = _to_float(value)
    if numeric_value is None:
        return default
    return f"{numeric_value:.2%}"


def safe_number(value: Any, digits: int = 2, default: str = "N/A") -> str:
    """Format a numeric value with a fixed number of decimal places."""
    numeric_value = _to_float(value)
    if numeric_value is None:
        return default
    return f"{numeric_value:.{digits}f}"


def build_stock_research_summary(
    snapshot: dict[str, object],
    rating_report: dict[str, object],
    selection_frequency: dict[str, object] | None = None,
) -> str:
    """Build a structured Chinese natural-language stock research summary."""
    selection_frequency = selection_frequency or {}
    name = safe_text(snapshot.get("name"))
    ts_code = safe_text(snapshot.get("ts_code"))
    latest_date = safe_text(snapshot.get("latest_date"))
    rating = safe_text(rating_report.get("investment_attractiveness_rating"))
    trend_reference = safe_text(rating_report.get("half_year_trend_reference"))
    research_score = safe_number(rating_report.get("research_score"), digits=2)
    percentile_label = safe_text(rating_report.get("percentile_label"))
    momentum_label = safe_text(rating_report.get("momentum_label"))
    volatility_label = safe_text(rating_report.get("volatility_label"))
    selected_latest = "进入" if snapshot.get("is_selected_latest") else "未进入"
    selected_periods = safe_text(selection_frequency.get("selected_periods"), default="0")
    selection_freq = safe_percent(selection_frequency.get("selection_frequency"), default="N/A")
    disclaimer = safe_text(
        rating_report.get("disclaimer"),
        default="以上结果基于历史样本中的多因子评分与量化规则生成，仅供研究参考，不代表未来表现，不构成投资建议。",
    )

    return (
        f"{name}（{ts_code}）在最新一期 {latest_date} 的模型分析中，"
        f"投资吸引力评级为{rating}，未来半年趋势参考为{trend_reference}，"
        f"research_score 为 {research_score}。综合评分位置处于{percentile_label}，"
        f"近期动量为{momentum_label}，6个月波动率为{volatility_label}。"
        f"该股票最新一期{selected_latest}模型选股组合，历史样本中进入模型组合 "
        f"{selected_periods} 次，入选频率为 {selection_freq}。"
        f"{disclaimer}"
    )


def build_factor_exposure_comment(snapshot: dict[str, object]) -> list[str]:
    """Build short comments for standardized factor exposures in a snapshot."""
    comments: list[str] = []
    for factor, (label, positive_text, negative_text) in FACTOR_COMMENTS.items():
        value = _to_float(snapshot.get(factor))
        if value is None:
            continue
        if value > 0:
            comments.append(f"{label}{positive_text}。")
        elif value < 0:
            comments.append(f"{label}{negative_text}。")
        else:
            comments.append(f"{label}接近样本均值。")
    return comments
