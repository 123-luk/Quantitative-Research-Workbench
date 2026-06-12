"""Single-stock research rating helpers based on local factor scores."""

from __future__ import annotations

import math
from typing import Any


DISCLAIMER = "以上结果基于历史样本中的多因子评分与量化规则生成，仅供研究参考，不代表未来表现，不构成投资建议。"


def safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None for missing or invalid values."""
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric_value):
        return None
    return numeric_value


def score_to_percentile_label(score_pct_rank: float | None) -> str:
    """Classify a score percentile rank into a sample-position label."""
    value = safe_float(score_pct_rank)
    if value is None:
        return "N/A"
    if value >= 0.8:
        return "样本前 20%"
    if value >= 0.6:
        return "样本前 40%"
    if value >= 0.4:
        return "样本中位附近"
    if value >= 0.2:
        return "样本后 40%"
    return "样本后 20%"


def classify_volatility(volatility_6m: float | None) -> str:
    """Classify standardized six-month volatility into a research label."""
    value = safe_float(volatility_6m)
    if value is None:
        return "N/A"
    if value <= -0.5:
        return "低波动"
    if value < 0.5:
        return "中等波动"
    return "高波动"


def classify_momentum(momentum_1m: float | None, momentum_3m: float | None) -> str:
    """Classify standardized one- and three-month momentum into a research label."""
    values = [safe_float(momentum_1m), safe_float(momentum_3m)]
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return "N/A"

    avg_momentum = sum(valid_values) / len(valid_values)
    if avg_momentum >= 0.5:
        return "动量较强"
    if avg_momentum >= 0:
        return "动量偏强"
    if avg_momentum >= -0.5:
        return "动量偏弱"
    return "动量较弱"


def _momentum_component(momentum_1m: float | None, momentum_3m: float | None) -> float:
    """Calculate the 0-25 momentum contribution from standardized factors."""
    values = [safe_float(momentum_1m), safe_float(momentum_3m)]
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return 12.5

    avg_momentum = sum(valid_values) / len(valid_values)
    if avg_momentum >= 1.0:
        return 25.0
    if avg_momentum >= 0.5:
        return 20.0
    if avg_momentum >= 0.0:
        return 15.0
    if avg_momentum >= -0.5:
        return 10.0
    return 5.0


def _volatility_component(volatility_6m: float | None) -> float:
    """Calculate the 0-15 volatility contribution from standardized volatility."""
    value = safe_float(volatility_6m)
    if value is None:
        return 7.5
    if value <= -1.0:
        return 15.0
    if value <= -0.5:
        return 12.0
    if value <= 0.5:
        return 8.0
    if value <= 1.0:
        return 5.0
    return 2.0


def _valuation_component(ep: float | None, bp: float | None, ps_inverse: float | None) -> float:
    """Calculate the 0-10 valuation contribution from standardized valuation factors."""
    values = [safe_float(ep), safe_float(bp), safe_float(ps_inverse)]
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return 5.0

    avg_valuation = sum(valid_values) / len(valid_values)
    if avg_valuation >= 0.5:
        return 10.0
    if avg_valuation >= 0.0:
        return 7.0
    if avg_valuation >= -0.5:
        return 5.0
    return 3.0


def calculate_trend_score(snapshot: dict[str, object]) -> dict[str, object]:
    """Calculate a 0-100 research score from the latest stock snapshot."""
    score_pct_rank = safe_float(snapshot.get("score_pct_rank"))
    score_component = score_pct_rank * 50.0 if score_pct_rank is not None else 25.0
    momentum_component = _momentum_component(
        snapshot.get("momentum_1m"),
        snapshot.get("momentum_3m"),
    )
    volatility_component = _volatility_component(snapshot.get("volatility_6m"))
    valuation_component = _valuation_component(
        snapshot.get("ep"),
        snapshot.get("bp"),
        snapshot.get("ps_inverse"),
    )
    research_score = (
        score_component
        + momentum_component
        + volatility_component
        + valuation_component
    )

    return {
        "research_score": research_score,
        "score_component": score_component,
        "momentum_component": momentum_component,
        "volatility_component": volatility_component,
        "valuation_component": valuation_component,
        "momentum_label": classify_momentum(
            snapshot.get("momentum_1m"),
            snapshot.get("momentum_3m"),
        ),
        "volatility_label": classify_volatility(snapshot.get("volatility_6m")),
        "percentile_label": score_to_percentile_label(score_pct_rank),
    }


def rating_from_score(research_score: float | None) -> str:
    """Convert a research score into an investment-attractiveness rating label."""
    value = safe_float(research_score)
    if value is None:
        return "N/A"
    if value >= 80:
        return "偏积极"
    if value >= 65:
        return "中性偏积极"
    if value >= 45:
        return "中性"
    if value >= 30:
        return "中性偏谨慎"
    return "偏谨慎"


def trend_reference_from_score(research_score: float | None) -> str:
    """Convert a research score into a half-year trend reference label."""
    value = safe_float(research_score)
    if value is None:
        return "N/A"
    if value >= 80:
        return "趋势参考偏强"
    if value >= 65:
        return "趋势参考略偏强"
    if value >= 45:
        return "趋势参考中性"
    if value >= 30:
        return "趋势参考略偏弱"
    return "趋势参考偏弱"


def generate_rating_explanation(
    snapshot: dict[str, object],
    trend_result: dict[str, object],
    selection_frequency: dict[str, object] | None = None,
) -> list[str]:
    """Generate neutral Chinese explanation lines for a stock research rating."""
    explanation = [
        f"模型显示综合评分处于{trend_result.get('percentile_label', 'N/A')}，用于描述历史样本中的相对位置。",
        f"近期动量标签为{trend_result.get('momentum_label', 'N/A')}，对趋势参考形成相应影响。",
        f"6个月波动率标签为{trend_result.get('volatility_label', 'N/A')}，反映历史样本中的相对风险暴露。",
    ]

    if snapshot.get("is_selected_latest"):
        explanation.append("模型显示该股票当前最新一期进入模型选股组合。")
    else:
        explanation.append("模型显示该股票当前最新一期未进入模型选股组合。")

    if selection_frequency:
        frequency = safe_float(selection_frequency.get("selection_frequency"))
        if frequency is not None:
            explanation.append(f"历史样本中进入模型组合的频率为 {frequency:.2%}。")
        else:
            explanation.append("历史样本中暂无法计算进入模型组合的频率。")

    explanation.append("return_next 仅作为历史回测标签，不作为未来收益预测。")
    return explanation


def build_stock_rating_report(
    snapshot: dict[str, object],
    selection_frequency: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a full neutral single-stock research rating report."""
    trend_result = calculate_trend_score(snapshot)
    research_score = trend_result.get("research_score")
    return {
        "research_score": research_score,
        "investment_attractiveness_rating": rating_from_score(research_score),
        "half_year_trend_reference": trend_reference_from_score(research_score),
        "percentile_label": trend_result.get("percentile_label"),
        "momentum_label": trend_result.get("momentum_label"),
        "volatility_label": trend_result.get("volatility_label"),
        "explanation": generate_rating_explanation(
            snapshot,
            trend_result,
            selection_frequency=selection_frequency,
        ),
        "disclaimer": DISCLAIMER,
    }
