"""Central localized terminology, parameter, dataset, and factor metadata."""

from __future__ import annotations

from dataclasses import dataclass

from src.data.contracts import ResearchFrequency
from src.factors.registry import FactorRegistry


@dataclass(frozen=True)
class ParameterMetadata:
    key: str
    zh_name: str
    en_name: str
    zh_help: str
    en_help: str
    unit_zh: str
    unit_en: str
    input_scale: str
    example: str

    def label(self, locale: str) -> str:
        name = self.zh_name if locale == "zh-CN" else self.en_name
        unit = self.unit_zh if locale == "zh-CN" else self.unit_en
        return f"{name}（{unit}）" if locale == "zh-CN" else f"{name} ({unit})"

    def help(self, locale: str) -> str:
        body = self.zh_help if locale == "zh-CN" else self.en_help
        return f"{body} Input: {self.input_scale}; example: {self.example}."


PARAMETERS = {
    "annual_risk_free_rate": ParameterMetadata("annual_risk_free_rate", "年化无风险利率", "Annual risk-free rate", "用于绩效指标的年化利率；界面按百分比输入，并在控制器边界转换一次。", "Annual rate used by performance metrics; entered as a percentage and converted once at the UI boundary.", "%/年", "% p.a.", "2 means 2%", "2.0"),
    "initial_nav": ParameterMetadata("initial_nav", "初始净值基准", "Initial NAV baseline", "无货币单位的净值起点，不代表投入金额。", "Dimensionless NAV starting baseline; it is not invested cash.", "无量纲", "dimensionless", "direct decimal", "1.0"),
    "lookback_trading_days": ParameterMetadata("lookback_trading_days", "回看窗口", "Lookback window", "截至形成日（含）的历史开市观察窗口。", "Historical open-session window ending at and including formation.", "交易日", "trading days", "positive integer", "60"),
    "top_n": ParameterMetadata("top_n", "入选数量", "Top N", "每个形成日按信号排序后保留的证券数量。", "Number of securities retained after signal ranking at each formation.", "只", "securities", "positive integer", "10"),
    "max_weight": ParameterMetadata("max_weight", "单只证券最大权重", "Maximum security weight", "界面按百分比输入，在配置边界转换为 0–1 小数一次。", "Entered as a percentage and converted once to a 0–1 decimal at the config boundary.", "%", "%", "20 means 20%", "20"),
    "forward_entry_lag_periods": ParameterMetadata("forward_entry_lag_periods", "远期收益入场滞后", "Forward-return entry lag", "形成日后延迟多少个开市观察期再进入。", "Number of open observation periods after formation before entry.", "观察期", "observation periods", "non-negative integer", "1"),
    "forward_holding_periods": ParameterMetadata("forward_holding_periods", "远期收益持有期", "Forward-return holding horizon", "从入场日到退出日跨越的真实开市观察期数。", "Open observation periods from entry to exit.", "观察期", "observation periods", "positive integer", "5"),
    "training_cutoff": ParameterMetadata("training_cutoff", "训练截止日", "Training cutoff", "只有 available_at 不晚于该日的标签才可训练。", "Only labels whose available_at is no later than this date may train the model.", "日期", "date", "YYYY-MM-DD", "2024-01-31"),
}


DATASETS = {
    "trade_cal": ("交易日历", "Trading calendar", "自然日"),
    "stock_basic": ("证券基础信息", "Security reference", "快照"),
    "daily": ("股票日行情", "Daily security market data", "交易日"),
    "daily_basic": ("每日基本面指标", "Daily fundamentals", "交易日"),
    "adj_factor": ("复权因子", "Adjustment factor", "交易日"),
    "suspend_d": ("停牌事件", "Suspension events", "交易日"),
    "index_daily": ("指数日行情", "Daily index market data", "交易日"),
    "index_weight": ("指数历史成分", "Historical index membership", "月份"),
}


FACTOR_FORMULAS = {
    "momentum_20d": "close / close.shift(20) - 1",
    "volatility_20d": "sample standard deviation of daily close returns over 20 observations (ddof=1)",
    "momentum_60d": "close / close.shift(60) - 1",
    "momentum_120d": "close / close.shift(120) - 1",
    "momentum_252_20d": "close.shift(20) / close.shift(252) - 1",
    "short_term_reversal_5d": "-(close / close.shift(5) - 1)",
    "price_52w_high": "close / rolling_max(close, 252)",
    "volatility_60d": "sample standard deviation of daily close returns over 60 observations (ddof=1)",
    "turnover_mean_20d": "rolling_mean(turnover_rate, 20)",
    "amihud_20d": "20-observation mean of absolute daily close return divided by amount",
    "ep_ttm": "1 / pe_ttm (positive values only)",
    "bp": "1 / pb (positive values only)",
    "sp_ttm": "1 / ps_ttm (positive values only)",
    "log_total_mv": "ln(total_mv) (positive values only)",
    "log_circ_mv": "ln(circ_mv) (positive values only)",
}


@dataclass(frozen=True)
class FactorExplanation:
    code: str
    name: str
    description: str
    formula: str
    source_fields: tuple[str, ...]
    lookback: int
    frequency: str
    direction: str
    unit: str
    availability_lag_days: int


def factor_explanations(registry: FactorRegistry, frequency: ResearchFrequency) -> tuple[FactorExplanation, ...]:
    rows: list[FactorExplanation] = []
    for metadata in registry.list_metadata():
        try:
            spec = metadata.frequency_spec(frequency)
        except Exception:
            continue
        rows.append(FactorExplanation(
            code=metadata.name,
            name=metadata.name.replace("_", " ").title(),
            description=metadata.description,
            formula=FACTOR_FORMULAS.get(metadata.name, metadata.description),
            source_fields=metadata.source_fields,
            lookback=metadata.lookback_days,
            frequency=spec.research_frequency.value,
            direction="higher" if metadata.direction == 1 else "lower",
            unit="dimensionless unless the source description states otherwise",
            availability_lag_days=metadata.availability_lag_days,
        ))
    return tuple(rows)


def dataset_label(dataset_id: str, locale: str) -> str:
    value = DATASETS.get(dataset_id)
    if value is None:
        return dataset_id
    return value[0] if locale == "zh-CN" else value[1]


def dataset_unit(dataset_id: str, locale: str) -> str:
    value = DATASETS.get(dataset_id)
    if value is None:
        return "units"
    return value[2] if locale == "zh-CN" else {"自然日": "calendar days", "快照": "snapshots", "交易日": "trading days", "月份": "months"}[value[2]]
