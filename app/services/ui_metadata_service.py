"""Central localized terminology, parameter, dataset, and factor metadata."""

from __future__ import annotations

from dataclasses import dataclass

from src.data.contracts import ResearchFrequency
from src.factors.registry import FactorRegistry


@dataclass(frozen=True)
class LocalizedValue:
    canonical: str
    zh_name: str
    en_name: str

    def label(self, locale: str) -> str:
        return self.zh_name if locale == "zh-CN" else self.en_name


DISPLAY_VALUES = {
    "research_workbench": LocalizedValue("research_workbench", "量化研究工作台", "Research Workbench"),
    "CUSTOM": LocalizedValue("CUSTOM", "自定义证券池", "Custom securities"),
    "INDEX": LocalizedValue("INDEX", "指数成分股", "Index constituents"),
    "ALL_A_SHARES": LocalizedValue("ALL_A_SHARES", "全部 A 股", "All A-shares"),
    "DAILY": LocalizedValue("DAILY", "日频", "Daily"),
    "MONTHLY": LocalizedValue("MONTHLY", "月频", "Monthly"),
    "ridge": LocalizedValue("ridge", "岭回归", "Ridge regression"),
    "elastic_net": LocalizedValue("elastic_net", "弹性网络回归", "Elastic Net regression"),
    "hist_gradient_boosting": LocalizedValue("hist_gradient_boosting", "直方图梯度提升回归", "Histogram gradient boosting"),
    "equal_weight": LocalizedValue("equal_weight", "等权重", "Equal weight"),
    "rank_weight": LocalizedValue("rank_weight", "排名加权", "Rank weight"),
    "inverse_volatility": LocalizedValue("inverse_volatility", "波动率倒数加权", "Inverse-volatility weight"),
    "minimum_variance": LocalizedValue("minimum_variance", "最小方差", "Minimum variance"),
    "sample_covariance": LocalizedValue("sample_covariance", "样本协方差", "Sample covariance"),
    "ledoit_wolf": LocalizedValue("ledoit_wolf", "Ledoit-Wolf 收缩协方差", "Ledoit-Wolf shrinkage covariance"),
    "equal": LocalizedValue("equal", "因子等权合成", "Equal factor composition"),
    "rolling_ic": LocalizedValue("rolling_ic", "滚动 IC 加权", "Rolling IC weighting"),
    "rolling_rank_ic": LocalizedValue("rolling_rank_ic", "滚动秩 IC 加权", "Rolling rank-IC weighting"),
    "none": LocalizedValue("none", "不合成", "No composition"),
    "descending": LocalizedValue("descending", "得分从高到低", "Highest score first"),
    "ascending": LocalizedValue("ascending", "得分从低到高", "Lowest score first"),
    "error": LocalizedValue("error", "证券不足时停止", "Stop when insufficient"),
    "allow_partial": LocalizedValue("allow_partial", "证券不足时使用可用证券", "Use available securities"),
    "rolling": LocalizedValue("rolling", "滚动窗口", "Rolling window"),
    "expanding": LocalizedValue("expanding", "扩展窗口", "Expanding window"),
    "auto": LocalizedValue("auto", "自动选择", "Automatic"),
    "cyclic": LocalizedValue("cyclic", "循环更新", "Cyclic"),
    "random": LocalizedValue("random", "随机更新", "Random"),
    "squared_error": LocalizedValue("squared_error", "平方误差", "Squared error"),
    "absolute_error": LocalizedValue("absolute_error", "绝对误差", "Absolute error"),
    "poisson": LocalizedValue("poisson", "泊松损失", "Poisson loss"),
    "quantile": LocalizedValue("quantile", "分位数损失", "Quantile loss"),
    "succeeded": LocalizedValue("succeeded", "已完成", "Completed"),
    "failed": LocalizedValue("failed", "失败", "Failed"),
    "valid": LocalizedValue("valid", "校验通过", "Valid"),
    "signal": LocalizedValue("signal", "交易信号", "Signals"),
    "holdings": LocalizedValue("holdings", "目标持仓", "Holdings"),
    "research_backtest": LocalizedValue("research_backtest", "研究回测", "Research backtest"),
    "ml": LocalizedValue("ml", "机器学习模型", "Machine-learning model"),
    "higher": LocalizedValue("higher", "数值越高越优", "Higher is preferred"),
    "lower": LocalizedValue("lower", "数值越低越优", "Lower is preferred"),
}


PARAMETER_NAMES = {
    "alpha": ("正则化强度 Alpha", "Regularization alpha"), "l1_ratio": ("L1 正则占比", "L1 ratio"),
    "fit_intercept": ("拟合截距", "Fit intercept"), "solver": ("求解器", "Solver"),
    "tol": ("收敛容差", "Tolerance"), "max_iter": ("最大迭代次数", "Maximum iterations"),
    "positive": ("仅允许非负系数", "Positive coefficients"), "random_state": ("随机种子", "Random seed"),
    "selection": ("坐标更新顺序", "Coordinate selection"), "warm_start": ("热启动", "Warm start"),
    "loss": ("损失函数", "Loss"), "quantile": ("目标分位数", "Quantile"),
    "learning_rate": ("学习率", "Learning rate"), "max_leaf_nodes": ("最大叶节点数", "Maximum leaf nodes"),
    "max_depth": ("最大树深度", "Maximum depth"), "min_samples_leaf": ("叶节点最少样本数", "Minimum samples per leaf"),
    "l2_regularization": ("L2 正则化强度", "L2 regularization"), "max_features": ("最大特征比例", "Maximum feature fraction"),
    "max_bins": ("最大分箱数", "Maximum bins"), "early_stopping": ("提前停止", "Early stopping"),
    "n_iter_no_change": ("无改进容忍轮数", "Iterations without improvement"), "verbose": ("训练日志级别", "Verbosity"),
}

PARAMETER_HELP_ZH = {
    "alpha": "控制整体正则化强度，数值越大约束越强。", "l1_ratio": "控制 L1 正则在组合惩罚中的比例。",
    "fit_intercept": "是否由模型估计截距项。", "solver": "选择岭回归使用的数值求解算法。",
    "tol": "优化或提前停止使用的最小收敛容差。", "max_iter": "限制模型训练的最大迭代次数。",
    "positive": "启用后将模型系数限制为非负。", "random_state": "用于可复现随机过程的种子；留空表示不指定。",
    "selection": "选择弹性网络逐个更新坐标的顺序。", "warm_start": "是否允许估计器复用上一次拟合状态。",
    "loss": "选择梯度提升模型优化的回归损失。", "quantile": "仅在选择分位数损失时使用，必须介于 0 和 1 之间。",
    "learning_rate": "控制每轮提升对最终模型的贡献。", "max_leaf_nodes": "限制每棵树的最大叶节点数；留空表示不限制。",
    "max_depth": "限制每棵树的最大深度；留空表示不限制。", "min_samples_leaf": "每个叶节点所需的最少训练样本数。",
    "l2_regularization": "对叶节点取值施加 L2 正则约束。", "max_features": "每个节点可考虑的最大特征比例。",
    "max_bins": "每个特征用于直方图训练的最大分箱数。", "early_stopping": "是否基于外部时间验证集提前停止。",
    "n_iter_no_change": "验证结果连续多少轮没有改善后停止。", "verbose": "控制模型训练输出的详细程度。",
}


def display_value(value: object, locale: str) -> str:
    canonical = str(value)
    item = DISPLAY_VALUES.get(canonical)
    return item.label(locale) if item else canonical.replace("_", " ").title()


def parameter_label(name: str, locale: str, fallback: str) -> str:
    value = PARAMETER_NAMES.get(name)
    return (value[0] if locale == "zh-CN" else value[1]) if value else fallback


def parameter_help(name: str, locale: str, fallback: str) -> str:
    return PARAMETER_HELP_ZH.get(name, "模型训练参数。") if locale == "zh-CN" else fallback


RESULT_SETTING_LABELS = {
    "Start Date": ("开始日期", "Start Date"), "End Date": ("结束日期", "End Date"),
    "Stock Pool": ("证券池", "Stock Pool"), "Factors": ("因子", "Factors"),
    "Benchmark": ("基准指数", "Benchmark"), "Model": ("模型", "Model"),
    "Signal Direction": ("信号方向", "Signal Direction"), "Top N": ("入选数量", "Top N"),
    "Portfolio Method": ("组合方法", "Portfolio Method"), "Risk Estimator": ("风险估计方法", "Risk Estimator"),
    "Lookback Trading Days": ("回看交易日", "Lookback Trading Days"),
    "Minimum Observations": ("最少观测数", "Minimum Observations"),
    "Maximum Weight": ("单只证券最大权重", "Maximum Weight"),
    "Transaction Cost Bps": ("交易成本（基点）", "Transaction Cost (bps)"),
    "Annual Risk-Free Rate": ("年化无风险利率", "Annual Risk-Free Rate"),
    "Annualization Days": ("年化交易日数", "Annualization Days"),
}


def result_setting_label(value: str, locale: str) -> str:
    names = RESULT_SETTING_LABELS.get(value)
    return (names[0] if locale == "zh-CN" else names[1]) if names else value


def display_config_value(value: object, locale: str) -> object:
    if isinstance(value, (list, tuple)):
        return ", ".join(display_value(item, locale) for item in value)
    if isinstance(value, str):
        return display_value(value, locale)
    return value


def display_result_value(setting: str, value: object, locale: str) -> object:
    if setting == "Factors" and isinstance(value, (list, tuple)):
        return ", ".join(factor_label(str(item), locale) for item in value)
    if setting == "Stock Pool" and isinstance(value, str):
        if value.startswith("CUSTOM:"):
            name = "自定义证券池" if locale == "zh-CN" else "Custom securities"
            return f"{name} ({value.removeprefix('CUSTOM:')})"
        if value == "ALL_A_SHARES":
            return display_value(value, locale)
    return display_config_value(value, locale)


def assert_registry_display_metadata(*, models: tuple[str, ...], portfolios: tuple[str, ...], risks: tuple[str, ...]) -> None:
    missing = set((*models, *portfolios, *risks)) - set(DISPLAY_VALUES)
    if missing:
        raise RuntimeError(f"Missing centralized display metadata: {sorted(missing)!r}")


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
        if locale == "zh-CN":
            return f"{body} 输入尺度：{self.input_scale}；示例：{self.example}。"
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


FACTOR_NAMES_ZH = {
    "momentum_20d": "20 日动量", "volatility_20d": "20 日波动率", "momentum_60d": "60 日动量",
    "momentum_120d": "120 日动量", "momentum_252_20d": "过去一年动量（跳过近 20 日）",
    "short_term_reversal_5d": "5 日短期反转", "price_52w_high": "52 周高点比率",
    "volatility_60d": "60 日波动率", "turnover_mean_20d": "20 日平均换手率",
    "amihud_20d": "20 日 Amihud 非流动性", "ep_ttm": "滚动市盈率倒数（EP）",
    "bp": "市净率倒数（BP）", "sp_ttm": "滚动市销率倒数（SP）",
    "log_total_mv": "总市值对数", "log_circ_mv": "流通市值对数",
    "debt_to_assets": "资产负债率", "dividend_yield_ttm": "滚动股息率",
    "gross_margin_ttm": "滚动毛利率", "net_margin_ttm": "滚动净利率",
    "net_profit_yoy": "净利润同比增长率", "operating_cf_to_assets": "经营现金流资产比",
    "revenue_yoy": "营业收入同比增长率", "roa_ttm": "滚动总资产收益率（ROA）",
    "roe_ttm": "滚动净资产收益率（ROE）",
}

FACTOR_DESCRIPTIONS_ZH = {
    "momentum_20d": "衡量最近 20 个交易日的价格趋势。", "volatility_20d": "衡量最近 20 个交易日收益率的波动程度。",
    "momentum_60d": "衡量最近 60 个交易日的价格趋势。", "momentum_120d": "衡量最近 120 个交易日的价格趋势。",
    "momentum_252_20d": "衡量过去约一年、剔除最近 20 个交易日后的价格趋势。",
    "short_term_reversal_5d": "衡量最近 5 个交易日价格走势的反向效应。", "price_52w_high": "衡量当前价格接近过去 52 周高点的程度。",
    "volatility_60d": "衡量最近 60 个交易日收益率的波动程度。", "turnover_mean_20d": "衡量最近 20 个交易日的平均换手水平。",
    "amihud_20d": "以价格变动相对成交额衡量最近 20 个交易日的非流动性。", "ep_ttm": "以滚动市盈率倒数衡量盈利收益率。",
    "bp": "以市净率倒数衡量账面价值相对价格。", "sp_ttm": "以滚动市销率倒数衡量销售收入相对价格。",
    "log_total_mv": "以总市值的自然对数衡量公司规模。", "log_circ_mv": "以流通市值的自然对数衡量可交易规模。",
    "debt_to_assets": "衡量负债相对资产的比例。", "dividend_yield_ttm": "衡量过去十二个月股息相对价格的收益率。",
    "gross_margin_ttm": "衡量过去十二个月营业收入的毛利水平。", "net_margin_ttm": "衡量过去十二个月营业收入的净利润水平。",
    "net_profit_yoy": "衡量净利润相对上年同期的增长。", "operating_cf_to_assets": "衡量经营现金流相对总资产的规模。",
    "revenue_yoy": "衡量营业收入相对上年同期的增长。", "roa_ttm": "衡量过去十二个月利润相对总资产的回报。",
    "roe_ttm": "衡量过去十二个月利润相对净资产的回报。",
}


def factor_label(code: str, locale: str) -> str:
    if locale == "zh-CN":
        return FACTOR_NAMES_ZH.get(code, code.upper())
    return code.replace("_", " ").title()


def factor_explanations(registry: FactorRegistry, frequency: ResearchFrequency, locale: str = "en") -> tuple[FactorExplanation, ...]:
    rows: list[FactorExplanation] = []
    for metadata in registry.list_metadata():
        try:
            spec = metadata.frequency_spec(frequency)
        except Exception:
            continue
        rows.append(FactorExplanation(
            code=metadata.name,
            name=FACTOR_NAMES_ZH.get(metadata.name, metadata.name.replace("_", " ").title()) if locale == "zh-CN" else metadata.name.replace("_", " ").title(),
            description=FACTOR_DESCRIPTIONS_ZH.get(metadata.name, metadata.description) if locale == "zh-CN" else metadata.description,
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
