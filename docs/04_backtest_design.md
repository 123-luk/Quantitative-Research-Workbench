# 回测设计

本文档记录第一版组合历史回测模块的输入、组合构建、收益计算、绩效指标和风险提示。

## 回测输入

- 输入文件：`selected_portfolio.csv`
- 数据含义：每月模型选出的 Top N 股票
- 标签字段：`return_next` 表示下一月收益

## 组合构建

- 调仓频率：月度调仓
- 权重设置：当期入选股票等权配置
- 简化假设：当前版本不考虑停牌、涨跌停无法成交、滑点等复杂交易约束，后续版本扩展

## 收益计算

- `gross_return = sum(weight * return_next)`
- `turnover` 根据相邻持仓权重变化计算
- `cost = turnover * transaction_cost`
- `net_return = gross_return - cost`
- `nav` 使用 `net_return` 复利累积

## 绩效指标

- `cumulative_return`
- `annual_return`
- `annual_volatility`
- `sharpe_ratio`
- `max_drawdown`
- `win_rate`
- `average_turnover`

## 风险提示

- 当前结果是历史样本回测，不代表未来表现。
- 当前版本为研究和工程验证用途。
- 后续可加入基准比较、交易成本细化、滑点、行业/个股权重约束、停牌处理等。

## 回测可视化

### 净值曲线

- 展示历史回测净值随时间变化。
- 基于 `net_return` 复利累积得到。

### 月度收益图

- 展示每期净收益。
- 用于观察收益波动和阶段性表现。

### 回撤曲线

- `drawdown = nav / historical_max_nav - 1`。
- 用于观察历史样本中的最大回撤和风险暴露。

### 风险提示

- 所有图表均为历史样本回测展示。
- 不代表未来收益。
- 后续可加入基准净值对比、超额收益曲线和行业暴露分析。