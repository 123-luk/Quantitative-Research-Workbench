# 数据字典

本文档记录量化因子系统使用的原始数据表、来源、用途与字段。

## stock_basic.csv

- 来源：TuShare `stock_basic`
- 用途：A 股股票基础信息与股票池基础表
- 字段：`ts_code`, `symbol`, `name`, `area`, `industry`, `market`, `list_date`

## trade_cal.csv

- 来源：TuShare `trade_cal`
- 用途：交易日历与后续月末交易日识别
- 字段：`exchange`, `cal_date`, `is_open`, `pretrade_date`

## hs300_index_weight.csv

- 来源：TuShare `index_weight`
- 用途：沪深 300 成分权重历史记录
- 字段：`index_code`, `con_code`, `trade_date`, `weight`

## hs300_components.csv

- 来源：由 `hs300_index_weight.csv` 去重生成
- 用途：第一版默认股票池
- 字段：`ts_code`

## monthly.csv

- 来源：TuShare `monthly`
- 用途：月线行情、收益率、动量、波动率等因子构建
- 字段：`ts_code`, `trade_date`, `open`, `high`, `low`, `close`, `pre_close`, `change`, `pct_chg`, `vol`, `amount`

## daily_basic.csv

- 来源：TuShare `daily_basic`
- 用途：估值、规模、换手率等因子构建
- 字段：`ts_code`, `trade_date`, `close`, `turnover_rate`, `volume_ratio`, `pe`, `pe_ttm`, `pb`, `ps`, `ps_ttm`, `dv_ratio`, `total_mv`, `circ_mv`

## 数据安全说明

- `data/raw` 不应提交到 GitHub。
- `.env` 不应提交到 GitHub。
- sample 数据后续单独生成，用于开源演示。

## factor_panel.csv

- 来源：`monthly.csv`, `daily_basic.csv`, `stock_basic.csv`, `hs300_components.csv` 加工生成
- 用途：后续因子检验、打分选股、回测主数据表
- 字段说明：
  - `date`：月末日期
  - `ts_code`：股票代码
  - `name`：股票名称
  - `industry`：所属行业
  - `close`：月收盘价
  - `monthly_return`：月收益率，`monthly_return = close / pre_close - 1`
  - `return_next`：下一期月收益标签
  - `ep`：市盈率倒数，`1 / pe_ttm`
  - `bp`：市净率倒数，`1 / pb`
  - `ps_inverse`：市销率倒数，`1 / ps_ttm`
  - `size_factor`：规模因子，`-log(total_mv)`
  - `turnover_factor`：换手率因子
  - `amount_factor`：成交额因子，`log(amount)`
  - `momentum_1m`：上一月收益率
  - `momentum_3m`：过去 3 个月滚动累计收益，不包含当前月
  - `volatility_6m`：过去 6 个月收益率滚动标准差，不包含当前月

- `factor_panel.csv` 属于 `data/processed`，不提交到 GitHub。
- `return_next` 是下月收益标签，构造时必须避免未来数据泄露。
- `pct_chg` 保留为 TuShare 原始字段，不作为 `monthly_return` 的计算来源。
