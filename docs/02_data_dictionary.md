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
