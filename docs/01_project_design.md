# 项目设计

本文件用于记录 quant-factor-system 的项目目标、模块边界和版本规划。

## V0 范围

- 建立目录结构
- 添加配置模板
- 添加源码占位模块
- 添加方法论文档框架

## 一键研究流水线

### 作用

`scripts/run_research_pipeline.py` 用于串联数据拉取、因子构建、因子预处理、因子检验、多因子打分、组合回测和可视化，形成完整的量化研究流水线。

### 示例命令

```powershell
python scripts/run_research_pipeline.py --start 20240101 --end 20241231 --max-stocks 50 --top-n 10
```

### skip-fetch 用法

- 当已有 `data/raw` 数据时，可以使用 `--skip-fetch` 跳过 TuShare 拉取，节省时间和接口调用次数。

### 输出文件

- `data/processed/factor_panel.csv`
- `data/processed/factor_panel_clean.csv`
- `reports/tables/ic_summary.csv`
- `reports/tables/selected_portfolio.csv`
- `reports/tables/backtest_nav.csv`
- `reports/tables/backtest_metrics.csv`
- `reports/figures/nav_curve.png`
- `reports/figures/monthly_return_bar.png`
- `reports/figures/drawdown_curve.png`

### 风险提示

- 当前结果为历史样本回测和研究输出。
- 不代表未来收益。
- 不构成投资建议。