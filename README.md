# quant-factor-system

A 股多因子选股、回测与股票趋势分析应用的开源项目脚手架。

## 项目简介

本项目计划构建一个面向 A 股市场的多因子研究系统，覆盖数据获取、因子加工、因子检验、组合构建、策略回测、股票趋势分析与可视化展示。

当前版本为 V0 初始化版本，仅包含项目结构、配置文件、文档模板和 Python 占位模块，不包含复杂业务逻辑。

## 核心功能规划

- A 股行情与基础数据获取
- 数据清洗与样本数据生成
- 多因子计算与预处理
- 因子有效性检验
- 多因子综合打分与股票筛选
- 组合构建与回测分析
- 股票趋势分析与可视化
- Streamlit 交互式应用

## 项目结构

```text
quant-factor-system/
├── app/                  # Streamlit 应用与服务层
├── config/               # 项目配置
├── data/                 # 原始、处理后、样本和输出数据
├── docs/                 # 项目设计与方法论文档
├── reports/              # 图表和表格输出
├── scripts/              # 数据获取和样本数据脚本
├── src/                  # 核心源码包
├── tests/                # 测试用例
├── main.py               # 命令入口
└── requirements.txt      # Python 依赖
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
pytest
streamlit run app/streamlit_app.py
```

## 数据说明

项目默认使用 Tushare 作为数据源，并通过 `.env` 中的 `TUSHARE_TOKEN` 读取访问令牌。V0 阶段不会写入真实 token，也不会生成真实投资收益结果。

`data/raw/`、`data/processed/` 和 `data/output/` 默认被 Git 忽略，适合存放本地数据文件。

## 免责声明

本项目仅用于量化金融工程学习、研究和软件开发示例，不构成任何投资建议。证券市场有风险，投资决策需由使用者自行判断并承担后果。
