# quant-factor-system

## Research Workbench (v0.10.0)

The Streamlit Research Workbench now provides substantive **Overview**, **New
Run**, **Results**, **Runs**, and **Data** pages. Results and run navigation use
one exact canonical `run_id`; metrics, NAV, benchmark, Holdings, configuration,
and lineage come from validated Artifacts without latest/mtime fallback. The
Data page is a read-only view of local cache readiness and performs no download
or update. See `docs/16_research_workbench.md` and
`docs/17_research_workbench_results.md`.

A 股多因子选股、回测、单股研究与 Streamlit 可视化 App。

## 1. 项目简介

本项目是一个面向 A 股市场的本地量化研究系统，覆盖从数据获取到可视化展示的完整研究链路。

系统支持 TuShare 数据获取、因子构建、因子检验、多因子打分选股、组合回测、单股研究和 Streamlit 可视化展示。项目定位是学习、研究和展示用途，不构成投资建议。

## 2. 核心功能

1. 数据获取
   - TuShare 股票基础信息、交易日历、沪深300成分、月线行情、日频基础指标等。

2. 因子工程
   - 估值、规模、流动性、动量、波动率等月频因子。

3. 因子有效性检验
   - RankIC、ICIR、分组收益、多空收益。

4. 多因子评分选股
   - 因子方向设定、标准化评分、Top N 模型组合。

5. 组合回测
   - 月度调仓、等权持仓、交易成本、净值曲线、回撤、换手率。

6. 单只股票研究
   - 股票名称/代码查询、模型评级、趋势参考、研究摘要、因子暴露、价格与收益走势。

7. Streamlit App
   - 一键运行研究流水线、推荐组合页面、单股分析页面、回测结果页面、因子研究页面。

8. 一键启动脚本
   - 支持 `run_app.bat` 和 `run_app.ps1` 启动本地 App。

## 3. App 页面说明

- 首页 Dashboard
  - 展示历史样本回测核心指标和主要回测图表，用于快速查看系统输出概览。

- 运行研究流水线
  - 在 App 内设置研究区间、股票池、Top N、交易成本等参数，并调用本地研究流水线。

- 推荐投资组合
  - 展示最新一期模型组合、组合概览、行业分布、个股权重分布和组合研究说明。

- 单只股票分析
  - 支持按股票名称或代码查询，展示模型评分、趋势参考、研究摘要、因子暴露、价格走势和历史入选情况。

- 回测结果
  - 展示历史样本回测指标、净值曲线、月度收益、回撤、换手率和相关数据表。

- 因子研究
  - 展示因子 IC 排名、Top 因子、历史分组收益、历史多空收益和原始因子研究表。

## 4. 项目结构

```text
quant-factor-system/
├── app/
│   ├── streamlit_app.py
│   └── services/
├── src/
│   ├── data/
│   ├── factors/
│   ├── models/
│   └── backtest/
├── scripts/
├── docs/
├── data/
├── reports/
├── config/
├── run_app.bat
├── run_app.ps1
├── README.md
└── AGENT.md
```

`data/raw`、`data/processed` 和 `reports` 通常用于本地运行输出。不建议提交真实数据、Token 或大体积研究结果到 GitHub。

## 5. 环境配置

1. 创建虚拟环境

```bash
python -m venv .venv
```

2. 激活虚拟环境

```bash
.venv\Scripts\activate
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 配置 `.env`

```text
TUSHARE_TOKEN=your_token_here
```

不要把真实 Token 提交到 GitHub。Token 应仅通过本地 `.env` 文件或系统环境变量读取。

## 6. 启动 App

### 方式一：双击启动

在 Windows 文件夹中双击：

```text
run_app.bat
```

### 方式二：PowerShell 启动

在项目根目录运行：

```powershell
.\run_app.ps1
```

如果 PowerShell 执行策略阻止脚本运行，可以改用 `run_app.bat`。

### 方式三：手动启动

在项目根目录运行：

```bash
streamlit run app/streamlit_app.py
```

`run_app.bat` 和 `run_app.ps1` 会尝试自动检测并激活项目根目录下的 `.venv`。如果没有 `.venv`，则使用当前 Python 环境启动。如果依赖未安装，请先运行：

```bash
pip install -r requirements.txt
```

## 7. 数据与研究流水线运行

### 方式一：命令行运行

```bash
python scripts/run_research_pipeline.py --start 20240101 --end 20241231 --universe hs300 --max-stocks 50 --top-n 10
```

### 方式二：Streamlit App 内运行

启动 App 后进入“运行研究流水线”页面。

如果 `skip_fetch=True`，系统将使用已有 `data/raw` 数据；如果 `skip_fetch=False`，系统会调用 TuShare，需要在 `.env` 中配置 `TUSHARE_TOKEN`。

## 8. 主要输出

输出目录：

- `data/raw/`
- `data/processed/`
- `reports/tables/`
- `reports/figures/`

主要输出文件：

- `factor_panel.csv`
- `factor_panel_clean.csv`
- `ic_summary.csv`
- `group_return.csv`
- `long_short_return.csv`
- `factor_score.csv`
- `selected_portfolio.csv`
- `backtest_metrics.csv`
- `backtest_nav.csv`
- `backtest_turnover.csv`
- `nav_curve.png`
- `monthly_return_bar.png`
- `drawdown_curve.png`

## 9. 方法说明

本项目以月频样本为主，基于 TuShare 原始数据构建因子面板。因子在预处理阶段进行缺失值填补、缩尾和横截面标准化。

因子检验使用 RankIC、ICIR、历史分组收益和历史多空收益评估因子在样本期内的排序关系。多因子模型使用方向调整后的等权评分，并按月选择 Top N 股票形成模型组合。

组合回测采用月度调仓、等权持仓和交易成本扣减，输出历史回测净值、回撤、换手率和绩效指标。

`return_next` 是历史回测标签，只用于历史回测检验，不是未来收益预测。

## 10. 风险声明

本项目仅用于量化研究、学习和项目展示。项目中的模型评分、趋势参考、组合结果和回测结果均基于历史样本与量化规则生成，不代表未来表现，不构成任何投资建议。使用者应自行承担投资决策风险。

## 11. 后续改进方向

- 扩展样本区间和股票池。
- 增加更多基本面和技术面因子。
- 增加机器学习模型。
- 增加基准指数对比。
- 增加行业/风格约束。
- 增加模型参数配置页面。
- 增加报告导出功能。
- 增加部署版本或 Docker 环境。

## V3 machine-learning experiments

The unified Pipeline entry point supports one optional, strictly
out-of-sample ML experiment:

```text
python scripts/run_pipeline.py --config config/ml_experiment.example.yaml
```

ML is disabled by default. The current model registry supports `ridge`,
`elastic_net`, and `hist_gradient_boosting`. LightGBM and XGBoost are not
currently installed or supported.

The ML input is one pre-merged Parquet modeling panel. Artifact persistence
is also opt-in; it writes validated JSON and Parquet files under the current
Pipeline run directory and does not save an estimator or model file.

CLI values override only explicitly supplied YAML leaves. For example:

```text
python scripts/run_pipeline.py --config config/ml_experiment.example.yaml --ml --ml-model ridge --ml-model-params "{\"alpha\":2.0}"
```

See [ML Experiment Guide](docs/07_ml_experiment_guide.md) for the panel
contract, configuration, CLI options, metrics, artifacts, and exit codes.

## V4 Modeling Panel Pipeline

Modeling Panel 将因子特征表和 forward returns 按安全的一对一时间契约合并，
并发布可审计的 ML 输入 Artifact。当前数据链路为：

```text
Factor Research → Modeling Panel → ML Experiment
```

可复制配置：
[config/modeling_panel_pipeline.example.yaml](config/modeling_panel_pipeline.example.yaml)。
详细输入契约、三种 source/chain 方式和常见错误见
[Modeling Panel Pipeline 使用指南](docs/05_modeling_panel_pipeline.md)。

在项目根目录使用 PowerShell 运行：

```powershell
& "E:\FINANCIAL ENGINEERING\.venv\quant-factor-system\Scripts\python.exe" `
  scripts/run_pipeline.py `
  --config "config/modeling_panel_pipeline.example.yaml"
```

默认 Artifact 位于 `<run_dir>/modeling_panel/`：

| 文件 | 内容 |
| --- | --- |
| `modeling_panel.parquet` | 已验证的训练面板 |
| `config.json` | Builder 配置快照 |
| `audit.json` | 匹配、缺失值、时间和标签审计 |
| `manifest.json` | Schema、dtype、大小与 SHA-256 |

Artifact Store 不覆盖既有目录；重复目标会明确失败。该边界降低结构性泄漏
风险，但不替代对 feature 生成时点和经济含义的人工审查。

## V5 Signal and Holdings development pipeline

The canonical development path extends the Pipeline through ML, Signal, and
Holdings while keeping business settings in the shared YAML schema:

    python scripts/run_pipeline.py --config config/signal_holdings_pipeline.example.yaml

The example chooses holdings.top_n: 10; this is a user-selected example value,
while the backend canonical default remains 20. The retained legacy root top_n
and legacy research/scoring CLI do not define V5 Holdings behavior.

See the [Signal and Holdings Pipeline Guide](docs/08_signal_holdings_pipeline.md)
for source modes, direction and ranking, Top-N and insufficient-universe
semantics, equal weighting, Artifact layouts, provenance, and output inspection.
This is a V5 development capability, not a final v0.6.0 release announcement.

## V6 Research Backtest backend

The v0.7 development path now chains the canonical Pipeline through the native
Research Backtest backend:

    python scripts/run_pipeline.py --config config/research_backtest_pipeline.example.yaml

The backend consumes exact validated Holdings targets, remains
frequency-agnostic, and publishes a seven-file Research Backtest Artifact with
direct Holdings lineage. It is a historical research evaluator, not a live
execution system. See the
[Research Backtest Pipeline Guide](docs/10_research_backtest_pipeline.md) for
source modes, timing, return and suspension rules, cost/NAV semantics, metrics,
Artifact validation, and non-goals.

Local v0.7.0 release-candidate scope, frozen semantics, compatibility, and
manual release steps are recorded in the
[v0.7.0 Release Readiness checklist](docs/11_v0.7.0_release_readiness.md).
This local readiness record does not claim a merged PR, passing remote CI, or a
published `v0.7.0` tag.

## V7 Portfolio Construction

The canonical research chain is now Signal -> Top-N selection -> Portfolio
Construction weighting -> Holdings -> Research Backtest. Portfolio Construction
is an internal Holdings capability, not a separate Pipeline stage or Artifact;
it preserves the exact Top-N security set and changes only target weights.

See the [Portfolio Construction Guide](docs/12_portfolio_construction.md) and
run the generic config entry point with
`config/portfolio_construction_pipeline.example.yaml`.

The frozen v0.8.0 semantics, compatibility gates, validation scope, and manual
release procedure are recorded in the
[v0.8.0 Local Release Readiness checklist](docs/13_v0.8.0_release_readiness.md).

## V8 Risk Model and Minimum Variance

The research chain now supports another registry-driven weighting method:

```text
Signal -> Top-N -> Portfolio Construction
                    |- Equal Weight
                    |- Rank Weight
                    |- Inverse Volatility
                    `- Minimum Variance -> Risk Model
                 -> Holdings -> Research Backtest
```

Minimum Variance preserves the exact Top-N set and uses a common complete-case
historical covariance with long-only, fully-invested SLSQP optimization. The
Risk Model remains an internal Holdings capability, not a Pipeline stage or
Artifact. Run the standard config-only CLI with
`config/minimum_variance_pipeline.example.yaml` and see
[Risk Model and Minimum Variance](docs/14_risk_model_optimizer.md) for the
frozen covariance, dependency, optimization, and compatibility semantics.

## V9 Research Workbench development

The registry-driven Streamlit Research Workbench is under development for
v0.10.0. P1 provides the five-page shell and canonical New Run execution
foundation; Artifact-backed Results, Runs, and Data views remain P2 work.

    streamlit run app/streamlit_app.py

See [V9 Quant Research Workbench](docs/16_research_workbench.md) for the audited
registries, exact run handoff, artifact contracts, and phase boundaries.
