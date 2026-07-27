# Modeling Panel Pipeline 使用指南

## 1. Modeling Panel 的作用

Factor Panel 是每个 `(trade_date, ts_code)` 对应的因子特征表；Forward
Returns 是未来收益标签以及 entry/exit 日期和价格审计字段。Modeling
Panel 将二者按唯一键一对一对齐，执行时间顺序、数据完整性和已知泄漏列
检查，形成可供 ML 读取的训练面板。

Modeling Panel 不训练模型、不自动选股、不自动调参，也不修改原始输入。
它提供结构化、可复现、可审计的工程边界，但不能仅凭列名证明不存在经济
意义上的未来函数；研究者仍需审查每个 feature 的生成时点和业务含义。

## 2. 数据流与两种 source mode

```text
Factor Research
  → final_factor_panel.parquet
  → forward_returns.parquet
  → ModelingPanelBuilder
  → modeling_panel.parquet
  → MLExperimentRunner
```

- `files`：从 YAML 中两条明确的 Parquet 路径读取。即使 Factor Research
  同时启用，也不会自动改用其输出。
- `factor_research`：只使用本次 Pipeline run 返回的 Factor Research
  published outputs，不读取旧 run，也不扫描“最新”目录。该模式要求
  `factor_research.enabled: true`，且 source 中两条文件路径必须为 null。

## 3. 输入表要求

Factor Panel 必须包含：

- `trade_date`
- `ts_code`
- 至少一个数值 feature

Factor Panel 不得包含 label 或 entry/exit 审计列。Forward Returns 必须包含：

- `trade_date`
- `ts_code`
- `entry_trade_date`
- `exit_trade_date`
- `entry_price`
- `exit_price`
- builder 配置指定的 label column，默认 `forward_return`

两侧 key 必须唯一。Feature、label 和价格采用数值 dtype，正负无穷会被
拒绝。Feature NaN 会进入审计；label NaN 是否允许由
`allow_missing_labels` 控制。非空 label 必须有完整的 entry/exit 信息。

## 4. 时间安全与 label 公式

signal date 即 `trade_date`。Entry 不得早于 signal；默认
`require_entry_after_signal: true` 时必须严格晚于 signal。Exit 必须严格晚于
entry。非空 label 必须在实现固定的数值容差内近似等于：

```text
exit_price / entry_price - 1
```

Builder 只验证该关系，不重算或覆盖用户 label。列名检查只能阻断已知的
结构性泄漏列，不能替代对 feature 经济含义的人工审查。

## 5. Unmatched policy

- `audit_and_drop`：记录未匹配观测并从输出中删除。
- `error`：任一方向出现未匹配观测即失败。

审计是双向的：`factor_only` 表示只有因子输入存在，`return_only` 表示只有
收益输入存在。审计记录行数、日期范围和确定性、有界的 sampled keys，
不会保存完整未匹配 DataFrame。

## 6. Artifact 布局与发布语义

默认输出位于：

```text
<run_dir>/modeling_panel/
├── modeling_panel.parquet
├── config.json
├── audit.json
└── manifest.json
```

Artifact 使用 manifest-last 和同父目录 staging 后原子发布。Manifest 记录
payload 的 SHA-256、文件大小和持久化 dtype。Store 拒绝 overwrite，不创建
backup；目录已存在时本次写入失败。Validator 会检查固定布局、hash、大小、
schema 和跨文件一致性。Artifact 不保存模型、估计器或原始输入表。

## 7. YAML 配置

可复制示例：
[modeling_panel_pipeline.example.yaml](../config/modeling_panel_pipeline.example.yaml)。
它使用 `PipelineConfig.from_dict()` 的完整 direct schema，默认状态为：

- Factor Research disabled
- Modeling Panel enabled，source 为 `files`
- ML disabled

Modeling Panel 配置块的真实结构：

```yaml
modeling_panel:
  enabled: true
  source:
    mode: files
    factor_panel_path: data/processed/modeling_factor_panel.parquet
    forward_returns_path: data/processed/modeling_forward_returns.parquet
  builder:
    label_column: forward_return
    include_features: null
    exclude_features: []
    unmatched_policy: audit_and_drop
    require_entry_after_signal: true
    allow_missing_labels: true
  output:
    save_artifact: true
    artifact_subdir: modeling_panel
    parquet_compression: zstd
    verify_after_write: true
```

`artifact_subdir` 只能是一个安全的相对目录名；实际目录为
`<run_dir>/<artifact_subdir>/`。ML enabled 时 panel 来源必须恰好一个：

- 链式模式：Modeling Panel enabled，`ml_experiment.panel_path: null`。
- 直接 ML：Modeling Panel disabled，显式提供 `.parquet` panel path。

Factor Research source 还要求 Factor Research enabled。Custom label 必须在
Factor Research published metadata、Modeling Panel builder 和 ML dataset
配置之间保持一致。

## 8. 三种运行方式

所有方式都使用同一命令，只修改 YAML：

```powershell
& "E:\FINANCIAL ENGINEERING\.venv\quant-factor-system\Scripts\python.exe" `
  scripts/run_pipeline.py `
  --config "config/modeling_panel_pipeline.example.yaml"
```

### A. files → Modeling Panel

```yaml
factor_research:
  enabled: false
modeling_panel:
  enabled: true
  source:
    mode: files
    factor_panel_path: data/processed/modeling_factor_panel.parquet
    forward_returns_path: data/processed/modeling_forward_returns.parquet
ml_experiment:
  enabled: false
  panel_path: null
```

不需要 ML `panel_path`。顺序为 cache → run_dir → Modeling Panel →
snapshots；输出位于 `<run_dir>/modeling_panel/`。

### B. Factor Research → Modeling Panel

```yaml
factor_research:
  enabled: true
  # 其余字段使用 FactorResearchPipelineConfig 的真实配置
modeling_panel:
  enabled: true
  source:
    mode: factor_research
    factor_panel_path: null
    forward_returns_path: null
ml_experiment:
  enabled: false
```

顺序为 Factor Research → Modeling Panel。Modeling Panel 只消费本次
Factor Research Result 的 published paths。

### C. Factor Research → Modeling Panel → ML

```yaml
factor_research:
  enabled: true
modeling_panel:
  enabled: true
  source:
    mode: factor_research
    factor_panel_path: null
    forward_returns_path: null
ml_experiment:
  enabled: true
  panel_path: null
  experiment:
    # 使用 MLExperimentConfig 的真实 dataset/walk_forward/training/evaluation 配置
```

顺序严格为 Factor Research → Modeling Panel → ML。用户无需复制
`modeling_panel.parquet` 路径；Runner 将本次 Result 返回的 absolute path
作为 override 传给 ML。

## 9. 路径语义

- `--config` 的相对路径基于启动 CLI 时的当前工作目录。
- 配置加载后 Pipeline 在项目根目录语义下运行；files source 的相对
  Parquet 路径和相对 `output_dir` 均基于项目根目录。
- ExperimentManager 在 `<output_dir>/runs/<run_id>/` 创建唯一 run_dir。
- Modeling Panel Artifact 位于 `<run_dir>/<artifact_subdir>/`。
- ML 读取 Store WriteResult 经 Pipeline Result 返回的实际
  `modeling_panel.parquet`，不会自行猜测文件名。

## 10. 查看与验证结果

CLI human summary 或 `--json` 输出包含 `modeling_panel` stage 摘要，包括
Artifact、panel path、features、label 和行数。进一步检查：

- `manifest.json`：固定布局、hash、大小、schema 和 dtype。
- `audit.json`：输入/输出行数、匹配覆盖、缺失值、时间和公式审计。

公开 Validator 示例：

```python
from src.modeling_panel import ModelingPanelArtifactStore

report = ModelingPanelArtifactStore().validate(
    "data/output/runs/<run_id>/modeling_panel"
)
print(report.is_valid)
print(report.issues)
```

## 11. 常见配置和数据错误

1. `factor_research` source 但 Factor Research disabled。
2. Modeling Panel 与 ML 同时开启，却又设置直接 ML `panel_path`。
3. ML enabled，但直接 path 与 Modeling Panel 两个来源都没有。
4. `files` mode 缺少 factor 或 returns path，或路径不是 `.parquet`。
5. `include_features` 与 published `feature_names` 不同序一致。
6. Custom label 在上游、Builder 或 ML dataset 中不一致。
7. `<run_dir>/<artifact_subdir>` 已存在；no-overwrite 会拒绝写入。
8. 任一输入存在 duplicate keys。
9. Entry/signal/exit 时间顺序错误。
10. Label 与 `exit_price / entry_price - 1` 不一致。

## 12. 与 V3 ML 的兼容性

固定 key、entry/exit audit 和 label 列不会成为 feature。Feature 顺序来自
Builder 验证后的审计结果。ML 读取已持久化的 Modeling Panel，不以内存
DataFrame 旁路 Artifact。旧的直接 ML `.parquet` panel path 模式保持兼容。

## 13. 当前非目标

当前不包含自动模型选择、自动超参数搜索、SHAP、portfolio construction、
实盘交易或 UI。Pipeline 输出用于研究和审计，不构成投资建议。
