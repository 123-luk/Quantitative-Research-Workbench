# ML Experiment Guide

## Overview

The unified Pipeline CLI can optionally run one strictly out-of-sample machine
learning experiment from a pre-merged modeling panel. ML, permutation
importance, and artifact persistence are all disabled by default.

The currently supported models are:

- `ridge`
- `elastic_net`
- `hist_gradient_boosting`

LightGBM and XGBoost are not currently supported. Their optional dependencies
have not been installed or validated in the target Python environment.

Start from `config/ml_experiment.example.yaml`. The portable entry point is:

```text
python scripts/run_pipeline.py --config config/ml_experiment.example.yaml
```

## Modeling panel contract

The input is one pre-merged Parquet file. CSV is not supported. The Pipeline
does not automatically merge `final_factor_panel.parquet` with
`forward_returns.parquet`, and it does not generate labels.

The panel requires:

- `trade_date`
- `ts_code`
- `entry_trade_date`
- `exit_trade_date`
- the configured label column, normally `forward_return`
- at least one feature column

Feature inference excludes `trade_date`, `ts_code`, `entry_trade_date`,
`exit_trade_date`, `entry_price`, `exit_price`, and the configured label.
Every other auxiliary column is treated as a candidate feature. Prepare the
panel carefully so audit or vendor metadata is not accidentally modeled.

An offline preparation workflow may join a factor table and forward-return
table on their intended keys, validate uniqueness and point-in-time alignment,
and write the result as Parquet. This project deliberately leaves that merge
outside the Pipeline.

Relative panel paths resolve from the project root. Absolute paths are accepted
by the reader, but portable YAML should use relative paths.

## Configuration

The top-level `ml_experiment` section controls Pipeline integration:

- `enabled` opts into ML execution.
- `panel_path` identifies the pre-merged Parquet.
- `save_artifacts` opts into JSON and Parquet artifacts.
- `artifact_root` is a safe relative child of the Pipeline run directory.
- `experiment_id` is required when artifacts are enabled.
- `parquet_compression` is `zstd`, `snappy`, or `none`.
- `experiment` contains dataset, walk-forward, training, evaluation, and
  optional importance configuration.

The `dataset.label_col` setting identifies the prediction target. The target
is not passed to model fitting or prediction.

`walk_forward` uses the real fields:

- `train_window_periods`
- `validation_periods`
- `window_type`: `rolling` or `expanding`
- `retrain_frequency`
- `embargo_periods`

Splits enforce chronological training, validation, embargo, prediction, and
label-availability boundaries. Evaluation uses only consolidated out-of-sample
predictions and reports MAE, RMSE, R², Pearson IC, and RankIC.

Model parameters live only under `experiment.training.model_params`. For
example, HistGradientBoosting may use:

```yaml
training:
  model_name: hist_gradient_boosting
  model_params:
    learning_rate: 0.05
    max_iter: 200
    max_depth: 4
    min_samples_leaf: 20
    random_state: 42
```

## Permutation importance

Permutation importance is disabled by default with:

```yaml
permutation_importance: null
```

When enabled, it independently retrains the walk-forward folds. Features are
permuted only within each prediction block and within the same `trade_date`.
The importance model and parameters always come from the training section.

## Artifacts

Artifacts are disabled by default. When enabled they are written below:

```text
<output_dir>/runs/<run_id>/<artifact_root>/<experiment_id>/
```

The store writes validated JSON metadata and Parquet tables. Existing
experiment directories are never overwritten. No estimator, Adapter, model
pickle, or joblib file is saved.

The directory contains configuration and audit JSON, prediction and evaluation
Parquet tables, a manifest, and—when enabled—permutation-importance artifacts.
Use the public `MLExperimentArtifactStore.validate(path)` API to validate a
published experiment directory.

## CLI options

The ML CLI options are:

```text
--ml | --no-ml
--ml-panel PATH
--ml-model NAME
--ml-model-params JSON_OBJECT
--ml-permutation-importance | --no-ml-permutation-importance
--ml-importance-repeats INT
--ml-importance-scoring {rmse,mae}
--ml-min-cross-section-size INT
--ml-save-artifacts | --no-ml-save-artifacts
--ml-artifact-root RELATIVE_PATH
--ml-experiment-id ID
--ml-parquet-compression {zstd,snappy,none}
```

Precedence is built-in defaults, then YAML, then explicitly supplied CLI
values. Unspecified CLI values never replace YAML. Supplying a panel, model,
or artifact option does not automatically enable ML. `--ml-model-params`
accepts a strict JSON object and replaces the complete YAML parameter mapping;
it is not a shallow merge. Do not put tokens, passwords, or other secrets in
model parameters.

PowerShell example:

```powershell
python scripts/run_pipeline.py `
  --config config/ml_experiment.example.yaml `
  --ml `
  --ml-panel data/processed/ml_modeling_panel.parquet `
  --ml-model ridge `
  --ml-model-params '{"alpha":2.0}' `
  --json
```

To persist artifacts:

```powershell
python scripts/run_pipeline.py `
  --config config/ml_experiment.example.yaml `
  --ml `
  --ml-save-artifacts `
  --ml-experiment-id ridge-demo
```

The fixed interpreter used for development validation is:

```powershell
& "E:\FINANCIAL ENGINEERING\.venv\quant-factor-system\Scripts\python.exe" `
  scripts/run_pipeline.py --help
```

## Output and errors

Without `--json`, an enabled ML run appends a compact summary containing the
model, fold and prediction counts, regression metrics, IC means, importance
status, and artifact status. With `--json`, the existing Pipeline response gains
a compact `ml_experiment` object only when ML actually ran. Predictions, full
model parameters, DataFrames, and the artifact manifest are not printed.

Typed ML failures use these exit codes:

| Exit code | Meaning |
| --- | --- |
| 2 | CLI or ML configuration error |
| 3 | Modeling-panel error |
| 4 | ML execution or integrity error |
| 5 | Artifact experiment directory already exists |
| 6 | Other artifact write or validation error |

Argparse syntax errors also use exit code 2. Existing non-ML exceptions retain
their previous propagation and traceback behavior.

## Current boundaries

The current integration does not include automatic tuning, multi-model
comparison, portfolio backtesting, a Streamlit ML UI, automatic panel merging,
LightGBM, or XGBoost.
