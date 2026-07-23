# Factor Research Workflow

## Overview

V2 adds a reproducible factor-research path to the existing pipeline entry
point. The current registry covers basic price/volume, valuation, profitability,
growth, quality, risk, and liquidity factors. Financial inputs must be
point-in-time (PIT) aligned before they enter the workflow.

The implemented flow is:

```text
raw input panels
  -> FactorEngine
  -> D1 cross-sectional preprocessing
  -> optional D2 industry/size neutralization
  -> G1 forward returns
  -> E1 IC and RankIC
  -> E2 quantile and long-short evaluation
  -> F1/F2 factor composition
  -> G3 Parquet tables and manifest
```

This is a factor-research and evaluation workflow, not an investment portfolio
backtest.

## Input Parquet panels

Three panels are required when `factor_research.enabled: true`:

- `factor_input`: full factor calculation history. It must contain
  `trade_date`, `ts_code`, and every raw field required by the selected factors.
  Price factors such as `momentum_20d` and `volatility_20d` require `close`.
- `score_panel`: the exact research keys, with `trade_date` and `ts_code`.
  Factor history must not be truncated to these dates before calculation.
- `price_panel`: market prices used only to build evaluation labels. It must
  contain `trade_date`, `ts_code`, and the configured price column (normally
  `close`).

`exposure_panel` is optional unless neutralization is enabled. It uses the same
keys and normally supplies `industry` and `log_total_mv`.

Financial factor fields must already use the project's standardized `fin_*`
names and must be PIT aligned. Vendor report dates must never be joined to
earlier score dates.

## Configuration and execution

Copy and edit `config/factor_research.example.yaml`. It is intentionally
disabled by default; prepare the input Parquet files, review every relative
path, and then set `factor_research.enabled: true`.

Relative research input paths resolve from the project root, independently of
the shell's current directory. Run the unified entry point in PowerShell:

```powershell
Set-Location "E:\FINANCIAL ENGINEERING\quant-factor-system"
& "E:\FINANCIAL ENGINEERING\.venv\quant-factor-system\Scripts\python.exe" scripts/run_pipeline.py --config config/factor_research.example.yaml
```

The compact success output reports the pipeline status and run directory. When
research is enabled it also reports the artifact directory, selected factors,
composition method, input shapes, major output shapes, and manifest
verification status. Add `--json` for a compact machine-readable summary.

Every invocation creates one independent run under:

```text
<data.output_dir>/runs/<timestamp>_<strategy>_<stock_pool>/
```

The research manifest is located at:

```text
run_dir/<artifact_subdir>/manifest.json
```

## Research artifacts

The `tables` directory can include:

- `raw_factor_panel`: selected raw factor values on score keys.
- `final_factor_panel`: values after D1 and optional D2.
- `forward_returns`: entry/exit audit fields and evaluation returns.
- `factor_ic_results`: component IC and RankIC by date.
- `factor_quantile_results`: component quantile and long-short results.
- `composite_scores`: equal, fixed, or rolling-IC composite scores.
- `composite_ic_results`: composite IC and RankIC results.

The manifest records schemas, shapes, relative file names, file sizes, and
SHA-256 checksums without embedding complete DataFrames. With
`verify_after_write: true`, publishing succeeds only after the staged artifact
passes integrity verification.

## Composition and forward returns

`equal` assigns equal base weights. `fixed` uses configured weights.
`rolling_ic` and `rolling_rank_ic` derive weights from historical IC or RankIC,
respectively. Dynamic weights for a score date only use evaluation observations
strictly earlier than that date; current-period IC never enters current-period
weights.

`entry_lag_periods` is the number of market observations between the score date
and entry. `holding_periods` is the number of market observations from entry to
exit. Forward returns are labels only and never become factor features.

## Current boundaries

The workflow does not implement automatic data download, historical universe
membership, Top N selection, holdings, rebalancing, transaction-cost or
slippage modeling, suspension matching, price-limit handling, or a real
portfolio equity curve.

## Troubleshooting

- Path not found: verify the project-relative path and Parquet file name.
- Factor not registered: choose names exposed by the current factor registry.
- `return_col` conflict: use the same value in evaluation, quantile, and
  forward-return sections.
- Insufficient samples: lower only valid research thresholds or provide more
  stocks per date; never bypass validation.
- Insufficient calendar: provide enough factor lookback and future price dates.
- Existing artifact with `overwrite: false`: use a new pipeline run directory
  or explicitly choose a different artifact target; do not overwrite silently.
