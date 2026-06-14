# Pipeline Design

This document records the V1 data-layer and pipeline skeleton design for
`quant-factor-system`.

## Local Data Cache

The project needs a local data cache because A-share research pipelines are
repeatedly run over the same date ranges and stock pools. Caching avoids
unnecessary remote API calls, makes research runs reproducible, and allows the
pipeline skeleton to validate data readiness without requiring a real TuShare
token.

## Parquet Storage

Parquet is used for local tabular market data because it is compact, fast to
read, preserves column types better than CSV, and works naturally with pandas.
The V1 `ParquetStore` keeps dataset names such as `daily/000001.SZ` mapped to
paths such as `data/raw/daily/000001.SZ.parquet`.

## Cache Metadata

`data_status.json` records which date range each local dataset currently covers.
For example, the `daily` dataset can record a `start_date`, `end_date`, and
`updated_at` timestamp. V1 tracks one continuous coverage range per dataset.
This is enough for the pipeline runner to decide whether local data is ready or
which date range is still missing.

## V1 Pipeline Flow

The V1 pipeline skeleton follows this flow:

1. Load `PipelineConfig` from `config/config.yaml` and command-line overrides.
2. Compute `required_start_date` and `required_end_date`.
3. Ask `DataManager` to check cache coverage for required datasets.
4. Create a unique experiment run directory.
5. Save `config_snapshot.yaml`, `run_info.json`, and `metrics.json`.
6. Return a run summary with cache status and missing ranges.

## Required Data Range

The required data range starts before the backtest window so training data and
factor lookback windows are available:

```text
required_start_date = backtest_start - train_years - max_lookback_months
required_end_date = backtest_end
```

For example, `backtest_start = 2024-01-01`, `train_years = 10`, and
`max_lookback_months = 12` produce `required_start_date = 2013-01-01`.

## V1 Completed Scope

- V1-A data cache layer skeleton.
- V1-B PipelineConfig and ExperimentManager.
- V1-C runner and command line entry.
- V1-D tests and documentation.

## V1 Boundaries

- No real full-market data download.
- No complete factor calculation.
- No machine learning.
- No Streamlit UI refactor.
- No Kronos integration.

## Roadmap

- V2: Extend the factor library and IC-weighted strategies.
- V3: Add machine learning models.
- V4: Improve backtest realism.
- V5: Improve the Streamlit UI.
- V6: Add deployment and advanced model extensions.
