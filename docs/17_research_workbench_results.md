# Research Workbench: Artifact-Backed Results, Runs, and Data

V9-P2 completes the read-only surfaces of the v0.10.0 Research Workbench. It
does not add research calculations or change any Artifact schema.

## Exact-run result loading

`ResultService` accepts one canonical `run_id`. `ExperimentManager` reconstructs
only the direct path `<output_dir>/runs/<run_id>` and rejects malformed,
missing, non-directory, or symbolic-link identities. There is no `latest`
alias, filesystem-mtime ordering, glob fallback, or cross-run recovery.

The service reads the run's canonical `config_snapshot.yaml` and
`run_info.json` when present. Enabled stage directories come only from that
validated configuration. A present Artifact must pass its existing store
validator before any payload is displayed. Missing enabled stages are modeled
as unavailable; an invalid present Artifact is an error and is never read as a
best-effort result.

## Results views and sources

Results always opens from `selected_run_id` and has five fixed tabs:

- **Overview** reads the 20 canonical values from Research Backtest
  `metrics.json`. NAV uses only `daily_portfolio.parquet` `net_nav`, while the
  comparison series uses only `benchmark.parquet` `benchmark_nav`, joined on
  exact Artifact dates.
- **Holdings** reads the exact Holdings `holdings.parquet`, filters one
  formation date for display, and preserves canonical rank order and selected
  zero-weight rows.
- **Returns** displays canonical daily accounting columns. Because the backend
  has no monthly-return Artifact, the monthly table is explicitly a derived
  display view using `prod(1 + net_return) - 1` by calendar month.
- **Config** displays only fields present in the canonical run snapshot and
  offers the detached raw mapping in a collapsed JSON view. It never uses the
  current UI draft.
- **Artifacts** reports validated Signal, Holdings, Research Backtest, and safe
  ML identity when available, including relative path, schema version, and
  recorded upstream lineage.

The Research Backtest schema has no canonical drawdown series. The chart is
therefore derived as `net_nav / net_nav.cummax() - 1` for presentation only.
The Max Drawdown card remains the exact `metrics.json` value. A mismatch is
reported and never overwrites canonical truth.

## Run catalog

`ExperimentManager.list_run_ids()` enumerates only valid direct child
identities in stable lexical order, and `resolve_run_dir()` validates every
selected path. `RunCatalogService` reads canonical run metadata and validated
Research Backtest metrics. It uses canonical `run_info.json.created_at` for
chronological ordering when that field is valid; entries without it retain a
deterministic run-ID order and are not described as latest.

Pipeline success currently persists `run_info.json` only at the normal end of
the runner. A failure before that point may leave the exact callback-created
run directory without a persisted status, failed stage, or reason. The catalog
shows unavailable fields as `N/A` and does not invent failure metadata.

Selecting **Open Results** writes the exact run ID to the small navigation
session state. No DataFrame or Artifact payload is retained in session state.

## Read-only data status

`DataStatusService` delegates to the existing read-only
`DataManager.prepare_data()`, `DataCache` metadata, and `ParquetStore` path and
existence contracts. The Data page can reliably show:

- configured, raw-data, and cache-metadata paths;
- required date range and datasets;
- ready/missing cache status and missing ranges;
- cached coverage and update timestamp when already recorded; and
- each exact dataset path and whether it exists.

The current backend has no efficient canonical API for row count, security
count, file size, or min/max trade date independent of cache coverage. The UI
does not scan the data lake to manufacture those fields. It also exposes no
download, refresh, repair, delete, provider, or token action and performs no
network call or write.

## Canonical run layout

For a successful configured run, the relevant direct locations are:

```text
<output_dir>/runs/<run_id>/
  config_snapshot.yaml
  run_info.json
  <signal.artifact_subdir>/
  <holdings.artifact_subdir>/
  <research_backtest.artifact_subdir>/
```

The default subdirectories are `signal`, `holdings`, and
`research_backtest`. Their file and column contracts remain unchanged.

