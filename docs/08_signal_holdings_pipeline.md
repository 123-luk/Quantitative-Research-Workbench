# Signal and Holdings Pipeline Guide

> Development capability for the planned V5/v0.6.0 line. This document does
> not announce a final v0.6.0 release or production trading readiness.

## Purpose and architecture

The canonical Pipeline can turn strictly out-of-sample ML predictions into a
deterministic Signal Artifact and then into long-only target Holdings:

```text
Data readiness
  -> Factor Research
  -> Modeling Panel
  -> ML Experiment
  -> Signal
  -> Holdings
  -> run metadata
```

Signal and Holdings have separate responsibilities:

- Signal maps one validated prediction column to `score`, applies the configured
  direction, and assigns a deterministic rank within each `trade_date`.
- Holdings selects existing Signal ranks and assigns target weights. It never
  recalculates scores or ranks.

The flow is therefore `prediction -> score -> rank -> holdings.top_n`.

## Canonical command and configuration

Run the V5 development path through the canonical Pipeline CLI:

```powershell
python scripts/run_pipeline.py --config config/signal_holdings_pipeline.example.yaml
```

The same command works in bash. Signal and Holdings business settings come
from the YAML parsed as `PipelineConfig`; no Signal/Holdings-specific CLI flags
exist. The older `--top-n` option on `scripts/run_pipeline.py` is a retained
legacy root-workflow override and is not the V5 Holdings source of truth.
Legacy `scripts/run_research_pipeline.py` and scoring commands are also not the
canonical V5 Holdings path.

The complete copyable example is
[`config/signal_holdings_pipeline.example.yaml`](../config/signal_holdings_pipeline.example.yaml).
Its relevant nested blocks are:

```yaml
signal:
  enabled: true
  source:
    mode: ml
    artifact_dir: null
  prediction_column: prediction
  signal_direction: descending
  artifact_subdir: signal

holdings:
  enabled: true
  top_n: 10
  insufficient_universe_policy: error
  weighting: equal_weight
  artifact_subdir: holdings
```

The example selects `10`; the backend canonical default in
`HoldingsPipelineConfig` remains `20`. The direct example also keeps the legacy
root `top_n` at 10 because enabled Holdings rejects conflicting root/nested
values. That root field remains part of the old workflow and does not become a
second V5 source of truth.

Both direct PipelineConfig YAML and the older grouped YAML layout remain
supported. In grouped YAML, `signal` and `holdings` are top-level nested blocks,
while the legacy root N remains under `pipeline.top_n`.

## Signal source modes

### Current-run ML source

```yaml
signal:
  enabled: true
  source:
    mode: ml
    artifact_dir: null
```

This consumes the exact `MLExperimentPipelineResult` produced by the current
run. `ml_experiment.enabled` must be true. A successful real run must persist
the ML Artifact (`save_artifacts: true`) so the result exposes an Artifact
directory. The Runner does not search historical runs, reconstruct a path, or
fall back to another source.

### Explicit files source

```yaml
signal:
  enabled: true
  source:
    mode: files
    artifact_dir: path/to/native/ml/artifact
```

`artifact_dir` must identify the complete native ML Artifact directory. It is
not a path to `predictions.parquet`; a bare Parquet file and a generic external
prediction file are rejected. The existing ML Artifact validator verifies the
directory before predictions are read. No latest-run, mtime, glob, or sibling
directory discovery occurs. The current run's ML stage may be disabled, but
Signal still executes in its canonical position after the skipped ML stage.

## Signal direction and deterministic ranking

- `descending`: larger scores rank first; rank 1 has the highest score.
- `ascending`: smaller scores rank first; rank 1 has the lowest score.
- Ties use `ts_code` ascending as the deterministic secondary key.

Direction describes score preference only; it is not a claim to “buy rising”
or “buy falling” securities. Ranking is isolated by `trade_date`, and appending
future dates does not alter past ranks.

## Holdings Top-N semantics

`holdings.top_n` is the maximum number of securities held on each rebalance
`trade_date`. With `top_n: 10`, a normal date selects Signal ranks 1 through 10;
it does not choose ten securities across the complete history. Selection does
not change Signal `score` or `rank`.

For a complete universe under `equal_weight`:

- `top_n: 10` produces 10 holdings at approximately 10% each.
- `top_n: 20` produces 20 holdings at approximately 5% each.

The approximation wording reflects floating-point representation. There is no
financial-return promise in these examples.

## Insufficient universe policy

`error` is the default and recommended strict mode. If any date has fewer than
the requested N valid Signals, the complete Holdings build fails; N is not
silently reduced.

`allow_partial` uses all K available securities on an insufficient date. The
requested N remains unchanged in config and audit, the selected count is below
N, weights use the actual K, and the audit records the partial date. For example,
with requested N=10 and seven valid Signals, `error` fails while
`allow_partial` holds seven securities at approximately `1/7` each.

## Weighting

V5 supports only `equal_weight`: every actually selected security receives
`target_weight = 1 / K`, where K is that date's selected count. Score weighting,
optimizers, risk parity, market-cap weighting, leverage, long-short portfolios,
and backtest execution are current non-goals.

## Artifact layouts and provenance

Signal publishes:

```text
signal/
  signals.parquet
  config.json
  audit.json
  manifest.json
```

Canonical Signal columns are `trade_date`, `ts_code`, `score`, and `rank`.

Holdings publishes:

```text
holdings/
  holdings.parquet
  config.json
  audit.json
  manifest.json
```

Canonical Holdings columns are `trade_date`, `ts_code`, `target_weight`,
`score`, and `rank`.

Each `config.json` stores the effective stage business config. Each `audit.json`
stores compact row/date and integrity observations. Each `manifest.json` stores
schema identity, direct source provenance, file sizes, and SHA-256 records.
Validators independently check the fixed layout and cross-file integrity.
Publication is atomic, uses a strict no-overwrite policy, and a downstream
failure does not delete already-published upstream Artifacts.

Signal provenance identifies the validated native ML Artifact and predictions
hash. Holdings provenance identifies the exact Signal Artifact and Signal hash
from the same run. Neither stage searches for a “latest” Artifact.

## Output inspection

The CLI JSON summary retains compact Signal and Holdings result fields,
including source mode, Artifact paths, row/date counts, direction, requested N,
policy, weighting, and schema version. The run's `config_snapshot.yaml` contains
the effective nested configuration; verify `holdings.top_n` there. Detailed
data remains in the two validated Artifacts rather than in the summary.

## Backward compatibility and UI note

Signal and Holdings default disabled, so V3 ML-only and V4 Modeling Panel/ML
YAML retain their prior execution and summary behavior. Enabling a stage never
auto-enables its dependencies; invalid dependency combinations fail during
config validation.

The existing UI and legacy research/scoring scripts are unchanged. V5-E4 is
expected to bridge UI parameters to this same canonical `PipelineConfig`
schema rather than creating another Top-N or direction implementation.
