# ResearchInputBuilder 1.0

## Purpose

P4C3 turns the P4B CURATED data, P4C1 point-in-time universe, and P4C2
calendar/frequency/adjusted-price contracts into deterministic inputs accepted
by the existing Factor Research and Modeling code. It does not download data.
Callers must first pass `ResearchInputPlan.requirements` to
`DataPreparationService`; missing canonical data then fails closed with
`ResearchInputDataUnavailable`.

## Plan

`ResearchInputPlanner` freezes the research frequency and interval, exact
formation dates, universe specification, selected factor order and frequency
specifications, forward-return contract, normalized data requirements, and
output filenames. Its `plan_id` is a SHA-256 of the canonical serialized plan.
Credentials and runtime objects are not part of the plan.

## Formation/Universe

`ResearchCalendar` supplies the exact DAILY or completed-month-end formation
schedule. `UniverseService` resolves independent point-in-time membership at
each formation, so historical changes remain visible and missing observations
cannot rewrite membership truth.

## Factors

Every selected factor is resolved through its registered
`FactorFrequencySpec`, source fields, calculator identity, and
`HistoryRequirement`. Existing calculators and preprocessing remain the only
owners of factor math.

## Warmup / Future Horizon

The research interval and acquisition interval are different. Factor history
extends before the first formation according to each factor's own
`HistoryRequirement`. Forward prices extend after the final formation through
the exact required future open date. A `LATEST_AS_OF` source is not expanded to
another factor's longer trading-day history. Monthly research changes only the
formation schedule; factor-owned daily lookbacks remain daily and are never
averaged into a generic monthly bar.

## Adjusted Prices

Feature prices and the forward-label compatibility panel use the P4C2 exact
same-key `raw OHLC * adj_factor` service. There is no dynamic end anchor,
generic fill, or synthetic suspended row.

## Forward Returns

`ForwardReturnSpec` is the explicit compatibility contract for the existing
`ForwardReturnBuilder`. P4C3 supports `TRADING_PERIODS` and uses the existing
formula unchanged:

```text
entry date = formation shifted by entry_lag_periods open sessions
exit date  = entry date shifted by horizon open sessions
forward_return = exit_price / entry_price - 1
```

The shift is exclusive of the anchor and follows `ResearchCalendar` open dates,
not calendar-day arithmetic. The compatibility price panel contains adjusted
close values under the configured existing price-column name (`close` by
default). This is a research-label input only. Research Backtest remains owned
by its existing `daily.pct_chg / 100` and `index_daily.pct_chg / 100` semantics.

## Modeling Panels

Every file is sorted deterministically by `(trade_date, ts_code)` and contains
no duplicate keys.

| File | Exact schema and owner |
| --- | --- |
| `factor_input.parquet` | `trade_date`, `ts_code`, then the sorted union of exact source fields required by the selected factors. This is the existing wide calculator input, not calculated factor output. |
| `price_panel.parquet` | `trade_date`, `ts_code`, configured price column (default `close`). Values are canonical adjusted close for the existing forward-return calculator. |
| `score_panel.parquet` | Exactly `trade_date`, `ts_code`. Repository audit proved that `FactorResearchRunner` uses this file as the external formation/universe selection-key schedule. It contains no score, prediction, rank, or signal value. |
| `modeling_factor_panel.parquet` | `trade_date`, `ts_code`, then exactly the selected factor IDs in plan order. |
| `modeling_forward_returns.parquet` | `trade_date`, `ts_code`, `entry_trade_date`, `exit_trade_date`, `entry_price`, `exit_price`, configured return column (default `forward_return`). |
| `labels_with_availability.parquet` | The modeling-forward-return columns plus `available_at`. This sidecar preserves realization metadata without changing the existing modeling input schema. |

Universe membership is resolved independently at every formation date through
`UniverseService`. Missing factor observations never rewrite membership.
`FactorResearchRunner` still owns factor calculation and preprocessing;
`ModelingPanelBuilder` still owns final feature/label row selection and missing
policy. `MLDatasetBuilder` validates the compatibility outputs. P4C3 adds no
factor formula, fill, imputation, eligibility filter, prediction, optimizer,
risk, signal, holdings, or backtest rule. Neutralization fails closed because
canonical point-in-time exposure-panel materialization is outside P4C3.

## score_panel Ownership

Despite its historical filename, `score_panel.parquet` is an input selection
schedule in the current Factor Research contract. ResearchInputBuilder emits
only those keys. ML predictions, ranks, signals, and portfolio scores remain
owned by their downstream stages and are never fabricated here.

## No-lookahead

For a label formed at `T`, `available_at` equals `exit_trade_date`: the close at
which the return is fully realized. `TrainingLabelAvailabilityGuard` returns
only rows satisfying `available_at <= training_cutoff`. A label whose formation
is before the cutoff but whose exit is after it is excluded. This sidecar and
guard prevent future realized returns from entering a historical training set.

## Materialization Identity

The materialization identity hashes the canonical plan, exact source
identities, hashes of compatibility inputs, calculator IDs and versions,
Factor Research configuration, and schema version. It never uses mtime,
filesystem order, `latest`, globbing, or a random identifier. An existing
content-addressed directory is reused only after its manifest, target set, and
every file SHA-256 validate; safe reuse skips factor recomputation.

New output is written to a unique staging directory, reread and frame-hash
checked, described by a manifest, and atomically renamed to the final identity.
Failure removes only that staging directory, leaves an older valid identity
intact, and publishes no partial target.

## Non-goals

P4D will own first-run orchestration: prepare only missing canonical data,
invoke this builder, and hand the exact generated paths to the existing
pipeline. P4C3 itself has no provider, RAW, Streamlit, token, or automatic-run
boundary. It also adds no eligibility rules, factor formulas, ML prediction
math, signal/holdings behavior, portfolio/risk/optimizer semantics, live
trading, or Research Backtest accounting change.
