# Research Backtest Pipeline

## Purpose and architecture

The V6 Research Backtest measures the historical behavior of portfolios already
selected by the upstream research chain:

```text
Factor Research -> Modeling Panel -> ML -> Signal -> Holdings -> Research Backtest
```

V5 owns ranking, Top-N selection, and target-weight construction. V6 consumes
the exact validated Holdings target weights; it does not rank securities, read
scores to select names, or optimize a new portfolio. This is a research
evaluation system, not a live-trading or execution system.

Run the canonical entry point from the repository root:

```text
python scripts/run_pipeline.py --config config/research_backtest_pipeline.example.yaml
```

No Research Backtest-specific command-line flags exist. Business assumptions
come from the YAML, and the top-level `backtest_end` remains the single explicit
evaluation end date for the complete Pipeline run.

## Source modes

`pipeline` mode consumes the exact native Holdings Artifact created in the same
`run_pipeline` call. It requires `holdings.enabled: true` and never searches the
run directory.

`files` mode consumes one explicitly named, complete native Holdings Artifact:

```yaml
research_backtest:
  enabled: true
  source:
    mode: files
    artifact_dir: "EXPLICIT_NATIVE_HOLDINGS_ARTIFACT_DIR"
```

The path must be the Artifact directory containing its payload, config, audit,
and manifest. A bare `holdings.parquet` is not a valid source. Neither mode uses
latest-run, timestamp, sibling-directory, or fallback discovery.

## Frequency and timing

The core is frequency-agnostic. Ordered `Holdings.trade_date` values own the
rebalance schedule; there is no monthly/weekly/daily backtest enum. Monthly-like,
weekly-like, consecutive, or irregular Holdings events use the same engine.

Canonical configuration identifiers are `holdings_dates`,
`next_trading_day`, `adjusted_close_to_close`,
`one_way_traded_notional`, and `strict_common_calendar`.

For Holdings date T, the target becomes effective at the next trading-day
post-close boundary. The old portfolio consumes that effective day's
close-to-close return. The new target begins consuming returns on the following
open trading date. Calendar coverage may extend beyond `backtest_end` only to
prove the next-open mapping; evaluation never extends beyond `backtest_end`.

## Returns and missing observations

Security returns are `tushare.daily.pct_chg / 100`. Benchmark returns are
`tushare.index_daily.pct_chg / 100`. The pipeline does not reconstruct returns
from closes or adjusted prices.

Only a missing security return with same-date proof of full-day suspension is
resolved to research return zero. Any unexplained missing observation fails
closed. Rows are not generically filled, forward-filled, or dropped.

## Cost, NAV, and benchmark fairness

Transaction cost uses one-way traded security notional:

```text
transaction_cost = traded_notional * cost_bps / 10000
```

`traded_notional` is the security-weight movement used for cost. The audited
complete-state `turnover` is a separate half-L1 measure that includes the cash
leg; cost is not computed from turnover.

Gross NAV compounds before-cost portfolio returns. Net NAV applies transaction
cost multiplicatively at each rebalance close. The explicit benchmark uses the
same daily evaluation calendar, and its first evaluation-day return is forced
to zero for a fair common starting boundary.

## Metrics

The V6-E daily analytics output contains exactly:

- `observation_count`, `rebalance_count`
- `gross_total_return`, `net_total_return`
- `gross_annualized_return`, `net_annualized_return`
- `net_annualized_volatility`, `net_sharpe_ratio`, `net_max_drawdown`
- `benchmark_total_return`, `benchmark_annualized_return`
- `excess_total_return`, `annualized_excess_return`
- `tracking_error`, `information_ratio`
- `average_turnover`, `total_turnover`, `total_traded_notional`
- `total_transaction_cost`, `transaction_cost_return_drag`

Annualization is controlled only by the explicit `annualization_days` setting.

## Native Artifact

The final schema `1.0` Artifact contains exactly:

```text
rebalances.parquet
daily_portfolio.parquet
benchmark.parquet
metrics.json
config.json
audit.json
manifest.json
```

It is staged, validated, published atomically, never overwritten, and directly
records the exact validated Holdings Artifact lineage. The manifest records
SHA-256 and byte size for all six payloads and is written last.

## Configuration example

The canonical example chooses 10 bps cost, `000300.SH`, and 0.0 annual
risk-free rate. These are user-visible example choices, not hidden backend
defaults. Top-N remains under `holdings`; Research Backtest has no Top-N field.

## Non-goals

V6 does not implement live trading, broker integration, order/fill simulation,
market impact, execution prices, settlement, portfolio optimization,
long-short construction, or factor neutralization.
