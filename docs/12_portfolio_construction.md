# Portfolio Construction

Portfolio Construction is the weighting capability inside the existing
Holdings stage. Signal ranking and Holdings Top-N decide which securities are
selected. Portfolio Construction receives that exact selected set and decides
only its target weights; it cannot add, drop, replace, or re-rank candidates.

## Configuration

The canonical owner is `holdings.portfolio_construction`:

```yaml
portfolio_construction:
  method: inverse_volatility
  params:
    lookback_trading_days: 60
    min_observations: 40
  constraints:
    - type: max_weight
      params:
        max_weight: 0.20
```

An old Holdings config without this field resolves to `equal_weight`, empty
params, and no constraints. `equal_weight` and `rank_weight` use empty params.
Rank weighting uses the internal selected-set `selection_position`, not raw
score magnitude or an assumption that an upstream rank is contiguous.

`max_weight` uses capped proportional water filling. It is not
clip-then-normalize, remains fully invested, and fails when the cap is
infeasible.

## Historical risk semantics

`inverse_volatility` estimates sample standard deviation with `ddof=1` and no
hidden epsilon or volatility floor. Its lookback is the latest exact number of
market open dates ending at `risk_cutoff`, where `risk_cutoff` is the latest
open trading date no later than formation.

The concrete service reuses the V6 calendar, decimal `tushare.daily.pct_chg`
return, lifecycle, suspension, and missing-return contracts. Pre-listing dates
are excluded from observations; a proven full-day suspension resolves to zero;
active unexplained missing data and unresolved delisting inconsistencies fail
closed. Data after formation cannot enter the window.

## Architecture and compatibility

Strategies and constraints are registry plugins. Constructors satisfy
constraints and the Engine validates exact candidate identity and final
invariants. Services are injected by declared capabilities; equal and rank do
not require a token or market client.

The Holdings payload remains exactly `trade_date`, `ts_code`, `target_weight`,
`score`, and `rank`. `selection_position`, volatility, raw weights, and method
metadata never enter the payload. Portfolio Construction adds no Pipeline stage
or Artifact. Research Backtest consumes the same validated Holdings targets and
lineage.

## Non-goals

This release does not implement an optimizer, covariance model, risk parity,
turnover constraint, sector/factor constraint, or live execution. New methods
can be added through strategy, constraint, and service plugins without adding
method dispatch to HoldingsBuilder or Pipeline Runner.
