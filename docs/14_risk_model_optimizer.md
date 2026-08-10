# Risk Model and Minimum Variance Portfolio Construction

V8 extends the existing Portfolio Construction registry with historical risk
estimation and constrained minimum-variance weighting. It does not add a
Pipeline stage or Artifact type. The dependency chain is:

```text
HistoricalReturnService
  -> RiskModelService
  -> RiskEstimator
  -> MinimumVarianceConstructor
  -> OptimizerBackend
```

For the registered strategy this is resolved as
`minimum_variance -> risk_model -> historical_returns`. Services and the market
client are run-scoped, constructed once, and shared with Research Backtest when
both capabilities are enabled.

## Selection and return window

Top-N owns security selection. Minimum Variance receives that exact ordered set
and may not add, drop, replace, or rerank a security. Holdings keeps its five
canonical columns: `trade_date`, `ts_code`, `target_weight`, `score`, and
`rank`.

The risk model reuses canonical resolved daily returns. It retains only the
common complete-case dates on which every selected asset has an observation.
There is no pairwise covariance, forward/backward fill, or zero fill.
Pre-listing dates are absent; a proven full-day suspension is a canonical zero
return and remains an observation. Unknown active missing data fails closed.

The historical window ends at the latest open trading date no later than the
formation date. Weekend and holiday formation dates therefore use the previous
open date, and later returns cannot affect the portfolio.

## Covariance estimators

- `sample_covariance` uses sample covariance with `ddof=1`.
- `ledoit_wolf` uses sklearn
  `LedoitWolf(assume_centered=False, store_precision=False)` and reports its
  shrinkage coefficient.

Both operate on daily, unannualized returns in exact asset order. Covariance
must be finite, symmetric, have a strictly positive diagonal, and be positive semidefinite
within the centralized relative tolerance. Singular positive
semidefinite matrices are allowed. Eigenvalues and condition status are
diagnostics only: the implementation validates but does not repair, clip, add
epsilon, or compute a nearest-PSD matrix.

## Optimization

The SciPy SLSQP backend minimizes `0.5 * w' Sigma w` with analytic gradient
`Sigma w`, subject to long-only weights, fully invested `sum(w)=1`, and the
existing `max_weight` constraint as a direct optimizer upper bound. The frozen
settings are `ftol=1e-12`, `maxiter=1000`, and `disp=False`.

Solver results fail closed. There is no post-solve clipping, normalization,
equal-weight fallback, or inverse-volatility fallback.

## Configuration

```yaml
portfolio_construction:
  method: minimum_variance
  params:
    risk_model:
      estimator: ledoit_wolf
      params: {}
      lookback_trading_days: 120
      min_observations: 80
  constraints:
    - type: max_weight
      params:
        max_weight: 0.20
```

These numbers are example and UI suggestion values, not hidden backend defaults.
`sample_covariance` is selected by changing only the estimator name. The normal
CLI remains `python scripts/run_pipeline.py --config <path>`; there are no risk
model or solver business flags.

The complete resolved Portfolio Construction config is stored naturally in the
existing Holdings `config.json`. No covariance matrix, optimizer payload,
RiskModel Artifact, or extra manifest file is persisted. Research Backtest
continues to consume only canonical `target_weight` values and Holdings lineage.

## Non-goals

Expected returns, mean variance, risk parity, turnover constraints, sector or
factor exposure constraints, tracking error, and live execution are not
implemented in V8.
