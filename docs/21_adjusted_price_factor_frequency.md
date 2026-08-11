# Adjusted Price and Factor Frequency

## Formation Time

A formation date `T` is a research decision point as of the close of `T`.
Features at `T` may use only observations available by that close. Rows after
`T` are outside the request and cannot affect its result.

## ResearchCalendar

`ResearchCalendar` consumes canonical `trade_cal` rows rather than natural
month ends or weekday approximations. DAILY formations are every proven open
date inside the requested interval. MONTHLY formations are the last proven
open trading day of each complete natural month and must fall in the interval.
A partial final month without enough month-end evidence is skipped; a fully
covered month with no open date fails explicitly. Duplicate dates, incomplete
requested coverage, and invalid `is_open` values fail closed.

## Adjusted Price

Canonical research adjusted prices use the exact same-key formula:

```text
adj_open  = open  * adj_factor
adj_high  = high  * adj_factor
adj_low   = low   * adj_factor
adj_close = close * adj_factor
```

There is no division by an end-date factor and no dynamic qfq/hfq chart anchor.
Extending a request or perturbing data after `T` cannot re-anchor values at or
before `T`. `vol` and `amount` remain raw observations and are not multiplied.

## Backtest Separation

Adjusted prices serve research features only. The frozen V6 Research Backtest
continues to derive security returns from `daily.pct_chg / 100` and benchmark
returns from `index_daily.pct_chg / 100`. It does not reconstruct accounting
returns with `adj_close.pct_change()`.

## HistoryRequirement

- `TRADING_DAYS(n)` selects exactly `n` canonical open dates, including the
  formation date as the final observation.
- `CALENDAR_MONTHS(n)` includes the formation month and preceding `n-1` months;
  it starts at the first proven open date in the earliest included month.
- `LATEST_AS_OF` has no numeric historical window and resolves formation only.

Insufficient evidence and non-open formation dates fail. Trading days are never
converted to a calendar-day timedelta.

## FactorFrequencySpec

`FactorMetadata` remains the single factor registry source of truth and now
owns at most one `FactorFrequencySpec` per supported frequency. A spec declares
datasets, fields, typed history, observation semantics, and calculator ID.
Unsupported frequencies fail closed. Legacy metadata adapts to DAILY only.

The existing `ep_ttm`, `bp`, `sp_ttm`, `log_total_mv`, and `log_circ_mv`
factors explicitly support DAILY and MONTHLY. MONTHLY means
the formation-date as-of `daily_basic` value, never a monthly mean. Other
existing factors retain legacy DAILY support pending explicit semantics. In
particular, `dividend_yield_ttm` remains legacy because its `dv_ttm` input is
not present in the currently frozen P4B `daily_basic` schema; P4C2 does not
invent a `dv_ratio` substitution or change its formula.

## Monthly Semantics

Frequency selects calculation dates; it does not globally resample fields.
Point-in-time fields, price windows, and future aggregation factors own their
as-of, lookback, mean, or sum semantics. P4C2 does not implement a generic
MonthlyMarketBar because P4C3 has no unambiguous need for one yet.

## Observation Availability

Coverage completeness does not imply that every universe member has a row.
AdjustedPriceService outputs only observed daily keys. A daily row without a
unique, finite, positive same-key `adj_factor` fails closed. Extra adjustment
rows do not manufacture observations. There is no generic zero-fill, forward
fill, backward fill, or interpolation policy.

## Data Requirements

Adjusted prices contribute exactly scoped `daily` and `adj_factor` requirements
for the caller-resolved range. Factor requirements come from
`FactorFrequencySpec` after calendar history resolution. They remain ordinary
P4B `DataRequirement` values and use existing coalescing. Repeating completed
requirements makes zero provider calls.

## Extensibility

A plugin factor registers normal `FactorMetadata`, frequency specs, and a
calculator. Generic calendar, dependency, and adjusted-price services require
no factor-name dispatch. Future history strategies can extend the typed
contract without changing formulas or the calendar.

## Non-goals

P4C2 does not build ResearchInputBuilder panels, forward-return labels,
modeling panels, GUI flows, eligibility rules, financial-statement PIT
alignment, or universal monthly aggregation.
