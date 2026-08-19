# Universe 1.0

## Types

Universe membership has three stable backend IDs:

- `CUSTOM`: an ordered user-supplied security set, constrained by lifecycle.
- `INDEX`: historical membership from one canonical index identity.
- `ALL_A_SHARES`: all canonical China A-share common equities active at formation.

Localized GUI labels may be added later, but persisted IDs remain unchanged.

## Contracts

`UniverseSpec` strictly parses one type and its exact parameters. `UniverseSnapshot` records exact formation date, ordered unique canonical securities, type, source identity, source as-of date, and immutable diagnostics. It never contains target weights, factor values, eligibility decisions, or observation-availability decisions.

`UniverseResolverRegistry` is fresh and explicitly populated. `UniverseService` dispatches through that registry; dependencies arrive through `UniverseDataSource`. Resolvers do not access TuShare, RAW files, Streamlit, or filesystem discovery.

## Point-in-Time

INDEX chooses the greatest provider `index_weight.trade_date` satisfying:

`snapshot trade_date <= formation date`

The chosen snapshot carries forward only until a later snapshot becomes eligible. Current members never backfill history. Future snapshots are never backward-filled, selected by nearest date, or discovered through latest/mtime/path ordering. Formation before the first available snapshot fails with `UniverseDataUnavailable`.

## Lifecycle

CUSTOM, INDEX, and ALL_A_SHARES apply the same membership boundary:

`list_date <= T < delist_date`

A null/empty `delist_date` has no upper boundary. Thus a security is excluded on its delist date. This differs intentionally from the V6 observation-availability rule that can classify an already-held security specially on delist date; membership and return resolution are separate layers.

## CUSTOM

Canonical inputs such as `600519.SH`, `000001.SZ`, and TuShare Beijing suffixes are accepted directly after validation against canonical `stock_basic`. A bare six-digit symbol is accepted only when `stock_basic.symbol` proves exactly one canonical match. No exchange is inferred from numeric prefixes. Zero matches and ambiguous matches fail closed.

Duplicate inputs collapse by first occurrence. After canonicalization, user order is retained across lifecycle filtering. Persisted canonical specs contain full `ts_code` identities.

## INDEX

Any canonical index code represented by prepared `index_weight` data is supported; the backend is not limited to a hard-coded UI list. Constituent membership is sorted by canonical `ts_code`, independent of Parquet row order.

Provider `weight` is used only for source diagnostics. It is never converted into a Holdings or Portfolio Construction target weight.

`stock_basic` supplies constituent lifecycle validation. Suspension, missing daily observations, ST names, liquidity, and listing age do not remove a valid member.

## ALL_A_SHARES

The V1 classifier relies on the TuShare `stock_basic` equity endpoint plus canonical metadata:

- `market` is one of 主板, 创业板, 科创板, 北交所;
- `exchange` is SSE, SZSE, or BSE;
- `curr_type` is CNY.

This includes Shanghai/Shenzhen main boards, ChiNext, STAR Market, and Beijing A shares. Non-CNY B shares and known non-A classifications such as CDR/B股 are excluded. Unknown market classifications fail closed instead of silently broadening membership.

ST, suspended, newly listed, and illiquid members remain included whenever lifecycle is valid.

## Separation

Universe 1.0 implements membership only:

`Membership -> future Eligibility Filters -> Observation Availability`

It does not read `suspend_d`, daily rows, factor coverage, turnover, prices, or execution flags. Consequently, an index member with no daily row remains in `UniverseSnapshot`.

## Research Frequency

Membership always resolves on an exact formation date. Schedule resolution reuses Data Layer 2.0 `formation_dates`: DAILY means every supplied canonical open date; MONTHLY means the last supplied open date per calendar month. Universe code does not calculate natural month-end independently.

## Data Requirements

- CUSTOM: `stock_basic` for canonical-code validation and lifecycle.
- INDEX: `stock_basic` plus scoped `index_weight`; one prior calendar month is requested so the first formation can use the latest prior snapshot without future fill.
- ALL_A_SHARES: `stock_basic` for classification and lifecycle.

Universe membership does not request `daily`, `daily_basic`, `adj_factor`, or `suspend_d`. Requirements are compatible with P4B `DataPreparationService`; repeated complete requirements produce zero provider calls.

## Source Identity

CUSTOM identities include the exact canonical input and lifecycle source identity. INDEX identities include index code, selected provider snapshot date, index coverage identity, and lifecycle identity. ALL_A_SHARES identities include the validated stock reference identity. These diagnostics are available for future lineage without changing current Artifact schemas.

## Known Limitation

TuShare `index_weight.trade_date` proves a provider snapshot date, not necessarily an announcement timestamp or legally exact effective instant. Universe 1.0 therefore documents a conservative point-in-time approximation: it uses only snapshots dated on or before formation and never uses future data. It does not invent announcement/effective-date fields.

ALL_A_SHARES classification relies on the canonical stock endpoint being an equity reference source plus its market, exchange, and currency metadata. Unsupported market classifications fail explicitly.

## Non-goals

- ST, suspension, tradability, liquidity, or minimum-listing-age filters
- saved-universe artifacts or GUI integration
- AdjustedPriceService or factor-frequency semantics
- ResearchInputBuilder, factor panels, labels, or pipeline integration
- live trading

Next: P4C2 AdjustedPriceService, FactorFrequencySpec, and research-calendar semantics.
