# Data Layer 2.0

## Architecture

Data flows through four separate layers:

`RAW provider responses -> CURATED canonical data -> RESEARCH data -> PIPELINE`

P4B implements only RAW, CURATED, coverage metadata, and fetch orchestration. It does not build factor inputs, scores, adjusted prices, modeling panels, forward returns, or ML labels.

## Frequencies

`ResearchFrequency.DAILY` uses every real open trading day. `ResearchFrequency.MONTHLY` uses the last real open trading day in each calendar month. Both are derived from the canonical trade calendar.

Research frequency is independent of provider frequency. Monthly research does not mean downloading TuShare monthly bars. `index_weight` remains a native monthly snapshot dataset.

## Dataset Registry V1

The fresh in-process registry contains exactly eight specifications:

| Dataset | Primary key | Native frequency | Scope | Coverage unit | Partition |
| --- | --- | --- | --- | --- | --- |
| `trade_cal` | `exchange, cal_date` | calendar day | exchange series | exchange/date | year |
| `stock_basic` | `ts_code` | reference snapshot | market reference | effective-through snapshot | snapshot |
| `daily` | `ts_code, trade_date` | trading day | market snapshot | trade date | year/month |
| `daily_basic` | `ts_code, trade_date` | trading day | market snapshot | trade date | year/month |
| `adj_factor` | `ts_code, trade_date` | trading day | market snapshot | trade date | year/month |
| `suspend_d` | `ts_code, trade_date` | trading-day event | market snapshot | trade date | year/month |
| `index_daily` | `ts_code, trade_date` | trading day | entity series | index/date | index/year/month |
| `index_weight` | `index_code, con_code, trade_date` | monthly snapshot | entity-month | index/month | index/year |

The registry rejects duplicates and unknown IDs. New datasets are added by registering a `DatasetSpec` and fetch/completeness strategy, without UI code or a central dataset-name dispatch.

## Coverage Ledger

`data/metadata/catalog.sqlite` is the completeness truth. `coverage_units` is uniquely keyed by `(dataset_id, scope_key, unit_key)` and records status, row count, schema version, deterministic content hash, token-free request fingerprint, and completion time. Only `COMPLETE` participates in coverage subtraction.

`fetch_events` records a token-free request summary, timestamps, status, row count, and error class. SQLite writes are transactional. The legacy min/max JSON cache remains compatibility-only and is never trusted by Data Layer 2.0.

Coverage is not observation availability. A complete daily market snapshot can legitimately omit a suspended security. A successfully queried `suspend_d` date with zero events is also COMPLETE. Listing, suspension, and unexplained missing-return semantics remain downstream and unchanged.

## Missing-only invariant

The planner expands requirements into exact natural dates, open trading dates, entity/date units, entity/month units, or reference snapshots. It computes:

`Required units - ledger COMPLETE units = Missing units`

Only missing units become fetch tasks. Disjoint gaps remain disjoint. After successful completion, an identical request makes zero provider calls. Ordinary preparation never refreshes a COMPLETE unit.

## Storage

RAW responses are stored under `data/raw/tushare/<dataset>/` with an immutable fetch identity. CURATED data is schema-selected, date-normalized, primary-key validated, deterministically sorted, and stored as partitioned Parquet under `data/curated/<dataset>/`.

Coverage and physical partition granularity differ intentionally: daily coverage is one date while physical storage is year/month.

## Atomicity

Canonical merge distinguishes identical idempotent duplicates from conflicting revisions. Identical payloads collapse; conflicts fail closed. A partition is written to a same-filesystem temporary Parquet, flushed, reread, hash-validated, and atomically replaced. Coverage becomes COMPLETE only after replacement succeeds. Fetch or write failure leaves prior canonical data intact and does not mark missing units complete.

## Revision

The production path is missing-only. Historical revision and explicit refresh are reserved contracts, not P4B behavior. Reference datasets declare explicit-refresh policy and are not silently refreshed during ordinary preparation.

## Token

`TushareClient` uses instance injection through `ts.pro_api(token)` and never calls `ts.set_token`. Tokens are excluded from fingerprints, SQLite, Parquet, metadata, logs, configs, and artifacts. The environment source remains isolated behind a credential provider for backward compatibility.

## Migration

Legacy min/max JSON is never imported as truth. The explicit migrator reads a named legacy Parquet, validates schema, keys, scope, and individually provable units, atomically imports them into CURATED storage, then records coverage. Ambiguous whole-market snapshots remain unknown and require a provider query.

## Non-goals

- ResearchInputBuilder and monthly factor materialization
- adjusted prices, factor inputs, modeling panels, or forward returns
- financial statements
- GUI integration
- automatic or explicit historical refresh/revision

The next phase is P4C ResearchInputBuilder design and implementation.
