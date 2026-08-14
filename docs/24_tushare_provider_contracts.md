# TuShare Provider Contracts and Audit

Audit date: 2026-08-14. Branch: `feature/research-workbench`.

The `stock_basic` official page was read successfully on 2026-08-14. Other
endpoint facts retain the preceding audit evidence. Facts present in the
official pages are encoded in `src/data/provider_contracts.py`; anything not stated is deliberately
`OFFICIAL_NOT_STATED` or `UNKNOWN`. The obsolete suspension page `doc_id=31`
returning 404 is not evidence that `suspend_d` was retired: the current endpoint
is documented at `doc_id=214`.

## Provider boundary

- `tushare_official`: TuShare SDK's normal official endpoint. Existing official
  data remains under `data/` for backward compatibility.
- `tushare_proxy`: third-party proxy at fixed HTTPS endpoint
  `https://tuaremax.top`. It is not an official TuShare 5000-point service.
  Proxy data is isolated under `data/providers/tushare_proxy/`.
- Tokens have separate process/session fields. Only the official provider may
  read the existing environment credential. There is no automatic failover,
  token conversion, arbitrary URL, HTTP downgrade, TLS disabling, or mixed
  provider task.
- The runtime SDK is 1.4.29 and `requirements.txt` pins it reproducibly. The
  proxy adapter contains the only accesses to `_DataApi__token` and
  `_DataApi__http_url`; missing fields fail with an explicit SDK compatibility
  message. The supplier-named 1.4.24 runtime was not installed or truly probed
  in this environment and remains a compatibility UAT gate.

## Actual call inventory and official contracts

Production wrappers call `stock_basic`, `trade_cal`, `index_weight`, `monthly`,
`daily`, `index_daily`, `suspend_d`, `daily_basic`, and `adj_factor`. `stk_limit`
is present for capability/quality auditing but is not a research dependency.
There are no `pro.query(...)` calls.

| API | Official document | Minimum points/permission | Rate | Rows | Update |
|---|---|---:|---|---:|---|
| stock_basic | [doc 25](https://tushare.pro/document/2?doc_id=25) | 2000 | 50/min | 6000 per call | OFFICIAL_NOT_STATED |
| trade_cal | [doc 26](https://tushare.pro/document/2?doc_id=26) | 2000 | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED |
| daily | [doc 27](https://tushare.pro/document/2?doc_id=27) | BASIC_DAILY_PERMISSION | OFFICIAL_NOT_STATED | 6000 | after close; exact time OFFICIAL_NOT_STATED |
| adj_factor | [doc 28](https://tushare.pro/document/2?doc_id=28) | 2000 | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED |
| daily_basic | [doc 32](https://tushare.pro/document/2?doc_id=32) | 2000 | OFFICIAL_NOT_STATED | 6000 | OFFICIAL_NOT_STATED |
| index_weight | [doc 96](https://tushare.pro/document/2?doc_id=96) | 2000 | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED |
| stk_limit | [doc 183](https://tushare.pro/document/2?doc_id=183) | 2000 | OFFICIAL_NOT_STATED | 5800 | OFFICIAL_NOT_STATED |
| suspend_d | [doc 214](https://tushare.pro/document/2?doc_id=214) | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED | irregular |
| index_daily | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED |
| monthly | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED | OFFICIAL_NOT_STATED |

General references: [permissions](https://tushare.pro/document/1?doc_id=108),
[points/rates](https://tushare.pro/document/1?doc_id=290), and
[FAQ](https://tushare.pro/document/1?doc_id=122). Endpoint documents take
precedence. Proxy rate/quota fields remain `PROXY_RULE_UNKNOWN`; the provider's
advertised “5000-level” claim is not official proof.

## `stock_basic` global-snapshot contract (UAT-032)

The GUI label `证券基础信息` maps to canonical dataset and provider endpoint
`stock_basic`. TuShare doc 25 defines optional `ts_code`, `name`, `market`,
`list_status`, `exchange`, and `is_hs` inputs; it does not define
`trade_date`, `start_date`, or `end_date`. The default `list_status` is `L`.
Consequently, the application makes four separate calls with
`list_status=L/D/P/G`, validates each response against the requested status,
then publishes one non-empty, deduplicated snapshot keyed by `ts_code`. A
single status may be empty. If sequential status calls expose one code more
than once, the explicit application deduplication priority is `D`, `P`, `L`,
then `G`; conflicting duplicates within one status fail closed.

Provider contract version 1.1 declares `GLOBAL_SNAPSHOT`; canonical schema
1.2 uses the fixed logical unit `GLOBAL`. Research interval length therefore
cannot change the number of `stock_basic` coverage units. The adapter never
passes a date argument, the canonical partition is `snapshot=GLOBAL`, and the
Ledger records only `stock_basic / GLOBAL`. A lightweight preflight checks the
Provider Contract, Dataset Registry, planned unit, fetch strategy, and token-
free request parameters before client creation or any provider call.

The result is a current reference snapshot retrieved by the application, not a
historical PIT snapshot. Historical eligibility is derived only as
`list_date <= T < delist_date`, with a null `delist_date` treated as no known
upper boundary. Current `list_status`, name, and industry are not represented
as historical PIT factor fields. Quotes, the trading calendar, and the
configured suspension policy remain responsible for final tradability.

Raw and canonical Parquet are staged, fsynced, verified, and atomically
replaced before the Ledger transaction marks `COMPLETE`. The canonical
manifest and fetch event record provider ID, endpoint, requested statuses,
actual token-free parameters, retrieval time, schema/contract versions, row
count, canonical hash, raw/canonical/manifest references, SDK version, and
quality conclusion. The merged whole-market snapshot must be non-empty and
cannot use an empty marker. Official and proxy roots, Ledgers, manifests, and
credentials stay isolated; there is no cross-provider fallback or repair.

Snapshot freshness is application policy, not an official TuShare update
frequency: one integrity-verified `GLOBAL` snapshot is reused within its UTC
retrieval day for the selected provider. A later day replans the same logical
unit and atomically replaces it. Legacy date-labelled schema 1.0/1.1
`stock_basic` units remain untouched as diagnostic evidence but cannot satisfy
the new contract. Retry replans one `GLOBAL` unit, preserves other verified
dataset units, and recomputes progress from the new coverage plan.

## Dependency map

- Momentum 20/60/120/252, reversal, 52-week-high and volatility use canonical
  `daily.close` plus adjusted-price inputs. Amihud uses local `daily.close` and
  `daily.amount`. Adjustment/forward-return services use `adj_factor`. No
  specialty factor API is used.
- EP uses `daily_basic.pe_ttm`; BP uses `daily_basic.pb`; SP uses
  `daily_basic.ps_ttm`; dividend yield uses `daily_basic.dv_ttm`; size uses
  `total_mv`/`circ_mv`; turnover uses `turnover_rate`.
- Registered `financial_pit` factors are not exposed in the GUI because no
  canonical contract exists. Announcement-date PIT alignment remains mandatory.
- All-A/custom universes depend on `stock_basic`; historical index universes
  add `index_weight`. Current constituents cannot backfill history.
- Inverse-volatility and minimum-variance portfolios require configured daily
  return warm-up. Models consume the PIT materialized panel and add no endpoint.
- Dates use `trade_cal`; security and benchmark returns use `daily` and
  `index_daily`. Strict suspension mode requires `suspend_d`.

## Suspension, quality, and real UAT

`STRICT_EVENT` blocks without verified `suspend_d`. `STANDARD_ROBUST` calls an
absent daily row only “unavailable; exact cause unconfirmed”, carries a holding
with the verified zero-return valuation convention, and freezes buys and sells.
More than 20% unexplained missing rows on a date is a quality incident and
blocks. No future/zero price or fabricated execution is introduced.

Validation covers schema, types, key uniqueness, dates, code format, OHLC,
non-negative volume/amount, positive adjustment factors and row limits.
Cross-provider comparison checks field/row/key sets and values at absolute
tolerance `1e-8`; important differences are reported, never reconciled.

Capability probes are serial minimal calls over every wrapper plus `stk_limit`,
with separate `suspend_d` dates 2020-03-12 and 2023-11-16. Timeout is a network
result, never inferred as points. Only a successful schema-valid zero response
may create an empty-event proof.

No real provider token was available to this process. No real capability probe,
HTTPS/TLS transaction, provider comparison, or end-to-end task was run. UAT-009
therefore remains open in strict mode; UAT-028 remains open until a non-empty
run ID opens. UAT-030 is implemented through task schema 1.1 in-memory migration
and formal diagnostics, but still requires packaged-app recheck.
