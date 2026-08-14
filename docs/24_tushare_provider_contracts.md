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
run ID opens. UAT-030 diagnostics introduced in task schema 1.1 are carried
forward by schema 1.2 in-memory migration, but still require packaged-app recheck.

## UAT-032 reopened: real-failure evidence and diagnostic gate

UAT-032 was reopened on 2026-08-14. Commit `ec26791cabc4eb990d8217e008b8cbce861f589c`
corrected the fictitious dated planning semantics, but did not pass real-provider
UAT. This investigation did not retry or edit a real task and did not call either
TuShare provider.

### Read-only evidence timeline

Both records have task schema 1.1, provider `tushare_proxy`, no run ID, and a
provider-scoped ledger at `data/providers/tushare_proxy/metadata/catalog.sqlite`.
Credentials are session/process-only and are not persisted; therefore the
credential value and a historical credential fingerprint are intentionally
unavailable. The provider scope is known, but credential-instance identity is
an evidence gap.

| Task | Task interval | Retry lineage | Task start (UTC) | GLOBAL fetch event | Fetch terminal (UTC) | Task terminal | Progress |
|---|---|---|---|---|---|---|---:|
| `e463823c-96ca-408e-ac61-2f16f8db738d` | 2023-01-01--2023-02-01 | new task retrying `5cf63598-f225-4320-be52-2b95f6143f41`; original ancestor `7c47ac50-6d90-45a0-843a-25b5727f1140` | 06:28:43.735712 | `306f9953629044d59510b2297fa54af8`, 06:30:36.985114 | FAILED 06:30:41.901523 | 06:30:41.915859 | 432/433 |
| `4b311baa-7867-4330-b87b-e64716bf59a7` | 2024-01-01--2024-02-03 | new task retrying `959e1aa3-a1b1-40cb-a883-5412d11d3dbb`; original ancestor `1afd8bce-9ede-474d-b1cf-ed5ecb7e7527` | 06:25:54.899005 | `5ee09f43065246c8bb97cf51d84f4a40`, 06:28:21.928596 | FAILED 06:28:26.098582 | 06:28:26.109679 | 491/492 |

The ancestor retries used the obsolete dated units `2023-01-31` and
`2024-02-03`. The two subject tasks are distinct child task records, not reused
tasks or hidden subtasks. Their planner progress and fetch events both use the
same identity:

```text
provider=tushare_proxy
dataset=stock_basic
scope={"scope":"CN_STOCK_REFERENCE"}
unit=GLOBAL
endpoint=stock_basic
schema=1.2
provider-contract=1.1
statuses=L,D,P,G
```

The proxy ledger has no `stock_basic` coverage row. For each subject task its
single fetch event persisted the four token-free calls, `rows=0`, terminal
`error_type=DataUnavailableError`, and no retrieval/schema/hash/raw/canonical/
manifest completion fields. Read-only filesystem inspection found no proxy
`stock_basic` raw directory, canonical directory, manifest, empty marker, or
temporary canonical file. The official-provider ledger contains unrelated old
dated snapshots and cannot satisfy or repair the proxy namespace.

### First divergent state

The first demonstrated divergence is after provider fetch/normalization/merge
and before raw staging, at `validate_quality()` in
`src/data/preparation.py::DataPreparationService.ensure`.

The conclusion follows from the persisted state and executable control flow:

1. `src/data/fetching.py::_reference` returns only after all four L/D/P/G calls,
   per-status `normalize_frame`, status matching, concatenation, and deduplication.
   A call exception would be persisted as `ProviderFetchError`; a frame/schema/
   status exception would be `CanonicalDataError`.
2. The real event instead persisted the inner type `DataUnavailableError`.
   Inside this transaction, that type can be raised at the quality gate or after
   raw staging when canonical readback is empty.
3. `RawParquetStore.save` precedes canonical merge/readback and never deletes a
   committed raw file. Both fetch-specific raw files and the entire proxy
   `stock_basic` raw directory are absent. The later empty-canonical branch is
   therefore excluded.
4. For the registered `stock_basic` fields, `validate_quality` can emit only
   `INVALID_SECURITY_CODE`; stock_basic has no OHLC, volume, adjustment factor,
   or configured row-limit rule. Thus the merged provider response reached the
   security-code quality gate and was rejected there.

This identifies the stopping gate and inferred safe category, but it does not
identify the offending provider value, per-status row counts, response fields,
or exact per-call timestamps. Those facts were not persisted. The investigation
therefore does **not** claim a complete payload root cause and makes no quality,
regex, proxy-adapter, or coverage-flow fix.

The GUI's generic local-ledger diagnosis is also explained: `ensure()` changes
`failure_origin` to `local` immediately before `validate_quality`; the outer
`DataUnavailableError` is then mapped to `COVERAGE_VALIDATION`, while task schema
1.1 discards the original quality category. This is a diagnostic loss, not
evidence of a ledger/canonical mismatch.

### Required-direction disposition

- Confirmed: both retries created new tasks; planner and fetch input are
  `stock_basic / CN_STOCK_REFERENCE / GLOBAL`; provider/schema/contract identity
  agrees; the four-call fetch/normalization/merge path completed; the merged
  response stopped at quality validation; GUI collapsed that state into a local
  coverage diagnosis.
- Excluded: a dated GLOBAL representation, cross-provider reuse, an empty merged
  result, raw/canonical written to another proxy path, an abandoned canonical
  temp file, manifest or ledger write using a legacy date, ledger transaction
  rollback, and a downstream readback identity mismatch. None of the storage or
  ledger-commit stages was reached.
- Evidence insufficient: historical credential fingerprint, L/D/P/G row counts
  and individual timings, exact returned field lists, offending security code,
  whether the proxy payload differs from the official payload, and why that
  value appeared. These require exactly one new manual retry with diagnostics.

### Diagnostic-only implementation

Future explicit-write ledgers add `coverage_transitions`, one row per normalized
coverage unit. It records `PLANNED`, provider-attempt and per-call
`FETCH_STARTED`, `FETCH_SUCCEEDED` or `FETCH_FAILED`, `RAW_STAGED`, temporary
canonical validation, atomic canonical and manifest commits,
`LEDGER_COMMITTED`, and `READBACK_VERIFIED`. Records include provider, endpoint,
canonical dataset/scope/unit identity, attempt, safe row/field metadata, schema,
safe error code, original exception type, direct `__cause__` type, safe summary,
and artifact reference. They never persist token values, request headers,
sensitive URLs, or arbitrary provider exception text.

Task schema 1.2 reads 1.0/1.1 in memory without rewriting historical task files.
On failure, the latest transition is copied into the task's technical details so
the GUI shows the actual stopping state and safe cause. Manifests now carry the
same canonical `coverage_identities` as diagnostic provenance. No provider call,
normalization rule, quality rule, retry rule, canonical content, or completion
decision was changed.

### Registry-driven identity audit

The offline parameterized audit covered all eight registered datasets and all
five registered granularities: GLOBAL snapshot, calendar date, market trade
date, entity trade date, and entity month. The generic path passes the same
planner `FetchTask` dataset/scope/units into fetch, storage partitioning,
manifest provenance, ledger transitions/records, and readback. No additional
functional identity divergence was found. A cross-dataset provenance gap was
confirmed: old manifests did not explicitly carry normalized unit identities;
future manifests now do. Legacy files are retained and were not rewritten.

Exact offline commands executed:

```powershell
E:\FINANCIAL ENGINEERING\.venv\quant-factor-system\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_coverage_transaction_diagnostics.py::test_quality_failure_records_exact_safe_stopping_state tests/test_coverage_transaction_diagnostics.py::test_success_records_fetch_to_readback_transaction_and_manifest_identity tests/test_coverage_transaction_diagnostics.py::test_registered_coverage_identity_is_registry_driven_and_layer_stable
E:\FINANCIAL ENGINEERING\.venv\quant-factor-system\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_stock_basic_snapshot_uat032.py::test_four_status_snapshot_has_no_date_calls_and_persists_provenance tests/test_provider_dual_contracts.py::test_task_1_0_read_migration_is_in_memory_and_adds_formal_diagnostics tests/test_gui_uat_consolidated.py::test_task_record_is_atomic_json_and_never_persists_token
```

Results were `10 passed in 4.40s` and `3 passed in 5.99s`. No real provider,
credential, real task retry, EXE, Streamlit/AppTest, complete pytest, test
directory, or long exact-run group was executed. One in-memory `compile()` check
of the ten changed Python files also passed without writing bytecode. UAT-032 remains open. The only
next action is for the user to manually retry one of the two failed tasks once,
then inspect its transaction state chain. UAT-009 and UAT-028 remain open.

## UAT-032 second evidence pass: invalid security identifiers

The supplied task ID for this pass was not present verbatim. The matching
persisted task is `5e00e5c4-68aa-4c2b-b862-b7503a6a0f4a` (rather than
`5e00e5c4-68aa-4c2b-b862-b7583a6aef4a`), and it points to transaction
`2fcbd31dc3864cd89c7766793eb1f139`. The task itself confirms provider
`tushare_proxy`, 432/433 progress, `stock_basic / GLOBAL`, no run ID, and a
pre-raw quality stop.

The transaction adds stronger fetch facts than the prior records: L returned
5542 rows, D returned 340, P and G returned zero, and all calls exposed the 11
registered stock-basic fields. The merged response had 5882 rows. Its terminal
event is intentionally represented as generic state `FETCH_FAILED`, qualified by
operation `QUALITY_VALIDATION`; the safe error is `QUALITY_VALIDATION_FAILED`,
the inner exception is `DataUnavailableError`, direct cause is null, and the
safe category is `INVALID_SECURITY_CODE`.

The rejecting code is `src/data/provider_quality.py::validate_quality`. It
normalizes the already merged frame and evaluates `ts_code` only against rule
`TS_CODE_6_DIGIT_CN_EXCHANGE_SUFFIX`, whose exact pattern is
`^[0-9]{6}\.(?:SH|SZ|BJ)$`. It does not combine `symbol` and `ts_code`, and does
not validate a derived exchange identifier. In `src/data/fetching.py::_reference`,
each L/D/P/G response is normalized and status-checked first; concatenation and
status-priority `ts_code` deduplication occur next; quality validation occurs
after that merged result returns to `DataPreparationService.ensure`.

Critically, the persisted task and transaction contain no offending identifier,
invalid count, invalid status/market/exchange, raw/normalized value pair, rule
ID, or direct cause. It is impossible to classify the value against the official
TuShare `stock_basic` contract or the project A-share Universe from the current
evidence. This pass therefore does not change the regex, provider adapter,
Universe policy, filtering, or quality gate and does not claim a root-cause fix.

Future quality failures retain only a bounded evidence object in the coverage
transition and task technical details. It contains invalid and reason counts;
at most 20 deduplicated, stable-sorted samples of public fields `ts_code`,
`symbol`, `list_status`, `market`, `exchange`, raw/normalized `ts_code`, and
rule ID; L/D/P/G row counts; pre/post-merge rows; and dedup count. A whitelist,
length bound, and second sanitization before GUI display exclude token, headers,
provider response bodies, names, and unrelated data.

The focused same-rule audit found an equivalent canonical security pattern in
`src/universe/contracts.py` and consumers in Universe resolvers and adjusted
price validation. The quality module still owns a duplicate literal rule, while
`app/services/stock_query_service.py::normalize_ts_code` recognizes SH/SZ but
not BJ. These are reported as potential consistency points only; without an
offending real value or failing boundary they were not changed or refactored.

The top-level user classification remains `COVERAGE_VALIDATION` because
`DataPreparationService.ensure` marks the origin local immediately before the
quality gate and `classify_data_unavailable_error` prioritizes that origin. The
transaction operation and safe quality summary are accurate, but the main
"local coverage" wording is not a provider-quality-specific diagnosis. It is
deferred rather than guessed in this diagnostic-only pass.

尚未确定完整根因，仅补充违规标识样本诊断，需要用户再重试一次。

## UAT-032 legacy reference identity contract

Task `8d75acc7-30be-4f2d-a6ce-7f013adf956f` and transaction
`4a9b7ca2f81047b19483259475900c3c` captured two exact proxy-returned D
identifiers: `T600018.SH` and `TS0018.SH`. The transaction retained their public
identifier/status/market/exchange fields but, by design, not the complete rows.
Consequently its task record cannot supply name, list date, or delist date.

A read-only comparison against the legacy official-provider stock-basic cache
found the following repeated row in six dated schema-1.1 snapshots:

```text
ts_code=T600018.SH
symbol=T600018
name=上港集箱(退)
area=null
industry=null
market=null
exchange=SSE
curr_type=CNY
list_status=D
list_date=2000-07-19
delist_date=2006-10-20
```

`TS0018.SH` was absent from that official cache. The isolated current official
provider directory has no stock-basic snapshot. No real endpoint was called.
The official doc-25 field contract identifies `ts_code`, `symbol`, name,
market/exchange, lifecycle status, list date, and delist date as stock-basic
reference fields, but it provides no entity-mapping field or rule authorizing
removal of `T`/`TS` prefixes. Numeric similarity is therefore not mapping proof.

For this task, the existing calendar and planning formulas prove the complete
required interval as `2022-11-17` through `2023-02-08`: the first requested open
formation is 2023-01-03; 33 required history periods start on 2022-11-17; the
effective last formation is 2023-01-31; one entry-lag plus five holding periods
end on 2023-02-08. The evidenced T entity ended in 2006 and is irrelevant to
this interval. TS lifecycle dates are not yet known and must fail closed if the
next response does not prove non-overlap.

### Registry-driven identity classes

The central classification contract is:

- `CANONICAL_TRADABLE`: exact `^[0-9]{6}\.(?:SH|SZ|BJ)$` identity. This remains
  mandatory for daily, daily_basic, adj_factor, suspend events, factors,
  signals, holdings, and backtests.
- `LEGACY_REFERENCE`: a non-canonical stock-basic identity with status D, valid
  ordered list/delist dates, and no overlap with the complete required interval.
  The row remains in the provider reference snapshot and receives stable
  exclusion evidence. It is not mapped, stripped, or exposed as a tradable
  Universe security.
- `INVALID`: a non-canonical identity overlapping the required interval, an
  L/P/G reference, or a row lacking lifecycle proof. Overlap uses
  `UNSUPPORTED_LEGACY_SECURITY_IDENTIFIER`; other unproven rows retain the
  invalid-security quality block.

This decision is dataset-contract driven: stock_basic is
`PROVIDER_REFERENCE`; daily, daily_basic, adj_factor, and suspend_d are
`CANONICAL_TRADABLE`; trade_cal and index identities do not reuse the stock
security rule. Standard numeric D rows are never removed. They participate in
historical eligibility under `list_date <= T < delist_date`, which preserves
delisted securities and prevents survivorship bias.

Because GLOBAL coverage can be reused across tasks, the Universe data boundary
repeats the same classification for the current plan's complete interval. It
removes only `LEGACY_REFERENCE` rows from the tradable slice and blocks an
overlapping cached reference. The interval is included in source identity, so a
classification made for one task is never treated as proof for another range.

### Quarantined raw and error boundary

After all four provider calls normalize and merge, the registered frame is
atomically written as `RAW_STAGED / UNVERIFIED_QUARANTINE` before quality
validation. Raw contains no Token, headers, or request credential. It is never
used by canonical reads, coverage planning, or reuse. A failed quality gate
retains that fetch-specific audit file but cannot create canonical, manifest, or
Ledger COMPLETE. A later successful retry receives a new fetch identity; the
old quarantine cannot masquerade as success.

Quality failures now carry origin `provider_quality` and stage
`quality_validation`. Ordinary invalid provider data maps to
`PROVIDER_DATA_QUALITY`; an overlapping unmapped legacy identity maps precisely
to `UNSUPPORTED_LEGACY_SECURITY_IDENTIFIER`. Only failures after the quality
gate in canonical publication, Ledger commit, or readback remain local Coverage
errors.

The same-rule audit found no other reference dataset incorrectly applying the
stock ts_code regex at ingestion. Index-weight constituent identities are
checked downstream against the strict canonical Universe contract. The SH/SZ-
only convenience behavior in `stock_query_service.py` still omits BJ; it remains
reported but unchanged because this failure did not exercise that code.

Verification used seven distinct exact pytest node IDs. Four affected nodes
were rerun after the final origin and cross-interval reuse boundaries changed;
all eleven invocations passed and each stayed below 7 seconds of pytest runtime.

## UAT-033 canonical reuse and completed research proof

The final UAT-033 task
`432b281e-cb94-406a-9afd-eb04213b5d11` proves the proxy-provider contract beyond
data preparation. All 433 required units were already COMPLETE and passed
canonical file, manifest, schema, content-hash, Ledger, and readback checks, so
the provider-call count was zero by the offline-first contract. This is reuse of
real local TuShare data, not fallback or synthetic data.

Fresh read-only planning reported every adj_factor, daily, daily_basic,
index_daily, stock_basic, and trade_cal requirement READY with zero missing
units. The `research_input_1.0` materialization was reusable and all six recorded
hashes matched. Its complete context is 2022-10-21 through 2023-02-08, with 48
formation dates from 2022-11-17 through 2023-01-31. Provider provenance records
`tushare_proxy`, endpoint names, dataset schema versions, PIT cutoffs, the
stock-basic 1.2 identity, and legacy-reference exclusions. Existing quarantine
evidence was retained and was not eligible for canonical reuse.

The exact run is
`20260814_172620_uat_full_lifecycle_custom_600000_sh_000001_sz_600001_sh_000002_sz`.
Its `run_info.json` records `status=succeeded`, `cache_status=ready`, an empty
missing-range map, and the proxy provider. Native Factor, Modeling Panel, ML,
Signal, Holdings, and Research Backtest validators plus ResultService all pass.
Artifact schemas are research_input 1.0, factor 1, and 1.0 for Modeling, ML,
Signal, Holdings, and Backtest.

This run used `STANDARD_ROBUST`; provenance explicitly marks that missing daily
rows would freeze affected trades. Therefore UAT-009 remains OPEN until a real
strict `suspend_d` (`STRICT_EVENT`) lifecycle succeeds. UAT-032 is CLOSED because
the real stock_basic GLOBAL snapshot completed canonical/Ledger/readback and the
same task continued through research. UAT-028 and UAT-033 are CLOSED because the
real task succeeded with an exact validated run that survives GUI refresh and
application restart.
