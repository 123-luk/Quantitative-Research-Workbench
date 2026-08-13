# Workbench First-Run Integration

## Startup

Start the sole default UI from the repository root:

```powershell
& "E:\FINANCIAL ENGINEERING\.venv\quant-factor-system\Scripts\python.exe" -m streamlit run app/streamlit_app.py
```

`app/streamlit_app.py` is the only entry point. Opening a page is read-only: it does not contact TuShare, create the coverage ledger, or materialize research inputs.

## Navigation

One explicit `st.Page` / `st.navigation` router registers exactly Overview, New Run, Results, Runs, and Data. The former custom router and all visible legacy dashboard entries are removed. Results shows an explicit empty state until an exact `selected_run_id` exists.

## Language

The centralized `app/i18n` catalog supports canonical locales `zh-CN` and `en`, defaults new sessions to Chinese, and requires identical nonblank keys. Language is session-only and does not change canonical IDs, numeric values, draft configuration, or the selected run.

## Credential

The sidebar accepts a TuShare token through a password widget. Resolution order is current session token, supported environment token, then none. The resolved secret is revealed only at the provider-construction boundary and is never written to config, artifacts, materialization identities, SQLite, fetch events, logs, or Git.

## Universe

New Run creates a canonical `UniverseSpec`:

- `CUSTOM`: validated canonical securities supplied by the user.
- `INDEX`: historical point-in-time membership for the selected canonical index code.
- `ALL_A_SHARES`: historical point-in-time common A-share membership from `stock_basic`; ST, suspended, and newly listed eligibility filters are intentionally not added.

Index weights define membership only; they are not portfolio weights.

## Frequency

Research frequency is the canonical `DAILY` or `MONTHLY` value. Monthly formation uses the final proven open trading day of each calendar month; it does not mean provider monthly data or a global monthly average.

## Data Readiness

The pre-run preview derives the exact `ResearchInputPlan` and Data Layer 2.0 requirements, then performs a local-only coverage-ledger inspection. It displays dataset, scope, range/unit count, missing count, status, action, and whether the exact research-input identity is reusable. An absent ledger is reported without creating one.

## Automatic Preparation

`FirstRunOrchestrator` delegates to the P4B `DataPreparationService`. Preparation computes `Required - COMPLETE`, downloads only missing units, validates coverage, and atomically merges canonical Parquet. INDEX requirements remain scoped to the selected index; CUSTOM and ALL_A_SHARES do not request `index_weight`.

## Research Inputs

The P4C3 builder automatically produces and validates the exact paths for `factor_input.parquet`, `price_panel.parquet`, `score_panel.parquet`, `modeling_factor_panel.parquet`, and `modeling_forward_returns.parquet`. Users no longer enter prepared-file paths. `score_panel` remains the two-column formation/universe key schedule (`trade_date`, `ts_code`), not an ML prediction or score.

## Run Flow

The application stages are: validate configuration, plan research inputs, check local data, download missing data when required, build/reuse research inputs, run the existing pipeline, validate artifacts, and complete. The orchestrator also includes enough pre-backtest formations for the configured leakage-safe walk-forward training, validation, embargo, and forward-label maturity windows.

The generated paths are bound to the existing Factor Research and Modeling file contracts. Existing ML, Signal, Holdings, Portfolio Construction, Risk, and Research Backtest implementations remain the sole quantitative owners. A successful pipeline callback supplies the exact run identity; Results never guesses latest/mtime.

## Offline-first

Complete canonical data needs no token. If materialization is absent, the builder can construct it entirely locally. Missing canonical data without a token fails before business-pipeline execution with a localized credential message and no fake successful run.

## Repeat Run

An identical second run reuses COMPLETE coverage and a validated deterministic materialization. The invariant is zero canonical provider calls. A missing tail or internal unit fetches only that unit; a changed source/config identity invalidates the materialization but does not refresh already COMPLETE provider data.

## Known Boundary

Canonical point-in-time neutralization exposure panels are not yet materialized. Selecting neutralization therefore fails closed before Run with a localized unsupported message. The default non-neutralized path is fully runnable.

## Non-goals

This release does not add refresh-all, repair/delete controls, implicit eligibility filters, a provider console, live trading, multi-user authentication, or a shared application database.

# GUI UAT Consolidated Fix

## Confirmed Root Causes

- Runs `20260811_114943_research_workbench_hs300`, `20260811_115619_research_workbench_hs300`, and `20260811_134409_research_workbench_hs300` are exact historical examples of directories created before failure without `run_info.json` or `config_snapshot.yaml`. The old catalog could enumerate them but could not explain their stage, configuration, or failure; the UI rendered a row of `N/A` values and still offered a result action.
- The Coverage Ledger showed `daily_basic` COMPLETE for 330 trading days and `index_daily` COMPLETE for 331 trading days. Those counts were not remaining missing coverage. The blocking fetch event was `suspend_d` on `2023-11-16`, recorded as `FAILED / CanonicalDataError`.
- TuShare legitimately returned an empty suspension-event snapshot with no columns. The completeness path normalized required columns before applying `allow_empty_complete`, so the valid zero-event day failed. Provider failures were then wrapped as generic `DataUnavailableError`, causing the GUI to report only stale/incomplete coverage.
- The New Run page synchronously called the full first-run orchestrator. Streamlit navigation, rerun, progress, failure context, and exact result readiness therefore all depended on one render call remaining alive.
- Overview and Runs treated canonical run directories as user tasks. They mixed historical partial directories with completed results, used broad `N/A` placeholders, exposed internal identifiers, and did not gate actions by validated result readiness.
- Readiness preview ran on every render and remained in session after configuration changes, allowing slow controls and stale coverage display.

## Final Run Lifecycle

`ResearchTaskService` creates a schema-versioned task record under `output/workbench_tasks`, writes it through staged JSON plus `os.replace`, and returns immediately. A single background worker owns the existing `FirstRunOrchestrator`; repeated clicks for the same active configuration reuse the same task fingerprint. Task state includes exact task/run identity, user name, timestamps, status, current/completed stages, real N/M data coverage progress, safe configuration summary, failure code/stage/dataset/range, and result readiness. It contains no credential.

The canonical runner accepts an optional observation-only stage callback and reports actual Factor, Modeling, ML, Signal, Portfolio/Holdings, and Research Backtest boundaries without changing stage order or calculations. Streamlit reruns read the atomic task record and never restart work. A process restart marks orphaned active tasks `PROCESS_INTERRUPTED`; if missing data still needs TuShare, the user must re-enter the in-memory token and retry.

Only `succeeded + validated exact Artifact + exact run_id` enables View Results. Failed, created, running, cancelled, and incomplete historical records cannot open results. Historical runs remain read-only and never have missing metadata invented.

## Data Readiness and Recovery

Requirements are consolidated by dataset and scope for display. The main table uses localized dataset names, explicit date ranges, missing counts with units, the system's next action, and research impact. Readiness runs only after an explicit user request and is tied to the exact draft fingerprint.

The `suspend_d` fix is deliberately narrow: only registry datasets declaring `allow_empty_complete=True` may convert a provider zero-row/no-column response into a canonical empty schema. All other empty datasets remain strict failures. COMPLETE ledger coverage remains missing-only and is never re-fetched.

Safe preparation failures preserve a structured dataset and unit range and distinguish missing credentials, invalid authentication, insufficient permission/points, rate limiting, network failure, provider-empty response, coverage validation, and other provider failure. The task page shows failed dataset/range/stage, whether research is blocked, the attempted action, and a recovery step. Provider text, traceback, scope JSON, and credentials are not exposed in the primary UI.

## Information Architecture, Metadata, and Performance

Overview is now a state-aware start page answering credential availability, active research, latest task status, and the next primary action. Research Tasks replaces the developer-oriented run table. Internal paths, raw config, schema, and IDs are collapsed under Technical Details.

`ui_metadata_service.py` centralizes parameter scale/unit/help, localized dataset terminology, and registry-backed factor explanations. Factor formulas were audited against implementations. Annual risk-free rate is entered as `%/year` and divided by 100 exactly once at the UI-to-canonical-config boundary. Initial NAV remains a dimensionless baseline.

Measured first-run synchronous work was `198.56s` in the deterministic Fake Provider E2E. Background submit now returns in under `0.5s` by contract and the page is immediately navigable. Five-route AppTest rendering took `2.42s` for the combined smoke; exact-run Workbench GUI E2E completed `41 passed in 16.93s`. Readiness planning is no longer repeated on every control rerun.

## Verification and UAT Status

- Targeted GUI/Run/Data/i18n: `85 passed`.
- Data root-cause regression: `58 passed`.
- Extended Workbench/Data/Pipeline/ResearchInput regression: `162 passed in 669.71s`.
- Heavy first-run/retry/offline exact-run E2E: `1 passed in 204.67s`; identical second run made zero provider calls.
- GUI smoke/exact-run E2E: `1 passed in 8.57s`; `41 passed in 16.93s`.
- Full pytest first run: `3169 passed, 4 skipped, 11 warnings, 3 failed`; all three were legacy UI-label/static compatibility checks and were corrected without reverting the background lifecycle or unit conversion.
- Final full pytest: `3172 passed, 4 skipped, 11 warnings, 0 failed in 914.29s`.

Status: UAT-000 through UAT-019 and UAT-021 are fixed in automated coverage. UAT-020 visual polish remains deferred. Real TuShare account-specific authentication/permission/rate-limit copy and subjective page responsiveness still require manual UAT; no real token is persisted or added to test artifacts. `v0.10.0` remains unpublished.

# GUI UAT Follow-up: UAT-009 Reopened and UAT-022 through UAT-025

## Production Evidence and Empty Event Flow

The persisted records for tasks `acb63909-e8c0-44e9-8b11-dda9da834c43` and its retry prove both failed at `download / suspend_d / 2023-11-16`, before a run identity existed. Coverage Ledger fetch events distinguish the original `CanonicalDataError` from two later `ProviderFetchError` events; the latter waited about 30 seconds at the Provider boundary, so completeness normalization was never reached. The worker uses `WorkbenchRuntime.preparation()` and the same registry-driven `DataPreparationService` as the tests; no alternate downloader exists.

The final production path is: `TushareClient.get_suspend_d` converts only an explicit empty, zero-column DataFrame to the endpoint's canonical columns; fetch completeness then permits that empty frame only because `suspend_d.allow_empty_complete` is true; canonical storage intentionally writes no event row; Coverage Ledger records the requested date as COMPLETE with zero rows and a deterministic empty-content hash. Subsequent planning subtracts that COMPLETE unit and makes zero Provider calls. Non-empty malformed frames, timeouts, authentication, permission, points, rate limiting, and other exceptions are never converted to empty success.

## Stable Navigation and Date Contract

All routes use centralized stable page keys and paths. Submit persists one fingerprinted task and switches to the `runs` page; concurrent/repeated clicks reuse an active exact fingerprint. Refresh and rerun read task state and never restart the worker.

`research_date_service.py` owns both UI and service validation. Today is derived from `Asia/Shanghai`; both date controls use that maximum. End must be strictly later than start. Invalid ranges show localized guidance, disable Run, and are rejected again before `ResearchTaskService` creates a directory, JSON record, or future.

## Localized Display and Diagnostics

Canonical model, model-parameter, factor, composition, signal, portfolio, risk, status, stage, result, and error values remain unchanged internally. `ui_metadata_service.py` supplies centralized Chinese and English display metadata to widgets, task summaries, result tables, and technical summaries. The catalogs remain strict equal-key bilingual maps. Chinese ordinary-user surfaces do not show snake_case values, raw config JSON, absolute paths, Provider repr, traceback, or sensitive parameters.

Failure diagnostics distinguish missing/invalid credentials, permission, points, rate limit, network, malformed Provider structure, unexpected ordinary empty data, local canonical/Coverage Ledger inconsistency, and Pipeline failure. The UI shows localized stage, dataset, date range, category, reason, attempted action, and category-specific recovery. Stored provider text is never rendered.

## Task Record Clearing

Clearing targets one validated direct-child `workbench_tasks/<task_id>.json` file. Active `created` or `running` tasks are rejected in the service and disabled in the UI. Terminal failed, cancelled, succeeded, interrupted, or historical task records require a second explicit confirmation. The operation is idempotent and rejects invalid IDs and symlinks.

For successful tasks, clearing removes only the task lifecycle record. The exact run directory, run metadata, index-visible result, and Artifacts are preserved and therefore continue under historical results. For failed/cancelled/incomplete tasks, only their selected task state and safe diagnostic are removed. The operation never traverses into runs, market data, Coverage Ledger, credentials, research inputs, another task, or arbitrary `data/` content.

Legacy run directories without a task record use hide-only clearing: a validated direct-child run identity receives an idempotent marker under `workbench_tasks/hidden_runs`. The legacy run directory and any existing files are not deleted, but the item remains absent from the Research Tasks list after refresh.

## Verification and Remaining Manual UAT

- UAT follow-up service/AppTest group: `21 passed`.
- TuShare adapter and Data Layer 2.0 preparation: `29 passed`.
- Existing consolidated GUI UAT: `9 passed`.
- Workbench shell and first-run integration: `33 passed`.
- Extended Workbench/config/Pipeline regression: `82 passed`.
- Existing Windows launcher smoke: exit code `0`; isolated port 18501 released.

Automated status: UAT-009 Reopened and UAT-022 through UAT-025 are fixed. Manual UAT must still confirm the real-account `2023-11-16` zero-event response is recorded COMPLETE, the next run advances beyond that date, Chinese wording is natural at the user's display scale, navigation lands on Research Tasks, and clear-confirm interactions match user expectations. No existing real task record was deleted during diagnosis or tests.

# GUI UAT Second Follow-up: UAT-009 and UAT-026 through UAT-029

## Durable completeness and targeted migration

A ledger row alone is no longer sufficient proof of COMPLETE. Every planned
unit is verified against readable canonical data, the current schema, row count,
and content hash. An allowed zero-row event additionally needs an exact,
schema-bound, scope-bound empty marker. The marker is written atomically before
the ledger transaction; a write failure cannot leave a new COMPLETE row.
Legacy COMPLETE rows with missing or damaged proof are treated as that one
missing unit and safely refetched. Ordinary empty datasets and non-empty rows
with missing fields still fail closed.

`daily_basic` schema 1.1 adds the provider-native numeric `dv_ttm` required by
the dividend-yield factor. Schema 1.0 partitions remain usable for factors that
do not request `dv_ttm`; only dividend-dependent units require an upgrade. This
fixes the former planning-only failure where the factor requirement was outside
the registered dataset schema.

The `2023-11-16` suspension date is a real warm-up boundary. The requested
`2024-01-01` start becomes the first open formation `2024-01-02`. The default
ML requirement totals 33 inclusive trading periods: 20 training, 5 validation,
1 embargo, 1 entry lag, 5 holding periods, and 1 current period. The 33rd date
is `2023-11-16`.

## Display, progress, polling, and retry contract

Task timestamps remain timezone-aware UTC at rest. The UI displays them in
`Asia/Shanghai` as `YYYY-MM-DD HH:MM:SS` in Chinese and the same value plus
`CST` in English. Durations use localized seconds/minutes. Corrupt historical
timestamps render a safe em dash.

The data-progress denominator is the count of all actual required coverage
units across the task. The numerator is verified reusable units plus units
whose canonical proof and ledger update have just completed. Progress is
atomically persisted, monotonic, and retained through refresh and terminal
failure. The detail identifies the localized current dataset and exact unit.

While any task is active, the task region uses a local Streamlit fragment at a
three-second interval. Each tick only rereads atomic task JSON. It does not
submit work, invoke buttons, or reset session confirmation state. When the last
task becomes succeeded, failed, or cancelled, one app rerun removes the timed
fragment and polling stops.

Transient connection/read timeouts, DNS, proxy, socket, and network failures
receive at most three total attempts. The two backoffs are 0.5 and 1.0 seconds,
each multiplied by light 0.9--1.1 jitter. Token/authentication, permission,
points, rate limiting/frequency, schema, and response-structure errors are
deterministic and are not retried. A retry always replans verified completion,
so it resumes at the first missing unit and does not redownload checkpoints.

## Real-provider status

The real task `21b2d569-7109-4317-b0fe-79ce538b2f99` exercised
`ResearchTaskService -> worker -> planner -> ledger/canonical verification ->
TuShare`. It resumed at `229/264`, requested only `stock_basic / 2024-01-30`,
and exhausted three roughly 30-second reads. Its terminal record is
`NETWORK_ERROR / READ_TIMEOUT`, attempts `3`, elapsed `170.10s`, with no
`run_id` and Results correctly disabled. The external endpoint remained
unavailable on the workstation's normal network route.

Consequently UAT-026, UAT-027, and UAT-029 are implemented, but UAT-009 remains
unresolved because `suspend_d / 2023-11-16` still has no provider-confirmed
canonical or empty-marker proof. UAT-028 also remains unresolved because this
round produced no real successful exact run or readable Artifact. These items
must not be marked fixed until a normal provider response allows an unmodified
real task to finish and the Results page to open it.
