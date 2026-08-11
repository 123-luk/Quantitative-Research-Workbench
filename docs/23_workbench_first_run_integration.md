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
