# V9 Quant Research Workbench

V9 targets `v0.10.0` and turns the existing research engine into an operable
Streamlit workbench. P1 establishes the product shell, registry-driven New Run
configuration, the canonical execution boundary, safe error handling, and
exact-run routing. It does not change quantitative or accounting semantics.

## Product shell and boundaries

The five main pages are **Overview**, **New Run**, **Results**, **Runs**, and
**Data**. Overview and New Run are functional in P1. Results accepts only an
exact `selected_run_id`; Runs and Data are stable P2 placeholders. The old
dashboard remains available in a separate compatibility expander.

The only supported execution flow is:

```text
UI form state -> PipelineConfig builder -> backend validation
              -> run_pipeline(config) -> exact returned run_dir/run_id
              -> Results(run_id)
```

The UI does not calculate factors, train models, rank securities, select Top-N,
construct weights, estimate covariance, optimize portfolios, or recompute
backtest metrics. Backend contracts remain the source of truth.

## Architecture audit

At the v0.9.0 baseline the Streamlit application was one large module with a
single sidebar navigation list. `app/services/pipeline_config_service.py` was a
UI-independent bridge that loaded a direct `PipelineConfig`, overlaid the V5-V8
Signal/Holdings/Portfolio/Research Backtest controls, and returned another
validated `PipelineConfig`. P1 retains those compatibility functions and adds
the single complete New Run builder in the same service.

`run_pipeline(config)` returns a dictionary. Its unconditional fields are
`status`, `run_dir`, `required_start_date`, `required_end_date`, `cache_status`,
`missing_ranges`, `strategy_name`, and `stock_pool`. Enabled stages add
`factor_research`, `modeling_panel`, `ml_experiment`, `signal`, `holdings`, and
`research_backtest` summaries. `run_dir` is the exact created run directory;
P1 derives `run_id` only as its final path component. The configured ML
`experiment_id` is already a validated identity and is preserved by RunService.
The runner also accepts an optional generic `run_created_callback(Path)` hook.
RunService uses it to retain the exact ID if a later stage fails; the hook does
not alter stage order, business behavior, the CLI, or the returned dictionary.

`ExperimentManager` can create a run directory and write config, run-info, and
placeholder metrics files. It has no public exact-`run_id` reader, safe run
enumeration method, or latest-run API. P1 therefore does not implement a latest
shortcut, Runs catalog, or filesystem discovery. DataManager does expose the
read-only `prepare_data()` readiness check (required range, cache status, and
missing ranges), while ParquetStore exposes exact-path `exists()` and `load()`;
there is no broader data-dashboard metadata API yet.

## Registry audit

The default capability names at the audited baseline are:

- FactorRegistry: `momentum_20d`, `volatility_20d`
- ModelRegistry: `elastic_net`, `hist_gradient_boosting`, `ridge`
- PortfolioConstructionRegistry: `equal_weight`, `inverse_volatility`,
  `minimum_variance`, `rank_weight`
- RiskEstimatorRegistry: `ledoit_wolf`, `sample_covariance`
- ConstraintRegistry: `max_weight`

The old `config/config.yaml` factor selection also contains `pe`, `pb`, and
`roe`. Those names are not in the default FactorRegistry, so the workbench does
not advertise them. CapabilityCatalogService consumes fresh backend registries
and maintains no second capability list.

Every registered model exposes `parameter_schema()`. The UI maps schema
`int`/`float` fields to numeric controls, `bool` to checkboxes,
`choice` to selectors, and strings to text controls. Optional fields have an
explicit enable control. Backend defaults, choices, bounds, steps, canonical
keys, and the `advanced` marker are preserved; backend model creation performs
the final validation.

## New Run configuration

The fixed sections are Data & Universe, Factor / Modeling, Signal & Selection,
Portfolio Construction, and Research Backtest. Engineering storage defaults
remain inherited from the backend config; panel paths required by the existing
file-backed Modeling Panel/Factor Research contracts live in advanced controls.

The builder is deterministic, detached, performs no financial calculation, and
rejects names absent from the registries. It constructs nested Modeling Panel,
ML walk-forward, Signal, Holdings, Portfolio Construction, and optional Research
Backtest configs before invoking `PipelineConfig.from_dict()`. Thus backend
validators continue to own Top-N types, date ranges, stage dependencies, model
parameters, portfolio parameters, max-weight feasibility, and frozen backtest
semantics.

Research Backtest exposes benchmark, cost in bps, annual risk-free rate,
annualization days, and initial NAV. The following semantics are informative,
not selectable: `next_trading_day`, `adjusted_close_to_close`,
`half_l1_pre_to_target`, `one_way_traded_notional`, and
`strict_common_calendar`.

## Run, error, and session contracts

RunService accepts a validated `PipelineConfig`, calls `run_pipeline()` once,
and returns a small `RunOutcome`. It never executes stages itself, retries,
scans directories, compares mtimes, or guesses a latest run. Errors expose only
the exception class, sanitized message, optional stage, and optional run ID;
tracebacks and secret-bearing messages are not rendered by default.

Session state is limited to `current_page`, `draft_config`, `current_run_id`,
`selected_run_id`, and `last_run_status`. DataFrames, model objects, covariance,
predictions, holdings history, and Artifact payloads remain in backend storage.

## Artifact-backed Results contract

P2 will read, validate, and present the exact artifacts identified by the
selected run. P1 audited and freezes these contracts:

- Signal files: `signals.parquet`, `config.json`, `audit.json`, `manifest.json`
- Signal columns: `trade_date`, `ts_code`, `score`, `rank`
- Holdings files: `holdings.parquet`, `config.json`, `audit.json`, `manifest.json`
- Holdings columns: `trade_date`, `ts_code`, `target_weight`, `score`, `rank`
- Research Backtest files: `rebalances.parquet`, `daily_portfolio.parquet`,
  `benchmark.parquet`, `metrics.json`, `config.json`, `audit.json`, `manifest.json`
- Daily portfolio columns: `trade_date`, `gross_return`, `transaction_cost`,
  `net_return`, `gross_nav`, `net_nav`, `is_rebalance`, `turnover`,
  `traded_notional`
- Benchmark columns: `trade_date`, `benchmark_code`, `benchmark_return`,
  `benchmark_nav`
- Rebalance columns: `holdings_trade_date`, `effective_date`, `ts_code`,
  `pre_rebalance_weight`, `target_weight`, `weight_change`, `pre_cash_weight`,
  `target_cash_weight`, `cash_weight_change`, `turnover`

The exact `metrics.json` keys are:

`observation_count`, `rebalance_count`, `gross_total_return`,
`net_total_return`, `gross_annualized_return`, `net_annualized_return`,
`net_annualized_volatility`, `net_sharpe_ratio`, `net_max_drawdown`,
`benchmark_total_return`, `benchmark_annualized_return`, `excess_total_return`,
`annualized_excess_return`, `tracking_error`, `information_ratio`,
`average_turnover`, `total_turnover`, `total_traded_notional`,
`total_transaction_cost`, and `transaction_cost_return_drag`.

## P1/P2/P3 and non-goals

P1 is the GUI/execution foundation. P2 adds Artifact-backed Results, the exact
Runs catalog after a safe public enumeration contract exists, and the read-only
Data dashboard. P3 adds GUI E2E coverage and v0.10.0 release readiness.

Non-goals are an asynchronous queue, database, multi-user/login system, cloud
deployment, downloader, live trading/execution, and new factors, models,
portfolio methods, risk estimators, constraints, or backtest semantics.
