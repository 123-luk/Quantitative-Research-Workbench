"""Registry-driven, five-section New Run page."""

from __future__ import annotations

from datetime import date

from app.components.errors import ErrorPresenter
from app.components.forms import ModelControlDescriptor, split_model_parameter_schema
from app.components.navigation import open_results
from app.services.capability_catalog_service import CapabilityCatalogService
from app.services.pipeline_config_service import build_pipeline_config
from app.services.run_service import RunService
from src.pipeline.config import PipelineConfig


def _label(value: object) -> str:
    return str(value).replace("_", " ").title()


def _render_model_control(
    st: object, model_name: str, control: ModelControlDescriptor
) -> object:
    key = f"workbench_model_{model_name}_{control.name}"
    enabled = True
    if control.optional:
        enabled = st.checkbox(
            f"Set {control.label}",
            value=control.default is not None,
            key=f"{key}_enabled",
            help=control.help,
        )
        if not enabled:
            return None
    if control.widget == "checkbox":
        return st.checkbox(
            control.label, value=bool(control.default), key=key, help=control.help
        )
    if control.widget == "selectbox":
        options = tuple(control.choices or ())
        index = options.index(control.default) if control.default in options else 0
        return st.selectbox(
            control.label, options, index=index, key=key, help=control.help
        )
    if control.widget == "text_input":
        return st.text_input(
            control.label, value=str(control.default), key=key, help=control.help
        )
    value = control.default
    if value is None:
        value = 0 if control.step is None else control.step
    kwargs: dict[str, object] = {
        "label": control.label,
        "value": value,
        "key": key,
        "help": control.help,
    }
    if control.minimum is not None:
        kwargs["min_value"] = control.minimum
    if control.maximum is not None:
        kwargs["max_value"] = control.maximum
    if control.step is not None:
        kwargs["step"] = control.step
    return st.number_input(**kwargs)


def _defaults() -> PipelineConfig:
    return PipelineConfig.from_yaml("config/config.yaml")


def render(st: object) -> None:
    st.title("New Run")
    st.caption("Configure one canonical research pipeline run. Backend validation owns business truth.")
    catalog = CapabilityCatalogService()
    defaults = _defaults()

    st.header("1. Data & Universe")
    data_left, data_right = st.columns(2)
    with data_left:
        start = st.date_input(
            "Start Date", value=date.fromisoformat(defaults.backtest_start), key="workbench_start"
        )
        stock_pool = st.text_input(
            "Stock Pool / Universe", value=defaults.stock_pool, key="workbench_stock_pool"
        )
    with data_right:
        end = st.date_input(
            "End Date", value=date.fromisoformat(defaults.backtest_end), key="workbench_end"
        )
        benchmark = st.text_input(
            "Benchmark", value=defaults.benchmark, key="workbench_benchmark"
        )
    with st.expander("Advanced Data Parameters"):
        train_years = int(
            st.number_input("Training Years", min_value=0, value=defaults.train_years, step=1)
        )
        max_lookback_months = int(
            st.number_input(
                "Maximum Lookback Months",
                min_value=0,
                value=defaults.max_lookback_months,
                step=1,
            )
        )

    st.header("2. Factor / Modeling")
    factor_names = catalog.list_factor_names()
    selected_factors = st.multiselect(
        "Factors", factor_names, default=factor_names, key="workbench_factors"
    )
    factor_research_enabled = st.checkbox(
        "Run Factor Research stage", value=False, key="workbench_factor_research"
    )
    factor_left, factor_right = st.columns(2)
    with factor_left:
        use_neutralization = st.checkbox("Neutralization", value=False)
    with factor_right:
        composition_method = st.selectbox(
            "Composition Method", ("equal", "rolling_ic", "rolling_rank_ic", "none")
        )
    with st.expander("Advanced Factor Parameters"):
        evaluate_components = st.checkbox("Evaluate Components", value=True)
        evaluate_composite = st.checkbox(
            "Evaluate Composite", value=composition_method != "none"
        )
        factor_input_path = st.text_input(
            "Factor Input Panel", value="data/processed/factor_input.parquet"
        )
        score_panel_path = st.text_input(
            "Score Panel", value="data/processed/score_panel.parquet"
        )
        price_panel_path = st.text_input(
            "Price Panel", value="data/processed/price_panel.parquet"
        )
        exposure_panel_path = st.text_input(
            "Exposure Panel", value="data/processed/exposure_panel.parquet"
        )
        factor_panel_path = st.text_input(
            "Modeling Factor Panel",
            value="data/processed/modeling_factor_panel.parquet",
        )
        forward_returns_path = st.text_input(
            "Modeling Forward Returns",
            value="data/processed/modeling_forward_returns.parquet",
        )

    model_name = st.selectbox("Model", catalog.list_model_names(), format_func=_label)
    ordinary, advanced = split_model_parameter_schema(
        catalog.get_model_parameter_schema(model_name)
    )
    model_params: dict[str, object] = {}
    for control in ordinary:
        model_params[control.name] = _render_model_control(st, model_name, control)
    with st.expander("Advanced Model Parameters"):
        for control in advanced:
            model_params[control.name] = _render_model_control(
                st, model_name, control
            )
    with st.expander("Advanced Walk-Forward Parameters"):
        train_window_periods = int(
            st.number_input("Training Window Periods", min_value=1, value=252, step=1)
        )
        validation_periods = int(
            st.number_input("Validation Periods", min_value=1, value=20, step=1)
        )
        window_type = st.selectbox("Window Type", ("rolling", "expanding"))
        retrain_frequency = int(
            st.number_input("Retrain Frequency", min_value=1, value=20, step=1)
        )
        embargo_periods = int(
            st.number_input("Embargo Periods", min_value=0, value=1, step=1)
        )

    st.header("3. Signal & Selection")
    signal_left, signal_right, signal_third = st.columns(3)
    with signal_left:
        signal_direction = st.selectbox(
            "Signal Direction", ("descending", "ascending")
        )
    with signal_right:
        top_n = int(st.number_input("Top N", min_value=1, value=20, step=1))
    with signal_third:
        insufficient_policy = st.selectbox(
            "Insufficient Universe Policy", ("error", "allow_partial")
        )

    st.header("4. Portfolio Construction")
    portfolio_method = st.selectbox(
        "Portfolio Method", catalog.list_portfolio_methods(), format_func=_label
    )
    lookback = 60
    min_observations = 40
    risk_estimator = "ledoit_wolf"
    risk_lookback = 120
    risk_min_observations = 80
    if portfolio_method == "inverse_volatility":
        left, right = st.columns(2)
        lookback = int(left.number_input("Lookback Trading Days", min_value=2, value=60))
        min_observations = int(
            right.number_input(
                "Minimum Observations", min_value=2, max_value=lookback, value=min(40, lookback)
            )
        )
    elif portfolio_method == "minimum_variance":
        risk_estimator = st.selectbox(
            "Risk Estimator", catalog.list_risk_estimators(), format_func=_label
        )
        left, right = st.columns(2)
        risk_lookback = int(
            left.number_input("Risk Lookback Trading Days", min_value=2, value=120)
        )
        risk_min_observations = int(
            right.number_input(
                "Risk Minimum Observations",
                min_value=2,
                max_value=risk_lookback,
                value=min(80, risk_lookback),
            )
        )
    max_weight_enabled = False
    max_weight_percent = 20.0
    if "max_weight" in catalog.list_constraints():
        max_weight_enabled = st.checkbox("Maximum Weight", value=False)
        if max_weight_enabled:
            max_weight_percent = float(
                st.number_input(
                    "Maximum Weight (%)", min_value=0.01, max_value=100.0, value=20.0
                )
            )

    st.header("5. Research Backtest")
    research_backtest_enabled = st.checkbox("Enabled", value=False, key="workbench_backtest")
    transaction_cost_bps = 10.0
    backtest_benchmark = benchmark
    annual_risk_free_rate = 0.0
    annualization_days = 252
    initial_nav = 1.0
    if research_backtest_enabled:
        left, right = st.columns(2)
        transaction_cost_bps = float(
            left.number_input("Transaction Cost (bps)", min_value=0.0, value=10.0)
        )
        backtest_benchmark = right.text_input("Backtest Benchmark", value=benchmark)
        annual_risk_free_rate = float(
            st.number_input("Annual Risk-Free Rate", value=0.0, format="%.4f")
        )
        with st.expander("Advanced Backtest Parameters"):
            annualization_days = int(
                st.number_input("Annualization Days", min_value=1, value=252, step=1)
            )
            initial_nav = float(st.number_input("Initial NAV", min_value=0.01, value=1.0))
        st.caption(
            "Execution is next trading day; returns are adjusted close-to-close; "
            "turnover is half-L1 pre-to-target; costs use one-way traded notional; "
            "benchmark alignment uses a strict common calendar."
        )

    form_state = {
        "backtest_start": start.isoformat(),
        "backtest_end": end.isoformat(),
        "train_years": train_years,
        "max_lookback_months": max_lookback_months,
        "stock_pool": stock_pool,
        "benchmark": benchmark,
        "selected_factors": selected_factors,
        "factor_research_enabled": factor_research_enabled,
        "use_neutralization": use_neutralization,
        "composition_method": composition_method,
        "evaluate_components": evaluate_components,
        "evaluate_composite": evaluate_composite,
        "factor_input_path": factor_input_path,
        "score_panel_path": score_panel_path,
        "price_panel_path": price_panel_path,
        "exposure_panel_path": exposure_panel_path if use_neutralization else None,
        "factor_panel_path": factor_panel_path,
        "forward_returns_path": forward_returns_path,
        "model_name": model_name,
        "model_params": model_params,
        "train_window_periods": train_window_periods,
        "validation_periods": validation_periods,
        "window_type": window_type,
        "retrain_frequency": retrain_frequency,
        "embargo_periods": embargo_periods,
        "signal_direction": signal_direction,
        "top_n": top_n,
        "insufficient_universe_policy": insufficient_policy,
        "portfolio_method": portfolio_method,
        "lookback_trading_days": lookback,
        "min_observations": min_observations,
        "risk_estimator": risk_estimator,
        "risk_lookback_trading_days": risk_lookback,
        "risk_min_observations": risk_min_observations,
        "max_weight_enabled": max_weight_enabled,
        "max_weight_percent": max_weight_percent,
        "research_backtest_enabled": research_backtest_enabled,
        "transaction_cost_bps": transaction_cost_bps,
        "research_backtest_benchmark": backtest_benchmark,
        "annual_risk_free_rate": annual_risk_free_rate,
        "annualization_days": annualization_days,
        "initial_nav": initial_nav,
    }
    try:
        config = build_pipeline_config(form_state, catalog=catalog, base_config=defaults)
        st.session_state["draft_config"] = config.to_dict()
        with st.expander("Canonical PipelineConfig Preview"):
            st.json(config.to_dict())
    except Exception as exc:
        config = None
        st.session_state["draft_config"] = None
        st.warning(str(exc))

    if st.button("Run Research", type="primary", disabled=config is None):
        with st.spinner("Running canonical research pipeline..."):
            outcome = RunService().run(config)
        st.session_state["last_run_status"] = outcome.status
        if outcome.success and outcome.run_id:
            st.session_state["current_run_id"] = outcome.run_id
            st.success(
                f"Research run {outcome.run_id} completed in {outcome.elapsed_seconds:.2f}s."
            )
            open_results(st.session_state, outcome.run_id)
            st.rerun()
        elif outcome.error is not None:
            ErrorPresenter.render(st, outcome.error)
