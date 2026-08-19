"""Canonical first-run configuration, readiness preview, and orchestration UI."""

from __future__ import annotations

from datetime import date
from hashlib import sha256
import json
import re
from typing import Callable

import pandas as pd

from app.components.forms import ModelControlDescriptor, split_model_parameter_schema
from app.components.navigation import page_path
from app.i18n import get_locale, t
from app.services.capability_catalog_service import CapabilityCatalogService
from app.services.credential_service import CredentialService
from app.services.first_run_service import FirstRunOrchestrator, WorkbenchErrorCode, WorkbenchRunDraft, WorkbenchRunError, create_workbench_factor_registry, validate_workbench_draft_feasibility
from app.services.pipeline_config_service import build_pipeline_config
from app.services.research_task_service import ResearchTaskService
from app.services.research_date_service import shanghai_today, validate_research_dates
from app.services.ui_metadata_service import PARAMETERS, assert_registry_display_metadata, dataset_label, dataset_unit, display_value, factor_explanations, factor_label, parameter_help, parameter_label
from src.data.contracts import ResearchFrequency
from src.data.dataset_registry import create_default_dataset_registry
from src.factors.frequency import FactorFrequencyError
from src.pipeline.config import PipelineConfig
from src.universe import UniverseSpec, UniverseType


INDEX_OPTIONS = (
    ("000016.SH", "上证50", "SSE 50"),
    ("000300.SH", "沪深300", "CSI 300"),
    ("000905.SH", "中证500", "CSI 500"),
    ("000852.SH", "中证1000", "CSI 1000"),
    ("000510.SH", "中证A500", "CSI A500"),
)


def _render_model_control(st: object, locale: str, model_name: str, control: ModelControlDescriptor) -> object:
    key = f"workbench_model_{model_name}_{control.name}"
    label = parameter_label(control.name, locale, control.label)
    help_text = parameter_help(control.name, locale, control.help)
    if control.optional:
        enabled = st.checkbox(t("new.enable_parameter", locale=locale, name=label), value=control.default is not None, key=f"{key}_enabled", help=help_text)
        if not enabled:
            return None
    if control.widget == "checkbox":
        return st.checkbox(label, value=bool(control.default), key=key, help=help_text)
    if control.widget == "selectbox":
        options = tuple(control.choices or ())
        return st.selectbox(label, options, index=options.index(control.default) if control.default in options else 0, key=key, format_func=lambda value: display_value(value, locale), help=help_text)
    if control.widget == "text_input":
        return st.text_input(label, value=str(control.default), key=key, help=help_text)
    value = control.default if control.default is not None else (0 if control.step is None else control.step)
    kwargs: dict[str, object] = {"label": label, "value": value, "key": key, "help": help_text}
    if control.minimum is not None:
        kwargs["min_value"] = control.minimum
    if control.maximum is not None:
        kwargs["max_value"] = control.maximum
    if control.step is not None:
        kwargs["step"] = control.step
    return st.number_input(**kwargs)


def _supported_factors(frequency: ResearchFrequency) -> tuple[str, ...]:
    factor_registry = create_workbench_factor_registry()
    datasets = set(create_default_dataset_registry().list_ids())
    result = []
    for name in factor_registry.list_names():
        metadata = factor_registry.get(name).metadata
        if not set(metadata.required_datasets).issubset(datasets):
            continue
        try:
            metadata.frequency_spec(frequency)
        except FactorFrequencyError:
            continue
        result.append(name)
    return tuple(result)


def _custom_codes(value: str) -> tuple[str, ...]:
    return tuple(item.upper() for item in re.split(r"[\s,;]+", value.strip()) if item)


def _universe_spec(kind: str, custom: str, index_code: str) -> UniverseSpec:
    if kind == UniverseType.CUSTOM.value:
        return UniverseSpec.custom(_custom_codes(custom))
    if kind == UniverseType.INDEX.value:
        return UniverseSpec.index(index_code)
    return UniverseSpec.all_a_shares()


def _stock_pool_adapter(spec: UniverseSpec) -> str:
    if spec.universe_type is UniverseType.INDEX:
        return str(spec.params["index_code"])
    if spec.universe_type is UniverseType.ALL_A_SHARES:
        return UniverseType.ALL_A_SHARES.value
    return "CUSTOM:" + ",".join(spec.params["securities"])  # type: ignore[arg-type]


def _draft_fingerprint(draft: WorkbenchRunDraft) -> str:
    payload = {
        "pipeline_config": draft.pipeline_config.to_dict(),
        "universe_spec": draft.universe_spec.to_dict(),
        "research_frequency": draft.research_frequency.value,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _render_consolidated_readiness(st: object, locale: str, preview: object) -> None:
    st.subheader(t("readiness.title", locale=locale))
    if preview.calendar_bootstrap_required:
        st.info(t("readiness.calendar_bootstrap", locale=locale))
    grouped: dict[tuple[str, tuple[tuple[str, str], ...]], list[object]] = {}
    for item in preview.rows:
        grouped.setdefault((item.dataset_id, item.scope), []).append(item)
    rows = []
    for (dataset_id, scope), items in grouped.items():
        starts = [item.required_start for item in items]
        ends = [item.required_end for item in items]
        missing = tuple(dict.fromkeys(unit for item in items for unit in item.missing_units))
        statuses = {item.status for item in items}
        status = "READY" if statuses == {"READY"} else ("PARTIAL" if "READY" in statuses or "PARTIAL" in statuses else "MISSING")
        rows.append({
            t("readiness.dataset", locale=locale): dataset_label(dataset_id, locale),
            t("readiness.scope", locale=locale): t("readiness.market_scope", locale=locale) if not scope or dict(scope) == {"scope": "CN_A"} else t("readiness.specific_scope", locale=locale),
            t("readiness.range", locale=locale): f"{min(starts)} – {max(ends)}",
            t("readiness.status", locale=locale): t({"READY": "readiness.ready", "PARTIAL": "readiness.partial", "MISSING": "readiness.missing_status"}[status], locale=locale),
            t("readiness.missing", locale=locale): f"{len(missing)} {dataset_unit(dataset_id, locale)}",
            t("readiness.action", locale=locale): t("readiness.reuse" if not missing else "readiness.download", locale=locale),
            t("readiness.impact", locale=locale): t("readiness.no_impact" if not missing else "readiness.blocks", locale=locale),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if preview.research_plan is None:
        st.caption(t("readiness.inputs_wait", locale=locale))
    else:
        st.caption(t("readiness.inputs_reusable" if preview.research_inputs_reusable else "readiness.inputs_build", locale=locale))


def _render_readiness(st: object, locale: str, preview: object) -> None:
    _render_consolidated_readiness(st, locale, preview)
    return
    st.subheader(t("readiness.title", locale=locale))
    if preview.calendar_bootstrap_required:
        st.info(t("readiness.calendar_bootstrap", locale=locale))
    rows = []
    for item in preview.rows:
        rows.append({
            t("readiness.dataset", locale=locale): item.dataset_id,
            t("readiness.scope", locale=locale): dict(item.scope),
            t("readiness.range", locale=locale): f"{item.required_start} → {item.required_end} ({item.required_units})",
            t("readiness.status", locale=locale): t({"READY": "readiness.ready", "PARTIAL": "readiness.partial", "MISSING": "readiness.missing_status"}.get(item.status, "readiness.unknown"), locale=locale),
            t("readiness.missing", locale=locale): len(item.missing_units),
            t("readiness.action", locale=locale): t("readiness.reuse" if item.action == "REUSE_LOCAL" else "readiness.download", locale=locale),
            t("readiness.provider", locale=locale): item.provider_id,
            t("readiness.endpoint", locale=locale): item.endpoint,
            t("readiness.points", locale=locale): item.official_minimum_points,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(t("readiness.inputs_reusable" if preview.research_inputs_reusable else "readiness.inputs_build", locale=locale))


def render(st: object, *, navigate: Callable[[str], None] | None = None, orchestrator: FirstRunOrchestrator | None = None) -> None:
    locale = get_locale(st.session_state)
    st.title(t("new.title", locale=locale))
    st.caption(t("new.subtitle", locale=locale))
    defaults = PipelineConfig.from_yaml("config/config.yaml")
    factor_registry = create_workbench_factor_registry()
    catalog = CapabilityCatalogService(factor_registry=factor_registry)
    assert_registry_display_metadata(models=catalog.list_model_names(), portfolios=catalog.list_portfolio_methods(), risks=catalog.list_risk_estimators())

    st.header(t("new.section.universe", locale=locale))
    left, right = st.columns(2)
    today = shanghai_today()
    default_start = min(date.fromisoformat(defaults.backtest_start), today)
    default_end = min(date.fromisoformat(defaults.backtest_end), today)
    start = left.date_input(t("new.start", locale=locale), value=default_start, max_value=today, key="workbench_start")
    end = right.date_input(t("new.end", locale=locale), value=default_end, max_value=today, key="workbench_end")
    date_validation = validate_research_dates(start, end, today=today)
    if not date_validation.valid:
        st.error(t(f"new.date_error.{date_validation.code}", locale=locale, today=today.isoformat()))
    universe_options = tuple(item.value for item in UniverseType)
    saved_kind = str(st.session_state.get("workbench_universe_canonical", UniverseType.CUSTOM.value))
    kind = left.selectbox(t("new.universe", locale=locale), universe_options, index=universe_options.index(saved_kind) if saved_kind in universe_options else 0, format_func=lambda value: t({"CUSTOM": "new.custom", "INDEX": "new.index", "ALL_A_SHARES": "new.all"}[value], locale=locale), key=f"workbench_universe_type_{locale}")
    st.session_state["workbench_universe_canonical"] = kind
    custom = "600000.SH, 000001.SZ, 600001.SH, 000002.SZ"
    index_code = "000300.SH"
    if kind == UniverseType.CUSTOM.value:
        custom = st.text_area(t("new.custom_codes", locale=locale), value=custom, key="workbench_custom_codes")
        if _custom_codes(custom):
            st.caption(f"{t('new.canonical_codes', locale=locale)}: {', '.join(_custom_codes(custom))}")
    elif kind == UniverseType.INDEX.value:
        index_options = tuple(item[0] for item in INDEX_OPTIONS)
        saved_index = str(st.session_state.get("workbench_index_canonical", index_code))
        index_code = st.selectbox(t("new.index_code", locale=locale), index_options, index=index_options.index(saved_index) if saved_index in index_options else 0, format_func=lambda code: next((item[1] if locale == "zh-CN" else item[2]) + f" ({code})" for item in INDEX_OPTIONS if item[0] == code), key=f"workbench_index_code_{locale}")
        st.session_state["workbench_index_canonical"] = index_code
        st.caption(t("new.index_history", locale=locale))
    else:
        st.caption(t("new.all_history", locale=locale))
    frequency_options = tuple(item.value for item in ResearchFrequency)
    saved_frequency = str(st.session_state.get("workbench_frequency_canonical", ResearchFrequency.DAILY.value))
    frequency = right.selectbox(t("new.frequency", locale=locale), frequency_options, index=frequency_options.index(saved_frequency) if saved_frequency in frequency_options else 0, format_func=lambda value: t("new.daily" if value == "DAILY" else "new.monthly", locale=locale), key=f"workbench_frequency_{locale}")
    st.session_state["workbench_frequency_canonical"] = frequency
    research_frequency = ResearchFrequency(frequency)
    if research_frequency is ResearchFrequency.MONTHLY:
        st.caption(t("new.monthly_help", locale=locale))
    benchmark = st.text_input(t("new.benchmark", locale=locale), value=defaults.benchmark, key="workbench_benchmark")
    with st.expander(t("new.advanced_data", locale=locale)):
        train_years = int(st.number_input(t("new.train_years", locale=locale), min_value=0, value=0, step=1))
        max_lookback_months = int(st.number_input(t("new.lookback_months", locale=locale), min_value=0, value=0, step=1))

    st.header(t("new.section.factor", locale=locale))
    factor_names = _supported_factors(research_frequency)
    default_factors = factor_names[: min(2, len(factor_names))]
    selected_factors = st.multiselect(t("new.factors", locale=locale), factor_names, default=default_factors, key=f"workbench_factors_{frequency}", format_func=lambda value: factor_label(value, locale))
    with st.expander(t("factor.library", locale=locale)):
        query = st.text_input(t("factor.search", locale=locale), key="factor_metadata_search").strip().lower()
        for item in factor_explanations(factor_registry, research_frequency, locale):
            if query and query not in item.code.lower() and query not in item.description.lower():
                continue
            st.markdown(f"**{item.name}**" + (f"（{item.code.upper()}）" if locale == "zh-CN" else f" (`{item.code}`)"))
            st.write(item.description)
            st.code(item.formula)
            st.caption(f"{t('factor.lookback', locale=locale)}: {item.lookback} {dataset_unit('daily', locale)} · {t('factor.direction', locale=locale)}: {display_value(item.direction, locale)}")
    use_neutralization = st.checkbox(t("new.neutralization", locale=locale), value=False)
    if use_neutralization:
        st.error(t("new.neutralization_unsupported", locale=locale))
    composition_method = st.selectbox(t("new.composition", locale=locale), ("equal", "rolling_ic", "rolling_rank_ic", "none"), format_func=lambda value: display_value(value, locale))
    with st.expander(t("new.advanced_factor", locale=locale)):
        evaluate_components = st.checkbox(t("new.evaluate_components", locale=locale), value=True)
        evaluate_composite = st.checkbox(t("new.evaluate_composite", locale=locale), value=composition_method != "none")
        a, b = st.columns(2)
        forward_entry = int(a.number_input(PARAMETERS["forward_entry_lag_periods"].label(locale), min_value=0, value=1, step=1, help=PARAMETERS["forward_entry_lag_periods"].help(locale)))
        forward_holding = int(b.number_input(PARAMETERS["forward_holding_periods"].label(locale), min_value=1, value=1 if research_frequency is ResearchFrequency.MONTHLY else 5, step=1, help=PARAMETERS["forward_holding_periods"].help(locale)))
    model_name = st.selectbox(t("new.model", locale=locale), catalog.list_model_names(), format_func=lambda value: display_value(value, locale))
    ordinary, advanced = split_model_parameter_schema(catalog.get_model_parameter_schema(model_name))
    model_params = {control.name: _render_model_control(st, locale, model_name, control) for control in ordinary}
    with st.expander(t("new.advanced_model", locale=locale)):
        for control in advanced:
            model_params[control.name] = _render_model_control(st, locale, model_name, control)
    with st.expander(t("new.advanced_walk", locale=locale)):
        train_window_periods = int(st.number_input(t("new.train_window", locale=locale), min_value=1, value=6 if research_frequency is ResearchFrequency.MONTHLY else 20, step=1))
        validation_periods = int(st.number_input(t("new.validation", locale=locale), min_value=1, value=2 if research_frequency is ResearchFrequency.MONTHLY else 5, step=1))
        window_type = st.selectbox(t("new.window", locale=locale), ("rolling", "expanding"), format_func=lambda value: display_value(value, locale))
        retrain_frequency = int(st.number_input(t("new.retrain", locale=locale), min_value=1, value=5, step=1))
        embargo_periods = int(st.number_input(t("new.embargo", locale=locale), min_value=0, value=1, step=1))

    st.header(t("new.section.signal", locale=locale))
    a, b, c = st.columns(3)
    signal_direction = a.selectbox(t("new.direction", locale=locale), ("descending", "ascending"), format_func=lambda value: display_value(value, locale))
    top_n = int(b.number_input(t("new.top_n", locale=locale), min_value=1, value=10, step=1, help=PARAMETERS["top_n"].help(locale)))
    insufficient_policy = c.selectbox(t("new.insufficient", locale=locale), ("error", "allow_partial"), format_func=lambda value: display_value(value, locale))

    st.header(t("new.section.portfolio", locale=locale))
    portfolio_method = st.selectbox(t("new.portfolio", locale=locale), catalog.list_portfolio_methods(), format_func=lambda value: display_value(value, locale))
    lookback, minimum, risk_estimator, risk_lookback, risk_minimum = 60, 40, "ledoit_wolf", 120, 80
    if portfolio_method == "inverse_volatility":
        a, b = st.columns(2)
        lookback = int(a.number_input(t("new.lookback_days", locale=locale), min_value=2, value=60, help=PARAMETERS["lookback_trading_days"].help(locale)))
        minimum = int(b.number_input(t("new.min_observations", locale=locale), min_value=2, max_value=lookback, value=min(40, lookback)))
    elif portfolio_method == "minimum_variance":
        risk_estimator = st.selectbox(t("new.risk_estimator", locale=locale), catalog.list_risk_estimators(), format_func=lambda value: display_value(value, locale))
        a, b = st.columns(2)
        risk_lookback = int(a.number_input(t("new.risk_lookback", locale=locale), min_value=2, value=120))
        risk_minimum = int(b.number_input(t("new.risk_min", locale=locale), min_value=2, max_value=risk_lookback, value=min(80, risk_lookback)))
    max_weight_enabled = st.checkbox(t("new.max_weight", locale=locale), value=False)
    max_weight_percent = float(st.number_input(t("new.max_weight_pct", locale=locale), min_value=0.01, max_value=100.0, value=20.0, help=PARAMETERS["max_weight"].help(locale))) if max_weight_enabled else 20.0

    st.header(t("new.section.backtest", locale=locale))
    backtest_enabled = st.checkbox(t("new.backtest_enabled", locale=locale), value=True, key="workbench_backtest")
    cost_bps, risk_free, annualization_days, initial_nav = 10.0, 0.0, 252, 1.0
    suspension_mode = "STRICT_EVENT"
    if backtest_enabled:
        a, b = st.columns(2)
        cost_bps = float(a.number_input(t("new.cost_bps", locale=locale), min_value=0.0, value=10.0))
        benchmark = b.text_input(t("new.benchmark", locale=locale), value=benchmark, key="workbench_backtest_benchmark")
        risk_free_percent = float(st.number_input(t("new.rf", locale=locale), value=0.0, format="%.4f", help=PARAMETERS["annual_risk_free_rate"].help(locale)))
        st.caption(t("new.rf_scale", locale=locale))
        suspension_mode = st.radio(
            t("new.suspension_mode", locale=locale),
            ("STANDARD_ROBUST", "STRICT_EVENT"),
            format_func=lambda value: t(f"new.suspension_mode.{value}", locale=locale),
            help=t("new.suspension_mode_help", locale=locale),
        )
        risk_free = risk_free_percent / 100.0
        with st.expander(t("new.section.backtest", locale=locale)):
            annualization_days = int(st.number_input(t("new.annualization", locale=locale), min_value=1, value=252, step=1))
            initial_nav = float(st.number_input(PARAMETERS["initial_nav"].label(locale), min_value=0.01, value=1.0, help=PARAMETERS["initial_nav"].help(locale)))

    config = None
    draft = None
    try:
        universe = _universe_spec(kind, custom, index_code)
        state = {
            "backtest_start": start.isoformat(), "backtest_end": end.isoformat(), "train_years": train_years,
            "max_lookback_months": max_lookback_months, "stock_pool": _stock_pool_adapter(universe), "benchmark": benchmark,
            "selected_factors": selected_factors, "factor_research_enabled": True, "use_neutralization": use_neutralization,
            "composition_method": composition_method, "evaluate_components": evaluate_components,
            "evaluate_composite": evaluate_composite, "model_name": model_name, "model_params": model_params,
            "forward_entry_lag_periods": forward_entry, "forward_holding_periods": forward_holding,
            "train_window_periods": train_window_periods, "validation_periods": validation_periods, "window_type": window_type,
            "retrain_frequency": retrain_frequency, "embargo_periods": embargo_periods, "signal_direction": signal_direction,
            "top_n": top_n, "insufficient_universe_policy": insufficient_policy, "portfolio_method": portfolio_method,
            "lookback_trading_days": lookback, "min_observations": minimum, "risk_estimator": risk_estimator,
            "risk_lookback_trading_days": risk_lookback, "risk_min_observations": risk_minimum,
            "max_weight_enabled": max_weight_enabled, "max_weight_percent": max_weight_percent,
            "research_backtest_enabled": backtest_enabled, "transaction_cost_bps": cost_bps,
            "research_backtest_benchmark": benchmark, "annual_risk_free_rate": risk_free,
            "annualization_days": annualization_days, "initial_nav": initial_nav,
            "suspension_mode": suspension_mode,
            "provider_id": st.session_state.get("selected_provider_id", "tushare_official"),
        }
        config = build_pipeline_config(state, catalog=catalog, base_config=defaults)
        draft = WorkbenchRunDraft(config, universe, research_frequency)
        validate_workbench_draft_feasibility(draft)
        st.session_state["draft_config"] = config.to_dict()
        with st.expander(t("new.config_preview", locale=locale)):
            st.dataframe(pd.DataFrame([
                {t("results.setting", locale=locale): t("new.start", locale=locale), t("results.value", locale=locale): config.backtest_start},
                {t("results.setting", locale=locale): t("new.end", locale=locale), t("results.value", locale=locale): config.backtest_end},
                {t("results.setting", locale=locale): t("new.factors", locale=locale), t("results.value", locale=locale): ", ".join(factor_label(value, locale) for value in config.selected_factors)},
                {t("results.setting", locale=locale): t("new.model", locale=locale), t("results.value", locale=locale): display_value(model_name, locale)},
                {t("results.setting", locale=locale): t("new.portfolio", locale=locale), t("results.value", locale=locale): display_value(portfolio_method, locale)},
            ]), width="stretch", hide_index=True)
    except Exception as exc:
        st.session_state["draft_config"] = None
        st.warning(t("new.invalid", locale=locale, message=str(exc)))

    service = orchestrator or FirstRunOrchestrator()
    if draft is not None and st.button(t("readiness.check", locale=locale)):
        with st.spinner(t("readiness.checking", locale=locale)):
            try:
                st.session_state["readiness_preview"] = service.preview(draft)
                st.session_state["readiness_fingerprint"] = _draft_fingerprint(draft)
            except Exception:
                st.session_state["readiness_preview"] = None
    preview = st.session_state.get("readiness_preview")
    if preview is not None and draft is not None and st.session_state.get("readiness_fingerprint") == _draft_fingerprint(draft):
        _render_readiness(st, locale, preview)

    # The ResearchTaskService worker, never this Streamlit render path, owns
    # the canonical service.run(draft, ...) call.
    matching_preview = preview is not None and draft is not None and st.session_state.get("readiness_fingerprint") == _draft_fingerprint(draft)
    local_preflight_valid = matching_preview and (
        preview.research_plan is not None or preview.calendar_bootstrap_required
    )
    missing_datasets = {
        item.dataset_id for item in preview.rows if item.missing_units
    } if matching_preview else set()
    capability = st.session_state.get("provider_capability_report")
    capability_valid = not missing_datasets or (
        capability is not None
        and capability.provider_id == draft.pipeline_config.provider_id
        and all(capability.status_for(dataset_id) == "AVAILABLE" for dataset_id in missing_datasets)
    )
    preflight_valid = local_preflight_valid and capability_valid
    if not preflight_valid and draft is not None:
        st.warning(t("readiness.must_check", locale=locale))
    if st.button(t("new.run", locale=locale), type="primary", disabled=draft is None or use_neutralization or not date_validation.valid or not preflight_valid):
        provider_id = draft.pipeline_config.provider_id
        token_key = "tushare_official_session_token" if provider_id == "tushare_official" else "tushare_proxy_session_token"
        credential = CredentialService().resolve(st.session_state.get(token_key), provider_id=provider_id).reveal_for_provider()
        task_service = ResearchTaskService(
            defaults.output_dir,
            orchestrator_factory=(lambda: orchestrator) if orchestrator is not None else None,
        )
        task = task_service.submit(draft, credential=credential)
        st.session_state["current_task_id"] = task.task_id
        st.session_state["last_run_status"] = task.status
        st.success(t("new.queued", locale=locale))
        st.info(t("new.background", locale=locale))
        if navigate is not None:
            navigate("runs")


if __name__ == "__main__":
    import streamlit as st
    render(st, navigate=lambda name: st.switch_page(page_path(name)))
