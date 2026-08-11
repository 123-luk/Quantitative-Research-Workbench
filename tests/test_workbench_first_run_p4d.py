from __future__ import annotations

from pathlib import Path
import inspect
import json
import sqlite3

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.i18n import LOCALES, TRANSLATIONS, get_locale, set_locale, t
from app.services.credential_service import CredentialService, ProviderErrorKind
from app.services.capability_catalog_service import CapabilityCatalogService
from app.services.first_run_service import FirstRunOrchestrator, WorkbenchErrorCode, WorkbenchRunDraft, WorkbenchRunError, WorkbenchRuntime, bind_materialized_inputs, create_workbench_factor_registry
from app.services.pipeline_config_service import build_pipeline_config
from app.services.result_service import ResultService
from src.data.contracts import ResearchFrequency
from src.pipeline.config import PipelineConfig
from src.universe import UniverseSpec


SECRET = "TEST_SECRET_TUSHARE_TOKEN_P4D_7A19"
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
STOCK_FIELDS = ("ts_code", "symbol", "name", "area", "industry", "market", "exchange", "curr_type", "list_status", "list_date", "delist_date")
DAILY_FIELDS = ("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount")
BASIC_FIELDS = ("ts_code", "trade_date", "close", "turnover_rate", "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "total_mv", "circ_mv")
CODES = ("600000.SH", "600001.SH", "000001.SZ", "000002.SZ")


class _Environment:
    def __init__(self, value: str | None) -> None:
        self.value = value
    def tushare_token(self) -> str | None:
        return self.value


class Provider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append(("trade_cal", (start_date, end_date)))
        dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="D")
        return pd.DataFrame({"exchange": "SSE", "cal_date": dates.strftime("%Y%m%d"), "is_open": [int(item.weekday() < 5) for item in dates], "pretrade_date": None})

    def get_stock_basic(self, list_status: str = "L") -> pd.DataFrame:
        self.calls.append(("stock_basic", list_status))
        if list_status != "L":
            return pd.DataFrame(columns=STOCK_FIELDS)
        rows = []
        for code in CODES:
            rows.append({"ts_code": code, "symbol": code[:6], "name": code, "area": "China", "industry": "Test", "market": "主板", "exchange": "SSE" if code.endswith("SH") else "SZSE", "curr_type": "CNY", "list_status": "L", "list_date": "20100101", "delist_date": None})
        return pd.DataFrame(rows, columns=STOCK_FIELDS)

    def get_daily(self, **kwargs: object) -> pd.DataFrame:
        day = str(kwargs["trade_date"])
        self.calls.append(("daily", day))
        index = pd.Timestamp(day).dayofyear
        rows = []
        for position, code in enumerate(CODES):
            close = 20.0 + index / 10 + position
            rows.append({"ts_code": code, "trade_date": day, "open": close - 0.2, "high": close + 0.3, "low": close - 0.4, "close": close, "pre_close": close - 0.1, "change": 0.1, "pct_chg": 0.5 + position / 10, "vol": 1000.0, "amount": 2000.0})
        return pd.DataFrame(rows, columns=DAILY_FIELDS)

    def get_daily_basic(self, **kwargs: object) -> pd.DataFrame:
        day = str(kwargs["trade_date"])
        self.calls.append(("daily_basic", day))
        rows = []
        for position, code in enumerate(CODES):
            row = {name: 1.0 for name in BASIC_FIELDS}
            row.update(ts_code=code, trade_date=day, pb=2.0 + position / 10)
            rows.append(row)
        return pd.DataFrame(rows, columns=BASIC_FIELDS)

    def get_adj_factor(self, **kwargs: object) -> pd.DataFrame:
        day = str(kwargs["trade_date"])
        self.calls.append(("adj_factor", day))
        return pd.DataFrame([{"ts_code": code, "trade_date": day, "adj_factor": 1.0} for code in CODES])

    def get_suspend_d(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("suspend_d", kwargs.get("trade_date")))
        return pd.DataFrame(columns=("ts_code", "trade_date", "suspend_timing", "suspend_type"))

    def get_index_daily(self, **kwargs: object) -> pd.DataFrame:
        code = str(kwargs["ts_code"])
        self.calls.append(("index_daily", code))
        dates = pd.date_range(pd.to_datetime(str(kwargs["start_date"])), pd.to_datetime(str(kwargs["end_date"])), freq="B")
        return pd.DataFrame([{"ts_code": code, "trade_date": day.strftime("%Y%m%d"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "pre_close": 99.9, "change": 0.1, "pct_chg": 0.1, "vol": 1.0, "amount": 1.0} for day in dates], columns=DAILY_FIELDS)

    def get_index_weight(self, index_code: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        self.calls.append(("index_weight", index_code))
        trade_date = str(end_date or start_date)
        return pd.DataFrame([
            {"index_code": index_code, "con_code": code, "trade_date": trade_date, "weight": 25.0}
            for code in CODES
        ])


def _base(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        backtest_start="2024-01-02", backtest_end="2024-02-29", train_years=0,
        max_lookback_months=0, stock_pool="CUSTOM", benchmark="000300.SH",
        strategy_name="p4d", selected_factors=["bp"], rebalance_frequency="D",
        top_n=2, transaction_cost=0.0, data_root=str(tmp_path / "data"),
        raw_data_dir=str(tmp_path / "data" / "raw"), processed_data_dir=str(tmp_path / "data" / "processed"),
        cache_dir=str(tmp_path / "data" / "cache"), output_dir=str(tmp_path / "output"),
        parquet_engine="pyarrow", required_datasets=[],
    )


def _draft(
    tmp_path: Path,
    *,
    backtest: bool = False,
    universe: UniverseSpec | None = None,
    frequency: ResearchFrequency = ResearchFrequency.DAILY,
    train_window: int = 8,
    validation: int = 3,
    forward_holding: int = 3,
) -> WorkbenchRunDraft:
    state = {
        "backtest_start": "2024-01-02", "backtest_end": "2024-02-29", "stock_pool": "CUSTOM",
        "benchmark": "000300.SH", "selected_factors": ["bp"], "factor_research_enabled": True,
        "composition_method": "none", "evaluate_components": False, "evaluate_composite": False,
        "forward_entry_lag_periods": 1, "forward_holding_periods": forward_holding,
        "model_name": "ridge", "model_params": {}, "train_window_periods": train_window, "validation_periods": validation,
        "retrain_frequency": 3, "embargo_periods": 1, "minimum_cross_section_size": 3,
        "signal_direction": "descending", "top_n": 2, "insufficient_universe_policy": "error",
        "portfolio_method": "equal_weight", "research_backtest_enabled": backtest,
        "transaction_cost_bps": 5.0, "research_backtest_benchmark": "000300.SH",
    }
    config = build_pipeline_config(state, base_config=_base(tmp_path), catalog=CapabilityCatalogService(factor_registry=create_workbench_factor_registry()))
    return WorkbenchRunDraft(config, universe or UniverseSpec.custom(CODES), frequency)


def _service(tmp_path: Path, provider: Provider) -> FirstRunOrchestrator:
    factors = create_workbench_factor_registry()
    return FirstRunOrchestrator(
        runtime_factory=lambda config: WorkbenchRuntime(config, root=tmp_path, factor_registry=factors, client_factory=lambda token: provider),
        preview_runtime_factory=lambda config: WorkbenchRuntime(config, root=tmp_path, factor_registry=factors, client_factory=lambda token: provider, read_only=True),
    )


def test_i18n_catalog_is_strict_complete_and_state_switch_preserves_ids() -> None:
    assert set(TRANSLATIONS) == set(LOCALES)
    assert set(TRANSLATIONS["zh-CN"]) == set(TRANSLATIONS["en"])
    assert all(value.strip() for catalog in TRANSLATIONS.values() for value in catalog.values())
    assert t("missing.key", locale="en") == "⟦missing.key⟧"
    for key in ("nav.overview", "nav.new_run", "nav.results", "nav.runs", "nav.data", "readiness.title", "error.permission"):
        assert TRANSLATIONS["zh-CN"][key] != TRANSLATIONS["en"][key]
    state = {"selected_run_id": "exact-run", "number": 1.25}
    set_locale(state, "en")
    assert get_locale(state) == "en" and state["selected_run_id"] == "exact-run" and state["number"] == 1.25


def test_credential_is_session_first_password_only_and_never_represented() -> None:
    service = CredentialService(environment=_Environment("environment-secret"), client_factory=lambda token: Provider())
    resolved = service.resolve(SECRET)
    assert resolved.source == "session" and resolved.available
    assert SECRET not in repr(resolved)
    assert service.resolve("").source == "environment"
    missing = CredentialService(environment=_Environment(None), client_factory=lambda token: Provider()).test_connection("")
    assert not missing.success and missing.error_kind is ProviderErrorKind.CREDENTIAL_MISSING
    source = APP.read_text(encoding="utf-8")
    assert 'type="password"' in source
    assert "Legacy dashboard" not in source and "legacy_page" not in source
    assert "st.navigation" in source and "sidebar.radio" not in source


def test_explicit_navigation_renders_all_pages_and_default_chinese() -> None:
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception and app.session_state["locale"] == "zh-CN"
    assert len([item for item in app.sidebar.text_input if item.label == "TuShare Token"]) == 1
    for name in ("overview", "new_run", "results", "runs", "data"):
        page = AppTest.from_file(str(ROOT / "app" / "views" / f"{name}.py"), default_timeout=30).run()
        assert not page.exception and page.title
    results = AppTest.from_file(str(ROOT / "app" / "views" / "results.py"), default_timeout=30).run()
    assert results.info


def test_missing_local_without_token_fails_before_pipeline(tmp_path: Path) -> None:
    provider = Provider()
    with pytest.raises(WorkbenchRunError) as raised:
        _service(tmp_path, provider).run(_draft(tmp_path), credential=None)
    assert raised.value.code is WorkbenchErrorCode.CREDENTIAL_MISSING
    assert provider.calls == []
    assert not (tmp_path / "output" / "runs").exists()


def test_readiness_preview_is_local_only_and_does_not_create_ledger(tmp_path: Path) -> None:
    provider = Provider()
    preview = _service(tmp_path, provider).preview(_draft(tmp_path))
    assert preview.calendar_bootstrap_required and preview.rows
    assert provider.calls == []
    assert not (tmp_path / "data" / "metadata" / "catalog.sqlite").exists()


def test_empty_local_custom_daily_then_identical_run_is_offline_and_secret_free(tmp_path: Path) -> None:
    provider = Provider()
    service = _service(tmp_path, provider)
    draft = _draft(tmp_path)
    first = service.run(draft, credential=SECRET)
    assert first.run.success and first.run.run_id
    assert first.provider_calls == 187 and first.materialization.reused is False
    calls = len(provider.calls)
    second = service.run(draft, credential=None)
    assert second.run.success and second.run.run_id != first.run.run_id
    assert second.provider_calls == 0 and len(provider.calls) == calls
    assert second.materialization.reused is True

    rebuilt = service.run(_draft(tmp_path, forward_holding=2), credential=None)
    assert rebuilt.run.success and rebuilt.provider_calls == 0
    assert rebuilt.materialization.reused is False

    ledger = tmp_path / "data" / "metadata" / "catalog.sqlite"
    with sqlite3.connect(ledger) as connection:
        units = [row[0] for row in connection.execute("SELECT unit_key FROM coverage_units WHERE dataset_id='daily_basic' AND status='COMPLETE' ORDER BY unit_key")]
        connection.execute("DELETE FROM coverage_units WHERE dataset_id='daily_basic' AND unit_key=?", (units[-1],))
    tail_calls = len(provider.calls)
    tail = service.run(draft, credential=SECRET)
    assert tail.run.success and tail.provider_calls == 1
    assert len(provider.calls) == tail_calls + 1

    with sqlite3.connect(ledger) as connection:
        connection.execute("DELETE FROM coverage_units WHERE dataset_id='daily_basic' AND unit_key=?", (units[len(units) // 2],))
    gap_calls = len(provider.calls)
    gap = service.run(draft, credential=SECRET)
    assert gap.run.success and gap.provider_calls == 1
    assert len(provider.calls) == gap_calls + 1
    assert ResultService(draft.pipeline_config.output_dir).load(second.run.run_id).run_id == second.run.run_id
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes()
    assert SECRET not in json.dumps(first.plan.to_dict())
    assert SECRET not in json.dumps(dict(first.materialization.diagnostics))


def test_materialized_binding_uses_all_existing_explicit_contracts(tmp_path: Path) -> None:
    provider = Provider()
    result = _service(tmp_path, provider).run(_draft(tmp_path), credential=SECRET)
    bound = bind_materialized_inputs(_draft(tmp_path).pipeline_config, result.materialization)
    assert bound.factor_research.factor_input_path == str(result.materialization.paths["factor_input.parquet"])
    assert bound.factor_research.score_panel_path == str(result.materialization.paths["score_panel.parquet"])
    assert bound.factor_research.price_panel_path == str(result.materialization.paths["price_panel.parquet"])
    assert str(bound.modeling_panel.source.factor_panel_path) == str(result.materialization.paths["modeling_factor_panel.parquet"])
    assert str(bound.modeling_panel.source.forward_returns_path) == str(result.materialization.paths["modeling_forward_returns.parquet"])
    score = pd.read_parquet(result.materialization.paths["score_panel.parquet"])
    assert tuple(score.columns) == ("trade_date", "ts_code")


def test_custom_daily_completes_existing_research_backtest_and_exact_results(tmp_path: Path) -> None:
    provider = Provider()
    draft = _draft(
        tmp_path,
        backtest=True,
        train_window=3,
        validation=1,
        forward_holding=1,
    )
    result = _service(tmp_path, provider).run(draft, credential=SECRET)
    assert result.run.success and result.run.run_id
    bundle = ResultService(draft.pipeline_config.output_dir).load(result.run.run_id)
    assert bundle.run_id == result.run.run_id
    assert bundle.research_backtest_available


def test_index_monthly_downloads_only_selected_index_scope_and_runs(tmp_path: Path) -> None:
    provider = Provider()
    draft = _draft(
        tmp_path,
        universe=UniverseSpec.index("000300.SH"),
        frequency=ResearchFrequency.MONTHLY,
        train_window=2,
        validation=1,
        forward_holding=1,
    )
    result = _service(tmp_path, provider).run(draft, credential=SECRET)
    assert result.run.success
    scopes = [value for name, value in provider.calls if name == "index_weight"]
    assert scopes and set(scopes) == {"000300.SH"}
    assert result.plan.research_frequency is ResearchFrequency.MONTHLY


def test_all_a_shares_runs_without_requesting_index_weight(tmp_path: Path) -> None:
    provider = Provider()
    draft = _draft(
        tmp_path,
        universe=UniverseSpec.all_a_shares(),
        train_window=3,
        validation=1,
        forward_holding=1,
    )
    result = _service(tmp_path, provider).run(draft, credential=SECRET)
    assert result.run.success
    assert not any(name == "index_weight" for name, _ in provider.calls)
