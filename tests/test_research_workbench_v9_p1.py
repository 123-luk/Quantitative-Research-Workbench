"""V9-P1 registry-driven Research Workbench gates."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.components.errors import ErrorPresenter
from app.components.forms import split_model_parameter_schema
from app.components.navigation import (
    NAVIGATION_ROUTES,
    initialize_session_state,
    open_results,
)
from app.services.capability_catalog_service import CapabilityCatalogService
from app.services.pipeline_config_service import build_pipeline_config
from app.services.run_service import RunService, SafeRunError
import app.services.run_service as run_service_module
from src.holdings.artifacts import HOLDINGS_ARTIFACT_FILENAMES
from src.holdings.contracts import HOLDINGS_OUTPUT_COLUMNS
from src.research_backtest.analytics import (
    BENCHMARK_DAILY_COLUMNS,
    PERFORMANCE_METRIC_KEYS,
)
from src.research_backtest.artifacts import RESEARCH_BACKTEST_ARTIFACT_FILENAMES
from src.research_backtest.portfolio import DAILY_PORTFOLIO_COLUMNS
from src.research_backtest.rebalance import REBALANCE_OUTPUT_COLUMNS
from src.signals.artifacts import SIGNAL_ARTIFACT_FILENAMES
from src.signals.contracts import SIGNAL_OUTPUT_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def _state(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "backtest_start": "2024-01-01",
        "backtest_end": "2024-12-31",
        "stock_pool": "hs300",
        "benchmark": "000300.SH",
        "selected_factors": ["momentum_20d", "volatility_20d"],
        "model_name": "ridge",
        "model_params": {},
        "top_n": 10,
        "portfolio_method": "equal_weight",
    }
    values.update(overrides)
    return values


def test_capability_catalog_is_exactly_registry_backed() -> None:
    catalog = CapabilityCatalogService()
    assert catalog.list_factor_names() == ("momentum_20d", "volatility_20d")
    assert catalog.list_model_names() == (
        "elastic_net",
        "hist_gradient_boosting",
        "ridge",
    )
    assert catalog.list_portfolio_methods() == (
        "equal_weight",
        "inverse_volatility",
        "minimum_variance",
        "rank_weight",
    )
    assert catalog.list_risk_estimators() == (
        "ledoit_wolf",
        "sample_covariance",
    )
    assert catalog.list_constraints() == ("max_weight",)
    source = inspect.getsource(CapabilityCatalogService)
    for capability in ("momentum_20d", "ridge", "minimum_variance", "ledoit_wolf"):
        assert capability not in source


@pytest.mark.parametrize(
    "model_name", ("ridge", "elastic_net", "hist_gradient_boosting")
)
def test_model_parameter_controls_preserve_backend_schema(model_name: str) -> None:
    catalog = CapabilityCatalogService()
    schema = catalog.get_model_parameter_schema(model_name)
    ordinary, advanced = split_model_parameter_schema(schema)
    controls = ordinary + advanced
    assert {item.name for item in controls} == {spec.name for spec in schema}
    assert {item.name for item in advanced} == {
        spec.name for spec in schema if spec.advanced
    }
    assert catalog.validate_model_parameters(model_name, {}) == {
        spec.name: spec.default for spec in schema
    }


@pytest.mark.parametrize(
    ("method", "expected_params"),
    [
        ("equal_weight", {}),
        ("rank_weight", {}),
        (
            "inverse_volatility",
            {"lookback_trading_days": 60, "min_observations": 40},
        ),
    ],
)
def test_builder_portfolio_methods(method: str, expected_params: dict[str, object]) -> None:
    config = build_pipeline_config(_state(portfolio_method=method))
    assert config.holdings.portfolio_construction.method == method
    assert dict(config.holdings.portfolio_construction.params) == expected_params


@pytest.mark.parametrize("estimator", ("sample_covariance", "ledoit_wolf"))
def test_builder_minimum_variance_risk_estimators(estimator: str) -> None:
    config = build_pipeline_config(
        _state(
            portfolio_method="minimum_variance",
            risk_estimator=estimator,
            risk_lookback_trading_days=90,
            risk_min_observations=60,
        )
    )
    assert config.holdings.portfolio_construction.to_dict()["params"] == {
        "risk_model": {
            "estimator": estimator,
            "params": {},
            "lookback_trading_days": 90,
            "min_observations": 60,
        }
    }


def test_builder_max_weight_and_backtest_are_canonical() -> None:
    config = build_pipeline_config(
        _state(
            max_weight_enabled=True,
            max_weight_percent=12.5,
            research_backtest_enabled=True,
            transaction_cost_bps=2.5,
            research_backtest_benchmark="000905.SH",
            annual_risk_free_rate=0.02,
            annualization_days=250,
            initial_nav=100.0,
        )
    )
    constraint = config.holdings.portfolio_construction.constraints[0]
    assert constraint.to_dict() == {
        "type": "max_weight",
        "params": {"max_weight": 0.125},
    }
    backtest = config.research_backtest
    assert backtest.transaction_cost.cost_bps == 2.5  # type: ignore[union-attr]
    assert backtest.benchmark.benchmark_code == "000905.SH"  # type: ignore[union-attr]
    assert backtest.performance.annualization_days == 250  # type: ignore[union-attr]
    assert backtest.portfolio.initial_nav == 100.0


def test_builder_is_deterministic_detached_and_preserves_model_defaults() -> None:
    state = _state(model_name="elastic_net", model_params={})
    before = deepcopy(state)
    first = build_pipeline_config(state)
    second = build_pipeline_config(state)
    assert first.to_dict() == second.to_dict()
    assert state == before
    assert first.ml_experiment.experiment is not None
    assert first.ml_experiment.experiment.training_config.model_params


@pytest.mark.parametrize(
    "updates",
    [
        {"selected_factors": ["pe"]},
        {"model_name": "random_forest"},
        {"portfolio_method": "risk_parity"},
        {"portfolio_method": "minimum_variance", "risk_estimator": "unknown"},
        {"top_n": 0},
        {"top_n": True},
    ],
)
def test_builder_rejects_unknown_or_invalid_values(updates: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_pipeline_config(_state(**updates))


def test_run_service_success_uses_exact_returned_identity() -> None:
    config = build_pipeline_config(_state())
    summary = {
        "status": "ready",
        "run_dir": "output/runs/20260811_exact_run",
        "signal": {"artifact_dir": "exact/signal"},
    }
    outcome = RunService(lambda received: summary).run(config)
    assert outcome.success
    assert outcome.run_id == "20260811_exact_run"
    assert outcome.status == "ready"
    assert outcome.elapsed_seconds >= 0.0
    assert outcome.artifact_summary == {"signal": {"artifact_dir": "exact/signal"}}


def test_run_service_failure_is_safe_and_never_discovers_latest() -> None:
    config = build_pipeline_config(_state())

    def fail(received: object) -> dict[str, object]:
        raise RuntimeError("TUSHARE_TOKEN=do-not-display")

    outcome = RunService(fail).run(config)  # type: ignore[arg-type]
    assert not outcome.success
    assert outcome.error is not None
    assert outcome.error.message == "Sensitive backend details were redacted."
    source = inspect.getsource(RunService)
    for forbidden in ("glob(", "rglob(", "mtime", "getmtime", "latest"):
        assert forbidden not in source.lower()


def test_run_service_keeps_exact_identity_when_backend_fails_after_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_after_create(config: object, *, run_created_callback: object) -> object:
        run_created_callback(Path("output/runs/exact-failed-run"))  # type: ignore[operator]
        raise RuntimeError("holdings failed")

    monkeypatch.setattr(run_service_module, "run_pipeline", fail_after_create)
    outcome = RunService().run(build_pipeline_config(_state()))
    assert not outcome.success
    assert outcome.run_id == "exact-failed-run"
    assert outcome.error is not None and outcome.error.run_id == "exact-failed-run"


def test_run_service_preserves_failed_stage_direct_cause_and_input_shape(
    tmp_path: Path,
) -> None:
    config = build_pipeline_config(_state())
    config.output_dir = str(tmp_path / "output")

    def fail(config: object, *, run_created_callback: object, stage_callback: object) -> object:
        run_dir = tmp_path / "output" / "runs" / "exact-failed-run"
        signal_dir = run_dir / "signal"
        signal_dir.mkdir(parents=True)
        (signal_dir / "audit.json").write_text(
            json.dumps(
                {
                    "input_rows": 30,
                    "output_rows": 30,
                    "trade_date_count": 10,
                    "min_trade_date": "2023-01-11",
                    "max_trade_date": "2023-01-31",
                }
            ),
            encoding="utf-8",
        )
        run_created_callback(run_dir)  # type: ignore[operator]
        stage_callback("portfolio", "STARTED")  # type: ignore[operator]
        try:
            raise ValueError(
                "insufficient universe: requested top_n=10, available_count=3"
            )
        except ValueError as cause:
            raise RuntimeError("Holdings build failed: ValueError") from cause

    outcome = RunService(fail, supports_identity_hook=True).run(
        config, stage_callback=lambda stage, status: None
    )

    assert not outcome.success and outcome.error is not None
    assert outcome.error.stage == "portfolio"
    assert outcome.error.cause_class == "ValueError"
    assert outcome.error.cause_message == (
        "insufficient universe: requested top_n=10, available_count=3"
    )
    assert outcome.error.input_shape == {
        "input_rows": 30,
        "output_rows": 30,
        "trade_date_count": 10,
        "min_trade_date": "2023-01-11",
        "max_trade_date": "2023-01-31",
    }
    assert outcome.error.output_shape is None
    assert outcome.error.retryable and outcome.error.retry_stage == "validate"


def test_navigation_and_exact_results_handoff_use_small_state() -> None:
    assert NAVIGATION_ROUTES == ("Overview", "New Run", "Results", "Runs", "Data")
    state: dict[str, object] = {}
    initialize_session_state(state)
    open_results(state, "exact-run-id")
    assert state["current_page"] == "Results"
    assert state["selected_run_id"] == "exact-run-id"
    assert set(state) == {
        "current_page",
        "draft_config",
        "current_run_id",
        "selected_run_id",
        "last_run_status",
        "locale",
    }
    with pytest.raises(ValueError):
        open_results(state, "")


def test_results_handoff_persists_exact_run_in_query_params() -> None:
    state: dict[str, object] = {}
    query_params: dict[str, object] = {}

    open_results(state, " exact-run-id ", query_params)

    assert state == {
        "selected_run_id": "exact-run-id",
        "current_page": "Results",
    }
    assert query_params == {"run_id": "exact-run-id"}


def test_error_presenter_contains_only_safe_fields() -> None:
    payload = ErrorPresenter.payload(
        SafeRunError("ValueError", "invalid Top N", "Holdings", "run-1")
    )
    assert payload["title"] == "Research Run Failed"
    assert set(payload["technical_details"]) == {
        "exception_class",
        "message",
        "stage",
        "run_id",
    }


def test_artifact_contract_audit_matches_backend_constants() -> None:
    assert SIGNAL_ARTIFACT_FILENAMES == (
        "signals.parquet", "config.json", "audit.json", "manifest.json"
    )
    assert SIGNAL_OUTPUT_COLUMNS == ("trade_date", "ts_code", "score", "rank")
    assert HOLDINGS_ARTIFACT_FILENAMES == (
        "holdings.parquet", "config.json", "audit.json", "manifest.json"
    )
    assert HOLDINGS_OUTPUT_COLUMNS == (
        "trade_date", "ts_code", "target_weight", "score", "rank"
    )
    assert RESEARCH_BACKTEST_ARTIFACT_FILENAMES == (
        "rebalances.parquet",
        "daily_portfolio.parquet",
        "benchmark.parquet",
        "metrics.json",
        "config.json",
        "audit.json",
        "manifest.json",
    )
    assert len(PERFORMANCE_METRIC_KEYS) == 20
    assert len(DAILY_PORTFOLIO_COLUMNS) == 9
    assert len(BENCHMARK_DAILY_COLUMNS) == 4
    assert len(REBALANCE_OUTPUT_COLUMNS) == 10


def test_app_ui_purity_static_gate() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "app").rglob("*.py")
    )
    for forbidden in (
        "np.cov",
        "DataFrame.cov",
        "LedoitWolf(",
        "scipy.optimize",
        "pct_change(",
        "open_latest_results",
    ):
        assert forbidden not in source


def test_streamlit_five_page_shell_and_registry_driven_new_run() -> None:
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert tuple(app.sidebar.selectbox[0].options) == ("中文", "English")
    app = AppTest.from_file(str(ROOT / "app" / "views" / "new_run.py"), default_timeout=30).run()
    app.session_state["locale"] = "en"
    app.run()
    assert not app.exception
    assert [item.value for item in app.header] == [
        "1. Data & Universe",
        "2. Factor & Modeling",
        "3. Signal & Selection",
        "4. Portfolio Construction",
        "5. Research Backtest",
    ]
    factors = next(item for item in app.multiselect if item.label == "Factors").options
    assert "Bp" in factors and factors
    assert len([item for item in app.button if item.label == "Run Research"]) == 1


@pytest.mark.parametrize(
    "model_name", ("ridge", "elastic_net", "hist_gradient_boosting")
)
def test_streamlit_dynamic_model_schemas_render(model_name: str) -> None:
    app = AppTest.from_file(str(ROOT / "app" / "views" / "new_run.py"), default_timeout=30).run()
    app.session_state["locale"] = "en"
    app.run()
    model = next(item for item in app.selectbox if item.label == "Model")
    model.set_value(model_name)
    app.run()
    assert not app.exception
    assert next(item for item in app.selectbox if item.label == "Model").value == model_name
