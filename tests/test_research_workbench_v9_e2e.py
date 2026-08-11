"""V9-P3 offline GUI user-journey and exact-run release gates."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import shutil

import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from app.components.errors import ErrorPresenter
from app.components.navigation import open_results
from app.services.data_status_service import DataStatusService
from app.services.pipeline_config_service import build_pipeline_config
from app.services.result_service import ResultService, ResultServiceError
from app.services.run_catalog_service import RunCatalogService
from app.services.run_service import RunOutcome, RunService
import app.services.run_service as run_service_module
from src.data.data_manager import DataManager
from src.holdings.artifacts import HoldingsArtifactStore
from src.ml.artifacts import MLExperimentArtifactStore
from src.pipeline.config import PipelineConfig
from src.pipeline.experiment import ExperimentManager
from src.pipeline.runner import run_pipeline as canonical_run_pipeline
from src.research_backtest.artifacts import ResearchBacktestArtifactStore
from src.signals.artifacts import SignalArtifactStore


FACTORS = ("momentum_20d", "volatility_20d")
CODES = ("S00", "S01", "S02", "S03")


class _OfflineMarketClient:
    """Deterministic TuShare-shaped provider used only by the real pipeline."""

    def __init__(self, *, return_scale: float = 1.0) -> None:
        self.return_scale = return_scale
        self.calls: list[str] = []

    @staticmethod
    def _dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
        return pd.date_range(
            pd.to_datetime(start_date, format="%Y%m%d"),
            pd.to_datetime(end_date, format="%Y%m%d"),
            freq="B",
        )

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append("calendar")
        dates = pd.date_range(
            pd.to_datetime(start_date, format="%Y%m%d"),
            pd.to_datetime(end_date, format="%Y%m%d"),
            freq="D",
        )
        return pd.DataFrame(
            {
                "cal_date": dates.strftime("%Y%m%d"),
                "is_open": [int(value.weekday() < 5) for value in dates],
            }
        )

    def get_daily(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        del trade_date
        assert ts_code is not None and start_date is not None and end_date is not None
        self.calls.append(f"daily:{ts_code}")
        code_index = CODES.index(ts_code)
        rows = [
            {
                "trade_date": date.strftime("%Y%m%d"),
                "ts_code": ts_code,
                "pct_chg": self.return_scale
                * (0.03 * (code_index + 1) + 0.002 * index),
            }
            for index, date in enumerate(self._dates(start_date, end_date))
        ]
        return pd.DataFrame(rows, columns=["trade_date", "ts_code", "pct_chg"])

    def get_index_daily(
        self,
        ts_code: str,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        del trade_date
        assert start_date is not None and end_date is not None
        self.calls.append(f"benchmark:{ts_code}")
        return pd.DataFrame(
            [
                {
                    "trade_date": date.strftime("%Y%m%d"),
                    "ts_code": ts_code,
                    "pct_chg": 0.01 * self.return_scale,
                }
                for date in self._dates(start_date, end_date)
            ]
        )

    def get_stock_basic(self, list_status: str = "L") -> pd.DataFrame:
        self.calls.append(f"stock_basic:{list_status}")
        if list_status != "L":
            return pd.DataFrame(
                columns=["ts_code", "list_status", "list_date", "delist_date"]
            )
        return pd.DataFrame(
            {
                "ts_code": list(CODES),
                "list_status": ["L"] * len(CODES),
                "list_date": ["20200101"] * len(CODES),
                "delist_date": [None] * len(CODES),
            }
        )

    def get_suspend_d(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        suspend_type: str | None = None,
    ) -> pd.DataFrame:
        del trade_date, start_date, end_date, suspend_type
        self.calls.append(f"suspend:{ts_code}")
        return pd.DataFrame(
            columns=["ts_code", "trade_date", "suspend_timing", "suspend_type"]
        )


def _write_modeling_inputs(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    factor_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(
        pd.date_range("2024-01-02", periods=18, freq="B")
    ):
        for code_index, code in enumerate(CODES):
            momentum = float(date_index + code_index / 10.0)
            volatility = float(code_index - date_index / 20.0)
            forward_return = 0.001 * (code_index + 1) + 0.0001 * date_index
            entry = trade_date + pd.Timedelta(days=1)
            exit_date = trade_date + pd.Timedelta(days=2)
            entry_price = 10.0 + code_index + date_index / 10.0
            factor_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "momentum_20d": momentum,
                    "volatility_20d": volatility,
                }
            )
            return_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "entry_trade_date": entry,
                    "exit_trade_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": entry_price * (1.0 + forward_return),
                    "forward_return": forward_return,
                }
            )
    factor_path = root / "factor_panel.parquet"
    returns_path = root / "forward_returns.parquet"
    pd.DataFrame(factor_rows).to_parquet(factor_path, engine="pyarrow", index=False)
    pd.DataFrame(return_rows).to_parquet(
        returns_path, engine="pyarrow", index=False
    )
    return factor_path, returns_path


def _base_config(root: Path) -> PipelineConfig:
    return PipelineConfig(
        backtest_start="2024-01-02",
        backtest_end="2024-02-15",
        train_years=0,
        max_lookback_months=0,
        stock_pool="synthetic",
        benchmark="TEST.IDX",
        strategy_name="research_workbench",
        selected_factors=list(FACTORS),
        rebalance_frequency="D",
        top_n=2,
        transaction_cost=0.0,
        data_root=str(root / "data"),
        raw_data_dir=str(root / "data" / "raw"),
        processed_data_dir=str(root / "data" / "processed"),
        cache_dir=str(root / "data" / "cache"),
        output_dir=str(root / "output"),
        parquet_engine="pyarrow",
        required_datasets=[],
    )


def _form_state(
    factor_path: Path,
    returns_path: Path,
    *,
    strategy: str,
    cost_bps: float,
) -> dict[str, object]:
    return {
        "backtest_start": "2024-01-02",
        "backtest_end": "2024-02-15",
        "train_years": 0,
        "max_lookback_months": 0,
        "stock_pool": "synthetic",
        "benchmark": "TEST.IDX",
        "strategy_name": strategy,
        "selected_factors": list(FACTORS),
        "factor_panel_path": str(factor_path),
        "forward_returns_path": str(returns_path),
        "model_name": "ridge",
        "model_params": {},
        "experiment_id": strategy,
        "train_window_periods": 4,
        "validation_periods": 2,
        "retrain_frequency": 2,
        "embargo_periods": 1,
        "minimum_cross_section_size": 3,
        "signal_direction": "descending",
        "top_n": 2,
        "insufficient_universe_policy": "error",
        "portfolio_method": "equal_weight",
        "research_backtest_enabled": True,
        "transaction_cost_bps": cost_bps,
        "research_backtest_benchmark": "TEST.IDX",
        "annual_risk_free_rate": 0.0,
        "annualization_days": 252,
        "initial_nav": 1.0,
    }


@dataclass(frozen=True)
class _Journey:
    root: Path
    output_root: Path
    config_a: PipelineConfig
    config_b: PipelineConfig
    outcome_a: RunOutcome
    outcome_b: RunOutcome
    provider_a: _OfflineMarketClient
    provider_b: _OfflineMarketClient


@pytest.fixture(scope="module")
def journey(tmp_path_factory: pytest.TempPathFactory) -> _Journey:
    root = tmp_path_factory.mktemp("v9-p3-journey")
    factor_path, returns_path = _write_modeling_inputs(root / "inputs")
    base = _base_config(root)
    config_a = build_pipeline_config(
        _form_state(factor_path, returns_path, strategy="journey_a", cost_bps=5.0),
        base_config=base,
    )
    config_b = build_pipeline_config(
        _form_state(factor_path, returns_path, strategy="journey_b", cost_bps=25.0),
        base_config=base,
    )
    providers = iter(
        (_OfflineMarketClient(return_scale=1.0), _OfflineMarketClient(return_scale=2.0))
    )
    used: list[_OfflineMarketClient] = []

    def execute(config: PipelineConfig, *, run_created_callback=None):
        provider = next(providers)
        used.append(provider)
        return canonical_run_pipeline(
            config,
            market_client_factory=lambda: provider,
            run_created_callback=run_created_callback,
        )

    patcher = pytest.MonkeyPatch()
    patcher.setattr(run_service_module, "run_pipeline", execute)
    try:
        outcome_a = RunService().run(config_a)
        outcome_b = RunService().run(config_b)
    finally:
        patcher.undo()
    assert outcome_a.success, outcome_a.error
    assert outcome_b.success, outcome_b.error
    return _Journey(
        root,
        Path(base.output_dir),
        config_a,
        config_b,
        outcome_a,
        outcome_b,
        used[0],
        used[1],
    )


def test_gate_l_full_user_journey_uses_existing_pipeline_and_exact_results(
    journey: _Journey,
) -> None:
    assert journey.outcome_a.run_id is not None
    run_id = journey.outcome_a.run_id
    run_dir = journey.output_root / "runs" / run_id
    assert run_dir.is_dir()
    assert (run_dir / "config_snapshot.yaml").is_file()
    assert (run_dir / "run_info.json").is_file()
    assert SignalArtifactStore().validate(run_dir / "signal").is_valid
    assert HoldingsArtifactStore().validate(run_dir / "holdings").is_valid
    assert ResearchBacktestArtifactStore().validate(
        run_dir / "research_backtest"
    ).is_valid
    ml_dir = run_dir / "ml_artifacts" / "journey_a"
    assert MLExperimentArtifactStore().validate(ml_dir).cross_file_integrity_verified

    result = ResultService(journey.output_root).load(run_id)
    assert result.run_id == run_id
    assert result.research_backtest_available
    assert result.holdings_available
    assert len(result.metrics) == 20
    assert not result.nav.empty and not result.monthly_returns.empty
    assert {item.artifact_type for item in result.artifacts} == {
        "ml",
        "signal",
        "holdings",
        "research_backtest",
    }


def test_gate_l_config_snapshot_is_exact_canonical_ui_mapping(journey: _Journey) -> None:
    run_id = journey.outcome_a.run_id
    assert run_id is not None
    snapshot = yaml.safe_load(
        (
            journey.output_root / "runs" / run_id / "config_snapshot.yaml"
        ).read_text(encoding="utf-8")
    )
    assert snapshot == journey.config_a.to_dict()
    assert snapshot["selected_factors"] == list(FACTORS)
    assert snapshot["ml_experiment"]["experiment"]["training"]["model_name"] == "ridge"
    assert snapshot["signal"]["signal_direction"] == "descending"
    assert snapshot["holdings"]["top_n"] == 2
    assert snapshot["holdings"]["portfolio_construction"]["method"] == "equal_weight"
    assert snapshot["research_backtest"]["benchmark"]["benchmark_code"] == "TEST.IDX"
    assert snapshot["research_backtest"]["transaction_cost"]["cost_bps"] == 5.0
    assert snapshot["research_backtest"]["performance"]["annualization_days"] == 252


def test_gate_l_runs_reopen_and_small_session_route_are_exact(journey: _Journey) -> None:
    run_id = journey.outcome_a.run_id
    assert run_id is not None
    rows = RunCatalogService(journey.output_root).list_runs()
    assert {item.run_id for item in rows} == {
        journey.outcome_a.run_id,
        journey.outcome_b.run_id,
    }
    state: dict[str, object] = {}
    open_results(state, run_id)
    assert state == {"selected_run_id": run_id, "current_page": "Results"}
    reopened = ResultService(journey.output_root).load(
        str(state["selected_run_id"])
    )
    assert reopened.run_id == run_id


def test_gate_l_data_status_uses_real_read_only_manager_contract(
    journey: _Journey,
) -> None:
    config_path = journey.root / "data-status.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "root": str(journey.root / "data"),
                    "raw_dir": str(journey.root / "data" / "raw"),
                    "cache_dir": str(journey.root / "data" / "cache"),
                    "parquet_engine": "pyarrow",
                    "required_start_date": "2024-01-02",
                    "backtest_end": "2024-02-15",
                    "required_datasets": ["daily", "adj_factor"],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    service = DataStatusService(config_path)
    before = set(journey.root.rglob("*"))
    status = service.get_status()
    after = set(journey.root.rglob("*"))
    assert status.cache_status == "missing"
    assert status.required_datasets == ("daily", "adj_factor")
    assert all(not item.exists for item in status.datasets)
    assert before == after


def test_gate_m_exact_run_isolation_for_metrics_nav_holdings_config_and_lineage(
    journey: _Journey,
) -> None:
    assert journey.outcome_a.run_id and journey.outcome_b.run_id
    a = ResultService(journey.output_root).load(journey.outcome_a.run_id)
    b = ResultService(journey.output_root).load(journey.outcome_b.run_id)
    assert a.run_id != b.run_id
    assert a.config_summary["Transaction Cost (bps)"] == 5.0
    assert b.config_summary["Transaction Cost (bps)"] == 25.0
    assert a.metrics["transaction_cost_return_drag"] != b.metrics[
        "transaction_cost_return_drag"
    ]
    assert not a.nav.equals(b.nav)
    assert a.holdings["trade_date"].equals(b.holdings["trade_date"])
    assert {item.relative_path for item in a.artifacts} == {
        "ml_artifacts/journey_a",
        "signal",
        "holdings",
        "research_backtest",
    }
    assert {item.relative_path for item in b.artifacts} == {
        "ml_artifacts/journey_b",
        "signal",
        "holdings",
        "research_backtest",
    }


def test_gate_m_metrics_cards_truth_is_exact_metrics_json_not_daily_recompute(
    journey: _Journey,
) -> None:
    run_id = journey.outcome_a.run_id
    assert run_id is not None
    run_dir = journey.output_root / "runs" / run_id
    canonical = json.loads(
        (run_dir / "research_backtest" / "metrics.json").read_text(encoding="utf-8")
    )
    result = ResultService(journey.output_root).load(run_id)
    assert dict(result.metrics) == canonical
    daily_mean = float(result.daily_returns["net_return"].mean())
    assert result.metrics["net_sharpe_ratio"] != pytest.approx(daily_mean)
    if result.drawdown_matches_metric is False:
        assert result.metrics["net_max_drawdown"] != pytest.approx(
            float(result.drawdown["drawdown"].min())
        )


def test_gate_m_nav_returns_and_detached_reads_preserve_artifacts(
    journey: _Journey,
) -> None:
    run_id = journey.outcome_a.run_id
    assert run_id is not None
    run_dir = journey.output_root / "runs" / run_id / "research_backtest"
    daily = pd.read_parquet(run_dir / "daily_portfolio.parquet", engine="pyarrow")
    benchmark = pd.read_parquet(run_dir / "benchmark.parquet", engine="pyarrow")
    result = ResultService(journey.output_root).load(run_id)
    assert result.nav["Portfolio Net NAV"].tolist() == daily["net_nav"].tolist()
    assert result.nav["Benchmark NAV"].tolist() == benchmark["benchmark_nav"].tolist()
    assert result.nav["trade_date"].is_monotonic_increasing
    expected_monthly = (
        daily.assign(month=daily["trade_date"].dt.to_period("M"))
        .groupby("month", sort=True)["net_return"]
        .agg(lambda values: (1.0 + values).prod() - 1.0)
        .tolist()
    )
    assert result.monthly_returns["net_return"].tolist() == pytest.approx(
        expected_monthly
    )
    result.daily_returns.loc[0, "net_return"] = 999.0
    assert ResultService(journey.output_root).load(run_id).daily_returns.loc[
        0, "net_return"
    ] != 999.0


def test_gate_l_current_session_failure_is_not_success_or_fake_results(
    journey: _Journey,
) -> None:
    created = journey.output_root / "runs" / "20240101_000001_failure_test"

    def fail(config: PipelineConfig, *, run_created_callback=None):
        created.mkdir()
        assert run_created_callback is not None
        run_created_callback(created)
        raise RuntimeError("synthetic holdings failure")

    patcher = pytest.MonkeyPatch()
    patcher.setattr(run_service_module, "run_pipeline", fail)
    try:
        outcome = RunService().run(journey.config_a)
    finally:
        patcher.undo()
    state: dict[str, object] = {"selected_run_id": journey.outcome_a.run_id}
    assert not outcome.success
    assert outcome.status == "failed"
    assert outcome.run_id == created.name
    assert outcome.error is not None
    assert "holdings failure" in outcome.error.message
    assert state["selected_run_id"] == journey.outcome_a.run_id
    partial = next(
        item
        for item in RunCatalogService(journey.output_root).list_runs()
        if item.run_id == created.name
    )
    assert partial.status is None and partial.created_at is None
    assert partial.backtest_status == "not_configured"


@pytest.mark.parametrize(
    ("method", "expected_params"),
    [
        ("equal_weight", {}),
        ("rank_weight", {}),
        (
            "inverse_volatility",
            {"lookback_trading_days": 15, "min_observations": 10},
        ),
    ],
)
def test_gate_l_ui_mapping_uses_canonical_portfolio_methods(
    tmp_path: Path, method: str, expected_params: dict[str, object]
) -> None:
    state = _form_state(
        Path("factor.parquet"),
        Path("returns.parquet"),
        strategy="mapping",
        cost_bps=10.0,
    )
    state.update(
        {
            "portfolio_method": method,
            "lookback_trading_days": 15,
            "min_observations": 10,
        }
    )
    config = build_pipeline_config(state, base_config=_base_config(tmp_path))
    portfolio = config.holdings.portfolio_construction
    assert portfolio.method == method
    assert dict(portfolio.params) == expected_params


@pytest.mark.parametrize("estimator", ["sample_covariance", "ledoit_wolf"])
def test_gate_l_ui_mapping_uses_canonical_minvar_risk_and_optional_cap(
    tmp_path: Path, estimator: str
) -> None:
    state = _form_state(
        Path("factor.parquet"),
        Path("returns.parquet"),
        strategy="minimum_variance",
        cost_bps=10.0,
    )
    state.update(
        {
            "top_n": 5,
            "portfolio_method": "minimum_variance",
            "risk_estimator": estimator,
            "risk_lookback_trading_days": 30,
            "risk_min_observations": 20,
            "max_weight_enabled": True,
            "max_weight_percent": 25.0,
        }
    )
    config = build_pipeline_config(state, base_config=_base_config(tmp_path))
    portfolio = config.holdings.portfolio_construction.to_dict()
    assert portfolio["method"] == "minimum_variance"
    assert portfolio["params"] == {
        "risk_model": {
            "estimator": estimator,
            "params": {},
            "lookback_trading_days": 30,
            "min_observations": 20,
        }
    }
    assert portfolio["constraints"] == [
        {"type": "max_weight", "params": {"max_weight": 0.25}}
    ]


@pytest.mark.parametrize(
    "updates",
    [
        {"backtest_start": "2024-03-01", "backtest_end": "2024-02-01"},
        {"top_n": 0},
        {"top_n": True},
        {"portfolio_method": "unsupported"},
        {"portfolio_method": "minimum_variance", "risk_estimator": "unsupported"},
        {"transaction_cost_bps": -1.0},
        {"annualization_days": 0},
    ],
)
def test_gate_l_invalid_form_values_fail_canonical_validation(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    state = _form_state(
        Path("factor.parquet"),
        Path("returns.parquet"),
        strategy="invalid",
        cost_bps=10.0,
    )
    state.update(updates)
    with pytest.raises((TypeError, ValueError)):
        build_pipeline_config(state, base_config=_base_config(tmp_path))


@pytest.mark.parametrize(
    "identity",
    [
        "",
        "latest",
        "../escape",
        "20240101_000001_alpha_pool/nested",
        "C:/absolute/run",
        "20240101_000001_onlyone",
        "20240101_000001_alpha_pool/../other",
    ],
)
def test_gate_n_result_identity_rejects_traversal_absolute_nested_and_malformed(
    tmp_path: Path, identity: str
) -> None:
    with pytest.raises(ResultServiceError):
        ResultService(tmp_path / "output").load(identity)


def test_gate_n_valid_prefix_or_nearest_run_never_falls_back(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "output")
    exact = manager.create_run_dir("alpha", "pool")
    with pytest.raises(ResultServiceError):
        ResultService(tmp_path / "output").load(exact.name[:-1])
    with pytest.raises(ResultServiceError):
        ResultService(tmp_path / "output").load(
            "20240101_000001_nearest_pool"
        )


def test_gate_n_file_with_valid_run_name_is_not_a_run(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "output")
    manager.runs_root.mkdir(parents=True)
    identity = "20240101_000001_file_pool"
    (manager.runs_root / identity).write_text("not a run", encoding="utf-8")
    assert identity not in manager.list_run_ids()
    with pytest.raises(ValueError, match="regular directory"):
        manager.resolve_run_dir(identity)


def test_gate_n_symlink_run_is_rejected_when_platform_supports_it(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "output")
    target = manager.create_run_dir("target", "pool")
    link = manager.runs_root / "20240101_000001_link_pool"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        assert not link.exists()
        return
    assert link.name not in manager.list_run_ids()
    with pytest.raises(ValueError, match="symbolic link"):
        manager.resolve_run_dir(link.name)


def test_gate_n_catalog_uses_only_canonical_time_and_stable_run_ids(
    tmp_path: Path,
) -> None:
    manager = ExperimentManager(tmp_path / "output")
    manager.runs_root.mkdir(parents=True)
    a = manager.runs_root / "20240101_000001_alpha_pool"
    b = manager.runs_root / "20240101_000002_beta_pool"
    c = manager.runs_root / "20240101_000003_gamma_pool"
    for directory in (a, b, c):
        directory.mkdir()
    (a / "run_info.json").write_text(
        json.dumps({"status": "ready", "created_at": "2024-01-01T00:00:00"}),
        encoding="utf-8",
    )
    (b / "run_info.json").write_text(
        json.dumps({"status": "ready", "created_at": "2024-02-01T00:00:00"}),
        encoding="utf-8",
    )
    (manager.runs_root / "not-a-run").mkdir()
    (manager.runs_root / "20240101_000004_file_pool").write_text(
        "file", encoding="utf-8"
    )
    os.utime(a, (1, 1))
    os.utime(b, (2, 2))
    os.utime(c, (999999999, 999999999))

    rows = RunCatalogService(tmp_path / "output").list_runs()
    assert [item.run_id for item in rows] == [b.name, a.name, c.name]
    assert rows[0].created_at == "2024-02-01T00:00:00"
    assert rows[2].created_at is None and rows[2].status is None
    assert rows[2].model is None and rows[2].net_total_return is None


def test_gate_n_all_missing_created_at_has_no_latest_truth(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "output")
    first = manager.create_run_dir("alpha", "pool")
    second = manager.create_run_dir("beta", "pool")
    rows = RunCatalogService(tmp_path / "output").list_runs()
    assert {item.run_id for item in rows} == {first.name, second.name}
    assert all(item.created_at is None for item in rows)
    assert next((item for item in rows if item.created_at is not None), None) is None


def _copied_run(journey: _Journey, tmp_path: Path) -> tuple[Path, str]:
    run_id = journey.outcome_a.run_id
    assert run_id is not None
    output = tmp_path / "output"
    target = output / "runs" / run_id
    target.parent.mkdir(parents=True)
    shutil.copytree(journey.output_root / "runs" / run_id, target)
    return output, run_id


@pytest.mark.parametrize(
    ("stage", "filename"),
    [
        ("holdings", "holdings.parquet"),
        ("research_backtest", "metrics.json"),
        ("research_backtest", "manifest.json"),
    ],
)
def test_gate_m_corrupt_or_missing_required_artifact_fails_closed(
    journey: _Journey, tmp_path: Path, stage: str, filename: str
) -> None:
    output, run_id = _copied_run(journey, tmp_path)
    path = output / "runs" / run_id / stage / filename
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, engine="pyarrow")
        frame.loc[0, "target_weight"] = -1.0
        frame.to_parquet(path, engine="pyarrow", index=False)
    elif filename == "manifest.json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifact_schema_version"] = "999.0"
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.unlink()
    with pytest.raises(ResultServiceError, match="Artifact validation failed"):
        ResultService(output).load(run_id)


def test_gate_m_missing_rb_keeps_holdings_config_and_lineage_available(
    journey: _Journey, tmp_path: Path
) -> None:
    output, run_id = _copied_run(journey, tmp_path)
    shutil.rmtree(output / "runs" / run_id / "research_backtest")
    result = ResultService(output).load(run_id)
    assert not result.research_backtest_available
    assert result.holdings_available and not result.holdings.empty
    assert result.config_summary["Stock Pool"] == "synthetic"
    assert {item.artifact_type for item in result.artifacts} == {
        "ml",
        "signal",
        "holdings",
    }


def test_gate_m_reopen_uses_snapshot_not_mutated_ui_draft(journey: _Journey) -> None:
    run_id = journey.outcome_a.run_id
    assert run_id is not None
    first = ResultService(journey.output_root).load(run_id)
    draft = journey.config_a.to_dict()
    draft["stock_pool"] = "changed-draft"
    draft["top_n"] = 999
    second = ResultService(journey.output_root).load(run_id)
    assert first.raw_config == second.raw_config
    assert second.config_summary["Stock Pool"] == "synthetic"
    assert second.config_summary["Top N"] == 2


class _SpyStore:
    def __init__(self, root: Path) -> None:
        self.root_dir = root

    def get_dataset_path(self, name: str) -> Path:
        return self.root_dir / f"{name}.parquet"

    def exists(self, name: str) -> bool:
        return name == "daily"

    def save(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("Data dashboard attempted a Parquet write")

    def load(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("Data dashboard attempted a full Parquet scan")


class _SpyCache:
    def __init__(self, root: Path) -> None:
        self.metadata_path = root / "data_status.json"
        self.metadata = {
            "daily": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "updated_at": "2024-02-01T00:00:00",
            }
        }

    def save(self) -> None:
        raise AssertionError("Data dashboard attempted a cache write")

    def update_range(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("Data dashboard attempted a cache update")


class _SpyDataManager:
    def __init__(self, root: Path) -> None:
        self.config = {"data": {"root": str(root.parent)}}
        self.cache = _SpyCache(root)
        self.parquet_store = _SpyStore(root)
        self.prepare_calls = 0

    def get_required_datasets(self) -> list[str]:
        return ["daily", "adj_factor"]

    def prepare_data(self) -> dict[str, object]:
        self.prepare_calls += 1
        return {
            "cache_status": "missing",
            "required_start_date": "2024-01-01",
            "required_end_date": "2024-01-31",
            "missing_ranges": {"adj_factor": [["2024-01-01", "2024-01-31"]]},
        }

    def fetch(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("Data dashboard attempted a provider fetch")

    def download(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("Data dashboard attempted a download")


def test_gate_n_data_status_is_read_only_without_scan_provider_or_write(
    tmp_path: Path,
) -> None:
    manager = _SpyDataManager(tmp_path / "raw")
    status = DataStatusService(manager=manager).get_status()  # type: ignore[arg-type]
    assert manager.prepare_calls == 1
    assert status.cache_status == "missing"
    assert status.datasets[0].exists
    assert not status.datasets[1].exists
    assert status.datasets[1].missing_ranges == (("2024-01-01", "2024-01-31"),)
    source = inspect.getsource(DataStatusService.get_status)
    for forbidden in ("read_parquet", ".load(", ".save(", "fetch(", "download("):
        assert forbidden not in source


def test_gate_l_failure_error_payload_is_diagnostic_without_fake_success() -> None:
    error = RunService(lambda config: (_ for _ in ()).throw(RuntimeError("boom"))).run(
        _base_config(Path("temporary"))
    ).error
    assert error is not None
    payload = ErrorPresenter.payload(error)
    assert payload["title"] == "Research Run Failed"
    assert payload["reason"] == "boom"
    assert payload["technical_details"]["exception_class"] == "RuntimeError"


def test_gate_n_streamlit_five_routes_reset_without_selected_run_guessing() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert app.session_state["selected_run_id"] is None
    for route in ("Overview", "New Run", "Results", "Runs", "Data"):
        app.sidebar.selectbox[0].set_value(route)
        app.run()
        assert not app.exception
    assert app.session_state["selected_run_id"] is None
    app.sidebar.selectbox[0].set_value("Results")
    app.run()
    assert any("exact run" in item.value.lower() for item in app.info)


def test_gate_n_workbench_ui_purity_has_only_allowed_display_derivations() -> None:
    root = Path(__file__).resolve().parents[1]
    workbench_paths = [
        *(root / "app" / "pages").glob("*.py"),
        root / "app" / "services" / "pipeline_config_service.py",
        root / "app" / "services" / "run_service.py",
        root / "app" / "services" / "result_service.py",
        root / "app" / "services" / "run_catalog_service.py",
        root / "app" / "services" / "data_status_service.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in workbench_paths)
    for forbidden in (
        "np.cov",
        "DataFrame.cov",
        "LedoitWolf(",
        "scipy.optimize",
        "pct_change(",
        "TushareClient",
        ".glob(",
        "rglob(",
        "getmtime",
        "max(mtime",
        "open_latest_results",
    ):
        assert forbidden not in source
    result_source = inspect.getsource(ResultService)
    assert "cummax()" in result_source
    assert "to_period(\"M\")" in result_source
    assert "net_sharpe_ratio" not in result_source
