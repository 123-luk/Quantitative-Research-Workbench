from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.views import data as data_page
from app.views import results as results_page
from app.views import runs as runs_page
from app.services.data_status_service import (
    CanonicalDatasetStatus,
    DataLayer2StatusService,
    DataLayer2StatusView,
    DataStatusService,
    DataStatusView,
    DatasetStatusView,
)
from app.services.formatting import (
    format_bps,
    format_count,
    format_float,
    format_percentage,
)
from app.services.result_service import ResultService, ResultServiceError
from app.services.run_catalog_service import RunCatalogService, RunSummary
from src.holdings.artifacts import (
    HoldingsArtifactConfig,
    HoldingsArtifactStore,
    SignalArtifactProvenance,
)
from src.pipeline.experiment import ExperimentManager
from src.holdings.builder import HoldingsBuilder, HoldingsBuildResult
from src.pipeline.config import PipelineConfig
from src.pipeline.holdings_config import HoldingsPipelineConfig
from src.pipeline.research_backtest_config import (
    BenchmarkConfig,
    PerformanceConfig,
    ResearchBacktestPipelineConfig,
)
from src.pipeline.signal_config import PredictionSourceConfig, SignalPipelineConfig
from src.research_backtest import (
    DAILY_PORTFOLIO_COLUMNS,
    REBALANCE_OUTPUT_COLUMNS,
    PerformanceAnalyticsEngine,
    PortfolioDailyAccountingResult,
    RebalanceAccountingResult,
    ResearchBacktestArtifactStore,
)


def _signals() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-04", "2024-01-04"]
            ),
            "ts_code": ["A.SZ", "B.SZ", "A.SZ", "B.SZ"],
            "score": [0.9, 0.6, 0.7, 0.8],
            "rank": [1, 2, 2, 1],
        }
    )
    frame["ts_code"] = frame["ts_code"].astype("string")
    frame["rank"] = frame["rank"].astype(np.int64)
    frame["trade_date"] = frame["trade_date"].astype("datetime64[ns]")
    return frame.sort_values(["trade_date", "rank", "ts_code"], kind="stable").reset_index(
        drop=True
    )


def _holdings_artifact(run_dir: Path, *, zero_weight: bool = False):
    source_dir = run_dir / "source-signal"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "signals.parquet"
    source_path.write_bytes(b"source-signal")

    result = HoldingsBuilder().build(
        _signals(), top_n=2, insufficient_universe_policy="error", weighting="equal_weight"
    )
    if zero_weight:
        frame = result.holdings
        for _, indexes in frame.groupby("trade_date", sort=False).groups.items():
            positions = list(indexes)
            frame.loc[positions[0], "target_weight"] = 1.0
            frame.loc[positions[1], "target_weight"] = 0.0
        result = HoldingsBuildResult(frame, result.audit)
    provenance = SignalArtifactProvenance(
        source_dir,
        source_path,
        "1.0",
        hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    artifact = HoldingsArtifactStore().write(
        result, provenance, HoldingsArtifactConfig(run_dir / "holdings")
    )
    assert artifact.validation.is_valid
    return artifact


def _rebalances() -> RebalanceAccountingResult:
    rows = []
    for index, (holdings_date, effective_date) in enumerate(
        (("2024-01-02", "2024-01-03"), ("2024-01-04", "2024-01-05"))
    ):
        for code in ("A.SZ", "B.SZ"):
            rows.append(
                {
                    "holdings_trade_date": pd.Timestamp(holdings_date),
                    "effective_date": pd.Timestamp(effective_date),
                    "ts_code": code,
                    "pre_rebalance_weight": 0.0 if index == 0 else 0.5,
                    "target_weight": 0.5,
                    "weight_change": 0.5 if index == 0 else 0.0,
                    "pre_cash_weight": 1.0 if index == 0 else 0.0,
                    "target_cash_weight": 0.0,
                    "cash_weight_change": -1.0 if index == 0 else 0.0,
                    "turnover": 1.0 if index == 0 else 0.0,
                }
            )
    return RebalanceAccountingResult(pd.DataFrame(rows, columns=list(REBALANCE_OUTPUT_COLUMNS)))


def _portfolio() -> PortfolioDailyAccountingResult:
    frame = pd.DataFrame(
        [
            ["2024-01-03", 0.0, 0.001, -0.001, 1.0, 0.999, True, 1.0, 1.0],
            ["2024-01-04", 0.0, 0.0, 0.0, 1.0, 0.999, False, 0.0, 0.0],
            ["2024-01-05", 0.0, 0.0, 0.0, 1.0, 0.999, True, 0.0, 0.0],
        ],
        columns=list(DAILY_PORTFOLIO_COLUMNS),
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return PortfolioDailyAccountingResult(
        frame,
        start_date=pd.Timestamp("2024-01-03"),
        end_date=pd.Timestamp("2024-01-05"),
        rebalance_count=2,
        initial_nav=1.0,
        cost_bps=10.0,
    )


def _analytics(portfolio: PortfolioDailyAccountingResult):
    benchmark = pd.DataFrame(
        [["2024-01-03", "TEST.IDX", 0.5], ["2024-01-04", "TEST.IDX", 0.0], ["2024-01-05", "TEST.IDX", 0.0]],
        columns=["trade_date", "benchmark_code", "return"],
    )
    return PerformanceAnalyticsEngine(
        BenchmarkConfig("TEST.IDX"), PerformanceConfig(0.0)
    ).run(portfolio=portfolio, benchmark_returns=benchmark)


def _root_config() -> PipelineConfig:
    return PipelineConfig(
        backtest_start="2024-01-01",
        backtest_end="2024-01-05",
        train_years=1,
        max_lookback_months=1,
        stock_pool="hs300",
        benchmark="TEST.IDX",
        strategy_name="research",
        selected_factors=[],
        rebalance_frequency="M",
        top_n=2,
        transaction_cost=0.001,
        data_root="data",
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        cache_dir="data/cache",
        output_dir="data/output",
        parquet_engine="auto",
        required_datasets=[],
        signal=SignalPipelineConfig(
            enabled=True,
            source=PredictionSourceConfig("files", Path("native-ml-artifact")),
            artifact_subdir="signal",
        ),
        holdings=HoldingsPipelineConfig(
            enabled=True,
            top_n=2,
            artifact_subdir="holdings",
        ),
        research_backtest=_research_config(),
    )


def _research_config() -> ResearchBacktestPipelineConfig:
    return ResearchBacktestPipelineConfig.from_dict(
        {
            "enabled": True,
            "source": {"mode": "pipeline", "artifact_dir": None},
            "schedule": {"mode": "holdings_dates"},
            "return_alignment": {
                "effective_rule": "next_trading_day",
                "return_convention": "adjusted_close_to_close",
            },
            "portfolio": {"initial_nav": 1.0, "turnover_definition": "half_l1_pre_to_target"},
            "transaction_cost": {"cost_bps": 10.0, "rate_basis": "one_way_traded_notional"},
            "benchmark": {"benchmark_code": "TEST.IDX", "alignment_policy": "strict_common_calendar"},
            "performance": {"annualization_days": 252, "annual_risk_free_rate": 0.0},
            "artifact_subdir": "research_backtest",
        }
    )


def _publish_research_backtest(run_dir: Path) -> None:
    holdings = _holdings_artifact(run_dir)
    portfolio = _portfolio()
    result = ResearchBacktestArtifactStore().publish(
        artifact_dir=run_dir / "research_backtest",
        rebalances=_rebalances(),
        portfolio=portfolio,
        analytics=_analytics(portfolio),
        config=_research_config(),
        holdings_artifact_dir=holdings.artifact_dir,
    )
    assert result.validation.is_valid


@pytest.fixture
def completed_run(tmp_path: Path) -> tuple[Path, str]:
    output_root = tmp_path / "output"
    manager = ExperimentManager(output_root)
    run_dir = manager.create_run_dir("research", "hs300")
    _publish_research_backtest(run_dir)
    config = _root_config()
    manager.save_config_snapshot(run_dir, config)
    manager.save_run_info(
        run_dir,
        {
            "run_id": run_dir.name,
            "status": "success",
            "created_at": "2024-01-10T00:00:00",
        },
    )
    return output_root, run_dir.name


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "N/A"), (float("nan"), "N/A"), (0.125, "12.50%"), (-0.01, "-1.00%")],
)
def test_format_percentage_is_null_and_finite_safe(value: float | None, expected: str) -> None:
    assert format_percentage(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "N/A"), (float("inf"), "N/A"), (1.23456, "1.23"), (-2.0, "-2.00")],
)
def test_format_float_is_null_and_finite_safe(value: float | None, expected: str) -> None:
    assert format_float(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "N/A"), (0, "0"), (1234, "1,234"), (12.9, "12")],
)
def test_format_count_is_safe(value: float | None, expected: str) -> None:
    assert format_count(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "N/A"), (10.0, "10.0 bps"), (-5.0, "-5.0 bps")],
)
def test_format_bps_is_safe(value: float | None, expected: str) -> None:
    assert format_bps(value) == expected


def test_experiment_manager_lists_only_valid_direct_run_directories(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "output")
    manager.runs_root.mkdir(parents=True)
    for name in ("20240101_000001_alpha_hs300", "20240101_000001_alpha_hs300_001"):
        (manager.runs_root / name).mkdir()
    (manager.runs_root / "notes").mkdir()
    (manager.runs_root / "20240101_000001_file_hs300").write_text("not a dir")

    assert manager.list_run_ids() == (
        "20240101_000001_alpha_hs300",
        "20240101_000001_alpha_hs300_001",
    )


@pytest.mark.parametrize("run_id", ["", "latest", "../escape", "a/b", "20240101_alpha"])
def test_experiment_manager_rejects_noncanonical_run_ids(tmp_path: Path, run_id: str) -> None:
    manager = ExperimentManager(tmp_path / "output")
    with pytest.raises(ValueError, match="canonical run id"):
        manager.resolve_run_dir(run_id)


def test_experiment_manager_resolves_exact_existing_run(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "output")
    run_dir = manager.create_run_dir("alpha", "hs300")
    assert manager.resolve_run_dir(run_dir.name) == run_dir.resolve()


def test_result_service_rejects_missing_exact_run(tmp_path: Path) -> None:
    service = ResultService(tmp_path / "output")
    with pytest.raises(ResultServiceError, match="Run not found"):
        service.load("20240101_000001_alpha_hs300")


def test_result_service_loads_canonical_metrics_and_frames(
    completed_run: tuple[Path, str],
) -> None:
    output_root, run_id = completed_run
    bundle = ResultService(output_root).load(run_id)

    assert bundle.run_id == run_id
    assert bundle.research_backtest_available is True
    assert bundle.holdings_available is True
    canonical = json.loads(
        (output_root / "runs" / run_id / "research_backtest" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert dict(bundle.metrics) == canonical
    assert bundle.daily_returns["net_return"].tolist() == [-0.001, 0.0, 0.0]
    assert bundle.holdings.shape[0] == 4
    assert bundle.rebalances.shape[0] == 4


def test_result_service_nav_is_exact_date_aligned(
    completed_run: tuple[Path, str],
) -> None:
    output_root, run_id = completed_run
    nav = ResultService(output_root).load(run_id).nav

    assert nav["trade_date"].is_monotonic_increasing
    assert nav["Portfolio Net NAV"].tolist() == [0.999, 0.999, 0.999]
    assert nav["Benchmark NAV"].tolist() == [1.0, 1.0, 1.0]


def test_result_service_monthly_returns_are_display_only_aggregation(
    completed_run: tuple[Path, str],
) -> None:
    output_root, run_id = completed_run
    monthly = ResultService(output_root).load(run_id).monthly_returns

    assert monthly.to_dict("records") == [
        {"month": "2024-01", "net_return": pytest.approx(-0.001)}
    ]


def test_result_service_drawdown_does_not_replace_canonical_metric(
    completed_run: tuple[Path, str],
) -> None:
    output_root, run_id = completed_run
    bundle = ResultService(output_root).load(run_id)

    assert bundle.metrics["net_max_drawdown"] == pytest.approx(-0.001)
    assert bundle.drawdown["drawdown"].min() == 0.0
    assert bundle.drawdown_matches_metric is False


def test_result_service_returns_detached_frames(completed_run: tuple[Path, str]) -> None:
    output_root, run_id = completed_run
    first = ResultService(output_root).load(run_id)
    first.daily_returns.loc[0, "net_return"] = 99.0
    second = ResultService(output_root).load(run_id)
    assert second.daily_returns.loc[0, "net_return"] == -0.001


def test_result_service_reports_missing_stage_without_fallback(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "output")
    run_dir = manager.create_run_dir("research", "hs300")
    manager.save_config_snapshot(run_dir, _root_config())
    manager.save_run_info(
        run_dir, {"status": "success", "created_at": "2024-01-10T00:00:00"}
    )

    bundle = ResultService(tmp_path / "output").load(run_dir.name)
    assert bundle.research_backtest_available is False
    assert bundle.holdings_available is False
    assert dict(bundle.metrics) == {}


def test_result_service_preserves_zero_weight_selected_holdings(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "output")
    run_dir = manager.create_run_dir("research", "hs300")
    _holdings_artifact(run_dir, zero_weight=True)
    raw = _root_config().to_dict()
    raw["research_backtest"] = ResearchBacktestPipelineConfig().to_dict()
    manager.save_config_snapshot(run_dir, PipelineConfig.from_dict(raw))

    bundle = ResultService(tmp_path / "output").load(run_dir.name)
    assert bundle.holdings_available is True
    assert bundle.research_backtest_available is False
    assert len(bundle.holdings) == 4
    assert int(bundle.holdings["target_weight"].eq(0.0).sum()) == 2


def test_result_service_rejects_present_but_invalid_artifact(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "output")
    run_dir = manager.create_run_dir("research", "hs300")
    manager.save_config_snapshot(run_dir, _root_config())
    (run_dir / "holdings").mkdir()
    (run_dir / "holdings" / "holdings.parquet").write_bytes(b"corrupt")

    with pytest.raises(ResultServiceError, match="Holdings Artifact validation failed"):
        ResultService(tmp_path / "output").load(run_dir.name)


def test_result_lineage_uses_validated_artifact_metadata(
    completed_run: tuple[Path, str],
) -> None:
    output_root, run_id = completed_run
    artifacts = ResultService(output_root).load(run_id).artifacts
    by_stage = {artifact.artifact_type: artifact for artifact in artifacts}

    assert by_stage["holdings"].status == "valid"
    assert by_stage["research_backtest"].status == "valid"
    assert by_stage["research_backtest"].relative_path == "research_backtest"
    assert by_stage["research_backtest"].schema_version == "1.0"


def test_run_catalog_orders_by_canonical_created_at_not_directory_name(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "output")
    older_name = manager.create_run_dir("zeta", "hs300")
    newer_name = manager.create_run_dir("alpha", "hs300")
    config = _root_config()
    manager.save_config_snapshot(older_name, config)
    manager.save_config_snapshot(newer_name, config)
    manager.save_run_info(
        older_name, {"status": "success", "created_at": "2024-01-01T00:00:00"}
    )
    manager.save_run_info(
        newer_name, {"status": "success", "created_at": "2024-02-01T00:00:00"}
    )

    rows = RunCatalogService(tmp_path / "output").list_runs()
    assert [row.run_id for row in rows] == [newer_name.name, older_name.name]


def test_run_catalog_preserves_partial_run_without_invented_status(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "output")
    run_dir = manager.create_run_dir("alpha", "hs300")

    row = RunCatalogService(tmp_path / "output").list_runs()[0]
    assert row.run_id == run_dir.name
    assert row.status is None
    assert row.created_at is None
    assert row.backtest_status == "not_configured"


class _FakeStore:
    def __init__(self, root: Path, existing: set[str]) -> None:
        self.root_dir = root
        self._existing = existing

    def get_dataset_path(self, dataset: str) -> Path:
        return self.root_dir / f"{dataset}.parquet"

    def exists(self, dataset: str) -> bool:
        return dataset in self._existing


class _FakeCache:
    def __init__(self, root: Path, existing: set[str]) -> None:
        self.metadata_path = root / "data_status.json"
        self.metadata = {
            "prices": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "updated_at": "2024-02-01T00:00:00Z",
            },
            "calendar": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-15",
                "updated_at": "2024-01-16T00:00:00Z",
            },
        }


class _FakeDataManager:
    def __init__(self, root: Path, existing: set[str]) -> None:
        self.config = {"data": {"root": str(root.parent)}}
        self.cache = _FakeCache(root, existing)
        self.parquet_store = _FakeStore(root, existing)
        self.calls = 0

    def get_required_datasets(self) -> list[str]:
        return ["prices", "calendar"]

    def prepare_data(self) -> dict[str, object]:
        self.calls += 1
        return {
            "cache_status": "missing",
            "required_start_date": "2024-01-01",
            "required_end_date": "2024-01-31",
            "missing_ranges": {"calendar": [["2024-01-16", "2024-01-31"]]},
        }


def test_data_status_service_is_read_only_and_reports_exact_paths(tmp_path: Path) -> None:
    manager = _FakeDataManager(tmp_path / "cache", {"prices"})
    status = DataStatusService(manager=manager).get_status()

    assert manager.calls == 1
    assert status.cache_status == "missing"
    assert status.raw_data_root == str((tmp_path / "cache").resolve())
    assert status.datasets[0].path == str((tmp_path / "cache" / "prices.parquet").resolve())
    assert status.datasets[0].exists is True
    assert status.datasets[1].exists is False


def test_data_status_service_preserves_missing_range_detail(tmp_path: Path) -> None:
    status = DataStatusService(manager=_FakeDataManager(tmp_path, {"prices"})).get_status()
    calendar = next(item for item in status.datasets if item.dataset == "calendar")
    assert calendar.cached_start == "2024-01-01"
    assert calendar.cached_end == "2024-01-15"
    assert calendar.missing_ranges == (("2024-01-16", "2024-01-31"),)


def test_result_bundle_raw_config_is_canonical_snapshot(
    completed_run: tuple[Path, str],
) -> None:
    output_root, run_id = completed_run
    bundle = ResultService(output_root).load(run_id)
    snapshot = output_root / "runs" / run_id / "config_snapshot.yaml"
    import yaml

    assert dict(bundle.raw_config or {}) == yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert bundle.config_summary["Stock Pool"] == "hs300"
    assert bundle.config_summary["Benchmark"] == "TEST.IDX"


def test_run_catalog_reads_exact_selected_run_metrics(
    completed_run: tuple[Path, str],
) -> None:
    output_root, run_id = completed_run
    row = RunCatalogService(output_root).list_runs()[0]
    assert row.run_id == run_id
    assert row.backtest_status == "available"
    assert row.net_total_return == pytest.approx(-0.001)
    assert row.net_sharpe_ratio is not None


def test_manifest_tampering_is_not_silently_accepted(
    completed_run: tuple[Path, str],
) -> None:
    output_root, run_id = completed_run
    metrics_path = output_root / "runs" / run_id / "research_backtest" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["net_total_return"] = 999.0
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResultServiceError, match="Research Backtest Artifact validation failed"):
        ResultService(output_root).load(run_id)


class _UIBlock:
    def __init__(self, ui: "_FakeUI") -> None:
        self.ui = ui

    def __enter__(self) -> "_UIBlock":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def metric(self, label: str, value: object) -> None:
        self.ui.metrics.append((label, value))

    def __getattr__(self, name: str):
        return getattr(self.ui, name)


class _FakeUI:
    def __init__(self, *, state: dict[str, object] | None = None, pressed: set[str] | None = None) -> None:
        self.session_state = {} if state is None else state
        self.pressed = set() if pressed is None else pressed
        self.metrics: list[tuple[str, object]] = []
        self.frames: list[pd.DataFrame] = []
        self.charts: list[pd.DataFrame] = []
        self.infos: list[str] = []
        self.tab_labels: tuple[str, ...] = ()
        self.rerun_called = False

    def title(self, value: str) -> None:
        self.title_value = value

    def caption(self, value: str) -> None:
        pass

    def subheader(self, value: str) -> None:
        pass

    def markdown(self, value: str) -> None:
        pass

    def info(self, value: str) -> None:
        self.infos.append(value)

    def warning(self, value: str) -> None:
        pass

    def error(self, value: str) -> None:
        pass

    def json(self, value: object) -> None:
        pass

    def columns(self, count: int) -> list[_UIBlock]:
        return [_UIBlock(self) for _ in range(count)]

    def tabs(self, labels: tuple[str, ...]) -> list[_UIBlock]:
        self.tab_labels = labels
        return [_UIBlock(self) for _ in labels]

    def expander(self, label: str) -> _UIBlock:
        return _UIBlock(self)

    def metric(self, label: str, value: object) -> None:
        self.metrics.append((label, value))

    def dataframe(self, value: pd.DataFrame, **kwargs: object) -> None:
        self.frames.append(value.copy(deep=True))

    def line_chart(self, value: pd.DataFrame) -> None:
        self.charts.append(value.copy(deep=True))

    def bar_chart(self, value: pd.DataFrame) -> None:
        self.charts.append(value.copy(deep=True))

    def selectbox(self, label: str, options: object, **kwargs: object):
        return tuple(options)[0]

    def button(self, label: str, **kwargs: object) -> bool:
        return label in self.pressed

    def rerun(self) -> None:
        self.rerun_called = True


def test_results_ui_requires_exact_selected_run() -> None:
    ui = _FakeUI(state={"locale": "en"})
    results_page.render(ui)
    assert ui.infos == ["Select or complete an exact run_id before opening Results."]


def test_results_ui_renders_all_tabs_and_preserves_zero_weight_row(
    completed_run: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root, run_id = completed_run
    bundle = ResultService(output_root).load(run_id)
    bundle.holdings.loc[bundle.holdings.index[0], "target_weight"] = 0.0

    class _Service:
        def __init__(self, output: object) -> None:
            pass

        def load(self, selected: str):
            assert selected == run_id
            return bundle

    monkeypatch.setattr(results_page, "ResultService", _Service)
    ui = _FakeUI(state={"selected_run_id": run_id, "locale": "en"})
    results_page.render(ui)

    assert ui.tab_labels == ("Overview", "Holdings", "Returns", "Config", "Artifacts")
    assert {label for label, _ in ui.metrics} >= {
        "Net Total Return",
        "Net Max Drawdown",
        "Total Transaction Cost",
    }
    holdings_view = next(frame for frame in ui.frames if "Target Weight" in frame.columns)
    assert bool(holdings_view["Target Weight"].eq(0.0).any())
    assert len(ui.charts) == 3


def test_runs_ui_opens_the_exact_selected_run(monkeypatch: pytest.MonkeyPatch) -> None:
    exact = "20240101_000001_alpha_hs300"
    summary = RunSummary(
        exact,
        "2024-01-01T00:00:00",
        "success",
        "ridge",
        10,
        "equal_weight",
        "000300.SH",
        "available",
        0.1,
        1.2,
    )

    class _Catalog:
        def __init__(self, output: object) -> None:
            pass

        def list_runs(self) -> tuple[RunSummary, ...]:
            return (summary,)

    monkeypatch.setattr(runs_page, "RunCatalogService", _Catalog)
    ui = _FakeUI(state={"locale": "en"}, pressed={"Open Results"})
    runs_page.render(ui)
    assert ui.session_state["selected_run_id"] == exact
    assert ui.session_state["current_page"] == "Results"
    assert ui.rerun_called is False


def test_data_ui_is_substantive_and_has_no_update_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = DataLayer2StatusView("C:/data", "C:/data/curated", "C:/data/metadata/catalog.sqlite", True, (CanonicalDatasetStatus("daily", "1.0", 22),))

    class _StatusService:
        def get_status(self) -> DataLayer2StatusView:
            return status

    monkeypatch.setattr(data_page, "DataLayer2StatusService", _StatusService)
    ui = _FakeUI(state={"locale": "en"})
    data_page.render(ui)
    assert ui.title_value == "Data"
    assert ("Coverage Ledger", "READY") in ui.metrics
    assert len(ui.frames) == 1
    assert ui.pressed == set()


def test_streamlit_all_five_workbench_routes_have_no_import_crash() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    for route in ("overview", "new_run", "results", "runs", "data"):
        page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "views" / f"{route}.py"), default_timeout=30).run()
        assert not page.exception
