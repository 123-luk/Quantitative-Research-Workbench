from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.pipeline.runner as runner_module
from app.services.research_backtest_ui_service import (
    load_research_backtest_dashboard,
)
from src.holdings import (
    HoldingsArtifactConfig,
    HoldingsArtifactStore,
    HoldingsBuilder,
    SignalArtifactProvenance,
)
from src.pipeline import (
    BacktestSourceConfig,
    BenchmarkConfig,
    HoldingsPipelineResult,
    PerformanceConfig,
    PipelineConfig,
    PredictionSourceConfig,
    ResearchBacktestPipelineConfig,
    ResearchBacktestPipelineExecutionError,
    ResearchBacktestPipelineExecutor,
    SignalPipelineConfig,
    TransactionCostConfig,
    HoldingsPipelineConfig,
    run_pipeline,
)
from src.portfolio_construction import (
    PortfolioConstructionConfig,
    PortfolioConstructionEngine,
    PortfolioConstructionServices,
)
from src.portfolio_construction.adapters import (
    ResearchBacktestHistoricalReturnService,
)
from src.research_backtest import ResearchBacktestArtifactStore
from src.risk_model import HistoricalCovarianceRiskModelService


def _holdings(
    tmp_path: Path,
    *,
    name: str = "holdings",
    dates: tuple[str, ...] = ("2024-01-02", "2024-01-04"),
    codes: tuple[str, ...] = ("A.SZ", "B.SZ"),
):
    rows = []
    for trade_date in dates:
        for rank, code in enumerate(codes, start=1):
            rows.append(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "ts_code": code,
                    "score": float(len(codes) - rank + 1),
                    "rank": rank,
                }
            )
    signals = pd.DataFrame(rows)
    signals["trade_date"] = signals["trade_date"].astype("datetime64[ns]")
    signals["ts_code"] = signals["ts_code"].astype("string")
    signals["rank"] = signals["rank"].astype(np.int64)
    built = HoldingsBuilder().build(
        signals,
        top_n=len(codes),
        insufficient_universe_policy="error",
        weighting="equal_weight",
    )
    signal_dir = tmp_path / f"{name}-signal"
    signal_dir.mkdir()
    signal_path = signal_dir / "signals.parquet"
    signal_path.write_bytes(b"source")
    provenance = SignalArtifactProvenance(
        signal_dir,
        signal_path,
        "1.0",
        hashlib.sha256(signal_path.read_bytes()).hexdigest(),
    )
    written = HoldingsArtifactStore().write(
        built,
        provenance,
        HoldingsArtifactConfig(tmp_path / name),
    )
    pipeline_result = HoldingsPipelineResult(
        enabled=True,
        source_signal_artifact_dir=signal_dir,
        artifact_dir=written.artifact_dir,
        holdings_path=written.holdings_path,
        manifest_path=written.manifest_path,
        rows=built.audit.output_rows,
        trade_date_count=built.audit.trade_date_count,
        requested_top_n=len(codes),
        insufficient_universe_policy="error",
        weighting="equal_weight",
        schema_version=written.schema_version,
    )
    return written, pipeline_result


def _rotating_holdings(tmp_path: Path):
    signals = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2024-01-02"),
                "ts_code": "A.SZ",
                "score": 2.0,
                "rank": 1,
            },
            {
                "trade_date": pd.Timestamp("2024-01-04"),
                "ts_code": "B.SZ",
                "score": 2.0,
                "rank": 1,
            },
        ]
    )
    signals["trade_date"] = signals["trade_date"].astype("datetime64[ns]")
    signals["ts_code"] = signals["ts_code"].astype("string")
    signals["rank"] = signals["rank"].astype(np.int64)
    built = HoldingsBuilder().build(
        signals,
        top_n=1,
        insufficient_universe_policy="error",
        weighting="equal_weight",
    )
    signal_dir = tmp_path / "rotating-signal"
    signal_dir.mkdir()
    signal_path = signal_dir / "signals.parquet"
    signal_path.write_bytes(b"source")
    provenance = SignalArtifactProvenance(
        signal_dir,
        signal_path,
        "1.0",
        hashlib.sha256(signal_path.read_bytes()).hexdigest(),
    )
    written = HoldingsArtifactStore().write(
        built,
        provenance,
        HoldingsArtifactConfig(tmp_path / "rotating-holdings"),
    )
    return written


class _Provider:
    def __init__(
        self,
        codes: tuple[str, ...] = ("A.SZ", "B.SZ"),
        *,
        missing_security: tuple[str, str] | None = None,
        suspended: tuple[str, str] | None = None,
        missing_benchmark_date: str | None = None,
        security_pct_chg: dict[tuple[str, str], float] | None = None,
        benchmark_pct_chg: dict[str, float] | None = None,
    ) -> None:
        self.codes = codes
        self.missing_security = missing_security
        self.suspended = suspended
        self.missing_benchmark_date = missing_benchmark_date
        self.security_pct_chg = security_pct_chg or {}
        self.benchmark_pct_chg = benchmark_pct_chg or {}
        self.daily_codes: list[str] = []
        self.benchmark_codes: list[str] = []

    @staticmethod
    def _dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
        return pd.date_range(
            pd.to_datetime(start_date, format="%Y%m%d"),
            pd.to_datetime(end_date, format="%Y%m%d"),
            freq="B",
        )

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.date_range(
            pd.to_datetime(start_date, format="%Y%m%d"),
            pd.to_datetime(end_date, format="%Y%m%d"),
            freq="D",
        )
        return pd.DataFrame(
            {
                "cal_date": dates.strftime("%Y%m%d"),
                "is_open": [int(item.weekday() < 5) for item in dates],
            }
        )

    def get_daily(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        assert ts_code is not None and start_date is not None and end_date is not None
        self.daily_codes.append(ts_code)
        rows = []
        for item in self._dates(start_date, end_date):
            identity = (item.strftime("%Y-%m-%d"), ts_code)
            if identity == self.missing_security or identity == self.suspended:
                continue
            rows.append(
                {
                    "trade_date": item.strftime("%Y%m%d"),
                    "ts_code": ts_code,
                    "pct_chg": self.security_pct_chg.get(identity, 0.0),
                }
            )
        return pd.DataFrame(rows, columns=["trade_date", "ts_code", "pct_chg"])

    def get_index_daily(
        self,
        ts_code: str,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        assert start_date is not None and end_date is not None
        self.benchmark_codes.append(ts_code)
        rows = [
            {
                "trade_date": item.strftime("%Y%m%d"),
                "ts_code": ts_code,
                "pct_chg": self.benchmark_pct_chg.get(
                    item.strftime("%Y-%m-%d"), 0.0
                ),
            }
            for item in self._dates(start_date, end_date)
            if item.strftime("%Y-%m-%d") != self.missing_benchmark_date
        ]
        return pd.DataFrame(rows)

    def get_stock_basic(self, list_status: str = "L") -> pd.DataFrame:
        if list_status != "L":
            return pd.DataFrame(
                columns=["ts_code", "list_status", "list_date", "delist_date"]
            )
        return pd.DataFrame(
            {
                "ts_code": list(self.codes),
                "list_status": ["L"] * len(self.codes),
                "list_date": ["20200101"] * len(self.codes),
                "delist_date": [None] * len(self.codes),
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
        columns = ["ts_code", "trade_date", "suspend_timing", "suspend_type"]
        if self.suspended is None or ts_code != self.suspended[1]:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(
            [
                {
                    "ts_code": ts_code,
                    "trade_date": self.suspended[0].replace("-", ""),
                    "suspend_timing": None,
                    "suspend_type": "S",
                }
            ],
            columns=columns,
        )


def _config(
    source: BacktestSourceConfig, *, cost_bps: float = 10.0
) -> ResearchBacktestPipelineConfig:
    return ResearchBacktestPipelineConfig(
        enabled=True,
        source=source,
        transaction_cost=TransactionCostConfig(cost_bps=cost_bps),
        benchmark=BenchmarkConfig(benchmark_code="TEST.IDX"),
        performance=PerformanceConfig(annual_risk_free_rate=0.0),
    )


def _assert_json_safe_without_frames(value: object) -> None:
    assert not isinstance(value, pd.DataFrame)
    if isinstance(value, dict):
        for item in value.values():
            _assert_json_safe_without_frames(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_json_safe_without_frames(item)


def test_pipeline_mode_end_to_end(tmp_path: Path) -> None:
    holdings, pipeline_result = _holdings(tmp_path)
    provider = _Provider()
    result = ResearchBacktestPipelineExecutor(
        _config(BacktestSourceConfig(mode="pipeline")), provider
    ).execute(
        artifact_dir=tmp_path / "backtest",
        end_date="2024-01-08",
        holdings_result=pipeline_result,
    )
    assert result.enabled
    assert result.start_date == "2024-01-03" and result.end_date == "2024-01-08"
    assert result.rebalance_count == 2 and result.observation_count == 4
    assert provider.daily_codes == ["A.SZ", "B.SZ"]
    assert provider.benchmark_codes == ["TEST.IDX"]
    assert ResearchBacktestArtifactStore().validate(result.artifact_dir).is_valid
    audit = json.loads(
        (result.artifact_dir / "audit.json").read_text(encoding="utf-8")
    )
    assert Path(audit["upstream_holdings"]["holdings_artifact_dir"]) == (
        holdings.artifact_dir
    )


def test_files_mode_end_to_end(tmp_path: Path) -> None:
    holdings, _ = _holdings(tmp_path)
    result = ResearchBacktestPipelineExecutor(
        _config(
            BacktestSourceConfig(mode="files", artifact_dir=holdings.artifact_dir)
        ),
        _Provider(),
    ).execute(
        artifact_dir=tmp_path / "backtest",
        end_date="2024-01-08",
    )
    assert ResearchBacktestArtifactStore().validate(result.artifact_dir).is_valid


def test_all_portfolio_methods_feed_exact_weights_to_same_v6_engine(
    tmp_path: Path,
) -> None:
    codes = tuple(f"S{index:02d}.SZ" for index in range(1, 11))
    rows = [
        {
            "trade_date": pd.Timestamp(trade_date),
            "ts_code": code,
            "score": float(11 - rank),
            "rank": rank,
        }
        for trade_date in ("2024-01-02", "2024-01-04")
        for rank, code in enumerate(codes, start=1)
    ]
    signals = pd.DataFrame(rows)
    signals["trade_date"] = signals["trade_date"].astype("datetime64[ns]")
    signals["ts_code"] = signals["ts_code"].astype("string")
    signals["rank"] = signals["rank"].astype(np.int64)
    risk_returns = {
        (trade_date, code): value
        for index, code in enumerate(codes, start=1)
        for trade_date, value in (
            ("2023-12-29", float(-index)),
            ("2024-01-01", 0.0),
            ("2024-01-02", float(index)),
            ("2024-01-03", 0.0),
            ("2024-01-04", float(-index)),
        )
    }
    portfolio_configs = {
        "equal": PortfolioConstructionConfig("equal_weight", {}),
        "rank": PortfolioConstructionConfig("rank_weight", {}),
        "inverse": PortfolioConstructionConfig(
            "inverse_volatility",
            {"lookback_trading_days": 3, "min_observations": 3},
        ),
        "minimum_variance": PortfolioConstructionConfig(
            "minimum_variance",
            {"risk_model": {
                "estimator": "ledoit_wolf",
                "params": {},
                "lookback_trading_days": 3,
                "min_observations": 3,
            }},
        ),
    }
    selected_sets: dict[str, set[str]] = {}
    first_date_weights: dict[str, np.ndarray] = {}

    for name, portfolio_config in portfolio_configs.items():
        provider = _Provider(codes=codes, security_pct_chg=risk_returns)
        service = ResearchBacktestHistoricalReturnService(provider)
        engine = PortfolioConstructionEngine(
            services=PortfolioConstructionServices(
                historical_returns=service,
                risk_model=HistoricalCovarianceRiskModelService(service),
            )
        )
        built = HoldingsBuilder(engine).build(
            signals,
            top_n=5,
            insufficient_universe_policy="error",
            weighting="equal_weight",
            portfolio_construction=portfolio_config,
        )
        holdings = built.holdings
        assert tuple(holdings.columns) == (
            "trade_date",
            "ts_code",
            "target_weight",
            "score",
            "rank",
        )
        first = holdings.loc[
            holdings["trade_date"] == pd.Timestamp("2024-01-02")
        ]
        selected_sets[name] = set(first["ts_code"])
        first_date_weights[name] = first["target_weight"].to_numpy()

        signal_dir = tmp_path / f"{name}-signal"
        signal_dir.mkdir()
        signal_path = signal_dir / "signals.parquet"
        signal_path.write_bytes(b"synthetic-signal")
        provenance = SignalArtifactProvenance(
            signal_dir,
            signal_path,
            "1.0",
            hashlib.sha256(signal_path.read_bytes()).hexdigest(),
        )
        written = HoldingsArtifactStore().write(
            built,
            provenance,
            HoldingsArtifactConfig(tmp_path / f"{name}-holdings"),
            portfolio_construction=portfolio_config,
        )
        assert HoldingsArtifactStore().validate(written.artifact_dir).is_valid
        stored_config = json.loads(written.config_path.read_text(encoding="utf-8"))
        assert stored_config["portfolio_construction"] == portfolio_config.to_dict()
        assert {path.name for path in written.artifact_dir.iterdir()} == {
            "holdings.parquet", "config.json", "audit.json", "manifest.json"
        }
        pipeline_result = HoldingsPipelineResult(
            enabled=True,
            source_signal_artifact_dir=signal_dir,
            artifact_dir=written.artifact_dir,
            holdings_path=written.holdings_path,
            manifest_path=written.manifest_path,
            rows=built.audit.output_rows,
            trade_date_count=built.audit.trade_date_count,
            requested_top_n=5,
            insufficient_universe_policy="error",
            weighting="equal_weight",
            schema_version=written.schema_version,
        )
        backtest = ResearchBacktestPipelineExecutor(
            _config(BacktestSourceConfig(mode="pipeline")), provider
        ).execute(
            artifact_dir=tmp_path / f"{name}-backtest",
            end_date="2024-01-08",
            holdings_result=pipeline_result,
        )
        assert ResearchBacktestArtifactStore().validate(
            backtest.artifact_dir
        ).is_valid
        rebalances = pd.read_parquet(
            backtest.artifact_dir / "rebalances.parquet"
        )
        actual = rebalances.loc[
            rebalances["holdings_trade_date"] == pd.Timestamp("2024-01-02"),
            ["ts_code", "target_weight"],
        ].sort_values("ts_code", ignore_index=True)
        expected = first.loc[:, ["ts_code", "target_weight"]].sort_values(
            "ts_code", ignore_index=True
        )
        assert actual["ts_code"].astype(str).tolist() == (
            expected["ts_code"].astype(str).tolist()
        )
        np.testing.assert_array_equal(
            actual["target_weight"].to_numpy(),
            expected["target_weight"].to_numpy(),
        )

    assert selected_sets["equal"] == selected_sets["rank"]
    assert selected_sets["equal"] == selected_sets["inverse"]
    assert selected_sets["equal"] == selected_sets["minimum_variance"]
    assert not np.allclose(first_date_weights["equal"], first_date_weights["rank"])
    assert not np.allclose(
        first_date_weights["equal"], first_date_weights["inverse"]
    )
    assert not np.allclose(
        first_date_weights["rank"], first_date_weights["minimum_variance"]
    )

    top_ten = HoldingsBuilder().build(
        signals,
        top_n=10,
        insufficient_universe_policy="error",
        weighting="equal_weight",
    ).holdings
    assert top_ten.groupby("trade_date").size().tolist() == [10, 10]


def test_result_is_json_safe_detached_and_has_no_frames(tmp_path: Path) -> None:
    holdings, _ = _holdings(tmp_path)
    result = ResearchBacktestPipelineExecutor(
        _config(
            BacktestSourceConfig(mode="files", artifact_dir=holdings.artifact_dir)
        ),
        _Provider(),
    ).execute(artifact_dir=tmp_path / "backtest", end_date="2024-01-08")
    snapshot = result.to_dict()
    _assert_json_safe_without_frames(snapshot)
    json.dumps(snapshot, allow_nan=False)
    snapshot["metrics"]["observation_count"] = 999  # type: ignore[index]
    assert result.to_dict()["metrics"]["observation_count"] == 4  # type: ignore[index]


def test_disabled_executor_has_no_provider_calls(tmp_path: Path) -> None:
    provider = _Provider()
    result = ResearchBacktestPipelineExecutor(
        ResearchBacktestPipelineConfig(), provider
    ).execute(artifact_dir=tmp_path / "unused", end_date="bad")
    assert result.to_dict()["enabled"] is False
    assert provider.daily_codes == []


def test_holdings_effective_after_end_date_rejected(tmp_path: Path) -> None:
    holdings, _ = _holdings(tmp_path, dates=("2024-01-08",))
    with pytest.raises(ResearchBacktestPipelineExecutionError, match="precede"):
        ResearchBacktestPipelineExecutor(
            _config(
                BacktestSourceConfig(
                    mode="files", artifact_dir=holdings.artifact_dir
                )
            ),
            _Provider(),
        ).execute(artifact_dir=tmp_path / "backtest", end_date="2024-01-08")


def test_benchmark_missing_date_fails_closed(tmp_path: Path) -> None:
    holdings, _ = _holdings(tmp_path)
    with pytest.raises(ResearchBacktestPipelineExecutionError):
        ResearchBacktestPipelineExecutor(
            _config(
                BacktestSourceConfig(
                    mode="files", artifact_dir=holdings.artifact_dir
                )
            ),
            _Provider(missing_benchmark_date="2024-01-04"),
        ).execute(artifact_dir=tmp_path / "backtest", end_date="2024-01-08")


def test_proven_full_day_suspension_resolves_missing_return(tmp_path: Path) -> None:
    holdings, _ = _holdings(tmp_path)
    identity = ("2024-01-04", "A.SZ")
    result = ResearchBacktestPipelineExecutor(
        _config(
            BacktestSourceConfig(mode="files", artifact_dir=holdings.artifact_dir)
        ),
        _Provider(suspended=identity),
    ).execute(artifact_dir=tmp_path / "backtest", end_date="2024-01-08")
    assert result.enabled


def test_unexplained_missing_return_fails_closed(tmp_path: Path) -> None:
    holdings, _ = _holdings(tmp_path)
    with pytest.raises(ResearchBacktestPipelineExecutionError):
        ResearchBacktestPipelineExecutor(
            _config(
                BacktestSourceConfig(
                    mode="files", artifact_dir=holdings.artifact_dir
                )
            ),
            _Provider(missing_security=("2024-01-04", "A.SZ")),
        ).execute(artifact_dir=tmp_path / "backtest", end_date="2024-01-08")


@pytest.mark.parametrize(
    ("list_date", "delist_date", "missing"),
    [
        ("20240105", None, None),
        ("20200101", "20240103", ("2024-01-04", "A.SZ")),
    ],
    ids=("pre-listing", "post-delist"),
)
def test_lifecycle_inconsistency_fails_closed_end_to_end(
    tmp_path: Path,
    list_date: str,
    delist_date: str | None,
    missing: tuple[str, str] | None,
) -> None:
    holdings, _ = _holdings(tmp_path, codes=("A.SZ",))

    class LifecycleProvider(_Provider):
        def get_stock_basic(self, list_status: str = "L") -> pd.DataFrame:
            expected = "D" if delist_date is not None else "L"
            if list_status != expected:
                return pd.DataFrame(
                    columns=["ts_code", "list_status", "list_date", "delist_date"]
                )
            return pd.DataFrame(
                {
                    "ts_code": ["A.SZ"],
                    "list_status": [expected],
                    "list_date": [list_date],
                    "delist_date": [delist_date],
                }
            )

    provider = LifecycleProvider(codes=("A.SZ",), missing_security=missing)
    with pytest.raises(ResearchBacktestPipelineExecutionError):
        ResearchBacktestPipelineExecutor(
            _config(
                BacktestSourceConfig(
                    mode="files", artifact_dir=holdings.artifact_dir
                )
            ),
            provider,
        ).execute(artifact_dir=tmp_path / "backtest", end_date="2024-01-08")


def test_timing_cost_benchmark_and_determinism_end_to_end(tmp_path: Path) -> None:
    holdings = _rotating_holdings(tmp_path)
    security_returns = {
        ("2024-01-03", "A.SZ"): 50.0,
        ("2024-01-04", "A.SZ"): 10.0,
        ("2024-01-05", "A.SZ"): 10.0,
        ("2024-01-05", "B.SZ"): 100.0,
        ("2024-01-08", "B.SZ"): 20.0,
    }
    benchmark_returns = {"2024-01-03": 75.0}

    def execute(name: str, cost_bps: float):
        provider = _Provider(
            security_pct_chg=security_returns,
            benchmark_pct_chg=benchmark_returns,
        )
        return ResearchBacktestPipelineExecutor(
            _config(
                BacktestSourceConfig(
                    mode="files", artifact_dir=holdings.artifact_dir
                ),
                cost_bps=cost_bps,
            ),
            provider,
        ).execute(artifact_dir=tmp_path / name, end_date="2024-01-08")

    first = execute("backtest-first", 10.0)
    repeated = execute("backtest-repeated", 10.0)
    zero_cost = execute("backtest-zero", 0.0)
    daily = pd.read_parquet(first.artifact_dir / "daily_portfolio.parquet").set_index(
        "trade_date"
    )
    benchmark = pd.read_parquet(first.artifact_dir / "benchmark.parquet").set_index(
        "trade_date"
    )
    assert daily.loc[pd.Timestamp("2024-01-03"), "gross_return"] == 0.0
    assert daily.loc[pd.Timestamp("2024-01-04"), "gross_return"] == pytest.approx(
        0.10
    )
    later = daily.loc[pd.Timestamp("2024-01-05")]
    assert later["gross_return"] == pytest.approx(0.10)
    assert later["turnover"] == pytest.approx(1.0)
    assert later["traded_notional"] == pytest.approx(2.0)
    assert later["transaction_cost"] == pytest.approx(0.002)
    assert later["net_return"] == pytest.approx((1.1 * 0.998) - 1.0)
    assert daily.loc[pd.Timestamp("2024-01-08"), "gross_return"] == pytest.approx(
        0.20
    )
    initial = daily.loc[pd.Timestamp("2024-01-03")]
    assert initial["traded_notional"] == pytest.approx(1.0)
    assert initial["transaction_cost"] == pytest.approx(0.001)
    assert benchmark.loc[pd.Timestamp("2024-01-03"), "benchmark_return"] == 0.0

    zero_daily = pd.read_parquet(
        zero_cost.artifact_dir / "daily_portfolio.parquet"
    )
    assert np.allclose(zero_daily["gross_nav"], zero_daily["net_nav"])
    for filename in (
        "rebalances.parquet",
        "daily_portfolio.parquet",
        "benchmark.parquet",
    ):
        left = pd.read_parquet(first.artifact_dir / filename)
        right = pd.read_parquet(repeated.artifact_dir / filename)
        pd.testing.assert_frame_equal(left, right)
    assert first.metrics == repeated.metrics


def test_no_overwrite_propagates_as_execution_failure(tmp_path: Path) -> None:
    holdings, _ = _holdings(tmp_path)
    executor = ResearchBacktestPipelineExecutor(
        _config(
            BacktestSourceConfig(mode="files", artifact_dir=holdings.artifact_dir)
        ),
        _Provider(),
    )
    target = tmp_path / "backtest"
    executor.execute(artifact_dir=target, end_date="2024-01-08")
    with pytest.raises(ResearchBacktestPipelineExecutionError):
        executor.execute(artifact_dir=target, end_date="2024-01-08")


def _pipeline_config(
    tmp_path: Path,
    research: ResearchBacktestPipelineConfig,
    *,
    upstream: bool,
) -> PipelineConfig:
    return PipelineConfig(
        backtest_start="2024-01-01",
        backtest_end="2024-01-08",
        train_years=1,
        max_lookback_months=1,
        stock_pool="test",
        benchmark="LEGACY.IDX",
        strategy_name="e2e",
        selected_factors=[],
        rebalance_frequency="M",
        top_n=2,
        transaction_cost=0.001,
        data_root="data",
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        cache_dir="data/cache",
        output_dir=str(tmp_path / "output"),
        parquet_engine="auto",
        required_datasets=[],
        signal=SignalPipelineConfig(
            enabled=upstream,
            source=PredictionSourceConfig("files", tmp_path / "ml"),
        ),
        holdings=HoldingsPipelineConfig(enabled=upstream, top_n=2),
        research_backtest=research,
    )


class _ReadyDataManager:
    def prepare_data(self, config: object) -> dict[str, object]:
        return {"cache_status": "ready", "missing_ranges": {}}


def test_runner_pipeline_source_uses_real_executor_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holdings, pipeline_result = _holdings(tmp_path)
    provider = _Provider()

    class SignalResult:
        def as_dict(self) -> dict[str, object]:
            return {"enabled": True}

    class SignalExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(self, run_dir: object, **kwargs: object) -> SignalResult:
            return SignalResult()

    class HoldingsExecutor:
        def __init__(self, config: object, engine: object = None) -> None:
            pass

        def execute(
            self, run_dir: object, **kwargs: object
        ) -> HoldingsPipelineResult:
            return pipeline_result

    monkeypatch.setattr(runner_module, "DataManager", _ReadyDataManager)
    monkeypatch.setattr(runner_module, "TushareClient", lambda: provider)
    monkeypatch.setattr(runner_module, "SignalPipelineExecutor", SignalExecutor)
    monkeypatch.setattr(runner_module, "HoldingsPipelineExecutor", HoldingsExecutor)
    summary = run_pipeline(
        _pipeline_config(
            tmp_path,
            _config(BacktestSourceConfig(mode="pipeline")),
            upstream=True,
        )
    )
    artifact_dir = Path(summary["research_backtest"]["artifact_dir"])  # type: ignore[index]
    assert ResearchBacktestArtifactStore().validate(artifact_dir).is_valid
    audit = json.loads((artifact_dir / "audit.json").read_text(encoding="utf-8"))
    assert Path(audit["upstream_holdings"]["holdings_artifact_dir"]) == (
        holdings.artifact_dir
    )
    payload = load_research_backtest_dashboard(
        summary["research_backtest"]  # type: ignore[arg-type]
    )
    assert payload.artifact_dir == artifact_dir
    assert payload.metrics == summary["research_backtest"]["metrics"]  # type: ignore[index]
    assert tuple(payload.nav.columns) == (
        "trade_date",
        "gross_nav",
        "net_nav",
        "benchmark_nav",
    )


def test_runner_files_source_uses_real_executor_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holdings, _ = _holdings(tmp_path, name="explicit")
    current, current_result = _holdings(
        tmp_path, name="current", codes=("X.SZ", "Y.SZ")
    )

    class SignalResult:
        def as_dict(self) -> dict[str, object]:
            return {"enabled": True}

    class SignalExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(self, run_dir: object, **kwargs: object) -> SignalResult:
            return SignalResult()

    class HoldingsExecutor:
        def __init__(self, config: object, engine: object = None) -> None:
            pass

        def execute(
            self, run_dir: object, **kwargs: object
        ) -> HoldingsPipelineResult:
            return current_result

    monkeypatch.setattr(runner_module, "DataManager", _ReadyDataManager)
    monkeypatch.setattr(runner_module, "TushareClient", _Provider)
    monkeypatch.setattr(runner_module, "SignalPipelineExecutor", SignalExecutor)
    monkeypatch.setattr(runner_module, "HoldingsPipelineExecutor", HoldingsExecutor)
    summary = run_pipeline(
        _pipeline_config(
            tmp_path,
            _config(
                BacktestSourceConfig(
                    mode="files", artifact_dir=holdings.artifact_dir
                )
            ),
            upstream=True,
        )
    )
    artifact_dir = Path(summary["research_backtest"]["artifact_dir"])  # type: ignore[index]
    assert ResearchBacktestArtifactStore().validate(artifact_dir).is_valid
    audit = json.loads((artifact_dir / "audit.json").read_text(encoding="utf-8"))
    lineage = Path(audit["upstream_holdings"]["holdings_artifact_dir"])
    assert lineage == holdings.artifact_dir
    assert lineage != current.artifact_dir


@pytest.mark.parametrize("top_n", [5, 10])
def test_executor_preserves_holdings_top_n_without_reselection(
    tmp_path: Path, top_n: int
) -> None:
    codes = tuple(f"S{number:03d}" for number in range(top_n))
    holdings, _ = _holdings(
        tmp_path,
        name=f"holdings-{top_n}",
        codes=codes,
    )
    result = ResearchBacktestPipelineExecutor(
        _config(
            BacktestSourceConfig(mode="files", artifact_dir=holdings.artifact_dir)
        ),
        _Provider(codes),
    ).execute(
        artifact_dir=tmp_path / f"backtest-{top_n}",
        end_date="2024-01-08",
    )
    rebalances = pd.read_parquet(result.artifact_dir / "rebalances.parquet")
    first = rebalances.loc[
        rebalances["holdings_trade_date"].eq(pd.Timestamp("2024-01-02"))
        & rebalances["target_weight"].gt(0)
    ]
    assert tuple(first["ts_code"]) == codes
    assert np.allclose(first["target_weight"], 1.0 / top_n)


@pytest.mark.parametrize(
    "dates",
    [
        ("2024-01-02", "2024-02-02"),
        ("2024-01-02", "2024-01-09"),
        ("2024-01-02", "2024-01-03"),
    ],
    ids=("monthly-like", "weekly-like", "daily-like"),
)
def test_executor_is_frequency_agnostic(
    tmp_path: Path, dates: tuple[str, str]
) -> None:
    holdings, _ = _holdings(tmp_path, dates=dates)
    result = ResearchBacktestPipelineExecutor(
        _config(
            BacktestSourceConfig(mode="files", artifact_dir=holdings.artifact_dir)
        ),
        _Provider(),
    ).execute(
        artifact_dir=tmp_path / "backtest",
        end_date="2024-02-05",
    )
    assert result.rebalance_count == len(dates)
