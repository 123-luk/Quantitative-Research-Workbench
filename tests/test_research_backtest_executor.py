from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.pipeline.runner as runner_module
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
from src.research_backtest import ResearchBacktestArtifactStore


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


class _Provider:
    def __init__(
        self,
        codes: tuple[str, ...] = ("A.SZ", "B.SZ"),
        *,
        missing_security: tuple[str, str] | None = None,
        suspended: tuple[str, str] | None = None,
        missing_benchmark_date: str | None = None,
    ) -> None:
        self.codes = codes
        self.missing_security = missing_security
        self.suspended = suspended
        self.missing_benchmark_date = missing_benchmark_date
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
                    "pct_chg": 0.0,
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
                "pct_chg": 0.0,
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


def _config(source: BacktestSourceConfig) -> ResearchBacktestPipelineConfig:
    return ResearchBacktestPipelineConfig(
        enabled=True,
        source=source,
        transaction_cost=TransactionCostConfig(cost_bps=10.0),
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
        def __init__(self, config: object) -> None:
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


def test_runner_files_source_uses_real_executor_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holdings, _ = _holdings(tmp_path)
    monkeypatch.setattr(runner_module, "DataManager", _ReadyDataManager)
    monkeypatch.setattr(runner_module, "TushareClient", _Provider)
    summary = run_pipeline(
        _pipeline_config(
            tmp_path,
            _config(
                BacktestSourceConfig(
                    mode="files", artifact_dir=holdings.artifact_dir
                )
            ),
            upstream=False,
        )
    )
    artifact_dir = Path(summary["research_backtest"]["artifact_dir"])  # type: ignore[index]
    assert ResearchBacktestArtifactStore().validate(artifact_dir).is_valid


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
