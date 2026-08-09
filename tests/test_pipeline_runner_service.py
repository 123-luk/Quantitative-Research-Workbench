"""Tests for the Streamlit-to-canonical-runner service boundary."""

from __future__ import annotations

import json
import inspect
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

import app.services.pipeline_runner_service as service
import src.pipeline.runner as runner_module
from app.services.pipeline_config_service import build_effective_pipeline_config
from src.holdings import HoldingsArtifactStore
from src.ml import (
    MLArtifactConfig,
    MLExperimentArtifactStore,
    MLExperimentConfig,
    MLExperimentRunner,
    MLDatasetConfig,
    ModelEvaluationConfig,
    WalkForwardConfig,
    WalkForwardTrainingConfig,
)
from src.pipeline import (
    HoldingsPipelineConfig,
    PipelineConfig,
    PredictionSourceConfig,
    SignalPipelineConfig,
)
from src.signals import SignalArtifactStore


def _experiment() -> MLExperimentConfig:
    return MLExperimentConfig(
        dataset_config=MLDatasetConfig(),
        walk_forward_config=WalkForwardConfig(
            train_window_periods=2,
            validation_periods=2,
            window_type="rolling",
            retrain_frequency=3,
            embargo_periods=1,
        ),
        training_config=WalkForwardTrainingConfig("ridge"),
        evaluation_config=ModelEvaluationConfig(minimum_cross_section_size=3),
    )


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, trade_date in enumerate(pd.date_range("2024-02-01", periods=12)):
        for stock_number in range(12):
            value = float(date_number + stock_number / 10)
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"U{stock_number:03d}",
                    "factor_a": value,
                    "factor_b": float(stock_number - date_number / 10),
                    "entry_trade_date": trade_date + pd.Timedelta(days=1),
                    "exit_trade_date": trade_date + pd.Timedelta(days=2),
                    "forward_return": value / 100,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def native_ml_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ui-native-ml")
    result = MLExperimentRunner().run(_panel(), _experiment())
    return MLExperimentArtifactStore().write(
        result, MLArtifactConfig(root, "ui-source")
    ).experiment_dir


def _base(output_dir: Path, native: Path) -> PipelineConfig:
    return PipelineConfig(
        backtest_start="2024-01-01",
        backtest_end="2024-12-31",
        train_years=1,
        max_lookback_months=1,
        stock_pool="hs300",
        benchmark="000300.SH",
        strategy_name="ui_service",
        selected_factors=[],
        rebalance_frequency="M",
        top_n=99,
        transaction_cost=0.001,
        data_root="data",
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        cache_dir="data/cache",
        output_dir=str(output_dir),
        parquet_engine="auto",
        required_datasets=[],
        signal=SignalPipelineConfig(
            source=PredictionSourceConfig("files", native)
        ),
        holdings=HoldingsPipelineConfig(top_n=20),
    )


def test_service_calls_canonical_runner_with_the_same_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_effective_pipeline_config(
        _base(tmp_path / "output", tmp_path / "native"), top_n=10
    )
    before = config.to_dict()
    expected = {"status": "ready", "run_dir": "run", "holdings": {"rows": 10}}
    seen: list[PipelineConfig] = []

    def fake_run(received: PipelineConfig) -> dict[str, object]:
        seen.append(received)
        assert received.holdings.top_n == 10
        assert received.signal.signal_direction == "descending"
        assert received.holdings.insufficient_universe_policy == "error"
        assert received.holdings.weighting == "equal_weight"
        return expected

    monkeypatch.setattr(service, "run_pipeline", fake_run)
    result = service.run_canonical_pipeline(config)
    assert result is expected
    assert seen == [config]
    assert config.to_dict() == before
    json.dumps(result, allow_nan=False)


def test_service_propagates_execution_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_effective_pipeline_config(
        _base(tmp_path / "output", tmp_path / "native"), top_n=10
    )

    def fail(received: PipelineConfig) -> dict[str, object]:
        raise RuntimeError("explicit canonical failure")

    monkeypatch.setattr(service, "run_pipeline", fail)
    with pytest.raises(RuntimeError, match="explicit canonical failure"):
        service.run_canonical_pipeline(config)
    with pytest.raises(TypeError, match="PipelineConfig"):
        service.run_canonical_pipeline({})  # type: ignore[arg-type]


def test_enabled_research_backtest_reaches_canonical_runner_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_effective_pipeline_config(
        _base(tmp_path / "output", tmp_path / "native"),
        top_n=10,
        research_backtest_enabled=True,
        research_backtest_cost_bps=2.5,
        research_backtest_benchmark="000905.SH",
        annual_risk_free_rate=-0.005,
    )
    seen: list[PipelineConfig] = []

    def fake_run(received: PipelineConfig) -> dict[str, object]:
        seen.append(received)
        return {
            "research_backtest": {
                "enabled": True,
                "artifact_dir": "exact-artifact",
                "metrics": {},
            }
        }

    monkeypatch.setattr(service, "run_pipeline", fake_run)
    result = service.run_canonical_pipeline(config)
    assert seen == [config]
    assert result["research_backtest"]["artifact_dir"] == "exact-artifact"  # type: ignore[index]


def test_canonical_service_has_no_legacy_or_business_logic() -> None:
    canonical = inspect.getsource(service.run_canonical_pipeline)
    assert "run_pipeline(config)" in canonical
    for forbidden in (
        "subprocess",
        "--top-n",
        "run_research_pipeline.py",
        "read_parquet",
        "to_parquet",
        "sort_values",
        "rank(",
        "target_weight",
        ".glob(",
        ".rglob(",
        "latest",
    ):
        assert forbidden not in canonical


def test_real_service_bridge_top_n_five_and_ten(
    tmp_path: Path,
    native_ml_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReadyDataManager:
        def prepare_data(self, config: object) -> dict[str, object]:
            return {"cache_status": "ready", "missing_ranges": {}}

    monkeypatch.setattr(runner_module, "DataManager", ReadyDataManager)
    runs: dict[int, dict[str, object]] = {}
    signal_frames: dict[int, pd.DataFrame] = {}
    holdings_frames: dict[int, pd.DataFrame] = {}

    for top_n in (5, 10):
        base = _base(tmp_path / f"output-{top_n}", native_ml_artifact)
        before = base.to_dict()
        effective = build_effective_pipeline_config(base, top_n=top_n)
        runs[top_n] = service.run_canonical_pipeline(effective)
        assert base.to_dict() == before

        signal_dir = Path(runs[top_n]["signal"]["artifact_dir"])  # type: ignore[index]
        holdings_dir = Path(runs[top_n]["holdings"]["artifact_dir"])  # type: ignore[index]
        assert SignalArtifactStore().validate(signal_dir).is_valid
        assert HoldingsArtifactStore().validate(holdings_dir).is_valid
        signal_frames[top_n] = pd.read_parquet(signal_dir / "signals.parquet")
        holdings_frames[top_n] = pd.read_parquet(
            holdings_dir / "holdings.parquet"
        )

        artifact_config = json.loads(
            (holdings_dir / "config.json").read_text(encoding="utf-8")
        )
        snapshot = yaml.safe_load(
            (Path(runs[top_n]["run_dir"]) / "config_snapshot.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert effective.holdings.top_n == top_n
        assert runs[top_n]["holdings"]["requested_top_n"] == top_n  # type: ignore[index]
        assert artifact_config["top_n"] == top_n
        assert snapshot["holdings"]["top_n"] == top_n
        assert snapshot["top_n"] == top_n
        assert holdings_frames[top_n].groupby("trade_date").size().eq(top_n).all()

    pdt.assert_frame_equal(signal_frames[5], signal_frames[10])
    prefix = holdings_frames[10].loc[
        holdings_frames[10]["rank"] <= 5,
        ["trade_date", "ts_code", "score", "rank"],
    ].reset_index(drop=True)
    pdt.assert_frame_equal(
        holdings_frames[5][["trade_date", "ts_code", "score", "rank"]],
        prefix,
    )
