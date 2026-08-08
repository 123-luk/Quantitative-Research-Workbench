"""V5-E2 tests for canonical Runner Signal/Holdings chaining."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

import src.pipeline.runner as runner_module
from src.holdings import HoldingsArtifactStore
from src.factors.research_pipeline import FactorResearchConfig
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
    FactorResearchPipelineConfig,
    HoldingsPipelineConfig,
    HoldingsPipelineExecutionError,
    MLExperimentPipelineConfig,
    ModelingPanelPipelineConfig,
    PipelineConfig,
    PredictionSourceConfig,
    SignalPipelineConfig,
    SignalPipelineExecutionError,
    run_pipeline,
)
from src.pipeline.experiment import ExperimentManager
from src.signals import SignalArtifactStore


class _ReadyDataManager:
    def prepare_data(self, config: object) -> dict[str, object]:
        return {"cache_status": "ready", "missing_ranges": {}}


def _ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "DataManager", _ReadyDataManager)


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


def _base_config(output_dir: Path, **updates: object) -> PipelineConfig:
    values: dict[str, object] = {
        "backtest_start": "2024-01-01",
        "backtest_end": "2025-03-31",
        "train_years": 10,
        "max_lookback_months": 12,
        "stock_pool": "hs300",
        "benchmark": "000300.SH",
        "strategy_name": "score",
        "selected_factors": ["pe"],
        "rebalance_frequency": "M",
        "top_n": 20,
        "transaction_cost": 0.001,
        "data_root": "data",
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "cache_dir": "data/cache",
        "output_dir": str(output_dir),
        "parquet_engine": "auto",
        "required_datasets": ["daily"],
    }
    values.update(updates)
    return PipelineConfig.from_dict(values)


def _native_panel(stock_count: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, trade_date in enumerate(pd.date_range("2024-02-01", periods=12)):
        for stock_number in range(stock_count):
            value = float(date_number + stock_number / 10)
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"S{stock_number:03d}",
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
    root = tmp_path_factory.mktemp("runner-native-ml")
    result = MLExperimentRunner().run(_native_panel(), _experiment())
    return MLExperimentArtifactStore().write(
        result, MLArtifactConfig(root, "runner-source")
    ).experiment_dir


def _v5_files_config(
    output_dir: Path,
    native: Path,
    *,
    top_n: int,
    policy: str = "error",
    holdings: bool = True,
) -> PipelineConfig:
    return _base_config(
        output_dir,
        top_n=top_n,
        signal=SignalPipelineConfig(
            enabled=True,
            source=PredictionSourceConfig("files", native),
            signal_direction="descending",
            artifact_subdir="signal",
        ),
        holdings=HoldingsPipelineConfig(
            enabled=holdings,
            top_n=top_n,
            insufficient_universe_policy=policy,
            artifact_subdir="holdings",
        ),
    )


def test_disabled_v5_stages_preserve_old_summary_and_do_not_construct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)

    class Forbidden:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("disabled V5 executor was constructed")

    monkeypatch.setattr(runner_module, "SignalPipelineExecutor", Forbidden)
    monkeypatch.setattr(runner_module, "HoldingsPipelineExecutor", Forbidden)
    config = _base_config(tmp_path / "output")
    before = config.to_dict()
    summary = run_pipeline(config)
    assert "signal" not in summary and "holdings" not in summary
    assert config.to_dict() == before
    snapshot = yaml.safe_load(
        (Path(summary["run_dir"]) / "config_snapshot.yaml").read_text()
    )
    assert snapshot["signal"]["enabled"] is False
    assert snapshot["holdings"]["enabled"] is False


def test_canonical_order_and_exact_same_run_result_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    events: list[tuple[str, Path]] = []
    identities: list[tuple[str, object]] = []

    @dataclass
    class Result:
        name: str
        enabled: bool = True
        panel_path: Path | None = None

        def to_dict(self) -> dict[str, object]:
            return {"enabled": True, "name": self.name}

        def as_dict(self) -> dict[str, object]:
            return {"enabled": True, "name": self.name}

    factor_result = Result("factor")
    model_result = Result("model")
    ml_result = Result("ml")
    signal_result = Result("signal")
    holdings_result = Result("holdings")

    class FactorExecutor:
        def __init__(self, config: object) -> None: pass
        def execute(self, run_dir: str | Path, **kwargs: object) -> Result:
            events.append(("factor", Path(run_dir)))
            return factor_result

    class ModelExecutor:
        def __init__(self, config: object) -> None: pass
        def execute(self, run_dir: str | Path, *, factor_research_result: object = None) -> Result:
            events.append(("model", Path(run_dir)))
            identities.append(("research", factor_research_result))
            model_result.panel_path = Path(run_dir) / "panel.parquet"
            return model_result

    class MLExecutor:
        def __init__(self, config: object) -> None: pass
        def execute(self, run_dir: str | Path, **kwargs: object) -> Result:
            events.append(("ml", Path(run_dir)))
            return ml_result

    class SignalExecutor:
        def __init__(self, config: object) -> None: pass
        def execute(self, run_dir: str | Path, *, ml_result: object = None) -> Result:
            events.append(("signal", Path(run_dir)))
            identities.append(("ml", ml_result))
            return signal_result

    class HoldingsExecutor:
        def __init__(self, config: object) -> None: pass
        def execute(self, run_dir: str | Path, *, signal_result: object = None) -> Result:
            events.append(("holdings", Path(run_dir)))
            identities.append(("signal", signal_result))
            return holdings_result

    monkeypatch.setattr(runner_module, "FactorResearchPipelineExecutor", FactorExecutor)
    monkeypatch.setattr(runner_module, "ModelingPanelPipelineExecutor", ModelExecutor)
    monkeypatch.setattr(runner_module, "MLExperimentPipelineExecutor", MLExecutor)
    monkeypatch.setattr(runner_module, "SignalPipelineExecutor", SignalExecutor)
    monkeypatch.setattr(runner_module, "HoldingsPipelineExecutor", HoldingsExecutor)
    factor = FactorResearchPipelineConfig(
        enabled=True,
        factor_input_path="factor.parquet",
        score_panel_path="score.parquet",
        price_panel_path="price.parquet",
        research=FactorResearchConfig(
            factor_names=("factor_a",),
            composition_method="equal",
        ),
    )
    modeling = ModelingPanelPipelineConfig.from_dict(
        {"enabled": True, "source": {"mode": "factor_research"}}
    )
    ml = MLExperimentPipelineConfig(
        enabled=True, panel_path=None, experiment=_experiment()
    )
    config = _base_config(
        tmp_path / "output",
        factor_research=factor,
        modeling_panel=modeling,
        ml_experiment=ml,
        signal=SignalPipelineConfig(enabled=True),
        holdings=HoldingsPipelineConfig(enabled=True, top_n=20),
    )
    before = config.to_dict()
    summary = run_pipeline(config)
    run_dir = Path(summary["run_dir"])
    assert [name for name, _ in events] == ["factor", "model", "ml", "signal", "holdings"]
    assert all(path == run_dir for _, path in events)
    assert identities == [("research", factor_result), ("ml", ml_result), ("signal", signal_result)]
    assert config.to_dict() == before


def test_files_mode_runs_after_skipped_ml_and_passes_none(
    tmp_path: Path, native_ml_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    received: list[object] = []
    real_executor = runner_module.SignalPipelineExecutor

    class RecordingSignalExecutor(real_executor):
        def execute(self, run_dir: str | Path, *, ml_result: object = None):
            received.append(ml_result)
            return super().execute(run_dir, ml_result=ml_result)  # type: ignore[arg-type]

    class Forbidden:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("disabled stage was constructed")

    monkeypatch.setattr(runner_module, "MLExperimentPipelineExecutor", Forbidden)
    monkeypatch.setattr(runner_module, "HoldingsPipelineExecutor", Forbidden)
    monkeypatch.setattr(runner_module, "SignalPipelineExecutor", RecordingSignalExecutor)
    summary = run_pipeline(
        _v5_files_config(
            tmp_path / "output", native_ml_artifact, top_n=5, holdings=False
        )
    )
    assert received == [None]
    assert "ml_experiment" not in summary and "holdings" not in summary
    assert SignalArtifactStore().validate(summary["signal"]["artifact_dir"]).is_valid


def test_real_runner_e2e_top_n_trace_and_run_isolation(
    tmp_path: Path, native_ml_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    results: dict[int, dict[str, object]] = {}
    holdings_frames: dict[int, pd.DataFrame] = {}
    for top_n in (5, 10):
        config = _v5_files_config(
            tmp_path / "output", native_ml_artifact, top_n=top_n
        )
        before = config.to_dict()
        summary = run_pipeline(config)
        results[top_n] = summary
        run_dir = Path(summary["run_dir"])
        signal = summary["signal"]
        holdings = summary["holdings"]
        assert Path(signal["artifact_dir"]).is_relative_to(run_dir)
        assert Path(holdings["artifact_dir"]).is_relative_to(run_dir)
        assert SignalArtifactStore().validate(signal["artifact_dir"]).is_valid
        assert HoldingsArtifactStore().validate(holdings["artifact_dir"]).is_valid
        assert holdings["requested_top_n"] == top_n
        holdings_frames[top_n] = pd.read_parquet(holdings["holdings_path"])
        snapshot = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text())
        assert snapshot["holdings"]["top_n"] == top_n
        assert snapshot["top_n"] == top_n
        artifact_config = json.loads(
            (Path(holdings["artifact_dir"]) / "config.json").read_text()
        )
        assert artifact_config["top_n"] == top_n
        assert config.to_dict() == before
        json.dumps(summary, allow_nan=False)
    assert results[5]["run_dir"] != results[10]["run_dir"]
    assert np.allclose(
        holdings_frames[5].groupby("trade_date")["target_weight"].first(), 1 / 5
    )
    assert np.allclose(
        holdings_frames[10].groupby("trade_date")["target_weight"].first(), 1 / 10
    )
    pdt.assert_frame_equal(
        holdings_frames[5].drop(columns="target_weight"),
        holdings_frames[10].loc[holdings_frames[10]["rank"] <= 5]
        .reset_index(drop=True)
        .drop(columns="target_weight"),
    )


def test_signal_no_overwrite_failure_blocks_holdings_and_preserves_target(
    tmp_path: Path, native_ml_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    run_dir = tmp_path / "fixed-run"
    signal_dir = run_dir / "signal"
    signal_dir.mkdir(parents=True)
    marker = signal_dir / "keep.txt"
    marker.write_bytes(b"keep")

    class FixedExperimentManager:
        def __init__(self, output_root: str | Path) -> None:
            self.delegate = ExperimentManager(output_root)
        def create_run_dir(self, strategy_name: str, stock_pool: str) -> Path:
            return run_dir
        def save_config_snapshot(self, *args: object, **kwargs: object) -> Path:
            return self.delegate.save_config_snapshot(*args, **kwargs)  # type: ignore[arg-type]
        def save_run_info(self, *args: object, **kwargs: object) -> Path:
            return self.delegate.save_run_info(*args, **kwargs)  # type: ignore[arg-type]
        def save_metrics(self, *args: object, **kwargs: object) -> Path:
            return self.delegate.save_metrics(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runner_module, "ExperimentManager", FixedExperimentManager)
    with pytest.raises(SignalPipelineExecutionError, match="Artifact write"):
        run_pipeline(
            _v5_files_config(
                tmp_path / "output", native_ml_artifact, top_n=5
            )
        )
    assert marker.read_bytes() == b"keep"
    assert not (run_dir / "holdings").exists()
    assert not (run_dir / "config_snapshot.yaml").exists()


def test_real_holdings_failure_preserves_valid_signal_artifact(
    tmp_path: Path, native_ml_artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    config = _v5_files_config(
        tmp_path / "output",
        native_ml_artifact,
        top_n=20,
        policy="error",
    )
    with pytest.raises(HoldingsPipelineExecutionError, match="Holdings build"):
        run_pipeline(config)
    runs = list((tmp_path / "output" / "runs").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    assert SignalArtifactStore().validate(run_dir / "signal").is_valid
    assert not (run_dir / "holdings").exists()
    assert not (run_dir / "config_snapshot.yaml").exists()


def test_runtime_missing_ml_result_fails_without_signal_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)

    class MissingMLExecutor:
        def __init__(self, config: object) -> None: pass
        def execute(self, run_dir: str | Path) -> None: return None

    monkeypatch.setattr(runner_module, "MLExperimentPipelineExecutor", MissingMLExecutor)
    config = _base_config(
        tmp_path / "output",
        ml_experiment=MLExperimentPipelineConfig(
            enabled=True, panel_path="panel.parquet", experiment=_experiment()
        ),
        signal=SignalPipelineConfig(enabled=True),
    )
    with pytest.raises(SignalPipelineExecutionError, match="returned no result"):
        run_pipeline(config)


def test_runtime_missing_signal_result_fails_without_holdings_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)

    class MissingSignalExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(
            self,
            run_dir: str | Path,
            *,
            ml_result: object = None,
        ) -> None:
            return None

    monkeypatch.setattr(runner_module, "SignalPipelineExecutor", MissingSignalExecutor)
    config = _base_config(
        tmp_path / "output",
        top_n=5,
        signal=SignalPipelineConfig(
            enabled=True,
            source=PredictionSourceConfig("files", tmp_path / "source"),
        ),
        holdings=HoldingsPipelineConfig(enabled=True, top_n=5),
    )
    with pytest.raises(HoldingsPipelineExecutionError, match="returned no result"):
        run_pipeline(config)