"""V5-F release-level offline end-to-end verification."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

import src.pipeline.runner as runner_module
from app.services.pipeline_config_service import (
    ERROR_IF_INSUFFICIENT,
    HIGH_SCORE_FIRST,
    build_effective_pipeline_config,
)
from app.services.pipeline_runner_service import run_canonical_pipeline
from src.holdings import HoldingsArtifactStore
from src.ml import (
    MLArtifactConfig,
    MLExperimentArtifactStore,
    MLExperimentConfig,
    MLExperimentRunner,
)
from src.modeling_panel import ModelingPanelArtifactStore
from src.pipeline import (
    HoldingsPipelineConfig,
    HoldingsPipelineExecutionError,
    HoldingsPipelineExecutor,
    MLExperimentPipelineConfig,
    ModelingPanelPipelineConfig,
    PipelineConfig,
    PredictionSourceConfig,
    SignalPipelineConfig,
    SignalPipelineExecutionError,
    SignalPipelineExecutor,
    run_pipeline,
)
from src.pipeline.experiment import ExperimentManager
from src.signals import (
    SIGNAL_FORBIDDEN_OUTPUT_COLUMNS,
    SIGNAL_OUTPUT_COLUMNS,
    SignalArtifactStore,
    SignalBuilder,
)


SIGNAL_FILES = {"signals.parquet", "config.json", "audit.json", "manifest.json"}
HOLDINGS_FILES = {"holdings.parquet", "config.json", "audit.json", "manifest.json"}


class _ReadyDataManager:
    def prepare_data(self, config: object) -> dict[str, object]:
        return {"cache_status": "ready", "missing_ranges": {}}


def _ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "DataManager", _ReadyDataManager)


def _experiment() -> MLExperimentConfig:
    return MLExperimentConfig.from_dict(
        {
            "dataset": {"label_col": "forward_return"},
            "walk_forward": {
                "train_window_periods": 2,
                "validation_periods": 2,
                "window_type": "rolling",
                "retrain_frequency": 3,
                "embargo_periods": 1,
            },
            "training": {
                "model_name": "ridge",
                "model_params": {"alpha": 1.0},
            },
            "evaluation": {"minimum_cross_section_size": 3},
            "permutation_importance": None,
        }
    )


def _panel(counts: tuple[int, ...] = (24,) * 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, (trade_date, count) in enumerate(
        zip(pd.date_range("2024-02-01", periods=len(counts)), counts, strict=True)
    ):
        for stock_number in range(count):
            factor_a = float(date_number + stock_number / 10)
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"V{stock_number:03d}",
                    "factor_a": factor_a,
                    "factor_b": float(stock_number - date_number / 10),
                    "entry_trade_date": trade_date + pd.Timedelta(days=1),
                    "exit_trade_date": trade_date + pd.Timedelta(days=2),
                    "forward_return": factor_a / 100,
                }
            )
    return pd.DataFrame(rows)


def _write_native_ml(root: Path, counts: tuple[int, ...] = (24,) * 12) -> Path:
    result = MLExperimentRunner().run(_panel(counts), _experiment())
    return MLExperimentArtifactStore().write(
        result, MLArtifactConfig(root, "v5f-source")
    ).experiment_dir


@pytest.fixture(scope="module")
def native_ml_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_native_ml(tmp_path_factory.mktemp("v5f-native-ml"))


def _base_config(
    output_dir: Path,
    *,
    top_n: int = 10,
    signal: SignalPipelineConfig | None = None,
    holdings: HoldingsPipelineConfig | None = None,
    modeling_panel: ModelingPanelPipelineConfig | None = None,
    ml_experiment: MLExperimentPipelineConfig | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        backtest_start="2024-01-01",
        backtest_end="2024-12-31",
        train_years=1,
        max_lookback_months=1,
        stock_pool="synthetic",
        benchmark="SYNTHETIC",
        strategy_name="v5f_release",
        selected_factors=["factor_a", "factor_b"],
        rebalance_frequency="D",
        top_n=top_n,
        transaction_cost=0.0,
        data_root="data",
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        cache_dir="data/cache",
        output_dir=str(output_dir),
        parquet_engine="pyarrow",
        required_datasets=[],
        modeling_panel=modeling_panel,
        ml_experiment=ml_experiment,
        signal=signal,
        holdings=holdings,
    )


def _files_config(
    output_dir: Path,
    native: Path,
    *,
    top_n: int,
    policy: str = "error",
    holdings_enabled: bool = True,
) -> PipelineConfig:
    return _base_config(
        output_dir,
        top_n=top_n,
        signal=SignalPipelineConfig(
            enabled=True,
            source=PredictionSourceConfig("files", native),
            signal_direction="descending",
        ),
        holdings=HoldingsPipelineConfig(
            enabled=holdings_enabled,
            top_n=top_n,
            insufficient_universe_policy=policy,
        ),
    )


def _write_modeling_inputs(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    panel = _panel()
    factors = panel.loc[
        :, ["trade_date", "ts_code", "factor_a", "factor_b"]
    ]
    returns = panel.loc[
        :,
        [
            "trade_date",
            "ts_code",
            "entry_trade_date",
            "exit_trade_date",
            "forward_return",
        ],
    ].copy()
    returns["entry_price"] = 20.0 + returns.index.to_numpy(dtype=float) / 100
    returns["exit_price"] = returns["entry_price"] * (
        1.0 + returns["forward_return"]
    )
    returns = returns.loc[
        :,
        [
            "trade_date",
            "ts_code",
            "entry_trade_date",
            "exit_trade_date",
            "entry_price",
            "exit_price",
            "forward_return",
        ],
    ]
    factor_path = root / "factor_panel.parquet"
    returns_path = root / "forward_returns.parquet"
    factors.to_parquet(factor_path, engine="pyarrow", index=False)
    returns.to_parquet(returns_path, engine="pyarrow", index=False)
    return factor_path, returns_path


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_dataframe(value: object) -> bool:
    if isinstance(value, pd.DataFrame):
        return True
    if isinstance(value, dict):
        return any(_contains_dataframe(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_dataframe(item) for item in value)
    return False


def _assert_valid_chain(summary: dict[str, object]) -> tuple[Path, Path, Path]:
    ml_dir = Path(summary["ml_experiment"]["artifact_dir"])  # type: ignore[index]
    signal_dir = Path(summary["signal"]["artifact_dir"])  # type: ignore[index]
    holdings_dir = Path(summary["holdings"]["artifact_dir"])  # type: ignore[index]
    ml_report = MLExperimentArtifactStore().validate(ml_dir)
    assert "predictions.parquet" in ml_report.validated_artifacts
    assert ml_report.cross_file_integrity_verified
    assert (ml_dir / "predictions.parquet").is_file()
    assert SignalArtifactStore().validate(signal_dir).is_valid
    assert HoldingsArtifactStore().validate(holdings_dir).is_valid
    assert {item.name for item in signal_dir.iterdir()} == SIGNAL_FILES
    assert {item.name for item in holdings_dir.iterdir()} == HOLDINGS_FILES
    return ml_dir, signal_dir, holdings_dir


def test_real_modeling_panel_ml_signal_holdings_runner_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    factor_path, returns_path = _write_modeling_inputs(tmp_path / "inputs")
    modeling = ModelingPanelPipelineConfig.from_dict(
        {
            "enabled": True,
            "source": {
                "mode": "files",
                "factor_panel_path": factor_path,
                "forward_returns_path": returns_path,
            },
            "builder": {"include_features": ["factor_a", "factor_b"]},
        }
    )
    ml = MLExperimentPipelineConfig(
        enabled=True,
        panel_path=None,
        save_artifacts=True,
        artifact_root="ml_artifacts",
        experiment_id="v5f-current-run",
        experiment=_experiment(),
    )
    config = _base_config(
        tmp_path / "output",
        top_n=10,
        modeling_panel=modeling,
        ml_experiment=ml,
        signal=SignalPipelineConfig(enabled=True),
        holdings=HoldingsPipelineConfig(enabled=True, top_n=10),
    )
    before = deepcopy(config.to_dict())

    summary = run_pipeline(config)

    run_dir = Path(summary["run_dir"])
    assert ModelingPanelArtifactStore().validate(
        summary["modeling_panel"]["artifact_dir"]  # type: ignore[index]
    ).is_valid
    ml_dir, signal_dir, holdings_dir = _assert_valid_chain(summary)
    assert ml_dir.is_relative_to(run_dir)
    assert signal_dir.parent == run_dir and holdings_dir.parent == run_dir
    assert summary["signal"]["source_artifact_dir"] == str(ml_dir)  # type: ignore[index]
    assert summary["holdings"]["source_signal_artifact_dir"] == str(signal_dir)  # type: ignore[index]
    assert summary["signal"]["signal_direction"] == "descending"  # type: ignore[index]
    assert summary["holdings"]["requested_top_n"] == 10  # type: ignore[index]
    holdings = pd.read_parquet(holdings_dir / "holdings.parquet")
    assert holdings.groupby("trade_date").size().eq(10).all()
    assert np.allclose(holdings["target_weight"], 0.1)
    snapshot = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text())
    assert snapshot["signal"]["signal_direction"] == "descending"
    assert snapshot["holdings"]["top_n"] == 10
    assert not _contains_dataframe(summary)
    assert config.to_dict() == before


def test_files_mode_explicit_source_ignores_newer_sibling_and_bare_file_fails(
    tmp_path: Path,
    native_ml_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready(monkeypatch)
    sibling = native_ml_artifact.parent / "newer-decoy"
    sibling.mkdir()
    (sibling / "predictions.parquet").write_bytes(b"not-an-artifact")
    future = native_ml_artifact.stat().st_mtime + 10_000
    os.utime(sibling, (future, future))

    summary = run_pipeline(
        _files_config(tmp_path / "output", native_ml_artifact, top_n=10)
    )
    assert "ml_experiment" not in summary
    assert summary["signal"]["source_mode"] == "files"  # type: ignore[index]
    assert Path(summary["signal"]["source_artifact_dir"]) == native_ml_artifact.resolve()  # type: ignore[index]
    assert "newer-decoy" not in json.dumps(summary)
    assert SignalArtifactStore().validate(summary["signal"]["artifact_dir"]).is_valid  # type: ignore[index]
    assert HoldingsArtifactStore().validate(summary["holdings"]["artifact_dir"]).is_valid  # type: ignore[index]

    bare = _files_config(
        tmp_path / "bare-output",
        native_ml_artifact / "predictions.parquet",
        top_n=10,
    )
    with pytest.raises(SignalPipelineExecutionError, match="source validation"):
        run_pipeline(bare)
    bare_run = next((tmp_path / "bare-output" / "runs").iterdir())
    assert not (bare_run / "signal").exists()
    assert not (bare_run / "holdings").exists()


def test_top_n_five_ten_twenty_trace_prefix_weights_and_determinism(
    tmp_path: Path,
    native_ml_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready(monkeypatch)
    summaries: dict[int, dict[str, object]] = {}
    signals: dict[int, pd.DataFrame] = {}
    holdings: dict[int, pd.DataFrame] = {}
    signal_metadata: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
    holdings_metadata: dict[int, tuple[dict[str, object], dict[str, object]]] = {}

    for top_n in (5, 10, 20):
        config = _files_config(
            tmp_path / f"output-{top_n}", native_ml_artifact, top_n=top_n
        )
        summary = run_pipeline(config)
        summaries[top_n] = summary
        run_dir = Path(summary["run_dir"])
        signal_dir = Path(summary["signal"]["artifact_dir"])  # type: ignore[index]
        holdings_dir = Path(summary["holdings"]["artifact_dir"])  # type: ignore[index]
        signals[top_n] = pd.read_parquet(signal_dir / "signals.parquet")
        holdings[top_n] = pd.read_parquet(holdings_dir / "holdings.parquet")
        signal_metadata[top_n] = (
            _json(signal_dir / "config.json"),
            _json(signal_dir / "audit.json"),
        )
        holdings_metadata[top_n] = (
            _json(holdings_dir / "config.json"),
            _json(holdings_dir / "audit.json"),
        )
        snapshot = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text())
        assert config.holdings.top_n == top_n
        assert snapshot["holdings"]["top_n"] == top_n
        assert snapshot["top_n"] == top_n
        assert summary["holdings"]["requested_top_n"] == top_n  # type: ignore[index]
        assert holdings_metadata[top_n][0]["top_n"] == top_n
        assert holdings_metadata[top_n][1]["requested_top_n"] == top_n
        assert _json(holdings_dir / "manifest.json")["top_n"] == top_n
        assert holdings[top_n].groupby("trade_date").size().eq(top_n).all()
        assert np.allclose(holdings[top_n]["target_weight"], 1.0 / top_n)
        assert holdings[top_n]["target_weight"].gt(0).all()
        assert np.allclose(
            holdings[top_n].groupby("trade_date")["target_weight"].sum(), 1.0
        )
        assert signal_dir.parent == run_dir and holdings_dir.parent == run_dir

    pdt.assert_frame_equal(signals[5], signals[10])
    pdt.assert_frame_equal(signals[10], signals[20])
    assert signal_metadata[5] == signal_metadata[10] == signal_metadata[20]
    for smaller, larger in ((5, 10), (10, 20)):
        prefix = holdings[larger].loc[
            holdings[larger]["rank"] <= smaller,
            ["trade_date", "ts_code", "score", "rank"],
        ].reset_index(drop=True)
        pdt.assert_frame_equal(
            holdings[smaller][["trade_date", "ts_code", "score", "rank"]],
            prefix,
        )
    assert len({summary["run_dir"] for summary in summaries.values()}) == 3

    repeated = run_pipeline(
        _files_config(tmp_path / "output-repeat", native_ml_artifact, top_n=10)
    )
    repeated_signal_dir = Path(repeated["signal"]["artifact_dir"])  # type: ignore[index]
    repeated_holdings_dir = Path(repeated["holdings"]["artifact_dir"])  # type: ignore[index]
    pdt.assert_frame_equal(
        signals[10], pd.read_parquet(repeated_signal_dir / "signals.parquet")
    )
    pdt.assert_frame_equal(
        holdings[10], pd.read_parquet(repeated_holdings_dir / "holdings.parquet")
    )
    assert _json(repeated_signal_dir / "config.json") == signal_metadata[10][0]
    assert _json(repeated_signal_dir / "audit.json") == signal_metadata[10][1]
    repeated_holdings_audit = _json(repeated_holdings_dir / "audit.json")
    baseline_holdings_audit = deepcopy(holdings_metadata[10][1])
    repeated_holdings_audit.pop("source_signal_provenance")
    baseline_holdings_audit.pop("source_signal_provenance")
    assert repeated_holdings_audit == baseline_holdings_audit


def test_ui_bridge_to_real_canonical_pipeline_top_n_ten(
    tmp_path: Path,
    native_ml_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready(monkeypatch)
    base = _files_config(
        tmp_path / "output", native_ml_artifact, top_n=20
    )
    base.signal = SignalPipelineConfig(
        enabled=False,
        source=PredictionSourceConfig("files", native_ml_artifact),
    )
    base.holdings = HoldingsPipelineConfig(enabled=False, top_n=20)
    base = PipelineConfig.from_dict(base.to_dict())
    before = deepcopy(base.to_dict())
    effective = build_effective_pipeline_config(
        base,
        top_n=10,
        signal_direction_label=HIGH_SCORE_FIRST,
        insufficient_policy_label=ERROR_IF_INSUFFICIENT,
    )
    assert effective.holdings.top_n == 10
    assert effective.top_n == effective.holdings.top_n

    summary = run_canonical_pipeline(effective)

    run_dir = Path(summary["run_dir"])
    holdings_dir = Path(summary["holdings"]["artifact_dir"])  # type: ignore[index]
    snapshot = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text())
    frame = pd.read_parquet(holdings_dir / "holdings.parquet")
    assert snapshot["holdings"]["top_n"] == 10
    assert summary["holdings"]["requested_top_n"] == 10  # type: ignore[index]
    assert _json(holdings_dir / "config.json")["top_n"] == 10
    assert frame.groupby("trade_date").size().eq(10).all()
    assert base.to_dict() == before


def test_insufficient_universe_error_and_allow_partial_runner_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    native = _write_native_ml(
        tmp_path / "mixed-source",
        (12, 12, 12, 12, 12, 12, 12, 7, 12, 7, 12, 7),
    )
    partial_summary = run_pipeline(
        _files_config(
            tmp_path / "partial-output",
            native,
            top_n=10,
            policy="allow_partial",
        )
    )
    partial_signal = pd.read_parquet(partial_summary["signal"]["signal_path"])  # type: ignore[index]
    available = partial_signal.groupby("trade_date").size()
    assert available.ge(10).any() and available.lt(10).any()
    holdings_dir = Path(partial_summary["holdings"]["artifact_dir"])  # type: ignore[index]
    frame = pd.read_parquet(holdings_dir / "holdings.parquet")
    counts = frame.groupby("trade_date").size()
    assert counts.eq(np.minimum(available, 10)).all()
    assert np.allclose(
        frame.groupby("trade_date")["target_weight"].sum(), 1.0
    )
    for trade_date, count in counts.items():
        assert np.allclose(
            frame.loc[frame["trade_date"] == trade_date, "target_weight"],
            1.0 / count,
        )
    audit = _json(holdings_dir / "audit.json")
    assert audit["requested_top_n"] == 10
    assert audit["partial_dates"]
    assert any(item["partial"] for item in audit["per_date_counts"])
    assert HoldingsArtifactStore().validate(holdings_dir).is_valid

    with pytest.raises(HoldingsPipelineExecutionError, match="Holdings build"):
        run_pipeline(
            _files_config(
                tmp_path / "error-output", native, top_n=10, policy="error"
            )
        )
    error_run = next((tmp_path / "error-output" / "runs").iterdir())
    assert SignalArtifactStore().validate(error_run / "signal").is_valid
    assert not (error_run / "holdings").exists()
    assert not (error_run / "config_snapshot.yaml").exists()


class _FixedExperimentManager:
    run_dir: Path

    def __init__(self, output_root: str | Path) -> None:
        self.delegate = ExperimentManager(output_root)

    def create_run_dir(self, strategy_name: str, stock_pool: str) -> Path:
        return self.run_dir

    def save_config_snapshot(self, *args: object, **kwargs: object) -> Path:
        return self.delegate.save_config_snapshot(*args, **kwargs)  # type: ignore[arg-type]

    def save_run_info(self, *args: object, **kwargs: object) -> Path:
        return self.delegate.save_run_info(*args, **kwargs)  # type: ignore[arg-type]

    def save_metrics(self, *args: object, **kwargs: object) -> Path:
        return self.delegate.save_metrics(*args, **kwargs)  # type: ignore[arg-type]


def test_signal_and_holdings_no_overwrite_preserve_existing_and_upstream(
    tmp_path: Path,
    native_ml_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready(monkeypatch)
    signal_run = tmp_path / "signal-collision"
    signal_target = signal_run / "signal"
    signal_target.mkdir(parents=True)
    signal_marker = signal_target / "owner.txt"
    signal_marker.write_bytes(b"signal-owner")
    _FixedExperimentManager.run_dir = signal_run
    monkeypatch.setattr(runner_module, "ExperimentManager", _FixedExperimentManager)
    with pytest.raises(SignalPipelineExecutionError, match="Artifact write"):
        run_pipeline(
            _files_config(tmp_path / "signal-output", native_ml_artifact, top_n=10)
        )
    assert signal_marker.read_bytes() == b"signal-owner"
    assert set(signal_target.iterdir()) == {signal_marker}
    assert not (signal_run / "holdings").exists()

    holdings_run = tmp_path / "holdings-collision"
    holdings_target = holdings_run / "holdings"
    holdings_target.mkdir(parents=True)
    holdings_marker = holdings_target / "owner.txt"
    holdings_marker.write_bytes(b"holdings-owner")
    _FixedExperimentManager.run_dir = holdings_run
    with pytest.raises(HoldingsPipelineExecutionError, match="Artifact write"):
        run_pipeline(
            _files_config(tmp_path / "holdings-output", native_ml_artifact, top_n=10)
        )
    assert SignalArtifactStore().validate(holdings_run / "signal").is_valid
    assert holdings_marker.read_bytes() == b"holdings-owner"
    assert set(holdings_target.iterdir()) == {holdings_marker}
    assert not (holdings_run / "config_snapshot.yaml").exists()


def test_tampered_ml_blocks_signal_and_tampered_signal_blocks_holdings(
    tmp_path: Path,
    native_ml_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready(monkeypatch)
    tampered_ml = tmp_path / "tampered-ml"
    shutil.copytree(native_ml_artifact, tampered_ml)
    prediction_path = tampered_ml / "predictions.parquet"
    prediction_path.write_bytes(prediction_path.read_bytes() + b"tamper")
    with pytest.raises(SignalPipelineExecutionError, match="source validation"):
        run_pipeline(
            _files_config(tmp_path / "tampered-output", tampered_ml, top_n=10)
        )
    tampered_run = next((tmp_path / "tampered-output" / "runs").iterdir())
    assert not (tampered_run / "signal").exists()
    assert not (tampered_run / "holdings").exists()

    source_run = tmp_path / "signal-source-run"
    source_run.mkdir()
    signal_result = SignalPipelineExecutor(
        SignalPipelineConfig(
            enabled=True,
            source=PredictionSourceConfig("files", native_ml_artifact),
        )
    ).execute(source_run)
    signal_result.signal_path.write_bytes(
        signal_result.signal_path.read_bytes() + b"tamper"
    )
    holdings_run = tmp_path / "tampered-signal-downstream"
    holdings_run.mkdir()
    with pytest.raises(HoldingsPipelineExecutionError, match="validation is invalid"):
        HoldingsPipelineExecutor(
            HoldingsPipelineConfig(enabled=True, top_n=10)
        ).execute(holdings_run, signal_result=signal_result)
    assert not (holdings_run / "holdings").exists()


def test_artifact_lineage_is_ml_to_signal_to_holdings(
    tmp_path: Path,
    native_ml_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready(monkeypatch)
    summary = run_pipeline(
        _files_config(tmp_path / "output", native_ml_artifact, top_n=10)
    )
    signal_dir = Path(summary["signal"]["artifact_dir"])  # type: ignore[index]
    holdings_dir = Path(summary["holdings"]["artifact_dir"])  # type: ignore[index]
    ml_manifest = MLExperimentArtifactStore().read_manifest(native_ml_artifact)
    signal_manifest = _json(signal_dir / "manifest.json")
    holdings_manifest = _json(holdings_dir / "manifest.json")
    signal_source = signal_manifest["source_provenance"]
    holdings_source = holdings_manifest["source_signal_provenance"]
    assert Path(signal_source["artifact_dir"]) == native_ml_artifact.resolve()
    assert signal_source["experiment_id"] == ml_manifest.experiment_id
    assert signal_source["model_name"] == ml_manifest.model_name
    assert len(signal_source["prediction_sha256"]) == 64
    assert Path(holdings_source["signal_artifact_dir"]) == signal_dir.resolve()
    assert Path(holdings_source["signal_path"]) == signal_dir / "signals.parquet"
    assert len(holdings_source["signal_sha256"]) == 64
    assert "experiment_id" not in holdings_source


def test_deterministic_direction_ties_shuffle_and_rank_contract() -> None:
    predictions = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-01"] * 4 + ["2024-01-02"] * 4
            ),
            "ts_code": ["D", "B", "A", "C"] * 2,
            "prediction": [2.0, 2.0, 3.0, 1.0, 1.0, 1.0, 0.0, 2.0],
        }
    )
    shuffled = predictions.sample(frac=1.0, random_state=17)
    for direction in ("descending", "ascending"):
        first = SignalBuilder().build(
            predictions,
            prediction_column="prediction",
            signal_direction=direction,
        ).signals
        second = SignalBuilder().build(
            shuffled,
            prediction_column="prediction",
            signal_direction=direction,
        ).signals
        pdt.assert_frame_equal(first, second)
        for _, group in first.groupby("trade_date", sort=False):
            assert group["rank"].tolist() == list(range(1, len(group) + 1))
            for _, tied in group.groupby("score"):
                assert tied["ts_code"].tolist() == sorted(tied["ts_code"])
    assert tuple(first.columns) == SIGNAL_OUTPUT_COLUMNS
    assert SIGNAL_FORBIDDEN_OUTPUT_COLUMNS.isdisjoint(first.columns)


def test_backward_compatibility_matrix_and_signal_only_execution(
    tmp_path: Path,
    native_ml_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready(monkeypatch)
    disabled = _base_config(tmp_path / "disabled")
    summary = run_pipeline(disabled)
    assert "signal" not in summary and "holdings" not in summary

    signal_only = _files_config(
        tmp_path / "signal-only",
        native_ml_artifact,
        top_n=10,
        holdings_enabled=False,
    )
    signal_summary = run_pipeline(signal_only)
    assert "signal" in signal_summary and "holdings" not in signal_summary
    assert "ml_experiment" not in signal_summary
    assert SignalArtifactStore().validate(
        signal_summary["signal"]["artifact_dir"]
    ).is_valid

    with pytest.raises(ValueError, match="requires ml_experiment"):
        _base_config(
            tmp_path / "invalid-ml",
            signal=SignalPipelineConfig(enabled=True),
        )
    with pytest.raises(ValueError, match="requires signal"):
        _base_config(
            tmp_path / "invalid-holdings",
            holdings=HoldingsPipelineConfig(enabled=True, top_n=10),
        )

    project_root = Path(__file__).resolve().parents[1]
    old_direct = yaml.safe_load(
        (project_root / "config" / "modeling_panel_pipeline.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    parsed_direct = PipelineConfig.from_dict(old_direct)
    parsed_grouped = PipelineConfig.from_yaml(project_root / "config" / "config.yaml")
    for parsed in (parsed_direct, parsed_grouped):
        assert not parsed.signal.enabled
        assert not parsed.holdings.enabled


def test_ui_root_mirror_and_canonical_runner_are_one_way_only() -> None:
    from app.services import pipeline_config_service, pipeline_runner_service

    bridge_source = inspect.getsource(
        pipeline_config_service.build_effective_pipeline_config
    )
    runner_source = inspect.getsource(runner_module.run_pipeline)
    service_source = inspect.getsource(pipeline_runner_service.run_canonical_pipeline)
    assert '"top_n": effective_top_n' in bridge_source
    assert 'values["top_n"] = effective_top_n' in bridge_source
    assert "config.holdings" in runner_source
    assert "config.top_n" not in runner_source
    assert "run_pipeline(config)" in service_source
    for forbidden in ("subprocess", "--top-n", "rank(", "target_weight"):
        assert forbidden not in service_source


def test_canonical_v5_paths_have_no_latest_or_sibling_discovery() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = (
        project_root / "src" / "signals" / "sources.py",
        project_root / "src" / "pipeline" / "runner.py",
        project_root / "src" / "pipeline" / "signal_execution.py",
        project_root / "src" / "pipeline" / "holdings_execution.py",
        project_root / "app" / "services" / "pipeline_config_service.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in ("getmtime", ".glob(", ".rglob(", "os.walk("):
        assert forbidden not in source
    assert "latest Artifact" not in source