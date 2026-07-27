"""Tests for merged-panel ML execution and optional Pipeline integration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pyarrow.parquet as pq
import pytest
import yaml

import src.pipeline.ml_execution as execution_module
from src.factors.research_pipeline import FactorResearchConfig
from src.ml import (
    MLExperimentArtifactStore,
    MLExperimentConfig,
    MLExperimentResult,
)
from src.pipeline import (
    FactorResearchPipelineConfig,
    MLExperimentPipelineConfig,
    MLExperimentPipelineExecutor,
    MLExperimentPipelineResult,
    MLPipelineArtifactError,
    MLPipelineExecutionError,
    MLPipelineIntegrityError,
    MLPipelinePanelError,
    PipelineConfig,
    read_ml_modeling_panel,
    run_pipeline,
)


def _frame(periods: int = 16, stocks: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, date in enumerate(
        pd.date_range("2024-01-01", periods=periods, freq="D")
    ):
        for stock_number in range(stocks):
            factor_a = float(date_number + stock_number)
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": f"S{stock_number:02d}",
                    "factor_a": factor_a,
                    "factor_b": float(stock_number - date_number / 10),
                    "entry_trade_date": date + pd.Timedelta(days=1),
                    "exit_trade_date": date + pd.Timedelta(days=2),
                    "entry_price": 10.0 + factor_a,
                    "exit_price": 10.1 + factor_a,
                    "forward_return": factor_a / 100.0
                    + (stock_number % 2) / 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _experiment(
    model_name: str = "ridge",
    *,
    importance: bool = False,
) -> MLExperimentConfig:
    params: dict[str, object]
    if model_name == "ridge":
        params = {"alpha": 1.0}
    elif model_name == "elastic_net":
        params = {"alpha": 0.05, "l1_ratio": 0.2}
    else:
        params = {
            "max_iter": 10,
            "min_samples_leaf": 2,
            "early_stopping": False,
            "random_state": 3,
        }
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
                "model_name": model_name,
                "model_params": params,
            },
            "evaluation": {"minimum_cross_section_size": 3},
            "permutation_importance": (
                {
                    "scoring": "rmse",
                    "n_repeats": 2,
                    "random_state": 7,
                    "permutation_scope": "within_trade_date",
                }
                if importance
                else None
            ),
        }
    )


def _write_panel(path: Path, frame: pd.DataFrame | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    (frame if frame is not None else _frame()).to_parquet(
        path, engine="pyarrow", index=False
    )
    return path


def _config(
    panel: str | Path,
    *,
    model_name: str = "ridge",
    importance: bool = False,
    save_artifacts: bool = False,
    experiment_id: str | None = None,
    artifact_root: str = "ml_artifacts",
    compression: str = "zstd",
) -> MLExperimentPipelineConfig:
    return MLExperimentPipelineConfig(
        enabled=True,
        panel_path=str(panel),
        save_artifacts=save_artifacts,
        artifact_root=artifact_root,
        experiment_id=experiment_id,
        parquet_compression=compression,
        experiment=_experiment(model_name, importance=importance),
    )


def _execute(
    tmp_path: Path,
    *,
    model_name: str = "ridge",
    importance: bool = False,
    save_artifacts: bool = False,
    experiment_id: str | None = None,
    compression: str = "zstd",
) -> tuple[MLExperimentPipelineExecutor, MLExperimentPipelineResult, Path, Path]:
    panel = _write_panel(tmp_path / "project" / "panel.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    executor = MLExperimentPipelineExecutor(
        _config(
            panel,
            model_name=model_name,
            importance=importance,
            save_artifacts=save_artifacts,
            experiment_id=experiment_id,
            compression=compression,
        ),
        project_root=tmp_path / "project",
    )
    return executor, executor.execute(run_dir), run_dir, panel


def test_panel_reader_relative_absolute_and_defensive(tmp_path: Path) -> None:
    root = tmp_path / "project"
    original = _frame().iloc[[3, 1, 2, 0]].copy()
    path = _write_panel(root / "inputs" / "panel.parquet", original)
    relative = read_ml_modeling_panel(
        "inputs/panel.parquet",
        project_root=root,
        label_col="forward_return",
    )
    absolute = read_ml_modeling_panel(
        path,
        project_root=root,
        label_col="forward_return",
    )
    pdt.assert_frame_equal(relative, original.reset_index(drop=True))
    pdt.assert_frame_equal(absolute, original.reset_index(drop=True))
    relative.iloc[0, 0] = pd.Timestamp("2000-01-01")
    pdt.assert_frame_equal(absolute, original.reset_index(drop=True))


@pytest.mark.parametrize(
    "mode",
    ["missing", "directory", "suffix", "corrupt", "empty"],
)
def test_panel_reader_rejects_invalid_paths_and_files(
    tmp_path: Path, mode: str
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    path = root / "panel.parquet"
    if mode == "directory":
        path.mkdir()
    elif mode == "suffix":
        path = root / "panel.csv"
        path.write_text("not parquet", encoding="utf-8")
    elif mode == "corrupt":
        path.write_bytes(b"not parquet")
    elif mode == "empty":
        _write_panel(path, _frame().iloc[:0])
    with pytest.raises(MLPipelinePanelError):
        read_ml_modeling_panel(
            path,
            project_root=root,
            label_col="forward_return",
        )


def test_panel_reader_rejects_empty_path_and_invalid_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(MLPipelinePanelError):
        read_ml_modeling_panel(
            "", project_root=tmp_path, label_col="forward_return"
        )
    with pytest.raises(MLPipelinePanelError, match="project_root"):
        read_ml_modeling_panel(
            "panel.parquet",
            project_root=tmp_path / "missing",
            label_col="forward_return",
        )


@pytest.mark.parametrize(
    "missing",
    [
        "trade_date",
        "ts_code",
        "entry_trade_date",
        "exit_trade_date",
        "forward_return",
    ],
)
def test_panel_reader_rejects_missing_required_columns(
    tmp_path: Path, missing: str
) -> None:
    frame = _frame().drop(columns=[missing])
    path = _write_panel(tmp_path / "panel.parquet", frame)
    with pytest.raises(MLPipelinePanelError, match="missing required"):
        read_ml_modeling_panel(
            path,
            project_root=tmp_path,
            label_col="forward_return",
        )


def test_panel_reader_rejects_no_features_even_with_prices(
    tmp_path: Path,
) -> None:
    frame = _frame().loc[
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
    path = _write_panel(tmp_path / "panel.parquet", frame)
    with pytest.raises(MLPipelinePanelError, match="no candidate feature"):
        read_ml_modeling_panel(
            path,
            project_root=tmp_path,
            label_col="forward_return",
        )


def test_panel_reader_rejects_duplicate_columns_from_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_panel(tmp_path / "panel.parquet")
    duplicated = _frame()
    duplicated.columns = [
        *duplicated.columns[:-1],
        duplicated.columns[-2],
    ]
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: duplicated)
    with pytest.raises(MLPipelinePanelError, match="unique"):
        read_ml_modeling_panel(
            path,
            project_root=tmp_path,
            label_col="forward_return",
        )


def test_panel_reader_rejects_symlink_when_supported(tmp_path: Path) -> None:
    target = _write_panel(tmp_path / "target.parquet")
    link = tmp_path / "link.parquet"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(MLPipelinePanelError, match="symlink"):
        read_ml_modeling_panel(
            link,
            project_root=tmp_path,
            label_col="forward_return",
        )


@pytest.mark.parametrize(
    "model_name",
    ["ridge", "elastic_net", "hist_gradient_boosting"],
)
def test_executor_real_models_without_artifacts(
    tmp_path: Path, model_name: str
) -> None:
    executor, result, run_dir, panel = _execute(
        tmp_path, model_name=model_name
    )
    assert isinstance(result, MLExperimentPipelineResult)
    assert result.enabled is True
    assert result.model_name == model_name
    assert result.n_folds > 0
    assert result.n_prediction_rows > 0
    assert result.n_prediction_dates > 0
    assert result.mae is not None and result.mae >= 0.0
    assert result.rmse is not None and result.rmse >= 0.0
    assert result.r2_valid is True
    assert result.r2 is not None
    if model_name == "hist_gradient_boosting":
        assert result.pearson_ic_mean is None
        assert result.rank_ic_mean is None
    else:
        assert result.pearson_ic_mean is not None
        assert result.rank_ic_mean is not None
    assert result.permutation_importance_enabled is False
    assert result.permutation_importance_completed is False
    assert result.artifacts_saved is False
    assert result.artifact_dir is None
    assert not (run_dir / "ml_artifacts").exists()
    assert set(vars(executor)) == {"config", "project_root"}
    json.dumps(result.to_dict(), allow_nan=False)
    assert panel.is_file()


def test_executor_importance_status_and_repeat_independence(
    tmp_path: Path,
) -> None:
    panel = _write_panel(tmp_path / "project" / "panel.parquet")
    first_run = tmp_path / "run-1"
    second_run = tmp_path / "run-2"
    first_run.mkdir()
    second_run.mkdir()
    executor = MLExperimentPipelineExecutor(
        _config(panel, importance=True),
        project_root=tmp_path / "project",
    )
    first = executor.execute(first_run)
    second = executor.execute(second_run)
    assert first.permutation_importance_enabled is True
    assert first.permutation_importance_completed is True
    assert first.to_dict() == second.to_dict()
    assert set(vars(executor)) == {"config", "project_root"}


def test_no_artifact_mode_never_constructs_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ForbiddenStore:
        def __init__(self) -> None:
            raise AssertionError("artifact store must not be constructed")

    monkeypatch.setattr(
        execution_module, "MLExperimentArtifactStore", ForbiddenStore
    )
    _, result, run_dir, _ = _execute(tmp_path)
    assert result.artifacts_saved is False
    assert not (run_dir / "ml_artifacts").exists()


@pytest.mark.parametrize(
    ("importance", "artifact_count"),
    [(False, 13), (True, 17)],
)
def test_executor_real_artifact_save_and_validation(
    tmp_path: Path, importance: bool, artifact_count: int
) -> None:
    _, result, run_dir, _ = _execute(
        tmp_path,
        importance=importance,
        save_artifacts=True,
        experiment_id=f"Demo-{importance}",
    )
    expected = (
        run_dir / "ml_artifacts" / f"Demo-{importance}"
    ).resolve()
    assert result.artifacts_saved is True
    assert result.artifact_dir == str(expected)
    manifest = MLExperimentArtifactStore().read_manifest(expected)
    assert manifest.artifact_count == artifact_count
    report = MLExperimentArtifactStore().validate(expected)
    assert report.cross_file_integrity_verified is True
    assert not any(
        path.suffix.lower()
        in {".pkl", ".pickle", ".joblib", ".bin", ".model"}
        for path in expected.rglob("*")
        if path.is_file()
    )


def test_artifact_compression_is_propagated(tmp_path: Path) -> None:
    _, result, _, _ = _execute(
        tmp_path,
        save_artifacts=True,
        experiment_id="snappy",
        compression="snappy",
    )
    parquet = Path(result.artifact_dir) / "predictions.parquet"  # type: ignore[arg-type]
    metadata = pq.ParquetFile(parquet).metadata
    assert metadata.row_group(0).column(0).compression == "SNAPPY"


def test_existing_artifact_is_not_overwritten(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "project" / "panel.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _config(
        panel,
        save_artifacts=True,
        experiment_id="existing",
    )
    executor = MLExperimentPipelineExecutor(
        config, project_root=tmp_path / "project"
    )
    first = executor.execute(run_dir)
    marker = Path(first.artifact_dir) / "marker.txt"  # type: ignore[arg-type]
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(MLPipelineArtifactError) as caught:
        executor.execute(run_dir)
    assert caught.value.__cause__ is not None
    assert marker.read_text(encoding="utf-8") == "keep"


def test_artifact_symlink_escape_is_rejected_when_supported(
    tmp_path: Path,
) -> None:
    panel = _write_panel(tmp_path / "project" / "panel.parquet")
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    run_dir.mkdir()
    outside.mkdir()
    link = run_dir / "ml_artifacts"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    executor = MLExperimentPipelineExecutor(
        _config(
            panel,
            save_artifacts=True,
            experiment_id="escape",
        ),
        project_root=tmp_path / "project",
    )
    with pytest.raises(MLPipelineIntegrityError, match="inside"):
        executor.execute(run_dir)
    assert list(outside.iterdir()) == []


def test_runner_failure_is_wrapped_and_chained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = _write_panel(tmp_path / "project" / "panel.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class FailingRunner:
        def run(self, *args: object, **kwargs: object) -> object:
            raise ValueError("injected")

    monkeypatch.setattr(execution_module, "MLExperimentRunner", FailingRunner)
    executor = MLExperimentPipelineExecutor(
        _config(panel), project_root=tmp_path / "project"
    )
    with pytest.raises(MLPipelineExecutionError) as caught:
        executor.execute(run_dir)
    assert isinstance(caught.value.__cause__, ValueError)


def test_invalid_runner_result_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = _write_panel(tmp_path / "project" / "panel.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class InvalidRunner:
        def run(self, *args: object, **kwargs: object) -> object:
            return object()

    monkeypatch.setattr(execution_module, "MLExperimentRunner", InvalidRunner)
    with pytest.raises(MLPipelineExecutionError, match="invalid result"):
        MLExperimentPipelineExecutor(
            _config(panel), project_root=tmp_path / "project"
        ).execute(run_dir)


def test_artifact_failure_is_wrapped_and_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = _write_panel(tmp_path / "project" / "panel.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls = 0

    class FailingStore:
        def write(self, *args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            raise OSError("injected")

    monkeypatch.setattr(
        execution_module, "MLExperimentArtifactStore", FailingStore
    )
    executor = MLExperimentPipelineExecutor(
        _config(
            panel,
            save_artifacts=True,
            experiment_id="fixed-id",
        ),
        project_root=tmp_path / "project",
    )
    with pytest.raises(MLPipelineArtifactError) as caught:
        executor.execute(run_dir)
    assert isinstance(caught.value.__cause__, OSError)
    assert calls == 1


def test_executor_rejects_disabled_config_and_invalid_run_dir(
    tmp_path: Path,
) -> None:
    with pytest.raises(MLPipelineExecutionError, match="enabled"):
        MLExperimentPipelineExecutor(
            MLExperimentPipelineConfig(), project_root=tmp_path
        )
    panel = _write_panel(tmp_path / "panel.parquet")
    executor = MLExperimentPipelineExecutor(
        _config(panel), project_root=tmp_path
    )
    with pytest.raises(MLPipelineExecutionError, match="does not exist"):
        executor.execute(tmp_path / "missing-run")
    file_run = tmp_path / "file-run"
    file_run.write_text("x", encoding="utf-8")
    with pytest.raises(MLPipelineExecutionError, match="directory"):
        executor.execute(file_run)


class _ReadyDataManager:
    def prepare_data(self, config: object) -> dict[str, object]:
        return {"cache_status": "ready", "missing_ranges": {}}


def _pipeline_config(
    output_dir: Path,
    ml: MLExperimentPipelineConfig | None = None,
    factor_research: FactorResearchPipelineConfig | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        backtest_start="2024-01-01",
        backtest_end="2025-03-31",
        train_years=10,
        max_lookback_months=12,
        stock_pool="hs300",
        benchmark="000300.SH",
        strategy_name="score",
        selected_factors=["pe"],
        rebalance_frequency="M",
        top_n=20,
        transaction_cost=0.001,
        data_root="data",
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        cache_dir="data/cache",
        output_dir=str(output_dir),
        parquet_engine="auto",
        required_datasets=["daily"],
        factor_research=factor_research or FactorResearchPipelineConfig(),
        ml_experiment=ml or MLExperimentPipelineConfig(),
    )


def _install_ready_data_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.pipeline.runner.DataManager", _ReadyDataManager
    )


def test_pipeline_default_disabled_preserves_exact_old_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_ready_data_manager(monkeypatch)

    class ForbiddenExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("disabled ML must not construct executor")

    monkeypatch.setattr(
        "src.pipeline.runner.MLExperimentPipelineExecutor",
        ForbiddenExecutor,
    )
    config = _pipeline_config(tmp_path / "output")
    before = config.to_dict()
    summary = run_pipeline(config)
    assert set(summary) == {
        "status",
        "run_dir",
        "required_start_date",
        "required_end_date",
        "cache_status",
        "missing_ranges",
        "strategy_name",
        "stock_pool",
    }
    assert summary.get("ml_experiment") is None
    assert config.to_dict() == before
    run_dir = Path(summary["run_dir"])
    assert not (run_dir / "ml_artifacts").exists()
    snapshot = yaml.safe_load(
        (run_dir / "config_snapshot.yaml").read_text(encoding="utf-8")
    )
    assert snapshot["ml_experiment"]["enabled"] is False


def test_pipeline_enabled_runs_real_ml_in_existing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_ready_data_manager(monkeypatch)
    panel = _write_panel(tmp_path / "panel.parquet")
    config = _pipeline_config(
        tmp_path / "output",
        _config(
            panel,
            save_artifacts=True,
            experiment_id="pipeline-demo",
        ),
    )
    before = config.to_dict()
    summary = run_pipeline(config)
    run_dir = Path(summary["run_dir"])
    ml_summary = summary["ml_experiment"]
    assert ml_summary["model_name"] == "ridge"
    assert ml_summary["artifact_dir"] == str(
        (run_dir / "ml_artifacts" / "pipeline-demo").resolve()
    )
    assert "predictions" not in ml_summary
    assert "model_params" not in ml_summary
    assert MLExperimentArtifactStore().validate(
        ml_summary["artifact_dir"]
    ).cross_file_integrity_verified
    assert config.to_dict() == before


def test_pipeline_factor_then_ml_order_and_same_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_ready_data_manager(monkeypatch)
    events: list[tuple[str, Path]] = []

    class FactorResult:
        def to_dict(self) -> dict[str, object]:
            return {"enabled": True}

    class FakeFactorExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(
            self, run_dir: str | Path, *, metadata: object = None
        ) -> FactorResult:
            events.append(("factor", Path(run_dir)))
            return FactorResult()

    class MLResult:
        def to_dict(self) -> dict[str, object]:
            return {"enabled": True, "model_name": "ridge"}

    class FakeMLExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(self, run_dir: str | Path) -> MLResult:
            events.append(("ml", Path(run_dir)))
            return MLResult()

    monkeypatch.setattr(
        "src.pipeline.runner.FactorResearchPipelineExecutor",
        FakeFactorExecutor,
    )
    monkeypatch.setattr(
        "src.pipeline.runner.MLExperimentPipelineExecutor",
        FakeMLExecutor,
    )
    factor = FactorResearchPipelineConfig(
        enabled=True,
        factor_input_path="factor.parquet",
        score_panel_path="score.parquet",
        price_panel_path="price.parquet",
        research=FactorResearchConfig(
            factor_names=("momentum_20d",),
            composition_method="equal",
        ),
    )
    ml = MLExperimentPipelineConfig(
        enabled=True,
        panel_path="panel.parquet",
        experiment=_experiment(),
    )
    summary = run_pipeline(
        _pipeline_config(tmp_path / "output", ml, factor)
    )
    assert [name for name, _ in events] == ["factor", "ml"]
    assert events[0][1] == events[1][1] == Path(summary["run_dir"])
    assert summary["factor_research"]["enabled"] is True
    assert summary["ml_experiment"]["model_name"] == "ridge"


def test_pipeline_ml_failure_propagates_without_silent_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_ready_data_manager(monkeypatch)

    class FailingExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(self, run_dir: str | Path) -> object:
            raise MLPipelineExecutionError("injected")

    monkeypatch.setattr(
        "src.pipeline.runner.MLExperimentPipelineExecutor",
        FailingExecutor,
    )
    config = _pipeline_config(
        tmp_path / "output",
        MLExperimentPipelineConfig(
            enabled=True,
            panel_path="panel.parquet",
            experiment=_experiment(),
        ),
    )
    with pytest.raises(MLPipelineExecutionError, match="injected"):
        run_pipeline(config)
    runs = list((tmp_path / "output" / "runs").iterdir())
    assert len(runs) == 1
    assert not (runs[0] / "config_snapshot.yaml").exists()
