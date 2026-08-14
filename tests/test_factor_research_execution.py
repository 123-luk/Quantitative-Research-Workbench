"""Tests for V2-G4B factor-research pipeline execution."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.factors.composition import FactorCompositionConfig
from src.factors.evaluation import FactorEvaluationConfig
from src.factors.forward_returns import ForwardReturnConfig
from src.factors.preprocessing import PreprocessingConfig
from src.factors.quantile_evaluation import QuantileEvaluationConfig
from src.factors.registry import create_default_registry
from src.factors.research_artifacts import (
    FactorResearchArtifactStore,
    ResearchArtifactConfig,
)
from src.factors.research_pipeline import FactorResearchConfig
from src.pipeline import (
    FactorResearchExecutionResult,
    FactorResearchPipelineConfig,
    FactorResearchPipelineExecutor,
    PipelineConfig,
)
from src.pipeline.runner import run_pipeline


NAMES = ("momentum_20d", "volatility_20d")


def _panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=32)
    codes = [f"S{index:02d}" for index in range(10)]
    factor_rows: list[dict] = []
    price_rows: list[dict] = []
    for date_index, trade_date in enumerate(dates):
        for stock_index, code in enumerate(codes):
            close = (
                100.0
                * (1.0 + 0.0015 * (stock_index + 1)) ** date_index
                * (1.0 + 0.002 * np.sin(date_index + stock_index))
            )
            price_rows.append(
                {"trade_date": trade_date, "ts_code": code, "close": close}
            )
            if date_index < 30:
                factor_rows.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": code,
                        "close": close,
                        "pe_ttm": 8.0 + stock_index,
                        "fin_roe_ttm": 0.05 + stock_index * 0.01,
                    }
                )
    score_panel = pd.DataFrame(
        [
            {"trade_date": trade_date, "ts_code": code}
            for trade_date in dates[22:26]
            for code in codes
        ]
    )
    exposure_panel = score_panel.copy()
    exposure_panel["industry"] = [
        "I1" if int(code[1:]) < 5 else "I2"
        for code in exposure_panel["ts_code"]
    ]
    exposure_panel["log_total_mv"] = [
        8.0 + int(code[1:]) * 0.1 for code in exposure_panel["ts_code"]
    ]
    return (
        pd.DataFrame(factor_rows),
        score_panel,
        pd.DataFrame(price_rows),
        exposure_panel,
    )


def _write_panels(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    names = ("factor_input", "score_panel", "price_panel", "exposure_panel")
    paths: dict[str, Path] = {}
    for name, frame in zip(names, _panels()):
        path = root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    return paths


def _research_config(
    paths: dict[str, Path | str],
    *,
    factor_names: tuple[str, ...] = NAMES,
    neutralization: bool = False,
    artifact_subdir: str = "factor_research",
    artifacts: ResearchArtifactConfig | None = None,
) -> FactorResearchPipelineConfig:
    return FactorResearchPipelineConfig(
        enabled=True,
        factor_input_path=str(paths["factor_input"]),
        score_panel_path=str(paths["score_panel"]),
        price_panel_path=str(paths["price_panel"]),
        exposure_panel_path=(
            str(paths["exposure_panel"])
            if "exposure_panel" in paths
            else None
        ),
        artifact_subdir=artifact_subdir,
        research=FactorResearchConfig(
            factor_names=factor_names,
            use_neutralization=neutralization,
            composition_method="equal",
        ),
        preprocessing=PreprocessingConfig(
            missing_method="none",
            winsor_method="none",
            standardize_method="zscore",
            min_cross_section_size=5,
        ),
        evaluation=FactorEvaluationConfig(min_cross_section_size=5),
        quantile=QuantileEvaluationConfig(
            quantiles=5,
            min_cross_section_size=5,
            min_group_size=1,
        ),
        composition=FactorCompositionConfig(method="equal"),
        forward_returns=ForwardReturnConfig(
            entry_lag_periods=1,
            holding_periods=1,
        ),
        artifacts=artifacts or ResearchArtifactConfig(),
    )


def _pipeline_config(
    output_dir: Path,
    research: FactorResearchPipelineConfig | None = None,
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
        factor_research=research or FactorResearchPipelineConfig(),
    )


def _install_ready_data_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    class ReadyDataManager:
        def prepare_data(self, request):
            return {"cache_status": "ready", "missing_ranges": {}}

    monkeypatch.setattr("src.pipeline.runner.DataManager", ReadyDataManager)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_execution_result_is_json_safe_stable_and_detached() -> None:
    manifest = {"metadata": {"values": [1, 2]}}
    result = FactorResearchExecutionResult(
        enabled=True,
        artifact_dir="run/factor_research",
        manifest=manifest,
        table_shapes={"scores": (2, 3)},
        input_shapes={"factor_input": {"rows": 4, "columns": 5}},
        factor_names=("momentum_20d",),
        composition_method="equal",
    )
    payload = result.to_dict()
    assert payload["table_shapes"]["scores"] == [2, 3]
    assert payload["factor_names"] == ["momentum_20d"]
    assert "DataFrame" not in repr(payload)
    json.dumps(payload)
    payload["manifest"]["metadata"]["values"].append(3)
    assert result.to_dict()["manifest"]["metadata"]["values"] == [1, 2]


def test_disabled_result_has_stable_structure() -> None:
    assert FactorResearchExecutionResult.disabled().to_dict() == {
        "enabled": False,
        "artifact_dir": None,
        "manifest": None,
        "table_shapes": {},
        "input_shapes": {},
        "factor_names": [],
        "composition_method": None,
    }


def test_executor_constructor_and_description(tmp_path: Path) -> None:
    config = FactorResearchPipelineConfig()
    first = FactorResearchPipelineExecutor(config, project_root=tmp_path)
    second = FactorResearchPipelineExecutor(config, project_root=str(tmp_path))
    assert first.project_root == tmp_path.resolve()
    assert second.project_root == tmp_path.resolve()
    assert first.describe_config() == config.to_dict()
    with pytest.raises(TypeError, match="config"):
        FactorResearchPipelineExecutor(object(), project_root=tmp_path)  # type: ignore[arg-type]
    file_root = tmp_path / "file"
    file_root.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        FactorResearchPipelineExecutor(config, project_root=file_root)
    with pytest.raises(FileNotFoundError, match="project_root"):
        FactorResearchPipelineExecutor(config, project_root=tmp_path / "missing")


def test_disabled_executor_reads_nothing_and_creates_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = FactorResearchPipelineConfig(
        factor_input_path="missing-factor.parquet",
        score_panel_path="missing-score.parquet",
        price_panel_path="missing-price.parquet",
    )

    def forbidden_read(*args, **kwargs):
        raise AssertionError("read_parquet must not be called")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    result = FactorResearchPipelineExecutor(
        config, project_root=tmp_path
    ).execute(run_dir)
    assert result.to_dict() == FactorResearchExecutionResult.disabled().to_dict()
    assert list(run_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("role", "path_key"),
    [
        ("factor_input", "factor_input"),
        ("score_panel", "score_panel"),
        ("price_panel", "price_panel"),
    ],
)
def test_missing_required_inputs_report_role(
    tmp_path: Path, role: str, path_key: str
) -> None:
    paths = _write_panels(tmp_path / "inputs")
    paths[path_key] = tmp_path / "missing.parquet"
    config = _research_config(paths)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=role):
        FactorResearchPipelineExecutor(
            config, project_root=tmp_path
        ).execute(run_dir)
    assert not (run_dir / "factor_research").exists()


def test_input_directory_and_corrupt_parquet_report_context(tmp_path: Path) -> None:
    paths = _write_panels(tmp_path / "inputs")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths["factor_input"] = tmp_path / "inputs"
    with pytest.raises(ValueError, match="factor_input.*regular file"):
        FactorResearchPipelineExecutor(
            _research_config(paths), project_root=tmp_path
        ).execute(run_dir)

    bad = tmp_path / "bad.data"
    bad.write_text("not parquet", encoding="utf-8")
    paths["factor_input"] = bad
    with pytest.raises(RuntimeError, match="factor_input Parquet input"):
        FactorResearchPipelineExecutor(
            _research_config(paths), project_root=tmp_path
        ).execute(run_dir)
    assert not (run_dir / "factor_research").exists()


def test_relative_paths_use_project_root_not_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_panels(tmp_path / "project" / "inputs")
    relative = {
        name: path.relative_to(tmp_path / "project").as_posix()
        for name, path in paths.items()
    }
    config = _research_config(relative)
    before = config.to_dict()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = FactorResearchPipelineExecutor(
        config, project_root=tmp_path / "project"
    ).execute(run_dir)
    assert result.enabled is True
    assert config.to_dict() == before


def test_absolute_paths_execute_successfully(tmp_path: Path) -> None:
    paths = _write_panels(tmp_path / "inputs")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = FactorResearchPipelineExecutor(
        _research_config(paths), project_root=tmp_path / "inputs"
    ).execute(run_dir)
    assert result.enabled is True


def test_exposure_is_ignored_when_neutralization_is_disabled(tmp_path: Path) -> None:
    paths = _write_panels(tmp_path / "inputs")
    paths["exposure_panel"] = tmp_path / "does-not-exist.parquet"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = FactorResearchPipelineExecutor(
        _research_config(paths, neutralization=False),
        project_root=tmp_path,
    ).execute(run_dir)
    assert "exposure_panel" not in result.input_shapes


def test_exposure_is_required_and_read_for_neutralization(tmp_path: Path) -> None:
    paths = _write_panels(tmp_path / "inputs")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = FactorResearchPipelineExecutor(
        _research_config(paths, neutralization=True),
        project_root=tmp_path,
    ).execute(run_dir)
    assert result.input_shapes["exposure_panel"] == {"rows": 40, "columns": 4}

    second_run = tmp_path / "second-run"
    second_run.mkdir()
    paths["exposure_panel"] = tmp_path / "missing-exposure.parquet"
    with pytest.raises(FileNotFoundError, match="exposure_panel"):
        FactorResearchPipelineExecutor(
            _research_config(paths, neutralization=True),
            project_root=tmp_path,
        ).execute(second_run)


def test_registry_is_independent_complete_and_does_not_pollute_defaults(
    tmp_path: Path,
) -> None:
    executor = FactorResearchPipelineExecutor(
        FactorResearchPipelineConfig(), project_root=tmp_path
    )
    before = create_default_registry().list_names()
    first = executor._build_registry()
    second = executor._build_registry()
    assert first is not second
    assert first.list_names() == second.list_names()
    for name in (
        "momentum_20d",
        "momentum_60d",
        "ep_ttm",
        "roe_ttm",
    ):
        assert first.contains(name)
    assert create_default_registry().list_names() == before


@pytest.mark.parametrize("factor_name", ["momentum_20d", "ep_ttm", "roe_ttm"])
def test_registered_factor_categories_run(
    tmp_path: Path, factor_name: str
) -> None:
    paths = _write_panels(tmp_path / "inputs")
    run_dir = tmp_path / f"run-{factor_name}"
    run_dir.mkdir()
    result = FactorResearchPipelineExecutor(
        _research_config(paths, factor_names=(factor_name,)),
        project_root=tmp_path,
    ).execute(run_dir)
    assert result.factor_names == (factor_name,)
    assert result.table_shapes["raw_factor_panel"][0] == 40


def test_unknown_factor_is_reported_before_g2(tmp_path: Path) -> None:
    paths = _write_panels(tmp_path / "inputs")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(KeyError, match="unknown_factor"):
        FactorResearchPipelineExecutor(
            _research_config(paths, factor_names=("unknown_factor",)),
            project_root=tmp_path,
        ).execute(run_dir)
    assert not (run_dir / "factor_research").exists()


def test_successful_execution_records_shapes_metadata_and_verified_artifact(
    tmp_path: Path,
) -> None:
    paths = _write_panels(tmp_path / "inputs")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    config = _research_config(paths, artifact_subdir="research/factors")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = FactorResearchPipelineExecutor(
        config, project_root=tmp_path
    ).execute(run_dir, metadata={"caller": "pytest"})

    artifact_dir = run_dir / "research" / "factors"
    assert result.artifact_dir == str(artifact_dir.resolve())
    assert result.factor_names == NAMES
    assert result.composition_method == "equal"
    assert result.input_shapes["factor_input"] == {"rows": 300, "columns": 5}
    assert result.input_shapes["score_panel"] == {"rows": 40, "columns": 2}
    assert result.table_shapes["forward_returns"][0] == 40
    assert (artifact_dir / "manifest.json").is_file()
    assert (artifact_dir / "tables").is_dir()
    report = FactorResearchArtifactStore(config.artifacts).verify(artifact_dir)
    assert report["valid"] is True
    manifest = result.manifest
    assert manifest["runner_config"]["research_config"]["factor_names"] == list(NAMES)
    assert manifest["metadata"]["caller"] == "pytest"
    assert manifest["metadata"]["pipeline_stage"] == "factor_research"
    assert manifest["metadata"]["input_shapes"] == result.input_shapes
    assert "DataFrame" not in repr(result.to_dict())
    assert {name: _sha256(path) for name, path in paths.items()} == hashes

    loaded = FactorResearchArtifactStore(config.artifacts).load_tables(artifact_dir)
    for frame in loaded.values():
        numeric = frame.select_dtypes(include="number")
        if not numeric.empty:
            assert not np.isinf(numeric.to_numpy(dtype=float)).any()


def test_artifact_overwrite_and_failure_cleanup_rules(tmp_path: Path) -> None:
    paths = _write_panels(tmp_path / "inputs")
    config = _research_config(paths)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    executor = FactorResearchPipelineExecutor(config, project_root=tmp_path)
    executor.execute(run_dir)
    with pytest.raises(FileExistsError):
        executor.execute(run_dir)
    assert (run_dir / "factor_research" / "manifest.json").is_file()

    bad_run = tmp_path / "bad-run"
    bad_run.mkdir()
    bad_scores = pd.DataFrame({"wrong": [1]})
    bad_scores.to_parquet(paths["score_panel"], index=False)
    with pytest.raises(RuntimeError, match="factor research execution failed"):
        FactorResearchPipelineExecutor(
            _research_config(paths), project_root=tmp_path
        ).execute(bad_run)
    assert not (bad_run / "factor_research").exists()


def test_repeated_runs_are_independent_and_equivalent(tmp_path: Path) -> None:
    paths = _write_panels(tmp_path / "inputs")
    executor = FactorResearchPipelineExecutor(
        _research_config(paths), project_root=tmp_path
    )
    first_run = tmp_path / "run-1"
    second_run = tmp_path / "run-2"
    first_run.mkdir()
    second_run.mkdir()
    first = executor.execute(first_run)
    second = executor.execute(second_run)
    assert first.table_shapes == second.table_shapes
    assert first.input_shapes == second.input_shapes
    assert first.artifact_dir != second.artifact_dir
    assert (first_run / "factor_research").is_dir()
    assert (second_run / "factor_research").is_dir()


def test_run_pipeline_disabled_preserves_original_summary_and_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_ready_data_manager(monkeypatch)

    class ForbiddenExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("disabled runner must not instantiate executor")

    monkeypatch.setattr(
        "src.pipeline.runner.FactorResearchPipelineExecutor",
        ForbiddenExecutor,
    )
    summary = run_pipeline(_pipeline_config(tmp_path / "output"))
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
    run_dir = Path(summary["run_dir"])
    assert not (run_dir / "factor_research").exists()
    assert (run_dir / "config_snapshot.yaml").is_file()
    assert (run_dir / "run_info.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    assert run_info["status"] == "succeeded"
    assert run_info["cache_status"] == "ready"


def test_run_pipeline_enabled_uses_one_existing_run_and_returns_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_ready_data_manager(monkeypatch)
    paths = _write_panels(tmp_path / "inputs")
    config = _pipeline_config(
        tmp_path / "output",
        _research_config(paths),
    )
    summary = run_pipeline(config)
    runs = list((tmp_path / "output" / "runs").iterdir())
    assert len(runs) == 1
    run_dir = Path(summary["run_dir"])
    assert run_dir == runs[0]
    assert summary["factor_research"]["enabled"] is True
    assert summary["factor_research"]["artifact_dir"] == str(
        (run_dir / "factor_research").resolve()
    )
    assert "raw_factor_panel" in summary["factor_research"]["table_shapes"]
    assert "FactorResearchResult" not in repr(summary)
    assert summary["required_start_date"] == "2013-01-01"
    assert summary["cache_status"] == "ready"
    assert FactorResearchArtifactStore().verify(
        run_dir / "factor_research"
    )["valid"]


def test_run_pipeline_research_failure_propagates_without_second_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_ready_data_manager(monkeypatch)
    paths = _write_panels(tmp_path / "inputs")
    paths["factor_input"] = tmp_path / "missing.parquet"
    config = _pipeline_config(
        tmp_path / "output",
        _research_config(paths),
    )
    with pytest.raises(FileNotFoundError, match="factor_input"):
        run_pipeline(config)
    runs = list((tmp_path / "output" / "runs").iterdir())
    assert len(runs) == 1
    assert not (runs[0] / "factor_research").exists()


def test_pipeline_config_from_dict_runs_complete_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_ready_data_manager(monkeypatch)
    paths = _write_panels(tmp_path / "inputs")
    config = _pipeline_config(tmp_path / "output", _research_config(paths))
    parsed = PipelineConfig.from_dict(config.to_dict())
    summary = run_pipeline(parsed)
    artifact_dir = Path(summary["factor_research"]["artifact_dir"])
    assert artifact_dir.is_relative_to(tmp_path)
    assert FactorResearchArtifactStore().verify(artifact_dir)["valid"] is True


def test_pipeline_public_execution_exports_remain_available() -> None:
    from src.pipeline import ExperimentManager, run_pipeline as exported_runner

    assert FactorResearchExecutionResult is not None
    assert FactorResearchPipelineExecutor is not None
    assert PipelineConfig is not None
    assert ExperimentManager is not None
    assert callable(exported_runner)
