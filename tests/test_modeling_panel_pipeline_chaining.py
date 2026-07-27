"""Tests for V4-E2 Pipeline stage chaining and explicit panel handoff."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import src.pipeline.runner as runner_module
from src.factors.research_pipeline import FactorResearchConfig
from src.ml import MLExperimentConfig
from src.pipeline import (
    FactorResearchPipelineConfig,
    MLExperimentPipelineConfig,
    MLPipelineExecutionError,
    ModelingPanelPipelineConfig,
    ModelingPanelPipelineExecutionError,
    PipelineConfig,
    run_pipeline,
)


class _ReadyDataManager:
    def prepare_data(self, config: object) -> dict[str, object]:
        return {"cache_status": "ready", "missing_ranges": {}}


class _FactorResult:
    def __init__(self, token: int) -> None:
        self.token = token

    def to_dict(self) -> dict[str, object]:
        return {"enabled": True, "token": self.token}


class _ModelResult:
    def __init__(self, panel_path: Path | None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.panel_path = panel_path

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "panel_path": (
                None if self.panel_path is None else str(self.panel_path)
            ),
        }


class _MLResult:
    def __init__(self, panel_path: Path | None) -> None:
        self.panel_path = panel_path

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": True,
            "model_name": "ridge",
            "panel_path": (
                None if self.panel_path is None else str(self.panel_path)
            ),
        }


def _experiment() -> MLExperimentConfig:
    return MLExperimentConfig.from_dict(
        {
            "dataset": {"label_col": "forward_return"},
            "walk_forward": {
                "train_window_periods": 2,
                "validation_periods": 2,
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


def _factor_config() -> FactorResearchPipelineConfig:
    return FactorResearchPipelineConfig(
        enabled=True,
        factor_input_path="factor.parquet",
        score_panel_path="score.parquet",
        price_panel_path="price.parquet",
        research=FactorResearchConfig(
            factor_names=("factor_a",),
            composition_method="equal",
        ),
    )


def _model_config(mode: str = "files") -> ModelingPanelPipelineConfig:
    source: dict[str, object] = {"mode": mode}
    if mode == "files":
        source.update(
            {
                "factor_panel_path": "factors.parquet",
                "forward_returns_path": "returns.parquet",
            }
        )
    return ModelingPanelPipelineConfig.from_dict(
        {"enabled": True, "source": source}
    )


def _ml_config(*, generated: bool) -> MLExperimentPipelineConfig:
    return MLExperimentPipelineConfig(
        enabled=True,
        panel_path=None if generated else "direct.parquet",
        experiment=_experiment(),
    )


def _config(
    output_dir: Path,
    *,
    factor: bool = False,
    modeling: str | None = None,
    ml: bool = False,
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
        factor_research=(
            _factor_config() if factor else FactorResearchPipelineConfig()
        ),
        modeling_panel=(
            _model_config(modeling)
            if modeling is not None
            else ModelingPanelPipelineConfig()
        ),
        ml_experiment=(
            _ml_config(generated=modeling is not None)
            if ml
            else MLExperimentPipelineConfig()
        ),
    )


def _ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "DataManager", _ReadyDataManager)


def test_all_disabled_preserves_existing_summary_omission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)

    class Forbidden:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("disabled stage must not construct executor")

    monkeypatch.setattr(runner_module, "FactorResearchPipelineExecutor", Forbidden)
    monkeypatch.setattr(runner_module, "ModelingPanelPipelineExecutor", Forbidden)
    monkeypatch.setattr(runner_module, "MLExperimentPipelineExecutor", Forbidden)
    summary = run_pipeline(_config(tmp_path / "output"))
    assert "factor_research" not in summary
    assert "modeling_panel" not in summary
    assert "ml_experiment" not in summary
    snapshot = yaml.safe_load(
        (Path(summary["run_dir"]) / "config_snapshot.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert snapshot["modeling_panel"]["enabled"] is False


def test_modeling_files_only_receives_none_and_adds_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    calls: list[tuple[Path, object]] = []

    class ModelingExecutor:
        def __init__(self, config: ModelingPanelPipelineConfig) -> None:
            assert config.source.mode == "files"

        def execute(
            self,
            run_dir: str | Path,
            *,
            factor_research_result: object = None,
        ) -> _ModelResult:
            run = Path(run_dir)
            panel = run / "modeling_panel" / "modeling_panel.parquet"
            panel.parent.mkdir()
            panel.write_bytes(b"panel")
            calls.append((run, factor_research_result))
            return _ModelResult(panel.resolve())

    monkeypatch.setattr(
        runner_module, "ModelingPanelPipelineExecutor", ModelingExecutor
    )
    summary = run_pipeline(
        _config(tmp_path / "output", modeling="files")
    )
    assert calls == [(Path(summary["run_dir"]), None)]
    assert summary["modeling_panel"]["enabled"] is True
    assert summary["modeling_panel"]["panel_path"] == str(
        (Path(summary["run_dir"]) / "modeling_panel" / "modeling_panel.parquet").resolve()
    )


def test_factor_modeling_ml_order_identity_override_and_same_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    events: list[tuple[str, Path]] = []
    factor_result = _FactorResult(7)
    received_research: list[object] = []
    received_override: list[Path | None] = []

    class FactorExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(
            self, run_dir: str | Path, *, metadata: object = None
        ) -> _FactorResult:
            events.append(("factor", Path(run_dir)))
            return factor_result

    class ModelingExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(
            self,
            run_dir: str | Path,
            *,
            factor_research_result: object = None,
        ) -> _ModelResult:
            run = Path(run_dir)
            events.append(("modeling", run))
            received_research.append(factor_research_result)
            panel = run / "modeling_panel" / "modeling_panel.parquet"
            panel.parent.mkdir()
            panel.write_bytes(b"panel")
            return _ModelResult(panel.resolve())

    class MLExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(
            self,
            run_dir: str | Path,
            *,
            panel_path_override: str | Path | None = None,
        ) -> _MLResult:
            run = Path(run_dir)
            events.append(("ml", run))
            override = (
                None
                if panel_path_override is None
                else Path(panel_path_override)
            )
            received_override.append(override)
            return _MLResult(override)

    monkeypatch.setattr(runner_module, "FactorResearchPipelineExecutor", FactorExecutor)
    monkeypatch.setattr(runner_module, "ModelingPanelPipelineExecutor", ModelingExecutor)
    monkeypatch.setattr(runner_module, "MLExperimentPipelineExecutor", MLExecutor)
    config = _config(
        tmp_path / "output",
        factor=True,
        modeling="factor_research",
        ml=True,
    )
    before = config.to_dict()
    summary = run_pipeline(config)
    run_dir = Path(summary["run_dir"])
    assert [name for name, _ in events] == ["factor", "modeling", "ml"]
    assert all(path == run_dir for _, path in events)
    assert received_research == [factor_result]
    expected_panel = (
        run_dir / "modeling_panel" / "modeling_panel.parquet"
    ).resolve()
    assert received_override == [expected_panel]
    assert summary["ml_experiment"]["panel_path"] == str(expected_panel)
    assert config.to_dict() == before


def test_factor_plus_files_modeling_never_passes_research_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    received: list[object] = []

    class FactorExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(
            self, run_dir: str | Path, *, metadata: object = None
        ) -> _FactorResult:
            return _FactorResult(1)

    class ModelingExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(
            self,
            run_dir: str | Path,
            *,
            factor_research_result: object = None,
        ) -> _ModelResult:
            received.append(factor_research_result)
            panel = Path(run_dir) / "panel.parquet"
            panel.write_bytes(b"panel")
            return _ModelResult(panel.resolve())

    monkeypatch.setattr(runner_module, "FactorResearchPipelineExecutor", FactorExecutor)
    monkeypatch.setattr(runner_module, "ModelingPanelPipelineExecutor", ModelingExecutor)
    run_pipeline(
        _config(tmp_path / "output", factor=True, modeling="files")
    )
    assert received == [None]


def test_direct_ml_mode_passes_no_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    received: list[object] = []

    class MLExecutor:
        def __init__(self, config: MLExperimentPipelineConfig) -> None:
            assert config.panel_path == "direct.parquet"

        def execute(
            self,
            run_dir: str | Path,
            *,
            panel_path_override: object = None,
        ) -> _MLResult:
            received.append(panel_path_override)
            return _MLResult(Path("direct.parquet"))

    monkeypatch.setattr(runner_module, "MLExperimentPipelineExecutor", MLExecutor)
    summary = run_pipeline(_config(tmp_path / "output", ml=True))
    assert received == [None]
    assert "modeling_panel" not in summary


def test_factor_failure_stops_modeling_and_ml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)

    class FailingFactor:
        def __init__(self, config: object) -> None:
            pass

        def execute(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("factor failed")

    class Forbidden:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("downstream stage must not run")

    monkeypatch.setattr(runner_module, "FactorResearchPipelineExecutor", FailingFactor)
    monkeypatch.setattr(runner_module, "ModelingPanelPipelineExecutor", Forbidden)
    monkeypatch.setattr(runner_module, "MLExperimentPipelineExecutor", Forbidden)
    with pytest.raises(RuntimeError, match="factor failed"):
        run_pipeline(
            _config(
                tmp_path / "output",
                factor=True,
                modeling="factor_research",
                ml=True,
            )
        )


def test_modeling_failure_or_missing_panel_stops_ml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)

    class ForbiddenML:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("ML must not run")

    class FailingModel:
        def __init__(self, config: object) -> None:
            pass

        def execute(self, *args: object, **kwargs: object) -> object:
            raise ModelingPanelPipelineExecutionError("model failed")

    monkeypatch.setattr(runner_module, "ModelingPanelPipelineExecutor", FailingModel)
    monkeypatch.setattr(runner_module, "MLExperimentPipelineExecutor", ForbiddenML)
    with pytest.raises(ModelingPanelPipelineExecutionError, match="model failed"):
        run_pipeline(
            _config(tmp_path / "failed", modeling="files", ml=True)
        )

    class MissingPanelModel:
        def __init__(self, config: object) -> None:
            pass

        def execute(self, *args: object, **kwargs: object) -> _ModelResult:
            return _ModelResult(None)

    monkeypatch.setattr(
        runner_module, "ModelingPanelPipelineExecutor", MissingPanelModel
    )
    with pytest.raises(ModelingPanelPipelineExecutionError, match="panel_path"):
        run_pipeline(
            _config(tmp_path / "missing", modeling="files", ml=True)
        )


def test_ml_failure_preserves_published_modeling_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    published: list[Path] = []

    class ModelingExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(self, run_dir: str | Path, **kwargs: object) -> _ModelResult:
            panel = Path(run_dir) / "modeling_panel" / "modeling_panel.parquet"
            panel.parent.mkdir()
            panel.write_bytes(b"keep")
            published.append(panel)
            return _ModelResult(panel.resolve())

    class FailingML:
        def __init__(self, config: object) -> None:
            pass

        def execute(self, *args: object, **kwargs: object) -> object:
            raise MLPipelineExecutionError("ml failed")

    monkeypatch.setattr(runner_module, "ModelingPanelPipelineExecutor", ModelingExecutor)
    monkeypatch.setattr(runner_module, "MLExperimentPipelineExecutor", FailingML)
    with pytest.raises(MLPipelineExecutionError, match="ml failed"):
        run_pipeline(
            _config(tmp_path / "output", modeling="files", ml=True)
        )
    assert len(published) == 1
    assert published[0].read_bytes() == b"keep"


def test_two_runs_use_only_current_results_and_distinct_run_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ready(monkeypatch)
    factor_results: list[_FactorResult] = []
    model_inputs: list[object] = []
    overrides: list[Path] = []

    class FactorExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(self, run_dir: str | Path, **kwargs: object) -> _FactorResult:
            result = _FactorResult(len(factor_results) + 1)
            factor_results.append(result)
            return result

    class ModelingExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(
            self,
            run_dir: str | Path,
            *,
            factor_research_result: object = None,
        ) -> _ModelResult:
            model_inputs.append(factor_research_result)
            panel = Path(run_dir) / "modeling_panel" / "modeling_panel.parquet"
            panel.parent.mkdir()
            panel.write_bytes(b"current")
            return _ModelResult(panel.resolve())

    class MLExecutor:
        def __init__(self, config: object) -> None:
            pass

        def execute(
            self,
            run_dir: str | Path,
            *,
            panel_path_override: str | Path | None = None,
        ) -> _MLResult:
            assert panel_path_override is not None
            path = Path(panel_path_override)
            overrides.append(path)
            return _MLResult(path)

    monkeypatch.setattr(runner_module, "FactorResearchPipelineExecutor", FactorExecutor)
    monkeypatch.setattr(runner_module, "ModelingPanelPipelineExecutor", ModelingExecutor)
    monkeypatch.setattr(runner_module, "MLExperimentPipelineExecutor", MLExecutor)
    output = tmp_path / "output"
    stale = output / "runs" / "older" / "modeling_panel"
    stale.mkdir(parents=True)
    (stale / "modeling_panel.parquet").write_bytes(b"stale")
    config = _config(
        output,
        factor=True,
        modeling="factor_research",
        ml=True,
    )
    first = run_pipeline(config)
    second = run_pipeline(config)
    first_run = Path(first["run_dir"])
    second_run = Path(second["run_dir"])
    assert first_run != second_run
    assert model_inputs == factor_results
    assert overrides[0].is_relative_to(first_run)
    assert overrides[1].is_relative_to(second_run)
    assert all(path != stale / "modeling_panel.parquet" for path in overrides)
