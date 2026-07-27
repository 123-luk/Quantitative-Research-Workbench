"""V4-E3 CLI, example YAML, and documentation consistency tests."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from src.factors.research_pipeline import FactorResearchConfig
from src.ml import MLExperimentConfig
from src.pipeline import (
    FactorResearchPipelineConfig,
    MLExperimentPipelineConfig,
    ModelingPanelPipelineExecutionError,
    PipelineConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "run_pipeline.py"
EXAMPLE = PROJECT_ROOT / "config" / "modeling_panel_pipeline.example.yaml"
DOC = PROJECT_ROOT / "docs" / "05_modeling_panel_pipeline.md"
README = PROJECT_ROOT / "README.md"


def _load_cli():
    spec = importlib.util.spec_from_file_location("modeling_panel_cli", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _example_mapping() -> dict[str, object]:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return deepcopy(raw)


def _write_yaml(path: Path, values: object) -> Path:
    path.write_text(
        yaml.safe_dump(values, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _modeling_summary() -> dict[str, object]:
    return {
        "enabled": True,
        "source_mode": "files",
        "artifact_dir": "run/modeling_panel",
        "panel_path": "run/modeling_panel/modeling_panel.parquet",
        "manifest_path": "run/modeling_panel/manifest.json",
        "feature_names": ["factor_a", "factor_b"],
        "label_column": "forward_return",
        "input_factor_rows": 10,
        "input_return_rows": 10,
        "output_rows": 10,
        "warnings": [],
    }


def _base_summary() -> dict[str, object]:
    return {
        "status": "ready",
        "run_dir": "run",
        "required_start_date": "2022-12-01",
        "required_end_date": "2024-12-31",
        "cache_status": "ready",
        "missing_ranges": {},
        "strategy_name": "modeling_panel_example",
        "stock_pool": "hs300",
        "modeling_panel": _modeling_summary(),
    }


def test_example_yaml_is_safe_direct_schema_and_roundtrips() -> None:
    assert EXAMPLE.is_file()
    text = EXAMPLE.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    assert isinstance(raw, dict)
    config = PipelineConfig.from_dict(raw)
    assert config.modeling_panel.enabled is True
    assert config.modeling_panel.source.mode == "files"
    assert config.modeling_panel.output.save_artifact is True
    assert config.modeling_panel.output.artifact_subdir == "modeling_panel"
    assert config.factor_research.enabled is False
    assert config.ml_experiment.enabled is False
    assert config.ml_experiment.panel_path is None
    assert PipelineConfig.from_dict(config.to_dict()) == config

    lowered = text.lower()
    for forbidden in (
        "c:\\users",
        "/users/",
        "e:\\",
        "http://",
        "https://",
        "password:",
        "private_key:",
        "tushare_token:",
    ):
        assert forbidden not in lowered


def test_cli_help_uses_existing_config_interface_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    monkeypatch.setattr(
        module,
        "run_pipeline",
        lambda config: (_ for _ in ()).throw(
            AssertionError("help must not execute Pipeline")
        ),
    )
    before = set(tmp_path.iterdir())
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config" in help_text
    assert "modeling_panel" in help_text
    assert set(tmp_path.iterdir()) == before
    assert all(
        flag not in help_text
        for flag in (
            "--enable-modeling-panel",
            "--modeling-panel-source",
            "--ml-use-modeling-panel",
        )
    )


def test_cli_valid_direct_yaml_calls_pipeline_once_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    values = _example_mapping()
    values["output_dir"] = str(tmp_path / "never-created")
    config_path = _write_yaml(tmp_path / "valid.yaml", values)
    before = config_path.read_bytes()
    calls: list[PipelineConfig] = []

    def fake_run(config: PipelineConfig) -> dict[str, object]:
        calls.append(config)
        return _base_summary()

    monkeypatch.setattr(module, "run_pipeline", fake_run)
    assert module.main(["--config", str(config_path), "--json"]) == 0
    assert len(calls) == 1
    assert isinstance(calls[0], PipelineConfig)
    assert calls[0].modeling_panel.enabled is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["modeling_panel"]["panel_path"].endswith(
        "modeling_panel.parquet"
    )
    assert payload["modeling_panel"]["feature_names"] == [
        "factor_a",
        "factor_b",
    ]
    assert config_path.read_bytes() == before
    assert not (tmp_path / "never-created").exists()


def test_cli_factor_research_modeling_ml_chain_is_config_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli()
    values = _example_mapping()
    values["factor_research"] = FactorResearchPipelineConfig(
        enabled=True,
        factor_input_path="factor.parquet",
        score_panel_path="score.parquet",
        price_panel_path="price.parquet",
        research=FactorResearchConfig(
            factor_names=("factor_a",),
            composition_method="equal",
        ),
    ).to_dict()
    modeling = values["modeling_panel"]
    assert isinstance(modeling, dict)
    modeling["source"] = {
        "mode": "factor_research",
        "factor_panel_path": None,
        "forward_returns_path": None,
    }
    experiment = MLExperimentConfig.from_dict(
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
    values["ml_experiment"] = MLExperimentPipelineConfig(
        enabled=True,
        panel_path=None,
        experiment=experiment,
    ).to_dict()
    path = _write_yaml(tmp_path / "chain.yaml", values)
    calls: list[PipelineConfig] = []

    def fake_run(config: PipelineConfig) -> dict[str, object]:
        calls.append(config)
        return _base_summary()

    monkeypatch.setattr(module, "run_pipeline", fake_run)
    assert module.main(["--config", str(path), "--json"]) == 0
    assert len(calls) == 1
    config = calls[0]
    assert config.factor_research.enabled is True
    assert config.modeling_panel.source.mode == "factor_research"
    assert config.ml_experiment.enabled is True
    assert config.ml_experiment.panel_path is None


@pytest.mark.parametrize(
    "case",
    [
        "research_disabled",
        "ml_conflict",
        "ml_missing",
        "missing_factor",
        "missing_returns",
        "unknown",
        "unsafe_subdir",
        "compression",
        "enabled_type",
    ],
)
def test_cli_invalid_modeling_configs_never_run_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    module = _load_cli()
    values = _example_mapping()
    modeling = values["modeling_panel"]
    assert isinstance(modeling, dict)
    source = modeling["source"]
    output = modeling["output"]
    assert isinstance(source, dict)
    assert isinstance(output, dict)
    if case == "research_disabled":
        source["mode"] = "factor_research"
        source["factor_panel_path"] = None
        source["forward_returns_path"] = None
    elif case == "ml_conflict":
        values["ml_experiment"] = {
            "enabled": True,
            "panel_path": "direct.parquet",
            "experiment": {
                "dataset": {"label_col": "forward_return"},
                "walk_forward": {
                    "train_window_periods": 2,
                    "validation_periods": 2,
                },
                "training": {"model_name": "ridge", "model_params": {}},
                "evaluation": {"minimum_cross_section_size": 3},
            },
        }
    elif case == "ml_missing":
        modeling["enabled"] = False
        values["ml_experiment"] = {
            "enabled": True,
            "panel_path": None,
            "experiment": {
                "dataset": {"label_col": "forward_return"},
                "walk_forward": {
                    "train_window_periods": 2,
                    "validation_periods": 2,
                },
                "training": {"model_name": "ridge", "model_params": {}},
                "evaluation": {"minimum_cross_section_size": 3},
            },
        }
    elif case == "missing_factor":
        source["factor_panel_path"] = None
    elif case == "missing_returns":
        source["forward_returns_path"] = None
    elif case == "unknown":
        modeling["invented"] = True
    elif case == "unsafe_subdir":
        output["artifact_subdir"] = "../escape"
    elif case == "compression":
        output["parquet_compression"] = "gzip"
    else:
        modeling["enabled"] = 1
    path = _write_yaml(tmp_path / f"{case}.yaml", values)
    calls = 0

    def forbidden_run(config: PipelineConfig) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(module, "run_pipeline", forbidden_run)
    try:
        exit_code = module.main(["--config", str(path)])
    except (TypeError, ValueError):
        exit_code = 1
    assert exit_code != 0
    assert calls == 0


@pytest.mark.parametrize("case", ["missing", "syntax", "list"])
def test_cli_yaml_loading_failures_never_run_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    module = _load_cli()
    path = tmp_path / f"{case}.yaml"
    if case == "syntax":
        path.write_text("modeling_panel: [", encoding="utf-8")
    elif case == "list":
        path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    calls = 0

    def forbidden_run(config: PipelineConfig) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(module, "run_pipeline", forbidden_run)
    with pytest.raises((OSError, ValueError, yaml.YAMLError)):
        module.main(["--config", str(path)])
    assert calls == 0


def test_cli_modeling_execution_failure_is_concise_and_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    path = _write_yaml(tmp_path / "valid.yaml", _example_mapping())
    calls = 0

    def failing_run(config: PipelineConfig) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise ModelingPanelPipelineExecutionError("injected\nfailure")

    monkeypatch.setattr(module, "run_pipeline", failing_run)
    assert module.main(["--config", str(path)]) == 4
    assert calls == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "Modeling Panel pipeline error: injected failure"
    ]
    assert "Traceback" not in captured.err
    assert not any(tmp_path.rglob("modeling_panel.parquet"))


def test_cli_scope_contains_no_stage_core_or_scanning_logic() -> None:
    source = CLI.read_text(encoding="utf-8")
    for forbidden in (
        "ModelingPanelBuilder",
        "ModelingPanelArtifactStore",
        "MLExperimentRunner",
        "pd.read_parquet",
        "read_parquet",
        ".glob(",
        ".rglob(",
        ".iterdir(",
        "os.walk",
        "latest",
        "mtime",
        "getmtime",
        "--enable-modeling-panel",
        "--modeling-panel-source",
        "--ml-use-modeling-panel",
    ):
        assert forbidden not in source


def test_readme_docs_and_example_contracts_are_consistent() -> None:
    readme = README.read_text(encoding="utf-8")
    docs = DOC.read_text(encoding="utf-8")
    example = EXAMPLE.read_text(encoding="utf-8")
    assert "modeling_panel_pipeline.example.yaml" in readme
    assert "05_modeling_panel_pipeline.md" in readme
    assert "modeling_panel_pipeline.example.yaml" in docs
    for term in (
        "files",
        "factor_research",
        "audit_and_drop",
        "`error`",
        "modeling_panel.parquet",
        "config.json",
        "audit.json",
        "manifest.json",
        "ModelingPanelArtifactStore",
        "--config",
        "artifact_subdir",
        "include_features",
        "exclude_features",
        "require_entry_after_signal",
        "allow_missing_labels",
    ):
        assert term in docs
    assert "scripts/run_pipeline.py" in docs
    assert "scripts/run_pipeline.py" in readme
    assert "modeling_panel:" in example
    assert "ml_experiment:" in example
    for false_claim in (
        "完全无泄漏",
        "自动模型选择已支持",
        "自动超参数搜索已支持",
        "SHAP 已支持",
        "支持实盘交易",
    ):
        assert false_claim not in docs
