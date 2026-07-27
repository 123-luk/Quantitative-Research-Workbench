"""Contract tests for the opt-in ML options on the unified Pipeline CLI."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from src.ml import MLArtifactExistsError
from src.pipeline import (
    MLCLIConfigError,
    MLCLIError,
    MLExperimentPipelineConfig,
    MLPipelineArtifactError,
    MLPipelineConfigError,
    MLPipelineExecutionError,
    MLPipelineIntegrityError,
    MLPipelinePanelError,
    PipelineConfig,
    exit_code_for_ml_error,
    format_ml_human_summary,
    merge_ml_cli_overrides,
    parse_ml_model_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "run_pipeline.py"
EXAMPLE = PROJECT_ROOT / "config" / "ml_experiment.example.yaml"


def _load_cli():
    spec = importlib.util.spec_from_file_location("ml_pipeline_cli", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _experiment(*, importance: bool = False) -> dict[str, object]:
    return {
        "dataset": {"label_col": "forward_return"},
        "walk_forward": {
            "train_window_periods": 4,
            "validation_periods": 2,
            "window_type": "rolling",
            "retrain_frequency": 2,
            "embargo_periods": 1,
        },
        "training": {
            "model_name": "ridge",
            "model_params": {"alpha": 1.0, "tol": 0.001},
        },
        "evaluation": {"minimum_cross_section_size": 3},
        "permutation_importance": (
            {
                "scoring": "rmse",
                "n_repeats": 7,
                "random_state": 9,
                "permutation_scope": "within_trade_date",
            }
            if importance
            else None
        ),
    }


def _ml_config(
    *,
    enabled: bool = False,
    importance: bool = False,
    save_artifacts: bool = False,
) -> MLExperimentPipelineConfig:
    return MLExperimentPipelineConfig.from_dict(
        {
            "enabled": enabled,
            "panel_path": "yaml-panel.parquet",
            "save_artifacts": save_artifacts,
            "artifact_root": "yaml-artifacts",
            "experiment_id": "yaml-run" if save_artifacts else None,
            "parquet_compression": "zstd",
            "experiment": _experiment(importance=importance),
        }
    )


def _pipeline_config(ml: MLExperimentPipelineConfig) -> PipelineConfig:
    return PipelineConfig(
        backtest_start="2024-01-01",
        backtest_end="2024-12-31",
        train_years=1,
        max_lookback_months=1,
        stock_pool="test",
        benchmark="000300.SH",
        strategy_name="cli",
        selected_factors=[],
        rebalance_frequency="M",
        top_n=5,
        transaction_cost=0.001,
        data_root="data",
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        cache_dir="data/cache",
        output_dir="data/output",
        parquet_engine="auto",
        required_datasets=["daily"],
        ml_experiment=ml,
    )


def _base_summary() -> dict[str, object]:
    return {
        "status": "ready",
        "run_dir": "run",
        "required_start_date": "2022-12-01",
        "required_end_date": "2024-12-31",
        "cache_status": "ready",
        "missing_ranges": {},
        "strategy_name": "cli",
        "stock_pool": "test",
    }


def _ml_summary(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "enabled": True,
        "model_name": "ridge",
        "n_folds": 3,
        "n_prediction_rows": 24,
        "n_prediction_dates": 6,
        "mae": 0.01234567,
        "rmse": 0.02345678,
        "r2": 0.3456789,
        "r2_valid": True,
        "r2_invalid_reason": None,
        "pearson_ic_mean": 0.1234567,
        "rank_ic_mean": None,
        "permutation_importance_enabled": False,
        "permutation_importance_completed": False,
        "artifacts_saved": False,
        "artifact_dir": None,
    }
    values.update(updates)
    return values


def test_parse_model_params_preserves_json_scalars_and_is_defensive() -> None:
    raw = '{"i":1,"f":2.5,"b":true,"s":"x","n":null}'
    first = parse_ml_model_params(raw)
    second = parse_ml_model_params(raw)
    assert first == {"i": 1, "f": 2.5, "b": True, "s": "x", "n": None}
    assert first is not second
    first["i"] = 99
    assert second["i"] == 1
    assert parse_ml_model_params("{}") == {}


@pytest.mark.parametrize(
    "raw",
    [
        "",
        " ",
        "[]",
        '"text"',
        "1",
        "true",
        "null",
        "{",
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
    ],
)
def test_parse_model_params_rejects_invalid_or_non_object_json(
    raw: str,
) -> None:
    with pytest.raises(MLCLIConfigError, match="model_params"):
        parse_ml_model_params(raw)


def test_empty_merge_returns_equivalent_new_config() -> None:
    original = _ml_config()
    merged = merge_ml_cli_overrides(original, {})
    assert merged == original
    assert merged is not original


def test_merge_top_level_leaves_without_implicit_enabling() -> None:
    original = _ml_config()
    overrides = {
        "panel_path": "cli-panel.parquet",
        "artifact_root": "cli-artifacts/nested",
        "experiment_id": "cli-run",
        "parquet_compression": "snappy",
    }
    snapshot = deepcopy(overrides)
    merged = merge_ml_cli_overrides(original, overrides)
    assert merged.enabled is False
    assert merged.save_artifacts is False
    assert merged.panel_path == "cli-panel.parquet"
    assert merged.artifact_root == "cli-artifacts/nested"
    assert merged.experiment_id == "cli-run"
    assert merged.parquet_compression == "snappy"
    assert overrides == snapshot
    assert original.panel_path == "yaml-panel.parquet"


def test_model_and_evaluation_overrides_replace_only_requested_leaves() -> None:
    original = _ml_config()
    merged = merge_ml_cli_overrides(
        original,
        {
            "model_name": "elastic_net",
            "model_params": {"alpha": 0.2, "l1_ratio": 0.4},
            "minimum_cross_section_size": 5,
        },
    )
    assert merged.enabled is False
    assert merged.experiment is not None
    assert merged.experiment.training_config.model_name == "elastic_net"
    assert dict(merged.experiment.training_config.model_params) == {
        "alpha": 0.2,
        "l1_ratio": 0.4,
    }
    assert "tol" not in merged.experiment.training_config.model_params
    assert merged.experiment.evaluation_config.minimum_cross_section_size == 5
    assert (
        merged.experiment.walk_forward_config
        == original.experiment.walk_forward_config  # type: ignore[union-attr]
    )


def test_importance_enable_builds_defaults_and_leaf_overrides() -> None:
    merged = merge_ml_cli_overrides(
        _ml_config(),
        {
            "permutation_importance_enabled": True,
            "importance_repeats": 3,
            "importance_scoring": "mae",
        },
    )
    assert merged.experiment is not None
    options = merged.experiment.permutation_importance
    assert options is not None
    assert options.n_repeats == 3
    assert options.scoring == "mae"
    assert options.random_state == 42
    assert options.permutation_scope == "within_trade_date"


def test_importance_existing_options_preserve_random_state_and_scope() -> None:
    merged = merge_ml_cli_overrides(
        _ml_config(importance=True),
        {"importance_repeats": 2, "importance_scoring": "mae"},
    )
    assert merged.experiment is not None
    options = merged.experiment.permutation_importance
    assert options is not None
    assert (options.n_repeats, options.scoring) == (2, "mae")
    assert (options.random_state, options.permutation_scope) == (
        9,
        "within_trade_date",
    )


def test_importance_explicit_disable_removes_yaml_options() -> None:
    merged = merge_ml_cli_overrides(
        _ml_config(importance=True),
        {"permutation_importance_enabled": False},
    )
    assert merged.experiment is not None
    assert merged.experiment.permutation_importance is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"importance_repeats": 2},
        {"importance_scoring": "mae"},
        {
            "permutation_importance_enabled": False,
            "importance_repeats": 2,
        },
    ],
)
def test_importance_leaf_overrides_require_enabled_options(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(MLCLIConfigError, match="importance"):
        merge_ml_cli_overrides(_ml_config(), overrides)


def test_artifact_flags_obey_existing_config_dependencies() -> None:
    enabled = merge_ml_cli_overrides(
        _ml_config(),
        {"enabled": True, "save_artifacts": True, "experiment_id": "run"},
    )
    assert enabled.enabled is enabled.save_artifacts is True
    disabled = merge_ml_cli_overrides(
        _ml_config(enabled=True, save_artifacts=True),
        {"save_artifacts": False},
    )
    assert disabled.enabled is True
    assert disabled.save_artifacts is False
    with pytest.raises(MLCLIConfigError):
        merge_ml_cli_overrides(
            _ml_config(), {"save_artifacts": True, "experiment_id": "run"}
        )


def test_merge_rejects_unknown_types_and_missing_experiment() -> None:
    with pytest.raises(MLCLIConfigError, match="unknown"):
        merge_ml_cli_overrides(_ml_config(), {"unknown": 1})
    with pytest.raises(MLCLIConfigError, match="config"):
        merge_ml_cli_overrides(object(), {})  # type: ignore[arg-type]
    with pytest.raises(MLCLIConfigError, match="Mapping"):
        merge_ml_cli_overrides(_ml_config(), [])  # type: ignore[arg-type]
    with pytest.raises(MLCLIConfigError, match="experiment"):
        merge_ml_cli_overrides(
            MLExperimentPipelineConfig(),
            {"model_name": "ridge"},
        )


def test_merged_config_is_json_safe_and_revalidated() -> None:
    merged = merge_ml_cli_overrides(
        _ml_config(), {"model_params": {"alpha": 2.0}}
    )
    json.dumps(merged.to_dict(), allow_nan=False)
    with pytest.raises(MLCLIConfigError) as captured:
        merge_ml_cli_overrides(
            _ml_config(), {"artifact_root": "../escape"}
        )
    assert isinstance(captured.value.__cause__, MLPipelineConfigError)


def test_parser_defaults_and_existing_options() -> None:
    module = _load_cli()
    args = module.parse_args(
        [
            "--backtest-start",
            "2024-02-01",
            "--top-n",
            "7",
            "--json",
        ]
    )
    assert args.backtest_start == "2024-02-01"
    assert args.top_n == 7
    assert args.json is True
    for name in (
        "ml_enabled",
        "ml_panel",
        "ml_model",
        "ml_model_params",
        "ml_permutation_importance",
        "ml_importance_repeats",
        "ml_importance_scoring",
        "ml_min_cross_section_size",
        "ml_save_artifacts",
        "ml_artifact_root",
        "ml_experiment_id",
        "ml_parquet_compression",
    ):
        assert getattr(args, name) is None


def test_parser_all_ml_options_and_sparse_builder() -> None:
    module = _load_cli()
    args = module.parse_args(
        [
            "--ml",
            "--ml-panel",
            "panel.parquet",
            "--ml-model",
            "ridge",
            "--ml-model-params",
            '{"alpha":2.0}',
            "--ml-permutation-importance",
            "--ml-importance-repeats",
            "4",
            "--ml-importance-scoring",
            "mae",
            "--ml-min-cross-section-size",
            "6",
            "--ml-save-artifacts",
            "--ml-artifact-root",
            "ml",
            "--ml-experiment-id",
            "demo",
            "--ml-parquet-compression",
            "snappy",
        ]
    )
    overrides = module.build_ml_cli_overrides(args)
    assert overrides == {
        "enabled": True,
        "panel_path": "panel.parquet",
        "model_name": "ridge",
        "model_params": {"alpha": 2.0},
        "permutation_importance_enabled": True,
        "importance_repeats": 4,
        "importance_scoring": "mae",
        "minimum_cross_section_size": 6,
        "save_artifacts": True,
        "artifact_root": "ml",
        "experiment_id": "demo",
        "parquet_compression": "snappy",
    }
    assert module.build_ml_cli_overrides(module.parse_args([])) == {}


@pytest.mark.parametrize(
    "argv",
    [
        ["--ml", "--no-ml"],
        [
            "--ml-permutation-importance",
            "--no-ml-permutation-importance",
        ],
        ["--ml-save-artifacts", "--no-ml-save-artifacts"],
        ["--ml-importance-scoring", "invalid"],
        ["--ml-parquet-compression", "gzip"],
    ],
)
def test_parser_rejects_conflicting_or_invalid_choices(
    argv: list[str],
) -> None:
    with pytest.raises(SystemExit) as captured:
        _load_cli().parse_args(argv)
    assert captured.value.code == 2


def test_human_summary_none_disabled_valid_and_invalid_metrics() -> None:
    assert format_ml_human_summary(None) == ()
    assert format_ml_human_summary({"enabled": False}) == ()
    valid = format_ml_human_summary(_ml_summary())
    text = "\n".join(valid)
    assert "ML model: ridge" in text
    assert "ML R²: 0.345679" in text
    assert "ML RankIC mean: N/A" in text
    assert "ML artifacts: not saved" in text
    assert all(
        forbidden not in text
        for forbidden in ("predictions", "model_params", "manifest")
    )
    invalid = format_ml_human_summary(
        _ml_summary(
            r2=None,
            r2_valid=False,
            r2_invalid_reason="constant target",
            artifacts_saved=True,
            artifact_dir="run/ml/demo",
            permutation_importance_completed=True,
        )
    )
    invalid_text = "\n".join(invalid)
    assert "N/A (constant target)" in invalid_text
    assert "completed" in invalid_text
    assert "run/ml/demo" in invalid_text


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (MLCLIConfigError("bad"), 2),
        (MLPipelineConfigError("bad"), 2),
        (MLPipelinePanelError("bad"), 3),
        (MLPipelineExecutionError("bad"), 4),
        (MLPipelineIntegrityError("bad"), 4),
        (MLPipelineArtifactError("bad"), 6),
        (RuntimeError("bad"), 1),
    ],
)
def test_exit_code_mapping(error: BaseException, expected: int) -> None:
    assert exit_code_for_ml_error(error) == expected


def test_artifact_exists_cause_and_cycle_are_safe() -> None:
    exists = MLArtifactExistsError("exists")
    wrapped = MLPipelineArtifactError("artifact failed")
    wrapped.__cause__ = exists
    assert exit_code_for_ml_error(wrapped) == 5
    first = MLPipelineExecutionError("first")
    second = RuntimeError("second")
    first.__context__ = second
    second.__context__ = first
    assert exit_code_for_ml_error(first) == 4


def test_build_output_disabled_is_legacy_and_enabled_is_compact() -> None:
    module = _load_cli()
    config = _pipeline_config(_ml_config())
    disabled = module.build_output(config, _base_summary())
    assert "ml_experiment" not in disabled
    enabled_summary = _base_summary()
    enabled_summary["ml_experiment"] = _ml_summary()
    enabled = module.build_output(config, enabled_summary)
    assert enabled["ml_experiment"]["model_name"] == "ridge"
    serialized = json.dumps(enabled, allow_nan=False)
    assert "model_params" not in serialized
    assert "predictions" not in serialized
    assert "manifest" not in serialized


def test_print_human_disabled_output_is_unchanged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    output = module.build_output(
        _pipeline_config(_ml_config()),
        _base_summary(),
    )
    module.print_human_summary(output)
    assert capsys.readouterr().out.splitlines() == [
        "Pipeline status: ready",
        "Run directory: run",
        "Required start: 2022-12-01",
        "Required end: 2024-12-31",
        "Factor research enabled: false",
    ]


def test_main_preserves_yaml_without_ml_flags_and_applies_cli_leaves(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    yaml_config = _pipeline_config(_ml_config())
    calls: list[object] = []

    def fake_from_yaml(*args: object, **kwargs: object) -> PipelineConfig:
        calls.append((args, kwargs))
        return PipelineConfig.from_dict(yaml_config.to_dict())

    def fake_run(config: PipelineConfig) -> dict[str, object]:
        calls.append(config)
        summary = _base_summary()
        if config.ml_experiment.enabled:
            summary["ml_experiment"] = _ml_summary(
                model_name=config.ml_experiment.experiment.training_config.model_name
            )
        return summary

    monkeypatch.setattr(module.PipelineConfig, "from_yaml", fake_from_yaml)
    monkeypatch.setattr(module, "run_pipeline", fake_run)
    assert module.main([]) == 0
    first_config = calls[-1]
    assert isinstance(first_config, PipelineConfig)
    assert first_config.ml_experiment.to_dict() == yaml_config.ml_experiment.to_dict()
    capsys.readouterr()

    calls.clear()
    assert (
        module.main(
            [
                "--ml",
                "--ml-panel",
                "cli.parquet",
                "--ml-model",
                "elastic_net",
                "--ml-model-params",
                '{"alpha":0.2,"l1_ratio":0.4}',
                "--backtest-start",
                "2024-02-01",
                "--json",
            ]
        )
        == 0
    )
    assert len([item for item in calls if isinstance(item, tuple)]) == 1
    configured = calls[-1]
    assert isinstance(configured, PipelineConfig)
    assert configured.backtest_start == "2024-01-01"
    assert configured.ml_experiment.enabled is True
    assert configured.ml_experiment.panel_path == "cli.parquet"
    training = configured.ml_experiment.experiment.training_config
    assert training.model_name == "elastic_net"
    assert dict(training.model_params) == {"alpha": 0.2, "l1_ratio": 0.4}
    payload = json.loads(capsys.readouterr().out)
    assert payload["ml_experiment"]["model_name"] == "elastic_net"


def test_main_typed_ml_error_is_one_line_and_non_ml_propagates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    config = _pipeline_config(_ml_config())
    monkeypatch.setattr(
        module.PipelineConfig,
        "from_yaml",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        module,
        "run_pipeline",
        lambda config: (_ for _ in ()).throw(
            MLPipelinePanelError("bad\npanel")
        ),
    )
    assert module.main([]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "ML pipeline error: bad panel"
    ]
    assert "Traceback" not in captured.err

    monkeypatch.setattr(
        module,
        "run_pipeline",
        lambda config: (_ for _ in ()).throw(RuntimeError("legacy")),
    )
    with pytest.raises(RuntimeError, match="legacy"):
        module.main([])


def test_example_yaml_is_safe_disabled_and_json_roundtrippable() -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    config = PipelineConfig.from_yaml(EXAMPLE)
    assert config.ml_experiment.enabled is False
    assert config.ml_experiment.save_artifacts is False
    assert config.ml_experiment.artifact_root == "ml_artifacts"
    assert config.ml_experiment.panel_path == (
        "data/processed/ml_modeling_panel.parquet"
    )
    assert config.ml_experiment.experiment is not None
    experiment = config.ml_experiment.experiment
    assert experiment.training_config.model_name == "ridge"
    assert dict(experiment.training_config.model_params) == {"alpha": 1.0}
    assert experiment.permutation_importance is None
    json.dumps(config.to_dict(), allow_nan=False)
    lowered = text.lower()
    for forbidden in (
        "c:\\users",
        "/users/",
        "e:\\",
        "token:",
        "password:",
        "lightgbm",
        "xgboost",
    ):
        assert forbidden not in lowered
