"""V5-E3 canonical CLI, YAML example, and documentation tests."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import src.pipeline.runner as runner_module
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
    ModelingPanelPipelineConfig,
    PipelineConfig,
    PredictionSourceConfig,
    SignalPipelineConfig,
)
from src.signals import SignalArtifactStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "run_pipeline.py"
EXAMPLE = PROJECT_ROOT / "config" / "signal_holdings_pipeline.example.yaml"
DOC = PROJECT_ROOT / "docs" / "08_signal_holdings_pipeline.md"
README = PROJECT_ROOT / "README.md"


def _load_cli():
    spec = importlib.util.spec_from_file_location("signal_holdings_cli", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, values: object) -> Path:
    path.write_text(
        yaml.safe_dump(values, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


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
    for date_number, trade_date in enumerate(pd.date_range("2024-04-01", periods=12)):
        for stock_number in range(12):
            value = float(date_number + stock_number / 10)
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"C{stock_number:03d}",
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
    root = tmp_path_factory.mktemp("cli-native-ml")
    result = MLExperimentRunner().run(_panel(), _experiment())
    return MLExperimentArtifactStore().write(
        result, MLArtifactConfig(root, "cli-source")
    ).experiment_dir


def _base(output_dir: Path, *, top_n: int = 5) -> PipelineConfig:
    return PipelineConfig(
        backtest_start="2024-01-01",
        backtest_end="2024-12-31",
        train_years=1,
        max_lookback_months=1,
        stock_pool="hs300",
        benchmark="000300.SH",
        strategy_name="cli_v5",
        selected_factors=[],
        rebalance_frequency="M",
        top_n=top_n,
        transaction_cost=0.001,
        data_root="data",
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        cache_dir="data/cache",
        output_dir=str(output_dir),
        parquet_engine="auto",
        required_datasets=[],
        modeling_panel=ModelingPanelPipelineConfig(),
    )


def test_canonical_example_uses_real_loader_and_roundtrips() -> None:
    module = _load_cli()
    text = EXAMPLE.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    before = deepcopy(raw)
    config = module.load_pipeline_config(EXAMPLE, {})
    assert raw == before
    assert config.modeling_panel.enabled is True
    assert config.ml_experiment.enabled is True
    assert config.ml_experiment.save_artifacts is True
    assert config.signal.enabled is True
    assert config.signal.source.mode == "ml"
    assert config.signal.source.artifact_dir is None
    assert config.signal.signal_direction == "descending"
    assert config.holdings.enabled is True
    assert config.holdings.top_n == config.top_n == 10
    assert config.holdings.insufficient_universe_policy == "error"
    assert config.holdings.weighting == "equal_weight"
    assert PipelineConfig.from_dict(config.to_dict()) == config
    lowered = text.lower()
    for forbidden in (
        "c:\\users",
        "/users/",
        "e:\\financial engineering",
        "password:",
        "private_key:",
        "tushare_token:",
    ):
        assert forbidden not in lowered


def test_grouped_and_old_yaml_loader_compatibility(tmp_path: Path) -> None:
    module = _load_cli()
    grouped = {
        "data": {
            "start_date": "20240101",
            "end_date": "20241231",
            "output_dir": "data/output",
            "required_datasets": [],
        },
        "pipeline": {
            "backtest_start": "2024-01-01",
            "backtest_end": "2024-12-31",
            "train_years": 1,
            "max_lookback_months": 1,
            "top_n": 10,
        },
        "factors": {"selected": []},
        "signal": {
            "enabled": True,
            "source": {
                "mode": "files",
                "artifact_dir": "path/to/native/ml/artifact",
            },
        },
        "holdings": {"enabled": True, "top_n": 10},
    }
    grouped_path = _write_yaml(tmp_path / "grouped.yaml", grouped)
    before = grouped_path.read_bytes()
    parsed = module.load_pipeline_config(grouped_path, {})
    assert parsed.signal.enabled and parsed.signal.source.mode == "files"
    assert parsed.holdings.enabled and parsed.holdings.top_n == 10
    assert grouped_path.read_bytes() == before

    old_grouped = module.load_pipeline_config(PROJECT_ROOT / "config" / "config.yaml", {})
    assert not old_grouped.signal.enabled and not old_grouped.holdings.enabled
    old_direct = module.load_pipeline_config(
        PROJECT_ROOT / "config" / "modeling_panel_pipeline.example.yaml", {}
    )
    assert not old_direct.signal.enabled and not old_direct.holdings.enabled


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"signal": {"enabled": True, "signal_direction": "sideways"}}, "signal_direction"),
        ({"holdings": {"enabled": True, "top_n": 0}}, "top_n"),
    ],
)
def test_invalid_v5_yaml_fails_before_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    message: str,
) -> None:
    module = _load_cli()
    values = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    values.update(changes)
    path = _write_yaml(tmp_path / "invalid.yaml", values)
    monkeypatch.setattr(
        module,
        "run_pipeline",
        lambda config: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    with pytest.raises((TypeError, ValueError), match=message):
        module.main(["--config", str(path)])


def test_legacy_and_holdings_top_n_conflict_and_same_value(tmp_path: Path) -> None:
    module = _load_cli()
    values = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    values["top_n"] = 20
    with pytest.raises(ValueError, match="conflicts"):
        module.load_pipeline_config(_write_yaml(tmp_path / "conflict.yaml", values), {})
    values["top_n"] = 10
    parsed = module.load_pipeline_config(_write_yaml(tmp_path / "same.yaml", values), {})
    assert parsed.top_n == parsed.holdings.top_n == 10


def test_help_has_no_new_signal_or_holdings_business_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config" in help_text
    for forbidden in (
        "--signal-direction",
        "--prediction-column",
        "--insufficient-universe-policy",
        "--weighting",
        "--signal-source",
        "--signal-artifact-dir",
        "--signal-enabled",
        "--holdings-enabled",
    ):
        assert forbidden not in help_text
    # Retained for V3/V4 backward compatibility; it changes only legacy root top_n.
    assert "--top-n" in help_text


def test_legacy_cli_top_n_cannot_override_nested_holdings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli()
    monkeypatch.setattr(
        module,
        "run_pipeline",
        lambda config: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    with pytest.raises(ValueError, match="conflicts"):
        module.main(["--config", str(EXAMPLE), "--top-n", "20"])


def test_cli_output_preserves_v5_compact_summaries_without_tables() -> None:
    module = _load_cli()
    config = _base(Path("output"), top_n=10)
    summary = {
        "status": "ready",
        "run_dir": "run",
        "required_start_date": "2022-12-01",
        "required_end_date": "2024-12-31",
        "cache_status": "ready",
        "missing_ranges": {},
        "strategy_name": "cli_v5",
        "stock_pool": "hs300",
        "signal": {
            "enabled": True,
            "source_mode": "ml",
            "source_artifact_dir": "run/ml_artifacts/example",
            "artifact_dir": "run/signal",
            "signal_path": "run/signal/signals.parquet",
            "rows": 120,
            "trade_date_count": 12,
            "prediction_column": "prediction",
            "signal_direction": "descending",
            "schema_version": "1.0",
        },
        "holdings": {
            "enabled": True,
            "source_signal_artifact_dir": "run/signal",
            "artifact_dir": "run/holdings",
            "holdings_path": "run/holdings/holdings.parquet",
            "rows": 100,
            "trade_date_count": 10,
            "requested_top_n": 10,
            "insufficient_universe_policy": "error",
            "weighting": "equal_weight",
            "schema_version": "1.0",
        },
    }
    output = module.build_output(config, summary)
    assert output["signal"] == summary["signal"]
    assert output["holdings"] == summary["holdings"]
    assert output["holdings"]["requested_top_n"] == 10
    assert "signals" not in output["signal"]
    assert "holdings" not in output["holdings"]
    json.dumps(output, allow_nan=False)


def test_real_cli_to_runner_files_signal_holdings(
    tmp_path: Path,
    native_ml_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class ReadyDataManager:
        def prepare_data(self, config: object) -> dict[str, object]:
            return {"cache_status": "ready", "missing_ranges": {}}

    monkeypatch.setattr(runner_module, "DataManager", ReadyDataManager)
    module = _load_cli()
    config = _base(tmp_path / "output", top_n=5)
    config.signal = SignalPipelineConfig(
        enabled=True,
        source=PredictionSourceConfig("files", native_ml_artifact),
    )
    config.holdings = HoldingsPipelineConfig(enabled=True, top_n=5)
    config = PipelineConfig.from_dict(config.to_dict())
    path = _write_yaml(tmp_path / "cli.yaml", config.to_dict())
    assert module.main(["--config", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["signal"]["source_mode"] == "files"
    assert payload["signal"]["signal_direction"] == "descending"
    assert payload["holdings"]["requested_top_n"] == 5
    assert payload["holdings"]["weighting"] == "equal_weight"
    assert SignalArtifactStore().validate(payload["signal"]["artifact_dir"]).is_valid
    assert HoldingsArtifactStore().validate(payload["holdings"]["artifact_dir"]).is_valid
    snapshot = yaml.safe_load(
        (Path(payload["run_dir"]) / "config_snapshot.yaml").read_text()
    )
    assert snapshot["holdings"]["top_n"] == 5


def test_cli_scope_and_documentation_contracts() -> None:
    source = CLI.read_text(encoding="utf-8")
    for forbidden in (
        "SignalBuilder",
        "HoldingsBuilder",
        "SignalArtifactStore",
        "HoldingsArtifactStore",
        "read_parquet",
        "to_parquet",
        ".glob(",
        ".rglob(",
        "latest",
        "mtime",
        "run_scoring_model",
    ):
        assert forbidden not in source
    readme = README.read_text(encoding="utf-8")
    docs = DOC.read_text(encoding="utf-8")
    example = EXAMPLE.read_text(encoding="utf-8")
    assert "docs/08_signal_holdings_pipeline.md" in readme
    assert "signal_holdings_pipeline.example.yaml" in readme
    assert "scripts/run_pipeline.py --config" in readme
    for term in (
        "holdings.top_n",
        "backend canonical default",
        "descending",
        "ascending",
        "allow_partial",
        "equal_weight",
        "predictions.parquet",
        "SHA-256",
        "no-overwrite",
        "Signal provenance",
        "Holdings provenance",
        "scripts/run_research_pipeline.py",
        "V5-E4",
    ):
        assert term in docs
    assert "top_n: 10" in example
    assert "not announce a final v0.6.0 release" in docs
    assert "supports optimizer" not in docs.lower()
