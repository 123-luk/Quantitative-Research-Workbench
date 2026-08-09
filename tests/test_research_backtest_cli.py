from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from src.pipeline import PipelineConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "run_pipeline.py"
EXAMPLE = PROJECT_ROOT / "config" / "research_backtest_pipeline.example.yaml"
DOC = PROJECT_ROOT / "docs" / "10_research_backtest_pipeline.md"
README = PROJECT_ROOT / "README.md"


def _load_cli():
    spec = importlib.util.spec_from_file_location("research_backtest_cli", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, value: object) -> Path:
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_example_yaml_uses_real_loader_and_roundtrips() -> None:
    module = _load_cli()
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    before = deepcopy(raw)
    config = module.load_pipeline_config(EXAMPLE, {})
    assert raw == before
    assert config.backtest_end == "2024-12-31"
    assert config.ml_experiment.enabled
    assert config.signal.enabled and config.signal.source.mode == "ml"
    assert config.holdings.enabled and config.holdings.top_n == 10
    research = config.research_backtest
    assert research.enabled and research.source.mode == "pipeline"
    assert research.transaction_cost.cost_bps == 10.0  # type: ignore[union-attr]
    assert research.benchmark.benchmark_code == "000300.SH"  # type: ignore[union-attr]
    assert research.performance.annual_risk_free_rate == 0.0  # type: ignore[union-attr]
    assert PipelineConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_files_mode_direct_yaml_parses_without_current_holdings(
    tmp_path: Path,
) -> None:
    module = _load_cli()
    values = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    values["research_backtest"]["source"] = {  # type: ignore[index]
        "mode": "files",
        "artifact_dir": "native/holdings",
    }
    values["holdings"]["enabled"] = False  # type: ignore[index]
    values["signal"]["enabled"] = False  # type: ignore[index]
    values["ml_experiment"]["enabled"] = False  # type: ignore[index]
    values["ml_experiment"]["save_artifacts"] = False  # type: ignore[index]
    values["modeling_panel"]["enabled"] = False  # type: ignore[index]
    config = module.load_pipeline_config(
        _write_yaml(tmp_path / "files.yaml", values), {}
    )
    assert config.research_backtest.source.mode == "files"
    assert not config.holdings.enabled


def test_generic_config_cli_passes_exact_nested_config_without_network(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    observed: list[PipelineConfig] = []

    def fake_run(config: PipelineConfig) -> dict[str, object]:
        observed.append(config)
        return {
            "status": "ready",
            "run_dir": "run",
            "required_start_date": config.required_start_date,
            "required_end_date": config.required_end_date,
            "cache_status": "ready",
            "missing_ranges": {},
            "strategy_name": config.strategy_name,
            "stock_pool": config.stock_pool,
            "research_backtest": {
                "enabled": True,
                "artifact_dir": "run/research_backtest",
                "benchmark_code": "000300.SH",
                "observation_count": 10,
                "metrics": {},
            },
        }

    monkeypatch.setattr(module, "run_pipeline", fake_run)
    assert module.main(["--config", str(EXAMPLE), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert len(observed) == 1
    assert observed[0].research_backtest.enabled
    assert output["research_backtest"]["benchmark_code"] == "000300.SH"


def test_help_adds_no_research_backtest_business_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config" in help_text
    for forbidden in (
        "--backtest-end-date",
        "--cost-bps",
        "--research-frequency",
        "--rebalance-frequency-for-backtest",
        "--holdings-artifact",
    ):
        assert forbidden not in help_text


def test_invalid_research_yaml_fails_before_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    values = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    values["research_backtest"]["source"] = {"mode": "latest"}  # type: ignore[index]
    path = _write_yaml(tmp_path / "invalid.yaml", values)
    monkeypatch.setattr(
        module,
        "run_pipeline",
        lambda config: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    assert module.main(["--config", str(path)]) == 2
    assert "Research Backtest config error" in capsys.readouterr().err


def test_legacy_cli_flags_do_not_override_nested_research_config() -> None:
    module = _load_cli()
    config = module.load_pipeline_config(
        EXAMPLE,
        {"benchmark": "LEGACY.OVERRIDE", "transaction_cost": 0.75},
    )
    assert config.benchmark == "LEGACY.OVERRIDE"
    assert config.transaction_cost == 0.75
    assert (
        config.research_backtest.benchmark.benchmark_code  # type: ignore[union-attr]
        == "000300.SH"
    )
    assert config.research_backtest.transaction_cost.cost_bps == 10.0  # type: ignore[union-attr]


def test_docs_and_readme_links_exist() -> None:
    assert CLI.is_file() and EXAMPLE.is_file() and DOC.is_file()
    readme = README.read_text(encoding="utf-8")
    assert "docs/10_research_backtest_pipeline.md" in readme
    assert "config/research_backtest_pipeline.example.yaml" in readme


def test_example_contains_no_machine_specific_or_secret_values() -> None:
    lowered = EXAMPLE.read_text(encoding="utf-8").lower()
    for forbidden in (
        "c:\\users",
        "/users/",
        "e:\\financial engineering",
        "password:",
        "private_key:",
        "tushare_token:",
    ):
        assert forbidden not in lowered
