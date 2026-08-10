"""Gate F tests for canonical PipelineConfig and generic Runner service wiring."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.pipeline.config import PipelineConfig
from src.pipeline.holdings_config import HoldingsPipelineConfig
from src.pipeline.runner import _portfolio_engine


def test_holdings_config_new_portfolio_roundtrip_is_detached() -> None:
    raw = {
        "enabled": True,
        "top_n": 10,
        "insufficient_universe_policy": "error",
        "weighting": "equal_weight",
        "artifact_subdir": "holdings",
        "portfolio_construction": {
            "method": "inverse_volatility",
            "params": {
                "lookback_trading_days": 60,
                "min_observations": 40,
            },
            "constraints": [
                {"type": "max_weight", "params": {"max_weight": 0.2}}
            ],
        },
    }
    config = HoldingsPipelineConfig.from_dict(raw)
    raw["portfolio_construction"]["method"] = "polluted"  # type: ignore[index]
    assert config.portfolio_construction.method == "inverse_volatility"
    assert HoldingsPipelineConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_equal_and_rank_do_not_construct_market_client() -> None:
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        raise AssertionError("market client must not be constructed")

    for method in ("equal_weight", "rank_weight"):
        engine, client = _portfolio_engine(
            method, factory, shared_client=None
        )
        assert engine is not None
        assert client is None
    assert calls == 0


def test_inverse_volatility_constructs_one_run_scoped_client() -> None:
    class Client:
        def get_trade_cal(self, *args, **kwargs): ...
        def get_daily(self, *args, **kwargs): ...
        def get_stock_basic(self, *args, **kwargs): ...
        def get_suspend_d(self, *args, **kwargs): ...

    client = Client()
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return client

    engine, actual = _portfolio_engine(
        "inverse_volatility", factory, shared_client=None
    )
    assert engine is not None
    assert actual is client
    assert calls == 1


def test_canonical_example_yaml_parses_exact_portfolio_config() -> None:
    path = (
        Path(__file__).parents[1]
        / "config"
        / "portfolio_construction_pipeline.example.yaml"
    )
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(values)
    portfolio = config.holdings.portfolio_construction
    assert portfolio.method == "inverse_volatility"
    assert portfolio.params == {
        "lookback_trading_days": 60,
        "min_observations": 40,
    }
    assert portfolio.constraints[0].to_dict() == {
        "type": "max_weight",
        "params": {"max_weight": 0.2},
    }


def test_cli_remains_generic_config_only_for_portfolio_settings() -> None:
    path = Path(__file__).parents[1] / "scripts" / "run_pipeline.py"
    source = path.read_text(encoding="utf-8")
    for forbidden in (
        "--portfolio-method",
        "--inverse-vol-lookback",
        "--max-weight",
    ):
        assert forbidden not in source
