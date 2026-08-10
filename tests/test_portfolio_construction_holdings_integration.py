"""Gate E tests for Top-N to Engine to canonical Holdings integration."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.holdings import HOLDINGS_OUTPUT_COLUMNS, HoldingsBuilder
from src.pipeline.holdings_config import HoldingsPipelineConfig
from src.portfolio_construction import (
    PortfolioConstructionConfig,
    PortfolioConstructionEngine,
    PortfolioConstructionRegistry,
    StrategyConstructionOutput,
    build_default_portfolio_construction_registry,
)


def signals(count: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.Series(
                [pd.Timestamp("2024-01-31")] * count,
                dtype="datetime64[ns]",
            ),
            "ts_code": [f"S{index}" for index in range(1, count + 1)],
            "score": np.arange(count, 0, -1, dtype=np.float64),
            "rank": np.arange(1, count + 1, dtype=np.int64),
        }
    )


def build(config: PortfolioConstructionConfig | None = None):
    return HoldingsBuilder().build(
        signals(),
        top_n=5,
        insufficient_universe_policy="error",
        weighting="equal_weight",
        portfolio_construction=config,
    )


def test_old_config_resolves_default_equal_weight() -> None:
    old = {
        "enabled": True,
        "top_n": 5,
        "insufficient_universe_policy": "error",
        "weighting": "equal_weight",
        "artifact_subdir": "holdings",
    }
    config = HoldingsPipelineConfig.from_dict(old)
    assert config.portfolio_construction.to_dict() == {
        "method": "equal_weight",
        "params": {},
        "constraints": [],
    }
    assert config.to_dict()["portfolio_construction"] == config.portfolio_construction.to_dict()


def test_default_equal_weight_preserves_v07_payload_exactly() -> None:
    actual = build().holdings
    expected = signals().head(5).copy(deep=True)
    expected.insert(2, "target_weight", np.full(5, 0.2, dtype=np.float64))
    expected = expected.loc[:, list(HOLDINGS_OUTPUT_COLUMNS)]
    pdt.assert_frame_equal(actual, expected)


def test_rank_weight_changes_only_weights_and_keeps_selection_position_internal() -> None:
    equal = build().holdings
    ranked = build(PortfolioConstructionConfig("rank_weight", {})).holdings
    pdt.assert_frame_equal(
        equal.drop(columns="target_weight"), ranked.drop(columns="target_weight")
    )
    np.testing.assert_allclose(
        ranked["target_weight"], np.asarray([5, 4, 3, 2, 1]) / 15.0
    )
    assert tuple(ranked.columns) == HOLDINGS_OUTPUT_COLUMNS


class CustomStrategy:
    name = "custom_strategy"
    supported_constraint_types = frozenset()
    required_services = frozenset()

    def parse_params(self, raw_params: Mapping[str, object]) -> object:
        assert not raw_params
        return None

    def construct(self, request, parsed_params, constraints, services):
        del parsed_params, constraints, services
        count = len(request.ts_codes)
        return StrategyConstructionOutput(
            pd.DataFrame(
                {
                    "ts_code": list(request.ts_codes),
                    "target_weight": np.full(count, 1.0 / count),
                }
            )
        )


def test_custom_registered_strategy_enters_builder_without_method_branch() -> None:
    registry = PortfolioConstructionRegistry()
    registry.register(CustomStrategy())
    result = HoldingsBuilder(
        PortfolioConstructionEngine(strategy_registry=registry)
    ).build(
        signals(),
        top_n=5,
        insufficient_universe_policy="error",
        weighting="equal_weight",
        portfolio_construction=PortfolioConstructionConfig(
            "custom_strategy", {}
        ),
    )
    assert tuple(result.holdings["ts_code"]) == tuple(signals().head(5)["ts_code"])


def test_required_services_are_generic_and_method_independent() -> None:
    registry = build_default_portfolio_construction_registry()
    assert registry.required_services("equal_weight") == frozenset()
    assert registry.required_services("rank_weight") == frozenset()
    assert registry.required_services("inverse_volatility") == frozenset(
        {"historical_returns"}
    )
