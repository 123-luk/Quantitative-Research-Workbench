"""Gate A tests for portfolio-construction contracts and registries."""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.portfolio_construction import (
    ConstraintRegistry,
    ConstraintSpec,
    EqualWeightStrategy,
    MaxWeightConstraint,
    PortfolioConstructionConfig,
    PortfolioConstructionConfigError,
    PortfolioConstructionRegistry,
    PortfolioConstructionRegistryError,
    PortfolioConstructionRequest,
    PortfolioConstructionValidationError,
    build_default_constraint_registry,
    build_default_portfolio_construction_registry,
)


def candidates(count: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [f"{index:06d}.SZ" for index in range(count)],
            "score": np.linspace(9.0, 1.0, count),
            "rank": [10 + index * 7 for index in range(count)],
            "selection_position": list(range(1, count + 1)),
        }
    )


def test_request_normalizes_date_sorts_position_and_is_defensive() -> None:
    source = candidates().iloc[[2, 0, 1]].reset_index(drop=True)
    original = source.copy(deep=True)
    request = PortfolioConstructionRequest("20240809", source)
    assert request.formation_date == pd.Timestamp("2024-08-09")
    assert request.ts_codes == tuple(candidates()["ts_code"])
    pdt.assert_frame_equal(source, original)
    exposed = request.candidates
    exposed.loc[0, "score"] = -999.0
    assert request.candidates.loc[0, "score"] == 9.0


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.assign(ts_code=["x", "x", "z"]),
        lambda frame: frame.assign(score=[1.0, np.inf, 3.0]),
        lambda frame: frame.assign(rank=[1, 0, 3]),
        lambda frame: frame.assign(selection_position=[1, 3, 4]),
        lambda frame: frame.rename(columns={"score": "factor_1"}),
    ],
)
def test_request_rejects_invalid_candidate_contract(mutator) -> None:
    with pytest.raises(PortfolioConstructionValidationError):
        PortfolioConstructionRequest("2024-08-09", mutator(candidates()))


@pytest.mark.parametrize(
    "value",
    ["2024-08-09T12:00:00", " 2024-08-09", pd.Timestamp("2024-08-09", tz="UTC")],
)
def test_request_rejects_noncanonical_dates(value: object) -> None:
    with pytest.raises(PortfolioConstructionValidationError):
        PortfolioConstructionRequest(value, candidates())


def test_config_roundtrip_is_json_safe_deterministic_and_detached() -> None:
    raw = {
        "method": "inverse_volatility",
        "params": {"min_observations": 40, "lookback_trading_days": 60},
        "constraints": [{"type": "max_weight", "params": {"max_weight": 0.4}}],
    }
    original = deepcopy(raw)
    config = PortfolioConstructionConfig.from_dict(raw)
    assert raw == original
    expected = {
        "method": "inverse_volatility",
        "params": {"lookback_trading_days": 60, "min_observations": 40},
        "constraints": [{"type": "max_weight", "params": {"max_weight": 0.4}}],
    }
    assert config.to_dict() == expected
    assert PortfolioConstructionConfig.from_dict(config.to_dict()).to_dict() == expected
    json.dumps(config.to_dict(), allow_nan=False, sort_keys=True)
    exposed = config.to_dict()
    exposed["params"]["lookback_trading_days"] = 2  # type: ignore[index]
    assert config.to_dict() == expected


@pytest.mark.parametrize(
    "raw",
    [
        {"method": "equal_weight", "params": {}},
        {"method": "equal_weight", "params": {}, "constraints": [], "abc": 1},
        {"method": " equal_weight", "params": {}, "constraints": []},
        {"method": "equal_weight", "params": {1: "bad"}, "constraints": []},
        {"method": "equal_weight", "params": {"x": np.nan}, "constraints": []},
        {"method": "equal_weight", "params": {}, "constraints": {}},
        {
            "method": "equal_weight",
            "params": {},
            "constraints": [
                {"type": "max_weight", "params": {"max_weight": 0.5}},
                {"type": "max_weight", "params": {"max_weight": 0.6}},
            ],
        },
        {
            "method": "equal_weight",
            "params": {},
            "constraints": [{"type": "max_weight", "params": {}, "abc": 1}],
        },
    ],
)
def test_config_rejects_unknown_non_json_and_duplicate_fields(raw: object) -> None:
    with pytest.raises(PortfolioConstructionConfigError):
        PortfolioConstructionConfig.from_dict(raw)


def test_strategy_registry_exact_duplicate_unknown_and_fresh_defaults() -> None:
    registry = PortfolioConstructionRegistry()
    strategy = EqualWeightStrategy()
    registry.register(strategy)
    assert registry.resolve("equal_weight") is strategy
    with pytest.raises(PortfolioConstructionRegistryError):
        registry.register(EqualWeightStrategy())
    for value in ("EQUAL_WEIGHT", " equal_weight", "unknown"):
        with pytest.raises(PortfolioConstructionRegistryError):
            registry.resolve(value)
    first = build_default_portfolio_construction_registry()
    second = build_default_portfolio_construction_registry()
    assert first is not second
    assert first.names() == (
        "equal_weight",
        "inverse_volatility",
        "minimum_variance",
        "rank_weight",
    )


def test_constraint_registry_exact_duplicate_unknown_and_typed_parse() -> None:
    registry = ConstraintRegistry()
    plugin = MaxWeightConstraint()
    registry.register(plugin)
    spec = ConstraintSpec("max_weight", {"max_weight": 0.5})
    parsed = registry.parse(spec)
    assert parsed.type == "max_weight"
    assert parsed.params.max_weight == 0.5
    with pytest.raises(PortfolioConstructionRegistryError):
        registry.register(MaxWeightConstraint())
    with pytest.raises(PortfolioConstructionRegistryError):
        registry.resolve("MAX_WEIGHT")
    assert build_default_constraint_registry().names() == ("max_weight",)


def test_built_in_strategy_params_are_strict() -> None:
    registry = build_default_portfolio_construction_registry()
    with pytest.raises(PortfolioConstructionConfigError):
        registry.resolve("equal_weight").parse_params({"abc": 1})
    with pytest.raises(PortfolioConstructionConfigError):
        registry.resolve("rank_weight").parse_params({"abc": 1})
    inverse = registry.resolve("inverse_volatility")
    with pytest.raises(PortfolioConstructionConfigError):
        inverse.parse_params(
            {"lookback_trading_days": 60, "min_observations": 40, "abc": 1}
        )
    with pytest.raises(PortfolioConstructionConfigError):
        inverse.parse_params({"lookback_trading_days": True, "min_observations": 2})
