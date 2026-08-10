"""Gate D tests for engine validation and plugin extensibility."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.portfolio_construction import (
    ConstraintRegistry,
    ConstraintSpec,
    MaxWeightConstraint,
    PortfolioConstructionConfig,
    PortfolioConstructionConstraintError,
    PortfolioConstructionEngine,
    PortfolioConstructionRegistry,
    PortfolioConstructionRequest,
    PortfolioConstructionResult,
    PortfolioConstructionServices,
    PortfolioConstructionValidationError,
    StrategyConstructionOutput,
)


def request() -> PortfolioConstructionRequest:
    return PortfolioConstructionRequest(
        "2024-01-10",
        pd.DataFrame(
            {
                "ts_code": ["A", "B", "C"],
                "score": [3.0, 2.0, 1.0],
                "rank": [10, 50, 90],
                "selection_position": [1, 2, 3],
            }
        ),
    )


class TestStrategy:
    name = "test_strategy"
    supported_constraint_types = frozenset({"test_constraint", "max_weight"})
    required_services = frozenset()

    def parse_params(self, raw_params: Mapping[str, object]) -> object:
        if set(raw_params) != {"weights"}:
            raise ValueError("strict test params")
        return tuple(raw_params["weights"])  # type: ignore[arg-type]

    def construct(self, request, parsed_params, constraints, services):
        del constraints, services
        return StrategyConstructionOutput(
            pd.DataFrame(
                {"ts_code": list(request.ts_codes), "target_weight": parsed_params}
            ),
            {"plugin": True},
        )


class TestConstraint:
    name = "test_constraint"

    def parse_params(self, raw_params: Mapping[str, object]) -> object:
        if raw_params:
            raise ValueError("strict test constraint")
        return None

    def validate(self, request, weights, parsed_params) -> None:
        del request, parsed_params
        if not weights["target_weight"].is_monotonic_decreasing:
            raise PortfolioConstructionConstraintError("test constraint violated")


def custom_engine() -> PortfolioConstructionEngine:
    strategies = PortfolioConstructionRegistry()
    strategies.register(TestStrategy())
    constraints = ConstraintRegistry()
    constraints.register(TestConstraint())
    constraints.register(MaxWeightConstraint())
    return PortfolioConstructionEngine(strategies, constraints)


def test_custom_strategy_and_constraint_run_without_engine_dispatch_change() -> None:
    config = PortfolioConstructionConfig(
        "test_strategy",
        {"weights": [0.5, 0.3, 0.2]},
        (ConstraintSpec("test_constraint", {}),),
    )
    result = custom_engine().construct(request(), config)
    np.testing.assert_allclose(result.weights["target_weight"], [0.5, 0.3, 0.2])
    assert result.diagnostics["strategy"] == {"plugin": True}


class BadStrategy(TestStrategy):
    name = "bad_strategy"

    def __init__(self, weights: pd.DataFrame | object) -> None:
        self.output = weights

    def parse_params(self, raw_params):
        return None

    def construct(self, request, parsed_params, constraints, services):
        del request, parsed_params, constraints, services
        if isinstance(self.output, StrategyConstructionOutput):
            return self.output
        return StrategyConstructionOutput(self.output)


def bad_engine(output: object) -> PortfolioConstructionEngine:
    registry = PortfolioConstructionRegistry()
    registry.register(BadStrategy(output))
    return PortfolioConstructionEngine(registry)


@pytest.mark.parametrize(
    "weights",
    [
        pd.DataFrame({"ts_code": ["A", "B"], "target_weight": [0.5, 0.5]}),
        pd.DataFrame(
            {"ts_code": ["A", "B", "X"], "target_weight": [0.4, 0.3, 0.3]}
        ),
        pd.DataFrame(
            {"ts_code": ["A", "A", "C"], "target_weight": [0.4, 0.3, 0.3]}
        ),
        pd.DataFrame(
            {"ts_code": ["A", "B", "C"], "target_weight": [0.4, 0.3, np.nan]}
        ),
        pd.DataFrame(
            {"ts_code": ["A", "B", "C"], "target_weight": [0.8, 0.3, -0.1]}
        ),
        pd.DataFrame(
            {"ts_code": ["A", "B", "C"], "target_weight": [0.4, 0.3, 0.2]}
        ),
        pd.DataFrame(
            {"target_weight": [0.4, 0.3, 0.3], "ts_code": ["A", "B", "C"]}
        ),
    ],
)
def test_engine_rejects_invalid_constructor_result(weights: pd.DataFrame) -> None:
    with pytest.raises(PortfolioConstructionValidationError):
        bad_engine(weights).construct(
            request(), PortfolioConstructionConfig("bad_strategy", {})
        )


def test_engine_canonicalizes_valid_constructor_row_order() -> None:
    weights = pd.DataFrame(
        {"ts_code": ["C", "A", "B"], "target_weight": [0.2, 0.5, 0.3]}
    )
    result = bad_engine(weights).construct(
        request(), PortfolioConstructionConfig("bad_strategy", {})
    )
    assert tuple(result.weights["ts_code"]) == request().ts_codes


def test_engine_rejects_unsupported_registered_constraint() -> None:
    class Unsupported(TestStrategy):
        name = "unsupported"
        supported_constraint_types = frozenset()

    strategies = PortfolioConstructionRegistry()
    strategies.register(Unsupported())
    constraints = ConstraintRegistry()
    constraints.register(TestConstraint())
    engine = PortfolioConstructionEngine(strategies, constraints)
    with pytest.raises(PortfolioConstructionConstraintError, match="does not support"):
        engine.construct(
            request(),
            PortfolioConstructionConfig(
                "unsupported",
                {"weights": [0.5, 0.3, 0.2]},
                (ConstraintSpec("test_constraint", {}),),
            ),
        )


def test_constraint_validator_catches_strategy_violation_without_clipping() -> None:
    config = PortfolioConstructionConfig(
        "test_strategy",
        {"weights": [0.8, 0.1, 0.1]},
        (ConstraintSpec("max_weight", {"max_weight": 0.5}),),
    )
    with pytest.raises(PortfolioConstructionConstraintError, match="violates"):
        custom_engine().construct(request(), config)


def test_engine_is_deterministic_and_inputs_outputs_are_isolated() -> None:
    candidate_source = request().candidates
    original = candidate_source.copy(deep=True)
    req = PortfolioConstructionRequest("2024-01-10", candidate_source)
    raw_config = {
        "method": "equal_weight",
        "params": {},
        "constraints": [],
    }
    config = PortfolioConstructionConfig.from_dict(raw_config)
    engine = PortfolioConstructionEngine(services=PortfolioConstructionServices())
    first = engine.construct(req, config)
    second = engine.construct(req, config)
    pdt.assert_frame_equal(first.weights, second.weights)
    pdt.assert_frame_equal(candidate_source, original)
    assert raw_config == {"method": "equal_weight", "params": {}, "constraints": []}
    exposed_weights = first.weights
    exposed_weights.loc[0, "target_weight"] = 1.0
    exposed_diagnostics = first.diagnostics
    exposed_diagnostics["method"] = "polluted"
    assert first.weights.loc[0, "target_weight"] == pytest.approx(1.0 / 3.0)
    assert first.diagnostics["method"] == "equal_weight"


def test_public_result_contract_rejects_invalid_direct_construction() -> None:
    with pytest.raises(PortfolioConstructionValidationError):
        PortfolioConstructionResult(
            pd.DataFrame(
                {"ts_code": ["A", "A"], "target_weight": [0.5, 0.5]}
            ),
            {},
        )


def test_factor_count_independence_candidate_contract_has_no_factor_columns() -> None:
    factors = pd.DataFrame(
        {f"factor_{index}": [float(index)] * 3 for index in range(100)}
    )
    selected = request().candidates
    enriched = pd.concat([selected, factors], axis=1)
    with pytest.raises(PortfolioConstructionValidationError):
        PortfolioConstructionRequest("2024-01-10", enriched)
    result = PortfolioConstructionEngine().construct(
        PortfolioConstructionRequest("2024-01-10", selected),
        PortfolioConstructionConfig("rank_weight", {}),
    )
    assert tuple(result.weights.columns) == ("ts_code", "target_weight")
