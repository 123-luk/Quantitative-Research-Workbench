"""Gate B tests for deterministic weighting and max-weight allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.portfolio_construction import (
    ConstraintSpec,
    PortfolioConstructionConfig,
    PortfolioConstructionConfigError,
    PortfolioConstructionConstraintError,
    PortfolioConstructionEngine,
    PortfolioConstructionRequest,
    capped_proportional_allocation,
)


def request(
    count: int, *, scores: list[float] | None = None
) -> PortfolioConstructionRequest:
    return PortfolioConstructionRequest(
        "2024-08-09",
        pd.DataFrame(
            {
                "ts_code": [f"S{index}" for index in range(1, count + 1)],
                "score": scores or list(np.linspace(100.0, 1.0, count)),
                "rank": [100 + index * 11 for index in range(count)],
                "selection_position": list(range(1, count + 1)),
            }
        ),
    )


@pytest.mark.parametrize("count", [1, 5, 10])
def test_equal_weight_matches_v5_math(count: int) -> None:
    result = PortfolioConstructionEngine().construct(
        request(count), PortfolioConstructionConfig("equal_weight", {})
    )
    np.testing.assert_allclose(
        result.weights["target_weight"], np.full(count, 1.0 / count), atol=1e-12
    )


def test_rank_weight_uses_selection_position_not_rank_or_score() -> None:
    first = PortfolioConstructionEngine().construct(
        request(5), PortfolioConstructionConfig("rank_weight", {})
    )
    second_request = request(5, scores=[-1e20, 7.0, 999.0, -8.0, 0.0])
    changed = second_request.candidates
    changed["rank"] = [900, 2, 4000, 7, 81]
    second = PortfolioConstructionEngine().construct(
        PortfolioConstructionRequest("2024-08-09", changed),
        PortfolioConstructionConfig("rank_weight", {}),
    )
    np.testing.assert_allclose(
        first.weights["target_weight"], np.asarray([5, 4, 3, 2, 1]) / 15.0
    )
    pdt.assert_frame_equal(first.weights, second.weights)


def test_allocator_without_cap_and_deterministic_order() -> None:
    raw = np.asarray([3.0, 1.0, 2.0])
    first = capped_proportional_allocation(raw)
    second = capped_proportional_allocation(raw.copy())
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, [0.5, 1.0 / 6.0, 1.0 / 3.0])


def test_allocator_cap_exactly_feasible() -> None:
    result = capped_proportional_allocation([9.0, 3.0, 1.0, 0.5], 0.25)
    np.testing.assert_allclose(result, np.full(4, 0.25), atol=1e-12)


def test_allocator_partly_binding_preserves_uncapped_ratio() -> None:
    result = capped_proportional_allocation([8.0, 4.0, 2.0, 1.0], 0.4)
    assert result[0] == pytest.approx(0.4)
    assert result[1] / result[2] == pytest.approx(2.0)
    assert result[2] / result[3] == pytest.approx(2.0)
    assert result.sum() == pytest.approx(1.0, abs=1e-12)


def test_allocator_multiple_binding_rounds() -> None:
    result = capped_proportional_allocation([100.0, 30.0, 5.0, 4.0, 3.0], 0.3)
    assert result[0] == pytest.approx(0.3)
    assert result[1] == pytest.approx(0.3)
    assert result[2] / result[3] == pytest.approx(5.0 / 4.0)
    assert result.max() <= 0.3 + 1e-12


@pytest.mark.parametrize(
    "raw,cap",
    [([], None), ([1.0, 0.0], None), ([1.0, np.nan], None), ([1.0, 1.0], 0.49)],
)
def test_allocator_rejects_invalid_or_infeasible_inputs(raw, cap) -> None:
    with pytest.raises(PortfolioConstructionConstraintError):
        capped_proportional_allocation(raw, cap)


@pytest.mark.parametrize("value", [True, 0.0, -0.1, 1.1, np.inf])
def test_max_weight_params_reject_invalid_values(value: object) -> None:
    with pytest.raises(
        (PortfolioConstructionConfigError, PortfolioConstructionConstraintError)
    ):
        config = PortfolioConstructionConfig(
            "equal_weight",
            {},
            (ConstraintSpec("max_weight", {"max_weight": value}),),
        )
        PortfolioConstructionEngine().construct(request(5), config)


def test_strategy_applies_cap_without_changing_candidates() -> None:
    config = PortfolioConstructionConfig(
        "rank_weight", {}, (ConstraintSpec("max_weight", {"max_weight": 0.25}),)
    )
    result = PortfolioConstructionEngine().construct(request(5), config)
    assert tuple(result.weights["ts_code"]) == request(5).ts_codes
    assert result.weights["target_weight"].max() <= 0.25 + 1e-12
    assert result.weights["target_weight"].sum() == pytest.approx(1.0, abs=1e-12)


def test_infeasible_cap_fails_closed_without_cash() -> None:
    config = PortfolioConstructionConfig(
        "equal_weight", {}, (ConstraintSpec("max_weight", {"max_weight": 0.15}),)
    )
    with pytest.raises(PortfolioConstructionConstraintError):
        PortfolioConstructionEngine().construct(request(5), config)
