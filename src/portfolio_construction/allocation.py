"""Deterministic proportional allocation with an optional hard cap."""

from __future__ import annotations

from numbers import Real

import numpy as np

from .contracts import WEIGHT_ABSOLUTE_TOLERANCE
from .errors import PortfolioConstructionConstraintError


def capped_proportional_allocation(
    raw_attractiveness: object,
    max_weight: float | None = None,
) -> np.ndarray:
    """Solve ``w_i=min(c*q_i, cap)`` by deterministic active-set filling."""
    try:
        raw = np.asarray(raw_attractiveness, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PortfolioConstructionConstraintError(
            "raw attractiveness must be finite positive real data."
        ) from exc
    if (
        raw.ndim != 1
        or not len(raw)
        or not np.isfinite(raw).all()
        or bool((raw <= 0).any())
    ):
        raise PortfolioConstructionConstraintError(
            "raw attractiveness must be a non-empty vector of finite positive values."
        )
    if max_weight is None:
        weights = raw / float(raw.sum())
    else:
        if isinstance(max_weight, bool) or not isinstance(max_weight, Real):
            raise PortfolioConstructionConstraintError(
                "max_weight must be finite numeric data."
            )
        cap = float(max_weight)
        if not np.isfinite(cap) or not 0.0 < cap <= 1.0:
            raise PortfolioConstructionConstraintError(
                "max_weight must satisfy 0 < max_weight <= 1."
            )
        if len(raw) * cap < 1.0 - WEIGHT_ABSOLUTE_TOLERANCE:
            raise PortfolioConstructionConstraintError(
                "max_weight is infeasible for the candidate count."
            )
        weights = np.zeros(len(raw), dtype=np.float64)
        active = list(range(len(raw)))
        remaining = 1.0
        while active:
            denominator = float(raw[active].sum())
            scale = remaining / denominator
            binding = [position for position in active if scale * raw[position] > cap]
            if not binding:
                weights[active] = scale * raw[active]
                remaining = 0.0
                break
            for position in binding:
                weights[position] = cap
                active.remove(position)
                remaining -= cap
            if remaining < -WEIGHT_ABSOLUTE_TOLERANCE:
                raise PortfolioConstructionConstraintError(
                    "capped allocation produced an invalid residual."
                )
        if active == [] and remaining > WEIGHT_ABSOLUTE_TOLERANCE:
            raise PortfolioConstructionConstraintError(
                "max_weight is infeasible for the candidate count."
            )
    if (
        not np.isfinite(weights).all()
        or bool((weights < 0).any())
        or not np.isclose(
            float(weights.sum()), 1.0, rtol=0.0, atol=WEIGHT_ABSOLUTE_TOLERANCE
        )
    ):
        raise PortfolioConstructionConstraintError(
            "allocation failed fully-invested long-only invariants."
        )
    if (
        max_weight is not None
        and float(weights.max())
        > float(max_weight) + WEIGHT_ABSOLUTE_TOLERANCE
    ):
        raise PortfolioConstructionConstraintError("allocation violates max_weight.")
    return weights
