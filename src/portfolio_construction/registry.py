"""Fresh-instance strategy and constraint registries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .contracts import (
    ConstraintSpec,
    PortfolioConstructionRequest,
    PortfolioConstructionServices,
    StrategyConstructionOutput,
)
from .errors import PortfolioConstructionRegistryError


class PortfolioConstructionStrategy(Protocol):
    """Stable constructor interface used by the registry-driven engine."""

    name: str
    supported_constraint_types: frozenset[str]

    def parse_params(self, raw_params: Mapping[str, object]) -> object: ...

    def construct(
        self,
        request: PortfolioConstructionRequest,
        parsed_params: object,
        constraints: tuple[ResolvedConstraint, ...],
        services: PortfolioConstructionServices,
    ) -> StrategyConstructionOutput: ...


class PortfolioConstraint(Protocol):
    """Strict parser and independent result validator for one constraint."""

    name: str

    def parse_params(self, raw_params: Mapping[str, object]) -> object: ...

    def validate(
        self,
        request: PortfolioConstructionRequest,
        weights: pd.DataFrame,
        parsed_params: object,
    ) -> None: ...


@dataclass(frozen=True)
class ResolvedConstraint:
    """One parsed constraint plus its validating plugin."""

    type: str
    params: object
    plugin: PortfolioConstraint


def _registry_name(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PortfolioConstructionRegistryError(
            f"{context} name must be a non-empty trimmed string."
        )
    return value


class PortfolioConstructionRegistry:
    """Resolve constructor plugins by exact canonical method name."""

    def __init__(self) -> None:
        self._strategies: dict[str, PortfolioConstructionStrategy] = {}

    def register(self, strategy: PortfolioConstructionStrategy) -> None:
        name = _registry_name(getattr(strategy, "name", None), context="strategy")
        if not callable(getattr(strategy, "parse_params", None)) or not callable(
            getattr(strategy, "construct", None)
        ):
            raise PortfolioConstructionRegistryError(
                "strategy must provide parse_params and construct."
            )
        supported = getattr(strategy, "supported_constraint_types", None)
        if not isinstance(supported, frozenset) or any(
            not isinstance(item, str) or not item for item in supported
        ):
            raise PortfolioConstructionRegistryError(
                "strategy supported_constraint_types must be a frozenset of names."
            )
        if name in self._strategies:
            raise PortfolioConstructionRegistryError(
                f"strategy {name!r} is already registered."
            )
        self._strategies[name] = strategy

    def resolve(self, method: object) -> PortfolioConstructionStrategy:
        name = _registry_name(method, context="strategy")
        try:
            return self._strategies[name]
        except KeyError as exc:
            raise PortfolioConstructionRegistryError(
                f"unknown portfolio-construction method {name!r}."
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._strategies))


class ConstraintRegistry:
    """Resolve strict constraint plugins by exact canonical type name."""

    def __init__(self) -> None:
        self._constraints: dict[str, PortfolioConstraint] = {}

    def register(self, constraint: PortfolioConstraint) -> None:
        name = _registry_name(getattr(constraint, "name", None), context="constraint")
        if not callable(getattr(constraint, "parse_params", None)) or not callable(
            getattr(constraint, "validate", None)
        ):
            raise PortfolioConstructionRegistryError(
                "constraint must provide parse_params and validate."
            )
        if name in self._constraints:
            raise PortfolioConstructionRegistryError(
                f"constraint {name!r} is already registered."
            )
        self._constraints[name] = constraint

    def resolve(self, constraint_type: object) -> PortfolioConstraint:
        name = _registry_name(constraint_type, context="constraint")
        try:
            return self._constraints[name]
        except KeyError as exc:
            raise PortfolioConstructionRegistryError(
                f"unknown portfolio constraint {name!r}."
            ) from exc

    def parse(self, spec: ConstraintSpec) -> ResolvedConstraint:
        if not isinstance(spec, ConstraintSpec):
            raise PortfolioConstructionRegistryError(
                "constraint spec must be ConstraintSpec."
            )
        plugin = self.resolve(spec.type)
        return ResolvedConstraint(
            type=spec.type,
            params=plugin.parse_params(spec.params),
            plugin=plugin,
        )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._constraints))


def build_default_portfolio_construction_registry() -> PortfolioConstructionRegistry:
    """Return an independently mutable registry with all V7-P1 strategies."""
    from .strategies.equal_weight import EqualWeightStrategy
    from .strategies.inverse_volatility import InverseVolatilityStrategy
    from .strategies.rank_weight import RankWeightStrategy

    registry = PortfolioConstructionRegistry()
    registry.register(EqualWeightStrategy())
    registry.register(RankWeightStrategy())
    registry.register(InverseVolatilityStrategy())
    return registry


def build_default_constraint_registry() -> ConstraintRegistry:
    """Return an independently mutable registry with V7-P1 constraints."""
    from .constraints.max_weight import MaxWeightConstraint

    registry = ConstraintRegistry()
    registry.register(MaxWeightConstraint())
    return registry
