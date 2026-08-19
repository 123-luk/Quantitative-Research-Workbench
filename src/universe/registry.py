"""Fresh resolver registry for Universe 1.0."""

from __future__ import annotations

from src.universe.contracts import UniverseType
from src.universe.resolvers import AllASharesUniverseResolver, CustomUniverseResolver, IndexUniverseResolver, UniverseResolver


class UniverseResolverRegistry:
    def __init__(self) -> None:
        self._resolvers: dict[UniverseType, UniverseResolver] = {}

    def register(self, resolver: UniverseResolver) -> None:
        kind = getattr(resolver, "universe_type", None)
        if not isinstance(kind, UniverseType) or not callable(getattr(resolver, "resolve", None)) or not callable(getattr(resolver, "requirements", None)):
            raise TypeError("resolver must implement the UniverseResolver contract.")
        if kind in self._resolvers:
            raise ValueError(f"Resolver for {kind.value!r} is already registered.")
        self._resolvers[kind] = resolver

    def get(self, universe_type: UniverseType) -> UniverseResolver:
        if not isinstance(universe_type, UniverseType):
            raise TypeError("universe_type must be a UniverseType.")
        try:
            return self._resolvers[universe_type]
        except KeyError as exc:
            raise KeyError(f"No resolver registered for {universe_type.value!r}.") from exc

    def list_types(self) -> tuple[UniverseType, ...]:
        return tuple(sorted(self._resolvers, key=lambda item: item.value))


def create_default_universe_registry() -> UniverseResolverRegistry:
    registry = UniverseResolverRegistry()
    for resolver in (CustomUniverseResolver(), IndexUniverseResolver(), AllASharesUniverseResolver()):
        registry.register(resolver)
    return registry
