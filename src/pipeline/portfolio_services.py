"""Run-scoped dependency graph for portfolio-construction capabilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType


class PortfolioServiceGraphError(ValueError):
    """Raised when a capability graph cannot be resolved safely."""


ServiceFactory = Callable[[Mapping[str, object]], object]


def _name(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PortfolioServiceGraphError(
            f"{context} must be a non-empty trimmed string."
        )
    return value


@dataclass(frozen=True)
class PortfolioServiceRegistration:
    """One in-process capability factory and its direct dependencies."""

    name: str
    dependencies: frozenset[str]
    factory: ServiceFactory


class PortfolioServiceFactoryRegistry:
    """Fresh-instance registry for explicit in-process service factories."""

    def __init__(self) -> None:
        self._registrations: dict[str, PortfolioServiceRegistration] = {}

    def register(
        self,
        name: str,
        *,
        dependencies: Iterable[str] = (),
        factory: ServiceFactory,
    ) -> None:
        canonical = _name(name, context="capability name")
        if isinstance(dependencies, (str, bytes)):
            raise PortfolioServiceGraphError(
                "capability dependencies must be an iterable of names."
            )
        dependency_set = frozenset(
            _name(item, context="dependency name") for item in dependencies
        )
        if not callable(factory):
            raise PortfolioServiceGraphError("capability factory must be callable.")
        if canonical in self._registrations:
            raise PortfolioServiceGraphError(
                f"capability {canonical!r} is already registered."
            )
        self._registrations[canonical] = PortfolioServiceRegistration(
            canonical, dependency_set, factory
        )

    def resolve_registration(self, name: object) -> PortfolioServiceRegistration:
        canonical = _name(name, context="capability name")
        try:
            return self._registrations[canonical]
        except KeyError as exc:
            raise PortfolioServiceGraphError(
                f"unknown portfolio service capability {canonical!r}."
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))


class PortfolioServiceResolver:
    """Resolve one deterministic graph with one instance per capability."""

    def __init__(self, registry: PortfolioServiceFactoryRegistry) -> None:
        if not isinstance(registry, PortfolioServiceFactoryRegistry):
            raise PortfolioServiceGraphError(
                "registry must be PortfolioServiceFactoryRegistry."
            )
        self._registry = registry
        self._resolved: dict[str, object] = {}
        self._stack: list[str] = []

    def resolve(self, required: Iterable[str]) -> dict[str, object]:
        if isinstance(required, (str, bytes)):
            raise PortfolioServiceGraphError(
                "required capabilities must be an iterable of names."
            )
        for name in sorted({_name(item, context="required capability") for item in required}):
            self._resolve_one(name)
        return dict(self._resolved)

    def _resolve_one(self, name: str) -> object:
        if name in self._resolved:
            return self._resolved[name]
        if name in self._stack:
            start = self._stack.index(name)
            cycle = self._stack[start:] + [name]
            raise PortfolioServiceGraphError(
                "portfolio service dependency cycle: " + " -> ".join(cycle)
            )
        registration = self._registry.resolve_registration(name)
        self._stack.append(name)
        try:
            dependencies = {
                dependency: self._resolve_one(dependency)
                for dependency in sorted(registration.dependencies)
            }
            instance = registration.factory(MappingProxyType(dependencies))
        finally:
            self._stack.pop()
        if instance is None:
            raise PortfolioServiceGraphError(
                f"capability {name!r} factory returned None."
            )
        self._resolved[name] = instance
        return instance
