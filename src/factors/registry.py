"""Registry for discoverable, independently testable research factors."""

from __future__ import annotations

from typing import Dict, List

from src.factors.base import Factor, FactorMetadata


class FactorRegistry:
    """Store factors by unique metadata name without shared global state."""

    def __init__(self) -> None:
        self._factors: Dict[str, Factor] = {}

    def register(self, factor: Factor, allow_override: bool = False) -> None:
        """Register a valid factor, optionally replacing an existing name."""
        metadata = getattr(factor, "metadata", None)
        if not isinstance(metadata, FactorMetadata):
            raise TypeError("Registered factors must provide valid FactorMetadata.")
        if not callable(getattr(factor, "compute", None)):
            raise TypeError("Registered factors must provide a callable compute method.")
        if metadata.name in self._factors and not allow_override:
            raise ValueError(f"Factor '{metadata.name}' is already registered.")
        self._factors[metadata.name] = factor

    def unregister(self, name: str) -> Factor:
        """Remove and return a factor, raising KeyError when it is absent."""
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' is not registered.")
        return self._factors.pop(name)

    def get(self, name: str) -> Factor:
        """Return a registered factor by name."""
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' is not registered.")
        return self._factors[name]

    def contains(self, name: str) -> bool:
        """Return whether a factor name is registered."""
        return name in self._factors

    def list_names(self) -> List[str]:
        """Return registered names in stable alphabetical order."""
        return sorted(self._factors)

    def list_metadata(self) -> List[FactorMetadata]:
        """Return metadata in the same stable order as list_names."""
        return [self._factors[name].metadata for name in self.list_names()]

    def get_by_category(self, category: str) -> List[Factor]:
        """Return factors in a category, ordered by factor name."""
        return [
            self._factors[name]
            for name in self.list_names()
            if self._factors[name].metadata.category == category
        ]

    def clear(self) -> None:
        """Remove all factors from this registry instance."""
        self._factors.clear()


def create_default_registry(include_examples: bool = True) -> FactorRegistry:
    """Create an independent registry, optionally populated with examples."""
    registry = FactorRegistry()
    if include_examples:
        from src.factors.examples import register_example_factors

        register_example_factors(registry)
    return registry
