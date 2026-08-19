"""Registry-driven Universe service and legacy configuration adapter."""

from __future__ import annotations

from typing import Iterable

from src.data.contracts import DataRequirement, ResearchFrequency, canonical_date, coalesce_requirements, formation_dates
from src.universe.contracts import UniverseConfigError, UniverseSnapshot, UniverseSpec
from src.universe.data import UniverseDataSource
from src.universe.registry import UniverseResolverRegistry, create_default_universe_registry


class UniverseService:
    def __init__(self, registry: UniverseResolverRegistry | None = None) -> None:
        self.registry = registry or create_default_universe_registry()

    def canonicalize_spec(self, spec: UniverseSpec, services: UniverseDataSource) -> UniverseSpec:
        resolver = self.registry.get(spec.universe_type)
        method = getattr(resolver, "canonicalize", None)
        if callable(method):
            return method(spec, services)
        return spec

    def resolve(self, spec: UniverseSpec, formation_date: object, services: UniverseDataSource) -> UniverseSnapshot:
        if not isinstance(spec, UniverseSpec):
            raise TypeError("spec must be a UniverseSpec.")
        return self.registry.get(spec.universe_type).resolve(spec, canonical_date(formation_date), services)

    def requirements(self, spec: UniverseSpec, *, start: object, end: object, frequency: ResearchFrequency) -> tuple[DataRequirement, ...]:
        if not isinstance(spec, UniverseSpec):
            raise TypeError("spec must be a UniverseSpec.")
        if not isinstance(frequency, ResearchFrequency):
            raise TypeError("frequency must be a ResearchFrequency.")
        return coalesce_requirements(self.registry.get(spec.universe_type).requirements(spec, canonical_date(start), canonical_date(end), frequency))

    def resolve_schedule(self, spec: UniverseSpec, *, frequency: ResearchFrequency, open_dates: Iterable[object], services: UniverseDataSource) -> tuple[UniverseSnapshot, ...]:
        if not isinstance(frequency, ResearchFrequency):
            raise TypeError("frequency must be a ResearchFrequency.")
        return tuple(self.resolve(spec, formation, services) for formation in formation_dates(frequency, open_dates))


def universe_spec_from_legacy(stock_pool: object) -> UniverseSpec:
    """Parse only the explicitly supported legacy stock-pool identities."""
    if not isinstance(stock_pool, str) or not stock_pool.strip():
        raise UniverseConfigError("legacy stock_pool must be a non-empty string.")
    value = stock_pool.strip().upper()
    aliases = {
        "HS300": UniverseSpec.index("000300.SH"),
        "CSI300": UniverseSpec.index("000300.SH"),
        "ALL": UniverseSpec.all_a_shares(),
        "ALL_A_SHARES": UniverseSpec.all_a_shares(),
    }
    if value in aliases:
        return aliases[value]
    if "." in value:
        return UniverseSpec.index(value)
    raise UniverseConfigError(f"Unsupported legacy stock_pool: {stock_pool!r}.")
