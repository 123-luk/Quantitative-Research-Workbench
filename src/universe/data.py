"""Validated CURATED-only data boundary for Universe resolvers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

import pandas as pd

from src.data.canonical_store import PartitionedParquetStore, content_hash, normalize_frame
from src.data.contracts import canonical_date, normalize_scope
from src.data.coverage_ledger import CoverageLedger
from src.data.coverage_planner import scope_key
from src.data.dataset_registry import DatasetRegistry
from src.universe.contracts import UniverseDataUnavailable


STOCK_BASIC_SCOPE = normalize_scope({"scope": "CN_STOCK_REFERENCE"})


@dataclass(frozen=True)
class CanonicalUniverseSlice:
    frame: pd.DataFrame
    dataset_id: str
    schema_version: str
    source_as_of: str
    source_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pd.DataFrame):
            raise UniverseDataUnavailable("canonical universe slice must contain a DataFrame.")
        object.__setattr__(self, "source_as_of", canonical_date(self.source_as_of))
        if not isinstance(self.source_identity, str) or not self.source_identity.strip():
            raise UniverseDataUnavailable("canonical source identity must not be empty.")
        object.__setattr__(self, "frame", self.frame.copy(deep=True))


class UniverseDataSource(Protocol):
    def stock_basic(self) -> CanonicalUniverseSlice: ...
    def index_weight(self, index_code: str, through_date: str) -> CanonicalUniverseSlice: ...


def _months(start: str, end: str) -> tuple[str, ...]:
    current = pd.Timestamp(canonical_date(start)).to_period("M")
    finish = pd.Timestamp(canonical_date(end)).to_period("M")
    return tuple(str(item) for item in pd.period_range(current, finish, freq="M"))


class CanonicalUniverseDataSource:
    """Read explicit ledger-proven partitions without discovery fallbacks."""

    def __init__(self, *, registry: DatasetRegistry, ledger: CoverageLedger, store: PartitionedParquetStore, stock_basic_as_of: str, index_weight_start: str) -> None:
        self.registry = registry
        self.ledger = ledger
        self.store = store
        self.stock_basic_as_of = canonical_date(stock_basic_as_of)
        self.index_weight_start = canonical_date(index_weight_start)

    def _read(self, dataset_id: str, scope: tuple[tuple[str, str], ...], units: tuple[str, ...], source_as_of: str) -> CanonicalUniverseSlice:
        spec = self.registry.get(dataset_id)
        if not units:
            raise UniverseDataUnavailable(f"No explicit units requested for {dataset_id}.")
        records = {record.unit_key: record for record in self.ledger.records(dataset_id) if record.scope_key == scope_key(scope) and record.status == "COMPLETE"}
        missing = tuple(unit for unit in units if unit not in records)
        if missing:
            raise UniverseDataUnavailable(f"Canonical {dataset_id} coverage is unavailable for units {missing!r}.")
        frames: list[pd.DataFrame] = []
        identities: list[str] = []
        for unit in units:
            rows = self.store.rows_for_unit(spec, unit=unit, scope=scope)
            record = records[unit]
            if len(rows) != record.row_count or content_hash(spec, rows) != record.content_hash:
                raise UniverseDataUnavailable(f"Canonical {dataset_id} integrity check failed for {unit}.")
            frames.append(rows)
            identities.append(f"{unit}:{record.content_hash}")
        frame = normalize_frame(spec, pd.concat(frames, ignore_index=True))
        identity_hash = sha256("|".join(identities).encode("utf-8")).hexdigest()
        return CanonicalUniverseSlice(frame, dataset_id, spec.schema_version, source_as_of, f"{dataset_id}:{spec.schema_version}:{identity_hash}")

    def stock_basic(self) -> CanonicalUniverseSlice:
        return self._read("stock_basic", STOCK_BASIC_SCOPE, (self.stock_basic_as_of,), self.stock_basic_as_of)

    def index_weight(self, index_code: str, through_date: str) -> CanonicalUniverseSlice:
        through = canonical_date(through_date)
        scope = normalize_scope({"index_code": index_code})
        return self._read("index_weight", scope, _months(self.index_weight_start, through), through)
