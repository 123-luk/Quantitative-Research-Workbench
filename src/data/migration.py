"""Conservative, explicit legacy-Parquet coverage import."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from src.data.canonical_store import PartitionedParquetStore, content_hash, normalize_frame
from src.data.contracts import CoverageKind, DatasetSpec, normalize_scope
from src.data.coverage_ledger import CoverageLedger, CoverageRecord
from src.data.coverage_planner import scope_key


class LegacyCoverageMigrator:
    def __init__(self, ledger: CoverageLedger, store: PartitionedParquetStore) -> None:
        self.ledger = ledger
        self.store = store

    def import_file(self, spec: DatasetSpec, path: str | Path, *, scope: object, effective_through: str | None = None) -> tuple[str, ...]:
        """Validate an explicit file; never trusts old min/max JSON metadata."""
        source = Path(path)
        if not source.is_file() or source.is_symlink():
            raise ValueError("legacy source must be an explicit regular Parquet file.")
        frame = normalize_frame(spec, pd.read_parquet(source))
        if spec.coverage_kind is CoverageKind.TRADE_DATE:
            return ()
        if spec.coverage_kind is CoverageKind.GLOBAL_SNAPSHOT:
            # A legacy date-labelled reference file cannot prove when the
            # current provider snapshot was retrieved. Do not relabel it as a
            # GLOBAL snapshot or preserve a fictitious PIT date.
            return ()
        if spec.coverage_kind is CoverageKind.REFERENCE_EFFECTIVE_THROUGH:
            if effective_through is None or set(frame.get("list_status", ())) != {"L", "D", "P"}:
                return ()
            groups = ((effective_through, frame),)
        else:
            column = "cal_date" if spec.coverage_kind is CoverageKind.CALENDAR_DATE else "trade_date"
            keys = frame[column].str[:7] if spec.coverage_kind is CoverageKind.ENTITY_MONTH else frame[column]
            groups = tuple((str(unit), frame.loc[keys.eq(unit)]) for unit in sorted(set(keys)))
        normalized_scope = normalize_scope(scope)
        scope_values = dict(normalized_scope)
        if spec.coverage_kind is CoverageKind.ENTITY_TRADE_DATE:
            expected = scope_values.get("index_code") or scope_values.get("ts_code")
            if not expected or set(frame["ts_code"]) != {expected}:
                return ()
        if spec.coverage_kind is CoverageKind.ENTITY_MONTH:
            if set(frame["index_code"]) != {scope_values.get("index_code")}:
                return ()
        if spec.coverage_kind is CoverageKind.CALENDAR_DATE:
            exchange = scope_values.get("exchange")
            if exchange and not set(frame["exchange"]).issubset({exchange, ""}):
                return ()
        unit_names = tuple(unit for unit, _rows in groups)
        self.store.merge(spec, frame, units=unit_names, scope=normalized_scope)
        records: list[CoverageRecord] = []
        imported: list[str] = []
        for unit, rows in groups:
            if rows.empty or (spec.coverage_kind is CoverageKind.CALENDAR_DATE and len(rows) != 1):
                continue
            canonical_rows = self.store.rows_for_unit(spec, unit=unit, scope=normalized_scope)
            digest = content_hash(spec, canonical_rows)
            fingerprint = sha256(f"migration:{spec.dataset_id}:{scope_key(normalized_scope)}:{unit}".encode()).hexdigest()
            records.append(CoverageRecord(spec.dataset_id, scope_key(normalized_scope), unit, "COMPLETE", len(canonical_rows), spec.schema_version, digest, fingerprint, datetime.now(timezone.utc).replace(microsecond=0).isoformat()))
            imported.append(unit)
        self.ledger.mark_complete(records)
        return tuple(imported)
