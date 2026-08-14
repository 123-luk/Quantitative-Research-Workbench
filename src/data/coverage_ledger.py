"""Transactional SQLite coverage truth for Data Layer 2.0."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
from typing import Iterable, Iterator, Mapping
from uuid import uuid4


@dataclass(frozen=True)
class CoverageRecord:
    dataset_id: str
    scope_key: str
    unit_key: str
    status: str
    row_count: int
    schema_version: str
    content_hash: str
    request_fingerprint: str
    completed_at: str
    provider_id: str = "tushare_official"


@dataclass(frozen=True)
class CoverageTransition:
    transition_id: int
    fetch_id: str
    dataset_id: str
    scope_key: str
    unit_key: str
    coverage_identity: str
    provider_id: str
    endpoint: str | None
    state: str
    operation: str | None
    occurred_at: str
    attempt: int | None
    rows: int | None
    fields: str
    schema_version: str | None
    error_code: str | None
    exception_type: str | None
    exception_cause_type: str | None
    safe_message: str | None
    artifact_reference: str | None
    quality_evidence: str


_TRANSITION_STATES = frozenset({
    "PLANNED", "FETCH_STARTED", "FETCH_SUCCEEDED", "FETCH_FAILED",
    "RAW_STAGED", "CANONICAL_VALIDATED", "CANONICAL_COMMITTED",
    "LEDGER_COMMITTED", "READBACK_VERIFIED",
})


def coverage_identity_key(dataset_id: str, scope_value: str, unit: str) -> str:
    """Return the one canonical, provider-neutral identity used in diagnostics."""
    try:
        scope = json.loads(scope_value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("scope_key must be canonical JSON.") from exc
    if not isinstance(scope, dict):
        raise ValueError("scope_key must encode a JSON object.")
    return json.dumps(
        {"dataset_id": dataset_id, "scope": scope, "unit": unit},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


class CoverageLedger:
    def __init__(self, path: str | Path, *, provider_id: str = "tushare_official") -> None:
        self.path = Path(path)
        self.provider_id = provider_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS coverage_units (
                    dataset_id TEXT NOT NULL, scope_key TEXT NOT NULL,
                    unit_key TEXT NOT NULL, status TEXT NOT NULL,
                    row_count INTEGER NOT NULL, schema_version TEXT NOT NULL,
                    content_hash TEXT NOT NULL, request_fingerprint TEXT NOT NULL,
                    completed_at TEXT NOT NULL, provider_id TEXT NOT NULL DEFAULT 'tushare_official',
                    PRIMARY KEY (dataset_id, scope_key, unit_key),
                    CHECK (status IN ('COMPLETE','PARTIAL','FAILED')),
                    CHECK (row_count >= 0)
                );
                CREATE TABLE IF NOT EXISTS fetch_events (
                    fetch_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL,
                    scope_key TEXT NOT NULL, requested_units TEXT NOT NULL,
                    started_at TEXT NOT NULL, finished_at TEXT,
                    status TEXT NOT NULL, rows INTEGER NOT NULL DEFAULT 0,
                    error_type TEXT, provider_id TEXT NOT NULL DEFAULT 'tushare_official',
                    endpoint TEXT, request_parameters TEXT NOT NULL DEFAULT '[]',
                    requested_statuses TEXT NOT NULL DEFAULT '[]', retrieved_at TEXT,
                    schema_version TEXT, contract_version TEXT, canonical_hash TEXT,
                    raw_reference TEXT, canonical_reference TEXT, manifest_reference TEXT,
                    sdk_version TEXT, quality_conclusion TEXT,
                    CHECK (status IN ('STARTED','COMPLETE','FAILED')),
                    CHECK (rows >= 0)
                );
                CREATE TABLE IF NOT EXISTS coverage_transitions (
                    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fetch_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
                    scope_key TEXT NOT NULL, unit_key TEXT NOT NULL,
                    coverage_identity TEXT NOT NULL, provider_id TEXT NOT NULL,
                    endpoint TEXT, state TEXT NOT NULL, operation TEXT,
                    occurred_at TEXT NOT NULL, attempt INTEGER, rows INTEGER,
                    fields TEXT NOT NULL DEFAULT '[]', schema_version TEXT,
                    error_code TEXT, exception_type TEXT, exception_cause_type TEXT,
                    safe_message TEXT, artifact_reference TEXT,
                    quality_evidence TEXT NOT NULL DEFAULT '{}',
                    CHECK (state IN (
                        'PLANNED','FETCH_STARTED','FETCH_SUCCEEDED','FETCH_FAILED',
                        'RAW_STAGED','CANONICAL_VALIDATED','CANONICAL_COMMITTED',
                        'LEDGER_COMMITTED','READBACK_VERIFIED'
                    )),
                    CHECK (attempt IS NULL OR attempt >= 1),
                    CHECK (rows IS NULL OR rows >= 0)
                );
                CREATE INDEX IF NOT EXISTS idx_coverage_transitions_identity
                    ON coverage_transitions(provider_id,dataset_id,scope_key,unit_key,transition_id);
            """)
            coverage_columns = {row[1] for row in connection.execute("PRAGMA table_info(coverage_units)")}
            if "provider_id" not in coverage_columns:
                connection.execute("ALTER TABLE coverage_units ADD COLUMN provider_id TEXT NOT NULL DEFAULT 'tushare_official'")
            fetch_columns = {row[1] for row in connection.execute("PRAGMA table_info(fetch_events)")}
            if "provider_id" not in fetch_columns:
                connection.execute("ALTER TABLE fetch_events ADD COLUMN provider_id TEXT NOT NULL DEFAULT 'tushare_official'")
            additions = {
                "endpoint": "TEXT", "request_parameters": "TEXT NOT NULL DEFAULT '[]'",
                "requested_statuses": "TEXT NOT NULL DEFAULT '[]'", "retrieved_at": "TEXT",
                "schema_version": "TEXT", "contract_version": "TEXT",
                "canonical_hash": "TEXT", "raw_reference": "TEXT",
                "canonical_reference": "TEXT", "manifest_reference": "TEXT",
                "sdk_version": "TEXT", "quality_conclusion": "TEXT",
            }
            for name, declaration in additions.items():
                if name not in fetch_columns:
                    connection.execute(f"ALTER TABLE fetch_events ADD COLUMN {name} {declaration}")
            transition_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(coverage_transitions)")
            }
            if "quality_evidence" not in transition_columns:
                connection.execute(
                    "ALTER TABLE coverage_transitions ADD COLUMN quality_evidence TEXT NOT NULL DEFAULT '{}'"
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def complete_units(self, dataset_id: str, scope_key: str, units: Iterable[str]) -> frozenset[str]:
        values = tuple(dict.fromkeys(units))
        if not values:
            return frozenset()
        placeholders = ",".join("?" for _ in values)
        with self._connect() as connection:
            rows = connection.execute(f"SELECT unit_key FROM coverage_units WHERE provider_id=? AND dataset_id=? AND scope_key=? AND status='COMPLETE' AND unit_key IN ({placeholders})", (self.provider_id, dataset_id, scope_key, *values)).fetchall()
        return frozenset(row["unit_key"] for row in rows)

    def records(self, dataset_id: str | None = None) -> tuple[CoverageRecord, ...]:
        sql = "SELECT * FROM coverage_units WHERE provider_id=?"
        params: tuple[str, ...] = (self.provider_id,)
        if dataset_id is not None:
            sql += " AND dataset_id=?"
            params = (self.provider_id, dataset_id)
        sql += " ORDER BY dataset_id, scope_key, unit_key"
        with self._connect() as connection:
            return tuple(CoverageRecord(**dict(row)) for row in connection.execute(sql, params))

    def mark_complete(self, records: Iterable[CoverageRecord], connection: sqlite3.Connection | None = None) -> None:
        values = tuple(records)
        if any(record.status != "COMPLETE" for record in values):
            raise ValueError("mark_complete accepts only COMPLETE records.")
        owner = connection is None
        target = self._connect() if owner else connection
        assert target is not None
        try:
            normalized = [record if record.provider_id == self.provider_id else CoverageRecord(**{**record.__dict__, "provider_id": self.provider_id}) for record in values]
            target.executemany("""INSERT INTO coverage_units(dataset_id,scope_key,unit_key,status,row_count,schema_version,content_hash,request_fingerprint,completed_at,provider_id) VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(dataset_id,scope_key,unit_key) DO UPDATE SET
                status=excluded.status,row_count=excluded.row_count,schema_version=excluded.schema_version,
                content_hash=excluded.content_hash,request_fingerprint=excluded.request_fingerprint,completed_at=excluded.completed_at,
                provider_id=excluded.provider_id""", [tuple(record.__dict__.values()) for record in normalized])
            if owner:
                target.commit()
        except Exception:
            if owner:
                target.rollback()
            raise
        finally:
            if owner:
                target.close()

    def start_fetch(self, dataset_id: str, scope_key: str, units: Iterable[str], started_at: str, *, endpoint: str | None = None, request_parameters: Iterable[object] = (), contract_version: str | None = None) -> str:
        fetch_id = uuid4().hex
        requested_units = tuple(units)
        parameters = tuple(request_parameters)
        statuses = tuple(
            str(item["list_status"])
            for item in parameters
            if isinstance(item, dict) and "list_status" in item
        )
        with self._connect() as connection:
            connection.execute("""INSERT INTO fetch_events(
                fetch_id,dataset_id,scope_key,requested_units,started_at,status,provider_id,
                endpoint,request_parameters,requested_statuses,contract_version
                ) VALUES (?,?,?,?,?,'STARTED',?,?,?,?,?)""", (
                    fetch_id, dataset_id, scope_key, json.dumps(requested_units), started_at,
                    self.provider_id, endpoint, json.dumps(parameters, sort_keys=True),
                    json.dumps(statuses), contract_version,
                ))
            self.record_transition(
                fetch_id, dataset_id, scope_key, requested_units,
                state="PLANNED", occurred_at=started_at, endpoint=endpoint,
                connection=connection,
            )
        return fetch_id

    def record_transition(
        self,
        fetch_id: str,
        dataset_id: str,
        scope_key: str,
        units: Iterable[str],
        *,
        state: str,
        occurred_at: str,
        endpoint: str | None = None,
        operation: str | None = None,
        attempt: int | None = None,
        rows: int | None = None,
        fields: Iterable[str] = (),
        schema_version: str | None = None,
        error_code: str | None = None,
        exception_type: str | None = None,
        exception_cause_type: str | None = None,
        safe_message: str | None = None,
        artifact_reference: str | None = None,
        quality_evidence: Mapping[str, object] | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if state not in _TRANSITION_STATES:
            raise ValueError("Unknown coverage transition state.")
        unit_values = tuple(dict.fromkeys(str(unit) for unit in units))
        if not unit_values:
            raise ValueError("coverage transitions require at least one unit.")
        field_values = tuple(dict.fromkeys(str(field) for field in fields))
        owner = connection is None
        target = self._connect() if owner else connection
        assert target is not None
        try:
            target.executemany(
                """INSERT INTO coverage_transitions(
                    fetch_id,dataset_id,scope_key,unit_key,coverage_identity,provider_id,
                    endpoint,state,operation,occurred_at,attempt,rows,fields,schema_version,
                    error_code,exception_type,exception_cause_type,safe_message,artifact_reference,
                    quality_evidence
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        fetch_id, dataset_id, scope_key, unit,
                        coverage_identity_key(dataset_id, scope_key, unit),
                        self.provider_id, endpoint, state, operation, occurred_at,
                        attempt, rows, json.dumps(field_values), schema_version,
                        error_code, exception_type, exception_cause_type,
                        safe_message, artifact_reference,
                        json.dumps(quality_evidence or {}, sort_keys=True, separators=(",", ":")),
                    )
                    for unit in unit_values
                ],
            )
            if owner:
                target.commit()
        except Exception:
            if owner:
                target.rollback()
            raise
        finally:
            if owner:
                target.close()

    def finish_fetch(self, fetch_id: str, *, status: str, finished_at: str, rows: int = 0, error_type: str | None = None, connection: sqlite3.Connection | None = None, retrieved_at: str | None = None, schema_version: str | None = None, canonical_hash: str | None = None, raw_reference: str | None = None, canonical_reference: str | None = None, manifest_reference: str | None = None, sdk_version: str | None = None, quality_conclusion: str | None = None) -> None:
        if status not in {"COMPLETE", "FAILED"}:
            raise ValueError("fetch status must be COMPLETE or FAILED.")
        owner = connection is None
        target = self._connect() if owner else connection
        assert target is not None
        try:
            cursor = target.execute("""UPDATE fetch_events SET
                finished_at=?,status=?,rows=?,error_type=?,retrieved_at=?,schema_version=?,
                canonical_hash=?,raw_reference=?,canonical_reference=?,manifest_reference=?,
                sdk_version=?,quality_conclusion=? WHERE fetch_id=?""", (
                    finished_at, status, rows, error_type, retrieved_at, schema_version,
                    canonical_hash, raw_reference, canonical_reference, manifest_reference,
                    sdk_version, quality_conclusion, fetch_id,
                ))
            if cursor.rowcount != 1:
                raise KeyError("Unknown fetch_id.")
            if owner:
                target.commit()
        except Exception:
            if owner:
                target.rollback()
            raise
        finally:
            if owner:
                target.close()

    def fetch_events(self) -> tuple[dict[str, object], ...]:
        with self._connect() as connection:
            return tuple(dict(row) for row in connection.execute("SELECT * FROM fetch_events WHERE provider_id=? ORDER BY started_at, fetch_id", (self.provider_id,)))

    def coverage_transitions(self, fetch_id: str | None = None) -> tuple[CoverageTransition, ...]:
        sql = "SELECT * FROM coverage_transitions WHERE provider_id=?"
        params: tuple[object, ...] = (self.provider_id,)
        if fetch_id is not None:
            sql += " AND fetch_id=?"
            params += (fetch_id,)
        sql += " ORDER BY transition_id"
        with self._connect() as connection:
            return tuple(
                CoverageTransition(**dict(row))
                for row in connection.execute(sql, params)
            )

    def latest_transition(
        self, dataset_id: str, scope_key: str, unit: str
    ) -> CoverageTransition | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM coverage_transitions
                WHERE provider_id=? AND dataset_id=? AND scope_key=? AND unit_key=?
                ORDER BY transition_id DESC LIMIT 1""",
                (self.provider_id, dataset_id, scope_key, unit),
            ).fetchone()
        return None if row is None else CoverageTransition(**dict(row))
