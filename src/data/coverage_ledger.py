"""Transactional SQLite coverage truth for Data Layer 2.0."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
from typing import Iterable, Iterator
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
                    CHECK (status IN ('STARTED','COMPLETE','FAILED')),
                    CHECK (rows >= 0)
                );
            """)
            coverage_columns = {row[1] for row in connection.execute("PRAGMA table_info(coverage_units)")}
            if "provider_id" not in coverage_columns:
                connection.execute("ALTER TABLE coverage_units ADD COLUMN provider_id TEXT NOT NULL DEFAULT 'tushare_official'")
            fetch_columns = {row[1] for row in connection.execute("PRAGMA table_info(fetch_events)")}
            if "provider_id" not in fetch_columns:
                connection.execute("ALTER TABLE fetch_events ADD COLUMN provider_id TEXT NOT NULL DEFAULT 'tushare_official'")

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

    def start_fetch(self, dataset_id: str, scope_key: str, units: Iterable[str], started_at: str) -> str:
        fetch_id = uuid4().hex
        with self._connect() as connection:
            connection.execute("INSERT INTO fetch_events(fetch_id,dataset_id,scope_key,requested_units,started_at,status,provider_id) VALUES (?,?,?,?,?,'STARTED',?)", (fetch_id, dataset_id, scope_key, json.dumps(tuple(units)), started_at, self.provider_id))
        return fetch_id

    def finish_fetch(self, fetch_id: str, *, status: str, finished_at: str, rows: int = 0, error_type: str | None = None) -> None:
        if status not in {"COMPLETE", "FAILED"}:
            raise ValueError("fetch status must be COMPLETE or FAILED.")
        with self._connect() as connection:
            cursor = connection.execute("UPDATE fetch_events SET finished_at=?,status=?,rows=?,error_type=? WHERE fetch_id=?", (finished_at, status, rows, error_type, fetch_id))
            if cursor.rowcount != 1:
                raise KeyError("Unknown fetch_id.")

    def fetch_events(self) -> tuple[dict[str, object], ...]:
        with self._connect() as connection:
            return tuple(dict(row) for row in connection.execute("SELECT * FROM fetch_events WHERE provider_id=? ORDER BY started_at, fetch_id", (self.provider_id,)))
