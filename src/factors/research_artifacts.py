"""Persist and validate :class:`FactorResearchResult` research artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from numbers import Real
import shutil
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from src.factors.research_pipeline import FactorResearchResult


ARTIFACT_TYPE = "factor_research"
RESEARCH_RESULT_TABLES = tuple(FactorResearchResult.TABLE_FIELDS)
_SHA256_CHUNK_SIZE = 1024 * 1024
_TABLE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_TABLE_ENTRY_FIELDS = (
    "name",
    "saved",
    "empty",
    "relative_path",
    "rows",
    "columns",
    "column_names",
    "dtypes",
    "file_size_bytes",
    "sha256",
)


def _validate_single_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
        or Path(value).is_absolute()
        or Path(value).name != value
    ):
        raise ValueError(f"{field_name} must be one safe path component.")
    return value


def _coerce_output_dir(output_dir: str | Path) -> Path:
    if not isinstance(output_dir, (str, Path)):
        raise TypeError("output_dir must be a str or pathlib.Path.")
    path = Path(output_dir)
    if not path.name:
        raise ValueError("output_dir must identify a directory.")
    return path


def _json_safe(value: Any, *, location: str) -> Any:
    """Return an explicit JSON-safe copy without silently stringifying objects."""
    if isinstance(value, np.generic):
        return _json_safe(value.item(), location=location)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite float.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} mapping keys must be strings.")
            converted[key] = _json_safe(item, location=f"{location}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{location} contains unsupported value type "
        f"{type(value).__name__!r}."
    )


def _mapping_snapshot(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a Mapping or None.")
    converted = _json_safe(value, location=name)
    if not isinstance(converted, dict):  # pragma: no cover - guarded by Mapping
        raise TypeError(f"{name} must serialize to a JSON object.")
    return converted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA256_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infinite_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        found = False
        for value in frame[column].array:
            if value is None or value is pd.NA or value is pd.NaT:
                continue
            if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
                try:
                    found = math.isinf(float(value))
                except (TypeError, ValueError, OverflowError):
                    found = False
            if found:
                columns.append(str(column))
                break
    return columns


def _safe_artifact_path(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("relative_path must be a non-empty string.")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe artifact relative_path: {relative_path!r}.")
    candidate = root / relative
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Artifact relative_path escapes output_dir: {relative_path!r}."
        ) from exc
    return candidate


@dataclass(frozen=True)
class ResearchArtifactConfig:
    """Configure staging-based, auditable research artifact persistence."""

    tables_dirname: str = "tables"
    manifest_filename: str = "manifest.json"
    compression: str | None = "snappy"
    include_empty_tables: bool = True
    overwrite: bool = False
    schema_version: str = "1"
    verify_after_write: bool = True

    def __post_init__(self) -> None:
        _validate_single_name(self.tables_dirname, "tables_dirname")
        manifest_filename = _validate_single_name(
            self.manifest_filename, "manifest_filename"
        )
        if not manifest_filename.lower().endswith(".json"):
            raise ValueError("manifest_filename must end with '.json'.")
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("schema_version must be a non-empty string.")
        if self.compression is not None and (
            not isinstance(self.compression, str) or not self.compression
        ):
            raise TypeError("compression must be None or a non-empty string.")
        for field_name in (
            "include_empty_tables",
            "overwrite",
            "verify_after_write",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration snapshot."""
        return asdict(self)


class FactorResearchArtifactStore:
    """Save, load, and verify research tables without recomputing them."""

    def __init__(self, config: ResearchArtifactConfig | None = None) -> None:
        if config is not None and not isinstance(config, ResearchArtifactConfig):
            raise TypeError("config must be ResearchArtifactConfig or None.")
        self.config = config or ResearchArtifactConfig()

    def describe_config(self) -> dict[str, Any]:
        """Return the active persistence configuration."""
        return self.config.to_dict()

    def save(
        self,
        result: FactorResearchResult,
        output_dir: str | Path,
        *,
        runner_config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a complete artifact to staging, verify it, then publish it."""
        if not isinstance(result, FactorResearchResult):
            raise TypeError("result must be a FactorResearchResult.")
        target = _coerce_output_dir(output_dir)
        if target.exists() and not target.is_dir():
            raise FileExistsError(f"output_dir is an existing file: {target}")
        if target.exists() and not self.config.overwrite:
            raise FileExistsError(f"output_dir already exists: {target}")

        runner_snapshot = _mapping_snapshot(runner_config, "runner_config")
        metadata_snapshot = _mapping_snapshot(metadata, "metadata")
        result_summary = _json_safe(result.to_dict(), location="result_summary")
        requirements = _json_safe(result.requirements, location="requirements")

        frames: dict[str, pd.DataFrame] = {}
        for table_name in RESEARCH_RESULT_TABLES:
            frame = getattr(result, table_name, None)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"FactorResearchResult.{table_name} must be a pandas DataFrame."
                )
            infinite_columns = _infinite_columns(frame)
            if infinite_columns:
                raise ValueError(
                    f"Table {table_name!r} contains infinity in columns "
                    f"{infinite_columns!r}."
                )
            frames[table_name] = frame

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
        backup: Path | None = None
        staging.mkdir(exist_ok=False)
        try:
            tables_dir = staging / self.config.tables_dirname
            tables_dir.mkdir()
            table_entries: list[dict[str, Any]] = []
            for table_name, frame in frames.items():
                entry = self._write_table(table_name, frame, tables_dir)
                table_entries.append(entry)

            manifest = {
                "schema_version": self.config.schema_version,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "artifact_type": ARTIFACT_TYPE,
                "result_summary": result_summary,
                "requirements": requirements,
                "runner_config": runner_snapshot,
                "metadata": metadata_snapshot,
                "tables": table_entries,
            }
            manifest_path = staging / self.config.manifest_filename
            with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    manifest,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")

            if self.config.verify_after_write:
                report = self.verify(staging)
                if not report["valid"]:
                    details = "; ".join(report["errors"])
                    raise ValueError(
                        f"Research artifact verification failed: {details}"
                    )

            if target.exists():
                backup = target.parent / f".{target.name}.backup-{uuid4().hex}"
                os.replace(target, backup)
                try:
                    os.replace(staging, target)
                except Exception:
                    os.replace(backup, target)
                    backup = None
                    raise
                shutil.rmtree(backup)
                backup = None
            else:
                os.replace(staging, target)
            return manifest
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
                backup = None
            raise
        finally:
            if backup is not None and backup.exists() and target.exists():
                shutil.rmtree(backup)

    def _write_table(
        self, table_name: str, frame: pd.DataFrame, tables_dir: Path
    ) -> dict[str, Any]:
        empty = frame.empty
        saved = self.config.include_empty_tables or not empty
        entry: dict[str, Any] = {
            "name": table_name,
            "saved": saved,
            "empty": empty,
            "relative_path": None,
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "column_names": [str(column) for column in frame.columns],
            "dtypes": {
                str(column): str(dtype)
                for column, dtype in frame.dtypes.items()
            },
            "file_size_bytes": None,
            "sha256": None,
        }
        if not saved:
            return entry

        file_path = tables_dir / f"{table_name}.parquet"
        frame.to_parquet(
            file_path,
            index=False,
            compression=self.config.compression,
        )
        entry.update(
            {
                "relative_path": (
                    Path(self.config.tables_dirname) / file_path.name
                ).as_posix(),
                "file_size_bytes": file_path.stat().st_size,
                "sha256": _sha256(file_path),
            }
        )
        return entry

    def load_manifest(self, output_dir: str | Path) -> dict[str, Any]:
        """Load and structurally validate an artifact manifest."""
        root = _coerce_output_dir(output_dir)
        if not root.exists():
            raise FileNotFoundError(f"Artifact output_dir does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Artifact output_dir is not a directory: {root}")
        manifest_path = root / self.config.manifest_filename
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Artifact manifest is missing: {manifest_path}")
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Artifact manifest is invalid JSON: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise ValueError("Artifact manifest root must be a JSON object.")
        if not isinstance(manifest.get("schema_version"), str) or not manifest[
            "schema_version"
        ]:
            raise ValueError("Artifact manifest has no valid schema_version.")
        if manifest.get("artifact_type") != ARTIFACT_TYPE:
            raise ValueError(
                f"Artifact manifest artifact_type must be {ARTIFACT_TYPE!r}."
            )
        tables = manifest.get("tables")
        if not isinstance(tables, list):
            raise ValueError("Artifact manifest tables must be a list.")

        names: set[str] = set()
        for index, entry in enumerate(tables):
            if not isinstance(entry, dict):
                raise ValueError(f"Artifact manifest table entry {index} must be an object.")
            name = entry.get("name")
            if not isinstance(name, str) or not _TABLE_NAME_PATTERN.fullmatch(name):
                raise ValueError(f"Artifact manifest table entry {index} has unsafe name.")
            if name in names:
                raise ValueError(f"Artifact manifest contains duplicate table {name!r}.")
            names.add(name)
            saved = entry.get("saved")
            if not isinstance(saved, bool):
                raise ValueError(f"Artifact manifest table {name!r} has invalid saved.")
            relative_path = entry.get("relative_path")
            if saved:
                _safe_artifact_path(root, relative_path)
            elif relative_path is not None:
                raise ValueError(
                    f"Unsaved artifact table {name!r} must have null relative_path."
                )
        return manifest

    def load_tables(
        self,
        output_dir: str | Path,
        table_names: Sequence[str] | None = None,
        *,
        verify: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Load all saved tables or a stable-order selected subset."""
        if not isinstance(verify, bool):
            raise TypeError("verify must be a bool.")
        root = _coerce_output_dir(output_dir)
        manifest = self.load_manifest(root)
        entries = manifest["tables"]
        entry_by_name = {entry["name"]: entry for entry in entries}

        requested: set[str] | None = None
        if table_names is not None:
            if isinstance(table_names, (str, bytes)) or not isinstance(
                table_names, Sequence
            ):
                raise TypeError("table_names must be a sequence of table names or None.")
            values = list(table_names)
            if any(not isinstance(name, str) or not name for name in values):
                raise ValueError("table_names cannot contain empty or non-string names.")
            if len(set(values)) != len(values):
                raise ValueError("table_names cannot contain duplicate names.")
            unknown = [name for name in values if name not in entry_by_name]
            if unknown:
                raise KeyError(f"Unknown artifact table names: {unknown!r}")
            requested = set(values)

        selected = [
            entry
            for entry in entries
            if (requested is None and entry["saved"])
            or (requested is not None and entry["name"] in requested)
        ]
        if verify:
            table_report = self._verify_table_entries(root, selected)
            if table_report["errors"]:
                raise ValueError(
                    "Artifact table verification failed: "
                    + "; ".join(table_report["errors"])
                )

        loaded: dict[str, pd.DataFrame] = {}
        for entry in selected:
            name = entry["name"]
            self._validate_table_entry(entry)
            if not entry["saved"]:
                loaded[name] = pd.DataFrame(columns=entry["column_names"])
                continue
            path = _safe_artifact_path(root, entry["relative_path"])
            frame = pd.read_parquet(path)
            self._validate_loaded_shape(name, frame, entry)
            loaded[name] = frame
        return loaded

    def verify(self, output_dir: str | Path) -> dict[str, Any]:
        """Return a complete integrity report without modifying the artifact."""
        report: dict[str, Any] = {
            "valid": False,
            "manifest_valid": False,
            "checked_tables": 0,
            "valid_tables": 0,
            "errors": [],
            "table_results": [],
        }
        try:
            root = _coerce_output_dir(output_dir)
            manifest = self.load_manifest(root)
        except Exception as exc:
            report["errors"].append(str(exc))
            return report

        report["manifest_valid"] = True
        checked = self._verify_table_entries(root, manifest["tables"])
        report["checked_tables"] = checked["checked_tables"]
        report["valid_tables"] = checked["valid_tables"]
        report["errors"].extend(checked["errors"])
        report["table_results"] = checked["table_results"]
        report["valid"] = not report["errors"]
        return report

    def _verify_table_entries(
        self, root: Path, entries: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        errors: list[str] = []
        table_results: list[dict[str, Any]] = []
        valid_tables = 0
        for index, entry in enumerate(entries):
            table_errors: list[str] = []
            name = (
                entry.get("name")
                if isinstance(entry, dict)
                and isinstance(entry.get("name"), str)
                else f"<entry-{index}>"
            )
            actual_dtypes: dict[str, str] | None = None
            try:
                self._validate_table_entry(entry)
                if entry["saved"]:
                    path = _safe_artifact_path(root, entry["relative_path"])
                    if not path.is_file():
                        table_errors.append(f"Table {name!r} file is missing: {path}")
                    else:
                        actual_size = path.stat().st_size
                        if actual_size != entry["file_size_bytes"]:
                            table_errors.append(
                                f"Table {name!r} file size mismatch: "
                                f"expected {entry['file_size_bytes']}, got {actual_size}."
                            )
                        actual_hash = _sha256(path)
                        if actual_hash != entry["sha256"]:
                            table_errors.append(f"Table {name!r} SHA-256 mismatch.")
                        try:
                            frame = pd.read_parquet(path)
                        except Exception as exc:
                            table_errors.append(
                                f"Table {name!r} Parquet cannot be read: {exc}"
                            )
                        else:
                            actual_dtypes = {
                                str(column): str(dtype)
                                for column, dtype in frame.dtypes.items()
                            }
                            try:
                                self._validate_loaded_shape(name, frame, entry)
                            except ValueError as exc:
                                table_errors.append(str(exc))
                elif not entry["empty"]:
                    table_errors.append(
                        f"Table {name!r} has saved=false but empty=false."
                    )
            except Exception as exc:
                table_errors.append(f"Table {name!r} manifest entry is invalid: {exc}")

            if not table_errors:
                valid_tables += 1
            errors.extend(table_errors)
            table_results.append(
                {
                    "name": name,
                    "valid": not table_errors,
                    "errors": table_errors,
                    "actual_dtypes": actual_dtypes,
                }
            )
        return {
            "checked_tables": len(entries),
            "valid_tables": valid_tables,
            "errors": errors,
            "table_results": table_results,
        }

    @staticmethod
    def _validate_table_entry(entry: object) -> None:
        if not isinstance(entry, dict):
            raise ValueError("table entry must be an object.")
        missing = [field for field in _TABLE_ENTRY_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"missing fields {missing!r}.")
        name = entry["name"]
        if not isinstance(name, str) or not _TABLE_NAME_PATTERN.fullmatch(name):
            raise ValueError("name must be safe snake_case.")
        for field_name in ("saved", "empty"):
            if not isinstance(entry[field_name], bool):
                raise ValueError(f"{field_name} must be a bool.")
        for field_name in ("rows", "columns"):
            if (
                not isinstance(entry[field_name], int)
                or isinstance(entry[field_name], bool)
                or entry[field_name] < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer.")
        if (
            not isinstance(entry["column_names"], list)
            or any(not isinstance(value, str) for value in entry["column_names"])
            or len(entry["column_names"]) != entry["columns"]
        ):
            raise ValueError("column_names must match columns.")
        if not isinstance(entry["dtypes"], dict):
            raise ValueError("dtypes must be an object.")
        if entry["saved"]:
            if (
                not isinstance(entry["file_size_bytes"], int)
                or entry["file_size_bytes"] <= 0
            ):
                raise ValueError("saved table file_size_bytes must be positive.")
            if not isinstance(entry["sha256"], str) or not re.fullmatch(
                r"[0-9a-f]{64}", entry["sha256"]
            ):
                raise ValueError("saved table sha256 must be lowercase hexadecimal.")
        elif entry["file_size_bytes"] is not None or entry["sha256"] is not None:
            raise ValueError("unsaved table file metadata must be null.")

    @staticmethod
    def _validate_loaded_shape(
        name: str, frame: pd.DataFrame, entry: Mapping[str, Any]
    ) -> None:
        if frame.shape[0] != entry["rows"]:
            raise ValueError(
                f"Table {name!r} row count mismatch: "
                f"expected {entry['rows']}, got {frame.shape[0]}."
            )
        if frame.shape[1] != entry["columns"]:
            raise ValueError(
                f"Table {name!r} column count mismatch: "
                f"expected {entry['columns']}, got {frame.shape[1]}."
            )
        actual_columns = [str(column) for column in frame.columns]
        if actual_columns != entry["column_names"]:
            raise ValueError(
                f"Table {name!r} column names mismatch: "
                f"expected {entry['column_names']!r}, got {actual_columns!r}."
            )
