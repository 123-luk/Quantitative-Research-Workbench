"""Read-only DataManager/DataCache/ParquetStore status adapter."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from src.data.data_manager import DataManager


@dataclass(frozen=True)
class DatasetStatusView:
    dataset: str
    path: str
    exists: bool
    cached_start: str | None
    cached_end: str | None
    updated_at: str | None
    missing_ranges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DataStatusView:
    configured_data_root: str
    raw_data_root: str
    cache_metadata_path: str
    cache_status: str
    required_start_date: str
    required_end_date: str
    required_datasets: tuple[str, ...]
    datasets: tuple[DatasetStatusView, ...]


class DataStatusService:
    """Expose reliable local readiness and exact configured paths without writes."""

    def __init__(
        self,
        config_path: str | Path = "config/config.yaml",
        *,
        manager: DataManager | None = None,
    ) -> None:
        self._manager = manager or DataManager(config_path)

    def get_status(self) -> DataStatusView:
        manager = self._manager
        readiness = manager.prepare_data()
        required = tuple(manager.get_required_datasets())
        metadata = deepcopy(manager.cache.metadata)
        missing = readiness.get("missing_ranges", {})
        rows: list[DatasetStatusView] = []
        for name in required:
            item = metadata.get(name, {})
            ranges = missing.get(name, []) if isinstance(missing, dict) else []
            path = manager.parquet_store.get_dataset_path(name).resolve()
            rows.append(DatasetStatusView(
                dataset=name,
                path=str(path),
                exists=manager.parquet_store.exists(name),
                cached_start=item.get("start_date") if isinstance(item.get("start_date"), str) else None,
                cached_end=item.get("end_date") if isinstance(item.get("end_date"), str) else None,
                updated_at=item.get("updated_at") if isinstance(item.get("updated_at"), str) else None,
                missing_ranges=tuple((str(value[0]), str(value[1])) for value in ranges),
            ))
        data_config = manager.config.get("data", {})
        return DataStatusView(
            configured_data_root=str(Path(data_config.get("root", "data")).resolve()),
            raw_data_root=str(manager.parquet_store.root_dir.resolve()),
            cache_metadata_path=str(manager.cache.metadata_path.resolve()),
            cache_status=str(readiness["cache_status"]),
            required_start_date=str(readiness["required_start_date"]),
            required_end_date=str(readiness["required_end_date"]),
            required_datasets=required,
            datasets=tuple(rows),
        )

