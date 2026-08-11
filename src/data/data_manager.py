"""Unified data preparation entrypoint for the V1 data layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from src.data.data_cache import DataCache, normalize_date
from src.data.parquet_store import ParquetStore


DEFAULT_REQUIRED_DATASETS = ["daily", "daily_basic", "adj_factor"]


class DataManager:
    """Coordinate local data cache checks before downstream research pipelines."""

    def __init__(self, config_path: str | Path = "config/config.yaml") -> None:
        """Initialize the manager from a YAML config file."""
        self.config_path = Path(config_path)
        self.config = self.load_config(self.config_path)
        data_config = self.config.get("data", {})
        cache_dir = Path(data_config.get("cache_dir", "data/cache"))
        raw_dir = Path(data_config.get("raw_dir", "data/raw"))
        parquet_engine = str(data_config.get("parquet_engine", "auto"))

        self.cache = DataCache(cache_dir / "data_status.json")
        self.parquet_store = ParquetStore(raw_dir, engine=parquet_engine)

    @staticmethod
    def load_config(config_path: str | Path) -> dict[str, Any]:
        """Load project YAML configuration."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a YAML mapping: {path}")
        return loaded

    def get_required_data_range(
        self,
        pipeline_config: Mapping[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Return the required start and end dates for local data preparation."""
        params = dict(pipeline_config or {})
        data_config = self.config.get("data", {})

        start_value = (
            params.get("required_start_date")
            or params.get("start_date")
            or data_config.get("required_start_date")
            or data_config.get("start_date")
        )
        end_value = (
            params.get("backtest_end")
            or params.get("end_date")
            or data_config.get("backtest_end")
            or data_config.get("end_date")
        )
        if start_value is None or end_value is None:
            raise ValueError("Both required start date and backtest end date are required.")

        return normalize_date(start_value), normalize_date(end_value)

    def get_required_datasets(
        self,
        pipeline_config: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Return datasets that must be present in the local cache."""
        params = dict(pipeline_config or {})
        data_config = self.config.get("data", {})
        datasets = (
            params.get("required_datasets")
            or data_config.get("required_datasets")
            or DEFAULT_REQUIRED_DATASETS
        )
        return [str(dataset) for dataset in datasets]

    def check_cache(
        self,
        start_date: str,
        end_date: str,
        datasets: list[str] | None = None,
    ) -> dict[str, list[list[str]]]:
        """Return missing cache ranges keyed by dataset name."""
        required_datasets = datasets or DEFAULT_REQUIRED_DATASETS
        missing_ranges: dict[str, list[list[str]]] = {}
        for dataset_name in required_datasets:
            ranges = self.cache.get_missing_ranges(dataset_name, start_date, end_date)
            if ranges:
                missing_ranges[dataset_name] = [[start, end] for start, end in ranges]
        return missing_ranges

    def prepare_data(
        self,
        pipeline_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Check local data readiness without downloading remote data.

        This V1 skeleton intentionally avoids TuShare calls. Later versions can
        use ``self.parquet_store`` and a TuShare-backed fetcher to fill the
        missing ranges returned here.
        """
        required_start, required_end = self.get_required_data_range(pipeline_config)
        datasets = self.get_required_datasets(pipeline_config)
        missing_ranges = self.check_cache(required_start, required_end, datasets)
        cache_status = "ready" if not missing_ranges else "missing"

        return {
            "cache_status": cache_status,
            "required_start_date": required_start,
            "required_end_date": required_end,
            "missing_ranges": missing_ranges,
        }

    def create_data_layer_2_service(self, *, open_dates=None, client_factory=None):
        """Create the explicit-write Data Layer 2.0 service on demand.

        ``DataManager`` construction and legacy ``prepare_data`` remain
        read-only. The SQLite catalog is created only at this explicit boundary.
        """
        from src.data.canonical_store import PartitionedParquetStore, RawParquetStore
        from src.data.coverage_ledger import CoverageLedger
        from src.data.dataset_registry import create_default_dataset_registry
        from src.data.preparation import DataPreparationService

        data_config = self.config.get("data", {})
        root = Path(data_config.get("root", "data"))
        curated_dir = Path(data_config.get("curated_dir", root / "curated"))
        metadata_dir = Path(data_config.get("metadata_dir", root / "metadata"))
        raw_dir = Path(data_config.get("raw_dir", root / "raw"))
        engine = str(data_config.get("parquet_engine", "auto"))
        parquet_engine = "pyarrow" if engine == "auto" else engine
        return DataPreparationService(
            registry=create_default_dataset_registry(),
            ledger=CoverageLedger(metadata_dir / "catalog.sqlite"),
            curated_store=PartitionedParquetStore(curated_dir, engine=parquet_engine),
            raw_store=RawParquetStore(raw_dir, engine=parquet_engine),
            open_dates=open_dates,
            client_factory=client_factory,
        )
