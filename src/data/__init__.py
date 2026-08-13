"""Data access, caching, and preparation package."""

from src.data.data_cache import DataCache, normalize_date
from src.data.data_manager import DataManager
from src.data.parquet_store import ParquetStore
from src.data.canonical_store import PartitionedParquetStore, RawParquetStore
from src.data.provider_registry import ProviderClientFactory, ProviderId, ProviderRegistry
from src.data.provider_contracts import ProviderContractRegistry
from src.data.contracts import DataRequirement, DatasetSpec, ResearchFrequency
from src.data.coverage_ledger import CoverageLedger
from src.data.coverage_planner import MissingDataPlanner
from src.data.dataset_registry import DatasetRegistry, create_default_dataset_registry
from src.data.preparation import DataPreparationService

__all__ = [
    "DataCache",
    "DataManager",
    "ParquetStore",
    "CoverageLedger",
    "DataPreparationService",
    "DataRequirement",
    "DatasetRegistry",
    "DatasetSpec",
    "MissingDataPlanner",
    "PartitionedParquetStore",
    "ProviderClientFactory",
    "ProviderContractRegistry",
    "ProviderId",
    "ProviderRegistry",
    "RawParquetStore",
    "ResearchFrequency",
    "create_default_dataset_registry",
    "normalize_date",
]
