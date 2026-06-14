"""Lightweight Parquet storage helpers for local market datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetStore:
    """Manage local parquet files under a dataset-oriented root directory."""

    def __init__(self, root_dir: str | Path, engine: str = "auto") -> None:
        """Initialize the store.

        Args:
            root_dir: Root directory for parquet datasets, such as ``data/raw``.
            engine: Pandas parquet engine. Use ``auto`` to let pandas choose
                between pyarrow and fastparquet.
        """
        self.root_dir = Path(root_dir)
        self.engine = engine

    def get_dataset_path(self, dataset_name: str) -> Path:
        """Return the parquet path for a dataset name.

        Examples:
            ``daily/000001.SZ`` becomes
            ``<root_dir>/daily/000001.SZ.parquet``.
        """
        clean_name = dataset_name.strip().replace("\\", "/")
        if not clean_name:
            raise ValueError("dataset_name must not be empty.")

        path = Path(clean_name)
        if path.is_absolute():
            raise ValueError("dataset_name must be a relative dataset path.")
        if path.suffix != ".parquet":
            path = path.with_suffix(".parquet")
        return self.root_dir / path

    def exists(self, dataset_name: str) -> bool:
        """Return whether a parquet dataset exists locally."""
        return self.get_dataset_path(dataset_name).exists()

    def save(self, dataset_name: str, df: pd.DataFrame) -> Path:
        """Save a DataFrame to parquet and return the output path.

        Raises:
            ImportError: If neither pyarrow nor fastparquet is available.
        """
        path = self.get_dataset_path(dataset_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path, index=False, engine=self.engine)
        except (ImportError, ValueError) as exc:
            self._raise_parquet_dependency_error(exc)
        return path

    def load(self, dataset_name: str) -> pd.DataFrame:
        """Load a parquet dataset into a DataFrame.

        Raises:
            FileNotFoundError: If the dataset does not exist.
            ImportError: If neither pyarrow nor fastparquet is available.
        """
        path = self.get_dataset_path(dataset_name)
        if not path.exists():
            raise FileNotFoundError(f"Parquet dataset not found: {path}")

        try:
            return pd.read_parquet(path, engine=self.engine)
        except (ImportError, ValueError) as exc:
            self._raise_parquet_dependency_error(exc)

    @staticmethod
    def _raise_parquet_dependency_error(exc: Exception) -> None:
        """Raise a clear dependency error when parquet support is unavailable."""
        message = str(exc).lower()
        if "pyarrow" in message or "fastparquet" in message or "parquet" in message:
            raise ImportError(
                "Parquet support requires installing either 'pyarrow' or "
                "'fastparquet'. Please install one of them and retry."
            ) from exc
        raise exc
