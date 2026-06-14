"""Metadata cache for local dataset date coverage."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Tuple


DateRange = Tuple[str, str]


class DataCache:
    """Track cached dataset coverage ranges in a JSON metadata file."""

    def __init__(self, metadata_path: str | Path = "data/cache/data_status.json") -> None:
        """Initialize the metadata cache.

        Args:
            metadata_path: JSON file used to persist dataset coverage metadata.
        """
        self.metadata_path = Path(metadata_path)
        self.metadata: dict[str, dict[str, str]] = self.load()

    def load(self) -> dict[str, dict[str, str]]:
        """Load metadata from disk, treating a missing file as an empty cache."""
        if not self.metadata_path.exists():
            return {}

        with self.metadata_path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        if not isinstance(loaded, dict):
            raise ValueError(f"Cache metadata must be a JSON object: {self.metadata_path}")
        return loaded

    def save(self) -> None:
        """Persist the current metadata to disk."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, ensure_ascii=False, indent=2)

    def update_range(self, dataset_name: str, start_date: str, end_date: str) -> None:
        """Update one dataset's cached coverage range and save metadata.

        Existing coverage is extended when the new range reaches farther left
        or right. V1 keeps only one continuous coverage interval per dataset.
        """
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        self._validate_range(start, end)

        existing = self.metadata.get(dataset_name)
        if existing:
            existing_start = self._parse_date(existing["start_date"])
            existing_end = self._parse_date(existing["end_date"])
            start = min(start, existing_start)
            end = max(end, existing_end)

        self.metadata[dataset_name] = {
            "start_date": self._format_date(start),
            "end_date": self._format_date(end),
            "updated_at": datetime.now().replace(microsecond=0).isoformat(),
        }
        self.save()

    def is_covered(self, dataset_name: str, start_date: str, end_date: str) -> bool:
        """Return whether a dataset fully covers the requested date range."""
        return len(self.get_missing_ranges(dataset_name, start_date, end_date)) == 0

    def get_missing_ranges(
        self,
        dataset_name: str,
        start_date: str,
        end_date: str,
    ) -> list[DateRange]:
        """Return missing date ranges for a requested dataset interval.

        V1 assumes each dataset has at most one continuous cached coverage
        interval, so the result contains zero, one, or two ranges.
        """
        required_start = self._parse_date(start_date)
        required_end = self._parse_date(end_date)
        self._validate_range(required_start, required_end)

        existing = self.metadata.get(dataset_name)
        if not existing:
            return [(self._format_date(required_start), self._format_date(required_end))]

        cached_start = self._parse_date(existing["start_date"])
        cached_end = self._parse_date(existing["end_date"])
        if cached_start <= required_start and cached_end >= required_end:
            return []

        missing: list[DateRange] = []
        if cached_start > required_start:
            left_end = min(required_end, cached_start - timedelta(days=1))
            if required_start <= left_end:
                missing.append((self._format_date(required_start), self._format_date(left_end)))

        if cached_end < required_end:
            right_start = max(required_start, cached_end + timedelta(days=1))
            if right_start <= required_end:
                missing.append((self._format_date(right_start), self._format_date(required_end)))

        return missing

    @staticmethod
    def _parse_date(value: str) -> date:
        """Parse a YYYY-MM-DD date string."""
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"Date must use YYYY-MM-DD format: {value}") from exc

    @staticmethod
    def _format_date(value: date) -> str:
        """Format a date as YYYY-MM-DD."""
        return value.strftime("%Y-%m-%d")

    @staticmethod
    def _validate_range(start: date, end: date) -> None:
        """Validate that a date range is ordered."""
        if start > end:
            raise ValueError("start_date must be earlier than or equal to end_date.")


def normalize_date(value: Any) -> str:
    """Normalize common config date formats to YYYY-MM-DD."""
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")
    return DataCache._format_date(DataCache._parse_date(text))
