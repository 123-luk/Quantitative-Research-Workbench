"""Tests for local data cache metadata."""

from __future__ import annotations

import json

from src.data.data_cache import DataCache, normalize_date


def test_data_cache_loads_empty_when_metadata_missing(tmp_path) -> None:
    """Missing metadata should be treated as an empty cache."""
    cache = DataCache(tmp_path / "data_status.json")

    assert cache.metadata == {}
    assert cache.load() == {}


def test_data_cache_update_range_saves_metadata(tmp_path) -> None:
    """update_range should persist metadata to JSON."""
    metadata_path = tmp_path / "data_status.json"
    cache = DataCache(metadata_path)

    cache.update_range("daily", "2020-01-01", "2025-12-31")

    loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert loaded["daily"]["start_date"] == "2020-01-01"
    assert loaded["daily"]["end_date"] == "2025-12-31"
    assert "updated_at" in loaded["daily"]


def test_data_cache_coverage_checks(tmp_path) -> None:
    """is_covered should reflect whether the requested range is fully cached."""
    cache = DataCache(tmp_path / "data_status.json")
    cache.update_range("daily", "2020-01-01", "2025-12-31")

    assert cache.is_covered("daily", "2021-01-01", "2025-03-31") is True
    assert cache.is_covered("daily", "2019-01-01", "2025-03-31") is False


def test_data_cache_missing_ranges_empty_cache(tmp_path) -> None:
    """An uncached dataset should report the full requested range as missing."""
    cache = DataCache(tmp_path / "data_status.json")

    assert cache.get_missing_ranges("daily", "2020-01-01", "2025-12-31") == [
        ("2020-01-01", "2025-12-31")
    ]


def test_data_cache_missing_ranges_when_covered(tmp_path) -> None:
    """A fully cached range should have no missing ranges."""
    cache = DataCache(tmp_path / "data_status.json")
    cache.update_range("daily", "2020-01-01", "2025-12-31")

    assert cache.get_missing_ranges("daily", "2021-01-01", "2025-03-31") == []


def test_data_cache_missing_ranges_left_and_right(tmp_path) -> None:
    """Partial coverage should report left-side and right-side gaps."""
    cache = DataCache(tmp_path / "data_status.json")
    cache.update_range("daily", "2020-01-01", "2025-12-31")

    assert cache.get_missing_ranges("daily", "2019-01-01", "2025-03-31") == [
        ("2019-01-01", "2019-12-31")
    ]
    assert cache.get_missing_ranges("daily", "2021-01-01", "2026-03-31") == [
        ("2026-01-01", "2026-03-31")
    ]


def test_normalize_date_accepts_yyyymmdd() -> None:
    """normalize_date should convert compact dates to ISO dates."""
    assert normalize_date("20240101") == "2024-01-01"
