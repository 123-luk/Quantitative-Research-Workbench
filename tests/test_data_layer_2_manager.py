from __future__ import annotations

from pathlib import Path

from src.data.data_manager import DataManager


def test_legacy_data_manager_construction_and_status_remain_read_only(tmp_path: Path) -> None:
    root = tmp_path / "data"
    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n"
        f"  root: '{root.as_posix()}'\n"
        f"  raw_dir: '{(root / 'raw').as_posix()}'\n"
        f"  cache_dir: '{(root / 'cache').as_posix()}'\n"
        f"  curated_dir: '{(root / 'curated').as_posix()}'\n"
        f"  metadata_dir: '{(root / 'metadata').as_posix()}'\n"
        "  start_date: '20240101'\n"
        "  end_date: '20240103'\n"
        "  required_datasets: [daily]\n",
        encoding="utf-8",
    )
    manager = DataManager(config)
    assert manager.prepare_data()["cache_status"] == "missing"
    assert not (root / "metadata" / "catalog.sqlite").exists()


def test_data_manager_explicit_factory_uses_configured_layer_paths(tmp_path: Path) -> None:
    root = tmp_path / "data"
    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n"
        f"  root: '{root.as_posix()}'\n"
        f"  raw_dir: '{(root / 'raw').as_posix()}'\n"
        f"  cache_dir: '{(root / 'cache').as_posix()}'\n"
        f"  curated_dir: '{(root / 'curated').as_posix()}'\n"
        f"  metadata_dir: '{(root / 'metadata').as_posix()}'\n"
        "  start_date: '20240101'\n"
        "  end_date: '20240103'\n",
        encoding="utf-8",
    )
    service = DataManager(config).create_data_layer_2_service(open_dates=lambda _s, _e: ())
    assert service.ledger.path == root / "metadata" / "catalog.sqlite"
    assert service.curated_store.root == root / "curated"
    assert service.raw_store.root == root / "raw"


def test_explicit_empty_required_datasets_does_not_probe_legacy_cache(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("data: {}\n", encoding="utf-8")
    manager = DataManager(config)

    status = manager.prepare_data(
        {
            "required_start_date": "2023-01-01",
            "backtest_end": "2023-02-01",
            "required_datasets": [],
        }
    )

    assert status["cache_status"] == "ready"
    assert status["missing_ranges"] == {}
