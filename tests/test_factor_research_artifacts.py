"""Tests for V2-G3 research artifact persistence and integrity checks."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.factors.evaluation import FactorEvaluationConfig
from src.factors.examples import register_example_factors
from src.factors.forward_returns import ForwardReturnConfig
from src.factors.preprocessing import PreprocessingConfig
from src.factors.quantile_evaluation import QuantileEvaluationConfig
from src.factors.registry import FactorRegistry
from src.factors.research_artifacts import (
    RESEARCH_RESULT_TABLES,
    FactorResearchArtifactStore,
    ResearchArtifactConfig,
)
from src.factors.research_pipeline import (
    FactorResearchConfig,
    FactorResearchResult,
    FactorResearchRunner,
)


def _result() -> FactorResearchResult:
    populated = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ts_code": pd.Series(
                ["000001.SZ", "000002.SZ"], index=[10, 20], dtype=object
            ),
            "value": [1.5, np.nan],
        },
        index=[10, 20],
    )
    empty = pd.DataFrame(
        {
            "trade_date": pd.Series(dtype="datetime64[ns]"),
            "factor_name": pd.Series(dtype=object),
            "weight": pd.Series(dtype=float),
        }
    )
    tables = {
        name: (empty.copy() if name == "weight_history" else populated.copy())
        for name in RESEARCH_RESULT_TABLES
    }
    return FactorResearchResult(
        requirements={"price_fields": ["trade_date", "ts_code", "close"]},
        **tables,
        factor_names=("momentum_20d", "volatility_20d"),
        used_neutralization=False,
        composition_method="equal",
        composite_score_col="composite_score",
        forward_return_col="forward_return",
    )


def _manifest_path(output: Path) -> Path:
    return output / "manifest.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _entry(manifest: dict, name: str) -> dict:
    return next(item for item in manifest["tables"] if item["name"] == name)


def _residuals(parent: Path, target_name: str, kind: str) -> list[Path]:
    return list(parent.glob(f".{target_name}.{kind}-*"))


def test_default_config_and_description_are_json_serializable() -> None:
    config = ResearchArtifactConfig()
    assert config.tables_dirname == "tables"
    assert config.manifest_filename == "manifest.json"
    assert config.compression == "snappy"
    assert config.include_empty_tables is True
    assert config.overwrite is False
    assert config.schema_version == "1"
    assert config.verify_after_write is True
    assert FactorResearchArtifactStore(config).describe_config() == config.to_dict()
    json.dumps(config.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("tables_dirname", "", ValueError),
        ("tables_dirname", ".", ValueError),
        ("tables_dirname", "..", ValueError),
        ("tables_dirname", "../tables", ValueError),
        ("tables_dirname", r"..\tables", ValueError),
        ("tables_dirname", "/absolute", ValueError),
        ("manifest_filename", "", ValueError),
        ("manifest_filename", "../manifest.json", ValueError),
        ("manifest_filename", "manifest.txt", ValueError),
        ("schema_version", "", ValueError),
        ("compression", "", TypeError),
        ("compression", 1, TypeError),
        ("include_empty_tables", 1, TypeError),
        ("overwrite", "false", TypeError),
        ("verify_after_write", 0, TypeError),
    ],
)
def test_invalid_config_values_raise(
    field: str, value: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        ResearchArtifactConfig(**{field: value})


def test_store_constructor_rejects_wrong_config() -> None:
    with pytest.raises(TypeError):
        FactorResearchArtifactStore(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("runner_config", []),
        ("metadata", []),
    ],
)
def test_save_requires_mapping_metadata(
    tmp_path: Path, argument: str, value: object
) -> None:
    with pytest.raises(TypeError):
        FactorResearchArtifactStore().save(
            _result(), tmp_path / "artifact", **{argument: value}
        )


@pytest.mark.parametrize("value", [object(), {1, 2}])
def test_unsupported_metadata_value_raises(tmp_path: Path, value: object) -> None:
    with pytest.raises(TypeError, match="metadata"):
        FactorResearchArtifactStore().save(
            _result(), tmp_path / "artifact", metadata={"bad": value}
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_metadata_value_raises(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        FactorResearchArtifactStore().save(
            _result(), tmp_path / "artifact", metadata={"bad": value}
        )


def test_invalid_result_output_type_and_existing_file_raise(tmp_path: Path) -> None:
    store = FactorResearchArtifactStore()
    with pytest.raises(TypeError):
        store.save(object(), tmp_path / "artifact")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        store.save(_result(), object())  # type: ignore[arg-type]
    output_file = tmp_path / "artifact"
    output_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        store.save(_result(), output_file)
    assert output_file.read_text(encoding="utf-8") == "occupied"


def test_save_round_trip_manifest_and_input_immutability(tmp_path: Path) -> None:
    result = _result()
    original_tables = {
        name: getattr(result, name).copy(deep=True)
        for name in RESEARCH_RESULT_TABLES
    }
    runner_config = {
        "method": "equal",
        "path": Path("relative/config.yaml"),
        "when": datetime(2024, 1, 2, 3, 4, 5),
        "numpy_int": np.int64(3),
    }
    metadata = {
        "label": "baseline",
        "day": date(2024, 1, 2),
        "nested": ("a", 2, True, None),
    }
    runner_before = deepcopy(runner_config)
    metadata_before = deepcopy(metadata)
    output = tmp_path / "artifact"

    manifest = FactorResearchArtifactStore().save(
        result,
        output,
        runner_config=runner_config,
        metadata=metadata,
    )

    assert output.is_dir()
    assert _manifest_path(output).is_file()
    assert (output / "tables").is_dir()
    assert manifest == _read_json(_manifest_path(output))
    assert manifest["artifact_type"] == "factor_research"
    assert manifest["schema_version"] == "1"
    assert datetime.fromisoformat(manifest["created_at_utc"]).tzinfo is not None
    assert manifest["requirements"] == result.requirements
    assert manifest["runner_config"]["path"] == "relative\\config.yaml"
    assert manifest["runner_config"]["when"] == "2024-01-02T03:04:05"
    assert manifest["runner_config"]["numpy_int"] == 3
    assert manifest["metadata"]["nested"] == ["a", 2, True, None]
    assert "table_shapes" in manifest["result_summary"]
    assert not any(
        isinstance(value, list)
        and value
        and isinstance(value[0], dict)
        for value in manifest["result_summary"].values()
    )
    assert runner_config == runner_before
    assert metadata == metadata_before
    for name, original in original_tables.items():
        pdt.assert_frame_equal(getattr(result, name), original)

    raw_text = _manifest_path(output).read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text
    assert str(tmp_path) not in raw_text
    assert not _residuals(tmp_path, "artifact", "staging")
    assert not _residuals(tmp_path, "artifact", "backup")


def test_table_metadata_and_registration_order_are_stable(tmp_path: Path) -> None:
    result = _result()
    output = tmp_path / "artifact"
    manifest = FactorResearchArtifactStore().save(result, output)

    assert RESEARCH_RESULT_TABLES == FactorResearchResult.TABLE_FIELDS
    assert [item["name"] for item in manifest["tables"]] == list(
        RESEARCH_RESULT_TABLES
    )
    for item in manifest["tables"]:
        source = getattr(result, item["name"])
        assert item["saved"] is True
        assert item["empty"] is source.empty
        assert item["rows"] == source.shape[0]
        assert item["columns"] == source.shape[1]
        assert item["column_names"] == [str(column) for column in source.columns]
        assert item["dtypes"] == {
            str(column): str(dtype) for column, dtype in source.dtypes.items()
        }
        assert item["relative_path"] == f"tables/{item['name']}.parquet"
        assert "\\" not in item["relative_path"]
        assert item["file_size_bytes"] > 0
        assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        assert (output / item["relative_path"]).is_file()


def test_load_tables_preserves_data_semantics_but_not_original_index(
    tmp_path: Path,
) -> None:
    result = _result()
    output = tmp_path / "artifact"
    store = FactorResearchArtifactStore()
    store.save(result, output)

    first = store.load_tables(output)
    second = store.load_tables(output)
    assert list(first) == list(RESEARCH_RESULT_TABLES)
    assert list(second) == list(RESEARCH_RESULT_TABLES)
    actual = first["raw_factor_panel"]
    expected = result.raw_factor_panel.reset_index(drop=True)
    assert actual.index.tolist() == [0, 1]
    assert len(actual) == len(expected)
    assert list(actual.columns) == list(expected.columns)
    pdt.assert_frame_equal(actual, expected, check_dtype=False)
    pdt.assert_frame_equal(actual, second["raw_factor_panel"])
    pdt.assert_series_equal(actual["trade_date"], expected["trade_date"])
    pdt.assert_series_equal(actual["value"], expected["value"])
    assert actual["ts_code"].tolist() == expected["ts_code"].tolist()
    assert actual["ts_code"].isna().tolist() == expected["ts_code"].isna().tolist()
    assert all(isinstance(value, str) for value in actual["ts_code"].dropna())
    assert pd.isna(actual.loc[1, "value"])


def test_include_empty_tables_false_records_but_skips_file(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    store = FactorResearchArtifactStore(
        ResearchArtifactConfig(include_empty_tables=False)
    )
    manifest = store.save(_result(), output)
    empty_entry = _entry(manifest, "weight_history")
    assert empty_entry["saved"] is False
    assert empty_entry["empty"] is True
    assert empty_entry["relative_path"] is None
    assert empty_entry["file_size_bytes"] is None
    assert empty_entry["sha256"] is None
    assert not (output / "tables" / "weight_history.parquet").exists()
    assert "weight_history" not in store.load_tables(output)

    loaded = store.load_tables(output, ["weight_history"])
    assert list(loaded) == ["weight_history"]
    assert loaded["weight_history"].empty
    assert list(loaded["weight_history"].columns) == empty_entry["column_names"]
    assert store.verify(output)["valid"] is True


def test_load_manifest_missing_malformed_and_wrong_structure_raise(
    tmp_path: Path,
) -> None:
    store = FactorResearchArtifactStore()
    output = tmp_path / "artifact"
    output.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest"):
        store.load_manifest(output)

    manifest_path = _manifest_path(output)
    manifest_path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        store.load_manifest(output)

    for document, match in [
        ({"artifact_type": "factor_research", "tables": []}, "schema_version"),
        (
            {"schema_version": "1", "artifact_type": "wrong", "tables": []},
            "artifact_type",
        ),
        (
            {
                "schema_version": "1",
                "artifact_type": "factor_research",
                "tables": {},
            },
            "tables",
        ),
    ]:
        _write_json(manifest_path, document)
        with pytest.raises(ValueError, match=match):
            store.load_manifest(output)


def test_manifest_duplicate_and_escaping_paths_are_rejected(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    store = FactorResearchArtifactStore()
    store.save(_result(), output)
    manifest = _read_json(_manifest_path(output))
    manifest["tables"].append(deepcopy(manifest["tables"][0]))
    _write_json(_manifest_path(output), manifest)
    with pytest.raises(ValueError, match="duplicate"):
        store.load_manifest(output)

    store = FactorResearchArtifactStore(
        ResearchArtifactConfig(overwrite=True, verify_after_write=False)
    )
    store.save(_result(), output)
    manifest = _read_json(_manifest_path(output))
    manifest["tables"][0]["relative_path"] = "../outside.parquet"
    _write_json(_manifest_path(output), manifest)
    with pytest.raises(ValueError, match="Unsafe|escapes"):
        store.load_manifest(output)


def test_load_selected_tables_validates_names_and_uses_manifest_order(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    store = FactorResearchArtifactStore()
    store.save(_result(), output)
    loaded = store.load_tables(
        output, ["forward_returns", "raw_factor_panel"]
    )
    assert list(loaded) == ["raw_factor_panel", "forward_returns"]
    with pytest.raises(ValueError, match="duplicate"):
        store.load_tables(output, ["raw_factor_panel", "raw_factor_panel"])
    with pytest.raises(ValueError, match="empty"):
        store.load_tables(output, [""])
    with pytest.raises(KeyError, match="unknown"):
        store.load_tables(output, ["unknown"])
    with pytest.raises(TypeError):
        store.load_tables(output, "raw_factor_panel")


def test_verify_complete_artifact_and_missing_file_reports_all_tables(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    store = FactorResearchArtifactStore()
    store.save(_result(), output)
    valid = store.verify(output)
    assert valid["valid"] is True
    assert valid["manifest_valid"] is True
    assert valid["checked_tables"] == len(RESEARCH_RESULT_TABLES)
    assert valid["valid_tables"] == len(RESEARCH_RESULT_TABLES)
    assert len(valid["table_results"]) == len(RESEARCH_RESULT_TABLES)
    assert all(item["actual_dtypes"] is not None for item in valid["table_results"])

    (output / "tables" / "raw_factor_panel.parquet").unlink()
    invalid = store.verify(output)
    assert invalid["valid"] is False
    assert invalid["checked_tables"] == len(RESEARCH_RESULT_TABLES)
    assert invalid["valid_tables"] == len(RESEARCH_RESULT_TABLES) - 1
    assert any("missing" in error for error in invalid["errors"])
    assert any(
        item["name"] == "forward_returns" and item["valid"]
        for item in invalid["table_results"]
    )
    with pytest.raises(ValueError, match="verification"):
        store.load_tables(output, ["raw_factor_panel"])


def test_verify_detects_file_size_and_hash_tampering(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    store = FactorResearchArtifactStore()
    store.save(_result(), output)
    table_path = output / "tables" / "raw_factor_panel.parquet"
    before_files = sorted(
        (path.relative_to(output).as_posix(), path.stat().st_size)
        for path in output.rglob("*")
        if path.is_file()
    )
    with table_path.open("ab") as handle:
        handle.write(b"tampered")
    invalid = store.verify(output)
    assert invalid["valid"] is False
    assert any("file size mismatch" in error for error in invalid["errors"])
    assert any("SHA-256 mismatch" in error for error in invalid["errors"])

    after_verify = sorted(
        (path.relative_to(output).as_posix(), path.stat().st_size)
        for path in output.rglob("*")
        if path.is_file()
    )
    assert after_verify == [
        (name, size + 8 if name.endswith("raw_factor_panel.parquet") else size)
        for name, size in before_files
    ]


@pytest.mark.parametrize(
    ("field", "mutator", "message"),
    [
        ("rows", lambda value: value + 1, "row count mismatch"),
        ("column_names", lambda value: list(reversed(value)), "column names mismatch"),
    ],
)
def test_verify_detects_manifest_shape_tampering(
    tmp_path: Path, field: str, mutator, message: str
) -> None:
    output = tmp_path / "artifact"
    store = FactorResearchArtifactStore()
    store.save(_result(), output)
    manifest = _read_json(_manifest_path(output))
    entry = _entry(manifest, "raw_factor_panel")
    entry[field] = mutator(entry[field])
    _write_json(_manifest_path(output), manifest)
    report = store.verify(output)
    assert report["valid"] is False
    assert any(message in error for error in report["errors"])


def test_verify_rejects_unsaved_nonempty_entry(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    store = FactorResearchArtifactStore(
        ResearchArtifactConfig(
            include_empty_tables=False, verify_after_write=False
        )
    )
    store.save(_result(), output)
    manifest = _read_json(_manifest_path(output))
    entry = _entry(manifest, "weight_history")
    entry["empty"] = False
    _write_json(_manifest_path(output), manifest)
    report = store.verify(output)
    assert report["valid"] is False
    assert any("saved=false but empty=false" in error for error in report["errors"])


def test_verify_unreadable_manifest_returns_stable_report(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    report = FactorResearchArtifactStore().verify(output)
    assert report == {
        "valid": False,
        "manifest_valid": False,
        "checked_tables": 0,
        "valid_tables": 0,
        "errors": [f"Artifact manifest is missing: {output / 'manifest.json'}"],
        "table_results": [],
    }


def test_overwrite_false_preserves_existing_artifact(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    store = FactorResearchArtifactStore()
    store.save(_result(), output)
    before = _manifest_path(output).read_bytes()
    with pytest.raises(FileExistsError):
        store.save(_result(), output)
    assert _manifest_path(output).read_bytes() == before
    assert not _residuals(tmp_path, "artifact", "staging")


def test_overwrite_true_replaces_artifact_without_residue(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    FactorResearchArtifactStore().save(
        _result(), output, metadata={"version": 1}
    )
    store = FactorResearchArtifactStore(ResearchArtifactConfig(overwrite=True))
    manifest = store.save(_result(), output, metadata={"version": 2})
    assert manifest["metadata"] == {"version": 2}
    assert store.verify(output)["valid"] is True
    assert not _residuals(tmp_path, "artifact", "staging")
    assert not _residuals(tmp_path, "artifact", "backup")


def test_write_failure_cleans_staging_and_preserves_old_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    FactorResearchArtifactStore().save(
        _result(), output, metadata={"version": 1}
    )
    before = _manifest_path(output).read_bytes()

    def fail_write(*args, **kwargs) -> None:
        raise RuntimeError("simulated parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    store = FactorResearchArtifactStore(ResearchArtifactConfig(overwrite=True))
    with pytest.raises(RuntimeError, match="simulated"):
        store.save(_result(), output, metadata={"version": 2})
    assert _manifest_path(output).read_bytes() == before
    assert not _residuals(tmp_path, "artifact", "staging")
    assert not _residuals(tmp_path, "artifact", "backup")


def test_manifest_is_written_only_after_all_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    original = pd.DataFrame.to_parquet
    calls = 0

    def observing_write(frame, path, *args, **kwargs):
        nonlocal calls
        staging = _residuals(tmp_path, "artifact", "staging")
        assert len(staging) == 1
        assert not (staging[0] / "manifest.json").exists()
        calls += 1
        return original(frame, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", observing_write)
    FactorResearchArtifactStore().save(_result(), output)
    assert calls == len(RESEARCH_RESULT_TABLES)
    assert _manifest_path(output).is_file()


def test_verify_after_write_is_invoked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    calls: list[Path] = []
    original = FactorResearchArtifactStore.verify

    def recording_verify(self, path):
        calls.append(Path(path))
        return original(self, path)

    monkeypatch.setattr(FactorResearchArtifactStore, "verify", recording_verify)
    FactorResearchArtifactStore().save(_result(), output)
    assert len(calls) == 1
    assert calls[0].name.startswith(".artifact.staging-")


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_infinity_in_result_table_is_rejected(
    tmp_path: Path, bad_value: float
) -> None:
    result = _result()
    result.raw_factor_panel.loc[0, "value"] = bad_value
    with pytest.raises(ValueError, match="raw_factor_panel.*value"):
        FactorResearchArtifactStore().save(result, tmp_path / "artifact")
    assert not _residuals(tmp_path, "artifact", "staging")


def test_object_strings_are_not_coerced_when_checking_infinity(tmp_path: Path) -> None:
    result = _result()
    result.raw_factor_panel["text"] = pd.Series(
        ["Infinity", "123"], index=result.raw_factor_panel.index, dtype=object
    )
    output = tmp_path / "artifact"
    store = FactorResearchArtifactStore()
    store.save(result, output)
    loaded = store.load_tables(output, ["raw_factor_panel"])
    assert loaded["raw_factor_panel"]["text"].tolist() == ["Infinity", "123"]


def _real_runner_and_result() -> tuple[FactorResearchRunner, FactorResearchResult]:
    registry = FactorRegistry()
    register_example_factors(registry)
    runner = FactorResearchRunner(
        registry,
        FactorResearchConfig(
            factor_names=("momentum_20d", "volatility_20d"),
            composition_method="equal",
        ),
        preprocessing_config=PreprocessingConfig(
            missing_method="none",
            winsor_method="none",
            standardize_method="zscore",
            min_cross_section_size=5,
        ),
        evaluation_config=FactorEvaluationConfig(min_cross_section_size=5),
        quantile_config=QuantileEvaluationConfig(
            quantiles=5,
            min_cross_section_size=5,
            min_group_size=1,
        ),
        forward_return_config=ForwardReturnConfig(
            entry_lag_periods=1, holding_periods=1
        ),
    )
    dates = pd.bdate_range("2024-01-02", periods=32)
    codes = [f"S{index:02d}" for index in range(10)]
    factor_rows: list[dict] = []
    price_rows: list[dict] = []
    for date_index, trade_date in enumerate(dates):
        for stock_index, code in enumerate(codes):
            close = (
                100.0
                * (1.0 + 0.001 * (stock_index + 1)) ** date_index
                * (1.0 + 0.002 * np.sin(date_index + stock_index))
            )
            price_rows.append(
                {"trade_date": trade_date, "ts_code": code, "close": close}
            )
            if date_index < 30:
                factor_rows.append(
                    {"trade_date": trade_date, "ts_code": code, "close": close}
                )
    score_panel = pd.DataFrame(
        [
            {"trade_date": trade_date, "ts_code": code}
            for trade_date in dates[22:26]
            for code in codes
        ]
    )
    result = runner.run(
        pd.DataFrame(factor_rows),
        score_panel,
        pd.DataFrame(price_rows),
    )
    return runner, result


def test_real_g2_to_g3_round_trip_uses_only_tmp_path(tmp_path: Path) -> None:
    runner, result = _real_runner_and_result()
    before = {
        name: getattr(result, name).copy(deep=True)
        for name in RESEARCH_RESULT_TABLES
    }
    output = tmp_path / "g2-g3-artifact"
    store = FactorResearchArtifactStore()
    manifest = store.save(
        result,
        output,
        runner_config=runner.describe_config(),
        metadata={"purpose": "G2-G3 integration"},
    )
    loaded = store.load_tables(output)

    assert len(loaded) == len(RESEARCH_RESULT_TABLES)
    assert list(loaded) == list(RESEARCH_RESULT_TABLES)
    assert manifest["result_summary"] == json.loads(
        json.dumps(result.to_dict(), allow_nan=False)
    )
    manifest_shapes = {
        item["name"]: [item["rows"], item["columns"]]
        for item in manifest["tables"]
    }
    assert manifest_shapes == {
        name: list(shape) for name, shape in result.table_shapes().items()
    }
    assert manifest["runner_config"] == json.loads(
        json.dumps(runner.describe_config(), allow_nan=False)
    )
    assert store.verify(output)["valid"] is True
    for name, original in before.items():
        pdt.assert_frame_equal(getattr(result, name), original)
    assert set(path.name for path in output.iterdir()) == {"manifest.json", "tables"}
