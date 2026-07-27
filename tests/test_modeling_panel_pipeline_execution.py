"""Tests for the V4-E1 Modeling Panel Pipeline executor."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pandas as pd
import pytest

from src.modeling_panel import ModelingPanelArtifactStore, ModelingPanelBuilder
from src.pipeline import (
    FactorResearchExecutionResult,
    FactorResearchPublishedOutputs,
    ModelingPanelOutputConfig,
    ModelingPanelPipelineConfig,
    ModelingPanelPipelineExecutionError,
    ModelingPanelPipelineExecutor,
    ModelingPanelPipelineResult,
    ModelingPanelSourceConfig,
)


FEATURES = ("factor_a", "factor_b")


def _factor_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"],
            "ts_code": ["B", "A", "A", "B"],
            "factor_a": [3.0, 1.0, 4.0, 2.0],
            "factor_b": [30.0, 10.0, 40.0, 20.0],
        }
    )


def _forward_returns(label: str = "forward_return") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "ts_code": ["B", "A", "A", "B"],
            "entry_trade_date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
            "exit_trade_date": ["2024-01-03", "2024-01-04", "2024-01-03", "2024-01-04"],
            "entry_price": [20.0, 40.0, 10.0, 30.0],
            "exit_price": [18.0, 44.0, 11.0, 33.0],
            label: [-0.1, 0.1, 0.1, 0.1],
        }
    )


def _write_inputs(root: Path, *, label: str = "forward_return") -> tuple[Path, Path]:
    root.mkdir(parents=True)
    factor_path = root / "factors.parquet"
    returns_path = root / "returns.parquet"
    _factor_panel().to_parquet(factor_path, index=False)
    _forward_returns(label).to_parquet(returns_path, index=False)
    return factor_path, returns_path


def _files_config(
    factor_path: str | Path,
    returns_path: str | Path,
    **builder: object,
) -> ModelingPanelPipelineConfig:
    return ModelingPanelPipelineConfig.from_dict(
        {
            "enabled": True,
            "source": {
                "mode": "files",
                "factor_panel_path": factor_path,
                "forward_returns_path": returns_path,
            },
            "builder": builder,
        }
    )


def _research_result(
    root: Path,
    *,
    label: str = "forward_return",
    feature_names: tuple[str, ...] = FEATURES,
) -> FactorResearchExecutionResult:
    factor_path, returns_path = _write_inputs(root, label=label)
    published = FactorResearchPublishedOutputs(
        artifact_dir=root,
        final_factor_panel_path=factor_path,
        forward_returns_path=returns_path,
        feature_names=feature_names,
        label_column=label,
    )
    return FactorResearchExecutionResult(
        enabled=True,
        published_outputs=published,
    )


def _research_config(**builder: object) -> ModelingPanelPipelineConfig:
    return ModelingPanelPipelineConfig.from_dict(
        {
            "enabled": True,
            "source": {"mode": "factor_research"},
            "builder": builder,
        }
    )


def test_executor_requires_config_and_valid_project_root_type() -> None:
    with pytest.raises(ModelingPanelPipelineExecutionError, match="config"):
        ModelingPanelPipelineExecutor(object())  # type: ignore[arg-type]
    with pytest.raises(ModelingPanelPipelineExecutionError, match="project_root"):
        ModelingPanelPipelineExecutor(
            ModelingPanelPipelineConfig(), project_root=object()  # type: ignore[arg-type]
        )


def test_disabled_is_no_io_compact_frozen_and_json_safe(tmp_path: Path) -> None:
    missing = tmp_path / "never-created" / "run"
    result = ModelingPanelPipelineExecutor(
        ModelingPanelPipelineConfig(),
        project_root=tmp_path / "also-missing",
    ).execute(missing, factor_research_result=object())  # type: ignore[arg-type]
    assert result == ModelingPanelPipelineResult.disabled()
    assert result.as_dict() == {
        "enabled": False,
        "source_mode": None,
        "artifact_dir": None,
        "panel_path": None,
        "manifest_path": None,
        "feature_names": [],
        "label_column": None,
        "input_factor_rows": 0,
        "input_return_rows": 0,
        "output_rows": 0,
        "warnings": [],
    }
    assert not missing.parent.exists()
    assert {item.name for item in fields(result)}.isdisjoint(
        {"panel", "config", "modeling_panel_result", "factor_research_result"}
    )
    with pytest.raises(FrozenInstanceError):
        result.enabled = True  # type: ignore[misc]


def test_files_mode_relative_paths_success_and_single_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    factor_path, returns_path = _write_inputs(project_root / "inputs")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    build_calls = 0
    write_calls = 0
    original_build = ModelingPanelBuilder.build
    original_write = ModelingPanelArtifactStore.write

    def counted_build(self: ModelingPanelBuilder, *args: object, **kwargs: object):
        nonlocal build_calls
        build_calls += 1
        return original_build(self, *args, **kwargs)

    def counted_write(self: ModelingPanelArtifactStore, *args: object, **kwargs: object):
        nonlocal write_calls
        write_calls += 1
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(ModelingPanelBuilder, "build", counted_build)
    monkeypatch.setattr(ModelingPanelArtifactStore, "write", counted_write)
    config = _files_config(
        factor_path.relative_to(project_root),
        returns_path.relative_to(project_root),
    )
    result = ModelingPanelPipelineExecutor(
        config, project_root=str(project_root)
    ).execute(run_dir)

    assert build_calls == write_calls == 1
    assert result.enabled is True
    assert result.source_mode == "files"
    assert result.artifact_dir == (run_dir / "modeling_panel").absolute()
    assert result.panel_path is not None and result.panel_path.is_file()
    assert result.manifest_path is not None and result.manifest_path.is_file()
    assert result.panel_path.parent == result.artifact_dir
    assert result.manifest_path.parent == result.artifact_dir
    assert result.feature_names == FEATURES
    assert result.label_column == "forward_return"
    assert result.input_factor_rows == result.input_return_rows == 4
    assert result.output_rows == 4
    assert result.warnings == ()
    persisted = pd.read_parquet(result.panel_path)
    assert list(persisted.columns) == [
        "trade_date",
        "ts_code",
        "factor_a",
        "factor_b",
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
        "forward_return",
    ]


def test_files_mode_absolute_paths_and_custom_output(tmp_path: Path) -> None:
    factor_path, returns_path = _write_inputs(tmp_path / "external")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = ModelingPanelPipelineConfig(
        enabled=True,
        source=ModelingPanelSourceConfig(
            factor_panel_path=factor_path.absolute(),
            forward_returns_path=returns_path.absolute(),
        ),
        output=ModelingPanelOutputConfig(
            artifact_subdir="panel_v1",
            parquet_compression="snappy",
            verify_after_write=False,
        ),
    )
    result = ModelingPanelPipelineExecutor(
        config, project_root=tmp_path
    ).execute(run_dir)
    assert result.artifact_dir == (run_dir / "panel_v1").absolute()


@pytest.mark.parametrize("root_kind", ["missing", "file"])
def test_files_mode_rejects_invalid_project_root(
    tmp_path: Path, root_kind: str
) -> None:
    root = tmp_path / "root"
    if root_kind == "file":
        root.write_text("not a directory", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _files_config("a.parquet", "b.parquet")
    with pytest.raises(ModelingPanelPipelineExecutionError, match="project_root"):
        ModelingPanelPipelineExecutor(config, project_root=root).execute(run_dir)


@pytest.mark.parametrize("bad_role", ["factor", "returns", "directory"])
def test_files_mode_rejects_missing_or_directory_inputs(
    tmp_path: Path, bad_role: str
) -> None:
    root = tmp_path / "project"
    factor_path, returns_path = _write_inputs(root / "inputs")
    if bad_role == "factor":
        factor_path.unlink()
    elif bad_role == "returns":
        returns_path.unlink()
    else:
        returns_path.unlink()
        returns_path.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ModelingPanelPipelineExecutionError, match="regular file"):
        ModelingPanelPipelineExecutor(
            _files_config(factor_path, returns_path), project_root=root
        ).execute(run_dir)
    assert not (run_dir / "modeling_panel").exists()


def test_files_mode_rejects_paths_resolving_to_same_file(tmp_path: Path) -> None:
    root = tmp_path / "project"
    factor_path, _ = _write_inputs(root / "inputs")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _files_config(
        Path("inputs") / "factors.parquet",
        Path("inputs") / ".." / "inputs" / "factors.parquet",
    )
    with pytest.raises(ModelingPanelPipelineExecutionError, match="different"):
        ModelingPanelPipelineExecutor(config, project_root=root).execute(run_dir)
    assert factor_path.is_file()

def test_files_mode_rejects_symlink_input_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "project"
    factor_path, returns_path = _write_inputs(root / "inputs")
    link = root / "inputs" / "linked.parquet"
    try:
        link.symlink_to(factor_path)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ModelingPanelPipelineExecutionError, match="symlink"):
        ModelingPanelPipelineExecutor(
            _files_config(link, returns_path), project_root=root
        ).execute(run_dir)


def test_read_and_builder_failures_are_wrapped_with_cause(tmp_path: Path) -> None:
    root = tmp_path / "project"
    factor_path, returns_path = _write_inputs(root / "inputs")
    factor_path.write_bytes(b"not parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ModelingPanelPipelineExecutionError, match="read") as read_exc:
        ModelingPanelPipelineExecutor(
            _files_config(factor_path, returns_path), project_root=root
        ).execute(run_dir)
    assert read_exc.value.__cause__ is not None

    _factor_panel().drop(columns=["trade_date"]).to_parquet(factor_path, index=False)
    with pytest.raises(ModelingPanelPipelineExecutionError, match="build") as build_exc:
        ModelingPanelPipelineExecutor(
            _files_config(factor_path, returns_path), project_root=root
        ).execute(run_dir)
    assert build_exc.value.__cause__ is not None


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_enabled_rejects_invalid_run_dir(tmp_path: Path, kind: str) -> None:
    factor_path, returns_path = _write_inputs(tmp_path / "inputs")
    run_dir = tmp_path / "run"
    if kind == "file":
        run_dir.write_text("file", encoding="utf-8")
    with pytest.raises(ModelingPanelPipelineExecutionError, match="run_dir"):
        ModelingPanelPipelineExecutor(
            _files_config(factor_path, returns_path)
        ).execute(run_dir)


def test_enabled_rejects_symlink_run_dir_when_supported(tmp_path: Path) -> None:
    factor_path, returns_path = _write_inputs(tmp_path / "inputs")
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ModelingPanelPipelineExecutionError, match="symlink"):
        ModelingPanelPipelineExecutor(
            _files_config(factor_path, returns_path)
        ).execute(link)


def test_files_mode_rejects_research_result_and_never_falls_back(
    tmp_path: Path,
) -> None:
    factor_path, returns_path = _write_inputs(tmp_path / "inputs")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ModelingPanelPipelineExecutionError, match="does not accept"):
        ModelingPanelPipelineExecutor(
            _files_config(factor_path, returns_path)
        ).execute(run_dir, factor_research_result=FactorResearchExecutionResult.disabled())


@pytest.mark.parametrize(
    "research_result",
    [None, object(), FactorResearchExecutionResult.disabled(), FactorResearchExecutionResult(enabled=True)],
)
def test_factor_research_mode_requires_successful_published_result(
    tmp_path: Path, research_result: object
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ModelingPanelPipelineExecutionError):
        ModelingPanelPipelineExecutor(_research_config()).execute(
            run_dir, factor_research_result=research_result  # type: ignore[arg-type]
        )


def test_factor_research_mode_uses_published_paths_and_default_metadata(
    tmp_path: Path,
) -> None:
    upstream = _research_result(tmp_path / "research")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = ModelingPanelPipelineExecutor(_research_config()).execute(
        run_dir, factor_research_result=upstream
    )
    assert result.source_mode == "factor_research"
    assert result.feature_names == FEATURES
    assert result.output_rows == 4


def test_factor_research_custom_label_include_and_exclude(tmp_path: Path) -> None:
    custom = _research_result(tmp_path / "custom", label="target")
    include_run = tmp_path / "include"
    include_run.mkdir()
    included = ModelingPanelPipelineExecutor(
        _research_config(label_column="target", include_features=FEATURES)
    ).execute(include_run, factor_research_result=custom)
    assert included.label_column == "target"
    assert included.feature_names == FEATURES

    excluded_run = tmp_path / "exclude"
    excluded_run.mkdir()
    excluded = ModelingPanelPipelineExecutor(
        _research_config(
            label_column="target",
            exclude_features=("factor_b",),
        )
    ).execute(excluded_run, factor_research_result=custom)
    assert excluded.feature_names == ("factor_a",)


@pytest.mark.parametrize(
    "builder",
    [
        {"label_column": "target"},
        {"include_features": ("factor_b", "factor_a")},
        {"exclude_features": ("missing_factor",)},
    ],
)
def test_factor_research_rejects_metadata_mismatch(
    tmp_path: Path, builder: dict[str, object]
) -> None:
    upstream = _research_result(tmp_path / "research")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ModelingPanelPipelineExecutionError):
        ModelingPanelPipelineExecutor(_research_config(**builder)).execute(
            run_dir, factor_research_result=upstream
        )


def test_no_overwrite_wraps_artifact_error_and_preserves_first_artifact(
    tmp_path: Path,
) -> None:
    factor_path, returns_path = _write_inputs(tmp_path / "inputs")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    executor = ModelingPanelPipelineExecutor(_files_config(factor_path, returns_path))
    first = executor.execute(run_dir)
    assert first.panel_path is not None
    original = first.panel_path.read_bytes()
    with pytest.raises(ModelingPanelPipelineExecutionError, match="Artifact") as exc:
        executor.execute(run_dir)
    assert exc.value.__cause__ is not None
    assert first.panel_path.read_bytes() == original
    assert not any(path.name.startswith(".tmp-") for path in run_dir.iterdir())
    assert not any("backup" in path.name.lower() for path in run_dir.iterdir())


def test_public_imports_are_the_concrete_types() -> None:
    from src.pipeline import (
        ModelingPanelPipelineExecutor as PublicExecutor,
        ModelingPanelPipelineResult as PublicResult,
    )

    assert PublicExecutor is ModelingPanelPipelineExecutor
    assert PublicResult is ModelingPanelPipelineResult
