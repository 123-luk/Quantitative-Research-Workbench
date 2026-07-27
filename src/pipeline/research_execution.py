"""Execute configured V2 factor research inside one existing pipeline run."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.factors.examples import register_example_factors
from src.factors.financial_factors import register_financial_factors
from src.factors.price_volume import register_price_volume_factors
from src.factors.registry import FactorRegistry
from src.factors.research_artifacts import FactorResearchArtifactStore
from src.factors.research_pipeline import FactorResearchResult, FactorResearchRunner
from src.factors.valuation import register_valuation_factors
from src.pipeline.research_config import FactorResearchPipelineConfig


_PUBLISHED_RESERVED_COLUMNS = {
    "trade_date",
    "ts_code",
    "entry_trade_date",
    "exit_trade_date",
    "entry_price",
    "exit_price",
}


def _absolute_path(value: str | Path, field_name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field_name} must be a str or pathlib.Path.")
    return Path(os.path.abspath(value))


def _reject_symlink_chain(path: Path, artifact_dir: Path, field_name: str) -> None:
    current = path
    while current != artifact_dir:
        if current.is_symlink():
            raise ValueError(f"{field_name} must not traverse a symlink.")
        parent = current.parent
        if parent == current:
            raise ValueError(f"{field_name} must remain inside artifact_dir.")
        current = parent


@dataclass(frozen=True)
class FactorResearchPublishedOutputs:
    """Validated references to this execution's Modeling Panel source tables."""

    artifact_dir: Path
    final_factor_panel_path: Path
    forward_returns_path: Path
    feature_names: tuple[str, ...]
    label_column: str

    def __post_init__(self) -> None:
        artifact_dir = _absolute_path(self.artifact_dir, "artifact_dir")
        if (
            not artifact_dir.exists()
            or not artifact_dir.is_dir()
            or artifact_dir.is_symlink()
        ):
            raise ValueError(
                "artifact_dir must be an existing non-symlink directory."
            )
        paths: list[Path] = []
        for field_name in ("final_factor_panel_path", "forward_returns_path"):
            path = _absolute_path(getattr(self, field_name), field_name)
            try:
                path.relative_to(artifact_dir)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name} must remain inside artifact_dir."
                ) from exc
            if path == artifact_dir:
                raise ValueError(f"{field_name} must identify a file.")
            _reject_symlink_chain(path, artifact_dir, field_name)
            if not path.exists() or not path.is_file() or path.is_symlink():
                raise ValueError(
                    f"{field_name} must be an existing non-symlink regular file."
                )
            if path.suffix.lower() != ".parquet":
                raise ValueError(f"{field_name} must have a .parquet suffix.")
            paths.append(path)
        if paths[0] == paths[1]:
            raise ValueError("Published Parquet paths must be different.")
        if not isinstance(self.feature_names, tuple):
            raise ValueError("feature_names must be a non-empty tuple of names.")
        features = self.feature_names
        if (
            not features
            or any(
                not isinstance(name, str)
                or not name.strip()
                or name != name.strip()
                for name in features
            )
            or len(features) != len(set(features))
        ):
            raise ValueError(
                "feature_names must contain unique non-empty trimmed names."
            )
        if not isinstance(self.label_column, str) or not self.label_column.strip():
            raise ValueError("label_column must be a non-empty string.")
        label = self.label_column.strip()
        conflicts = [
            name for name in features if name in _PUBLISHED_RESERVED_COLUMNS
        ]
        if conflicts:
            raise ValueError(
                f"feature_names contains reserved columns: {conflicts!r}."
            )
        if label in features:
            raise ValueError("label_column must not appear in feature_names.")
        if label in _PUBLISHED_RESERVED_COLUMNS:
            raise ValueError("label_column conflicts with a reserved column.")
        object.__setattr__(self, "artifact_dir", artifact_dir)
        object.__setattr__(self, "final_factor_panel_path", paths[0])
        object.__setattr__(self, "forward_returns_path", paths[1])
        object.__setattr__(self, "feature_names", features)
        object.__setattr__(self, "label_column", label)

    def as_dict(self) -> dict[str, Any]:
        """Return detached JSON-safe references without reading the files."""
        return {
            "artifact_dir": str(self.artifact_dir),
            "final_factor_panel_path": str(self.final_factor_panel_path),
            "forward_returns_path": str(self.forward_returns_path),
            "feature_names": list(self.feature_names),
            "label_column": self.label_column,
        }


def _json_safe(value: Any) -> Any:
    """Return a detached JSON-safe structure for known summary values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return deepcopy(value)


@dataclass(frozen=True)
class FactorResearchExecutionResult:
    """Compact pipeline-facing summary that never retains research DataFrames."""

    enabled: bool
    artifact_dir: str | None = None
    manifest: dict[str, Any] | None = None
    table_shapes: dict[str, tuple[int, int]] = field(default_factory=dict)
    input_shapes: dict[str, dict[str, int]] = field(default_factory=dict)
    factor_names: tuple[str, ...] = ()
    published_outputs: FactorResearchPublishedOutputs | None = None
    composition_method: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        if self.artifact_dir is not None and not isinstance(self.artifact_dir, str):
            raise TypeError("artifact_dir must be a string or None.")
        if self.manifest is not None and not isinstance(self.manifest, dict):
            raise TypeError("manifest must be a dict or None.")
        object.__setattr__(
            self,
            "manifest",
            deepcopy(self.manifest) if self.manifest is not None else None,
        )
        object.__setattr__(
            self,
            "table_shapes",
            {
                str(name): (int(shape[0]), int(shape[1]))
                for name, shape in dict(self.table_shapes).items()
            },
        )
        object.__setattr__(
            self,
            "input_shapes",
            {
                str(name): {
                    "rows": int(shape["rows"]),
                    "columns": int(shape["columns"]),
                }
                for name, shape in dict(self.input_shapes).items()
            },
        )
        object.__setattr__(self, "factor_names", tuple(self.factor_names))
        if self.published_outputs is not None and not isinstance(
            self.published_outputs, FactorResearchPublishedOutputs
        ):
            raise TypeError(
                "published_outputs must be FactorResearchPublishedOutputs or None."
            )

    @classmethod
    def disabled(cls) -> "FactorResearchExecutionResult":
        """Return the stable no-op result used when research is disabled."""
        return cls(enabled=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe summary without full input or result data."""
        values: dict[str, Any] = {
            "enabled": self.enabled,
            "artifact_dir": self.artifact_dir,
            "manifest": self.manifest,
            "table_shapes": self.table_shapes,
            "input_shapes": self.input_shapes,
            "factor_names": self.factor_names,
            "composition_method": self.composition_method,
        }
        if self.published_outputs is not None:
            values["published_outputs"] = self.published_outputs.as_dict()
        return _json_safe(values)


def _metadata_names(value: object, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise RuntimeError(f"{context} must be a sequence of names.")
    try:
        names = tuple(value)
    except TypeError as exc:
        raise RuntimeError(f"{context} must be a sequence of names.") from exc
    if (
        not names
        or any(
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            for name in names
        )
        or len(names) != len(set(names))
    ):
        raise RuntimeError(f"{context} contains invalid feature names.")
    return names


def _metadata_label(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty label name.")
    return value.strip()


def _find_manifest_table(
    manifest: Mapping[str, Any], logical_name: str
) -> Mapping[str, Any]:
    tables = manifest.get("tables")
    if not isinstance(tables, list):
        raise RuntimeError("factor research manifest tables must be a list.")
    matches = [
        item
        for item in tables
        if isinstance(item, Mapping) and item.get("name") == logical_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"manifest must contain exactly one {logical_name!r} table record."
        )
    record = matches[0]
    if record.get("saved") is not True:
        raise RuntimeError(f"manifest table {logical_name!r} was not published.")
    return record


def _resolve_published_output_path(
    artifact_dir: Path,
    record: Mapping[str, Any],
    logical_name: str,
) -> Path:
    relative_value = record.get("relative_path")
    if not isinstance(relative_value, str) or not relative_value:
        raise RuntimeError(
            f"manifest table {logical_name!r} has no valid relative_path."
        )
    if (
        "\\" in relative_value
        or ":" in relative_value
        or "://" in relative_value
        or PurePosixPath(relative_value).is_absolute()
        or PureWindowsPath(relative_value).is_absolute()
        or PureWindowsPath(relative_value).drive
    ):
        raise RuntimeError(
            f"manifest table {logical_name!r} has an unsafe relative_path."
        )
    raw_parts = relative_value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise RuntimeError(
            f"manifest table {logical_name!r} has an unsafe relative_path."
        )
    relative = PurePosixPath(relative_value)
    if any(part in {".", ".."} for part in relative.parts):
        raise RuntimeError(
            f"manifest table {logical_name!r} has an unsafe relative_path."
        )
    if relative.name != f"{logical_name}.parquet":
        raise RuntimeError(
            f"manifest table {logical_name!r} has an unexpected basename."
        )
    path = _absolute_path(
        artifact_dir.joinpath(*relative.parts),
        f"{logical_name} path",
    )
    try:
        path.relative_to(artifact_dir)
    except ValueError as exc:
        raise RuntimeError(
            f"manifest table {logical_name!r} escapes artifact_dir."
        ) from exc
    try:
        _reject_symlink_chain(path, artifact_dir, logical_name)
    except ValueError as exc:
        raise RuntimeError(
            f"manifest table {logical_name!r} traverses a symlink."
        ) from exc
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise RuntimeError(
            f"manifest table {logical_name!r} is not a regular published file."
        )
    if path.suffix.lower() != ".parquet":
        raise RuntimeError(
            f"manifest table {logical_name!r} is not a Parquet file."
        )
    return path


def _parquet_schema_names(path: Path, logical_name: str) -> tuple[str, ...]:
    try:
        names = tuple(pq.read_schema(path).names)
    except Exception as exc:
        raise RuntimeError(
            f"published table {logical_name!r} schema cannot be read."
        ) from exc
    if len(names) != len(set(names)):
        raise RuntimeError(
            f"published table {logical_name!r} has duplicate columns."
        )
    return names


def _build_published_outputs(
    result: FactorResearchResult,
    manifest: Mapping[str, Any],
    artifact_dir: Path,
    configured_feature_names: tuple[str, ...],
    configured_label_column: str,
) -> FactorResearchPublishedOutputs:
    if not isinstance(result, FactorResearchResult):
        raise TypeError("result must be a FactorResearchResult.")
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a Mapping.")
    root = _absolute_path(artifact_dir, "artifact_dir")
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise RuntimeError(
            "factor research artifact_dir is not a non-symlink directory."
        )
    summary = manifest.get("result_summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError("factor research manifest result_summary is invalid.")
    feature_sources = (
        _metadata_names(result.factor_names, "result.factor_names"),
        _metadata_names(configured_feature_names, "configured factor_names"),
        _metadata_names(summary.get("factor_names"), "manifest factor_names"),
    )
    if feature_sources[1:] != feature_sources[:-1]:
        raise RuntimeError(
            "factor_names metadata differs across result, config, and manifest."
        )
    label_sources = (
        _metadata_label(result.forward_return_col, "result.forward_return_col"),
        _metadata_label(configured_label_column, "configured return_col"),
        _metadata_label(
            summary.get("forward_return_col"),
            "manifest forward_return_col",
        ),
    )
    if label_sources[1:] != label_sources[:-1]:
        raise RuntimeError(
            "label metadata differs across result, config, and manifest."
        )
    features = feature_sources[0]
    label = label_sources[0]
    final_record = _find_manifest_table(manifest, "final_factor_panel")
    returns_record = _find_manifest_table(manifest, "forward_returns")
    final_path = _resolve_published_output_path(
        root, final_record, "final_factor_panel"
    )
    returns_path = _resolve_published_output_path(
        root, returns_record, "forward_returns"
    )
    final_columns = _parquet_schema_names(final_path, "final_factor_panel")
    returns_columns = _parquet_schema_names(returns_path, "forward_returns")
    for logical_name, record, actual_columns in (
        ("final_factor_panel", final_record, final_columns),
        ("forward_returns", returns_record, returns_columns),
    ):
        recorded_columns = record.get("column_names")
        if (
            not isinstance(recorded_columns, list)
            or tuple(recorded_columns) != actual_columns
        ):
            raise RuntimeError(
                f"manifest table {logical_name!r} columns differ from Parquet."
            )
    final_required = {"trade_date", "ts_code", *features}
    missing_final = final_required - set(final_columns)
    forbidden_final = {
        label,
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
    } & set(final_columns)
    if missing_final or forbidden_final:
        raise RuntimeError(
            "published table 'final_factor_panel' schema is incompatible."
        )
    returns_required = {
        "trade_date",
        "ts_code",
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
        label,
    }
    if returns_required - set(returns_columns):
        raise RuntimeError(
            "published table 'forward_returns' schema is incompatible."
        )
    return FactorResearchPublishedOutputs(
        artifact_dir=root,
        final_factor_panel_path=final_path,
        forward_returns_path=returns_path,
        feature_names=features,
        label_column=label,
    )


class FactorResearchPipelineExecutor:
    """Load configured panels, run G2, and persist the result through G3."""

    def __init__(
        self,
        config: FactorResearchPipelineConfig,
        *,
        project_root: str | Path | None = None,
    ) -> None:
        if not isinstance(config, FactorResearchPipelineConfig):
            raise TypeError("config must be a FactorResearchPipelineConfig.")
        if project_root is None:
            root = Path(__file__).resolve().parents[2]
        elif isinstance(project_root, (str, Path)):
            root = Path(project_root).expanduser().resolve()
        else:
            raise TypeError("project_root must be a str, pathlib.Path, or None.")
        if not root.exists():
            raise FileNotFoundError(f"project_root does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"project_root must be a directory: {root}")
        self.config = config
        self.project_root = root

    def describe_config(self) -> dict[str, Any]:
        """Return the detached G4A configuration snapshot."""
        return self.config.to_dict()

    def execute(
        self,
        run_dir: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> FactorResearchExecutionResult:
        """Execute one stateless G2-to-G3 research flow in an existing run."""
        run_path = self._existing_run_dir(run_dir)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a Mapping or None.")
        if not self.config.enabled:
            return FactorResearchExecutionResult.disabled()

        inputs, resolved_paths = self._load_inputs()
        input_shapes = {
            name: {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])}
            for name, frame in inputs.items()
        }
        registry = self._build_registry()
        research_config = self.config.research
        if research_config is None:  # guarded by G4A, retained defensively
            raise ValueError("research configuration is required when enabled.")
        missing = [
            name
            for name in research_config.factor_names
            if not registry.contains(name)
        ]
        if missing:
            raise KeyError(
                "Configured factors are not registered: " + ", ".join(missing)
            )

        runner = FactorResearchRunner(
            registry=registry,
            config=research_config,
            preprocessing_config=self.config.preprocessing,
            neutralization_config=self.config.neutralization,
            evaluation_config=self.config.evaluation,
            quantile_config=self.config.quantile,
            composition_config=self.config.composition,
            rolling_config=self.config.rolling,
            forward_return_config=self.config.forward_returns,
        )
        try:
            result = runner.run(
                inputs["factor_input"],
                inputs["score_panel"],
                inputs["price_panel"],
                inputs.get("exposure_panel"),
            )
        except Exception as exc:
            raise RuntimeError(
                f"factor research execution failed: {exc}"
            ) from exc

        artifact_dir = self._artifact_dir(run_path)
        pipeline_metadata = dict(metadata or {})
        pipeline_metadata.update(
            {
                "pipeline_stage": "factor_research",
                "factor_names": list(research_config.factor_names),
                "composition_method": research_config.composition_method,
                "configured_input_paths": {
                    "factor_input": self.config.factor_input_path,
                    "score_panel": self.config.score_panel_path,
                    "price_panel": self.config.price_panel_path,
                    "exposure_panel": (
                        self.config.exposure_panel_path
                        if research_config.use_neutralization
                        else None
                    ),
                },
                "resolved_input_paths": {
                    name: str(path) for name, path in resolved_paths.items()
                },
                "input_shapes": deepcopy(input_shapes),
            }
        )
        store = FactorResearchArtifactStore(self.config.artifacts)
        try:
            manifest = store.save(
                result,
                artifact_dir,
                runner_config=runner.describe_config(),
                metadata=pipeline_metadata,
            )
        except FileExistsError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"factor research artifact save failed: {exc}"
            ) from exc

        try:
            published_outputs = _build_published_outputs(
                result,
                manifest,
                artifact_dir,
                research_config.factor_names,
                self.config.forward_returns.return_col,
            )
        except Exception as exc:
            raise RuntimeError(
                f"factor research published outputs validation failed: {exc}"
            ) from exc

        return FactorResearchExecutionResult(
            enabled=True,
            artifact_dir=str(artifact_dir),
            manifest=manifest,
            table_shapes=result.table_shapes(),
            input_shapes=input_shapes,
            factor_names=research_config.factor_names,
            published_outputs=published_outputs,
            composition_method=research_config.composition_method,
        )

    def _load_inputs(
        self,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
        roles = (
            ("factor_input", self.config.factor_input_path),
            ("score_panel", self.config.score_panel_path),
            ("price_panel", self.config.price_panel_path),
        )
        if self.config.research and self.config.research.use_neutralization:
            roles += (("exposure_panel", self.config.exposure_panel_path),)

        frames: dict[str, pd.DataFrame] = {}
        paths: dict[str, Path] = {}
        for role, configured_path in roles:
            path = self._resolve_input_path(role, configured_path)
            frames[role] = self._read_parquet_input(role, path)
            paths[role] = path
        return frames, paths

    def _resolve_input_path(self, role: str, configured_path: str | None) -> Path:
        if configured_path is None:
            raise ValueError(f"{role} path is required.")
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"{role} input does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"{role} input must be a regular file: {path}")
        return path

    @staticmethod
    def _read_parquet_input(role: str, path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read {role} Parquet input at {path}: {exc}"
            ) from exc

    @staticmethod
    def _build_registry() -> FactorRegistry:
        registry = FactorRegistry()
        register_example_factors(registry)
        register_price_volume_factors(registry)
        register_valuation_factors(registry)
        register_financial_factors(registry)
        return registry

    def _artifact_dir(self, run_dir: Path) -> Path:
        artifact_dir = (run_dir / self.config.artifact_subdir).resolve()
        try:
            artifact_dir.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError("artifact_dir must remain inside run_dir.") from exc
        if artifact_dir == run_dir:
            raise ValueError("artifact_dir must be a child of run_dir.")
        return artifact_dir

    @staticmethod
    def _existing_run_dir(run_dir: str | Path) -> Path:
        if not isinstance(run_dir, (str, Path)):
            raise TypeError("run_dir must be a str or pathlib.Path.")
        path = Path(run_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"run_dir does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"run_dir must be a directory: {path}")
        return path
