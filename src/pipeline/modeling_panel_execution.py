"""Execute one independent Modeling Panel Pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.modeling_panel import (
    ModelingPanelArtifactConfig,
    ModelingPanelArtifactError,
    ModelingPanelArtifactStore,
    ModelingPanelBuilder,
    ModelingPanelError,
)
from src.pipeline.modeling_panel_config import (
    ModelingPanelPipelineConfig,
    ModelingPanelPipelineExecutionError,
)
from src.pipeline.research_execution import FactorResearchExecutionResult


def _absolute(value: str | Path) -> Path:
    return Path(os.path.abspath(value))


@dataclass(frozen=True)
class ModelingPanelPipelineResult:
    """Compact immutable summary without DataFrames or in-memory contracts."""

    enabled: bool
    source_mode: str | None = None
    artifact_dir: Path | None = None
    panel_path: Path | None = None
    manifest_path: Path | None = None
    feature_names: tuple[str, ...] = ()
    label_column: str | None = None
    input_factor_rows: int = 0
    input_return_rows: int = 0
    output_rows: int = 0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ModelingPanelPipelineExecutionError("enabled must be a bool.")
        try:
            features = tuple(self.feature_names)
            warnings = tuple(self.warnings)
        except TypeError as exc:
            raise ModelingPanelPipelineExecutionError(
                "result tuple fields are invalid."
            ) from exc
        for field_name in (
            "input_factor_rows",
            "input_return_rows",
            "output_rows",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingPanelPipelineExecutionError(
                    f"{field_name} must be a non-negative integer."
                )
        if not self.enabled:
            if (
                self.source_mode is not None
                or self.artifact_dir is not None
                or self.panel_path is not None
                or self.manifest_path is not None
                or features
                or self.label_column is not None
                or self.input_factor_rows
                or self.input_return_rows
                or self.output_rows
                or warnings
            ):
                raise ModelingPanelPipelineExecutionError(
                    "disabled result fields must use empty defaults."
                )
        else:
            if self.source_mode not in {"files", "factor_research"}:
                raise ModelingPanelPipelineExecutionError(
                    "enabled result source_mode is invalid."
                )
            if not features or len(features) != len(set(features)):
                raise ModelingPanelPipelineExecutionError(
                    "enabled result requires unique feature_names."
                )
            if not isinstance(self.label_column, str) or not self.label_column:
                raise ModelingPanelPipelineExecutionError(
                    "enabled result requires label_column."
                )
            if self.output_rows <= 0:
                raise ModelingPanelPipelineExecutionError(
                    "enabled result requires positive output_rows."
                )
            if (
                self.artifact_dir is None
                or self.panel_path is None
                or self.manifest_path is None
            ):
                raise ModelingPanelPipelineExecutionError(
                    "enabled result requires Artifact paths."
                )
            artifact = _absolute(self.artifact_dir)
            panel = _absolute(self.panel_path)
            manifest = _absolute(self.manifest_path)
            if (
                not artifact.is_dir()
                or artifact.is_symlink()
                or not panel.is_file()
                or panel.is_symlink()
                or not manifest.is_file()
                or manifest.is_symlink()
                or panel.parent != artifact
                or manifest.parent != artifact
            ):
                raise ModelingPanelPipelineExecutionError(
                    "enabled result Artifact paths are invalid."
                )
            object.__setattr__(self, "artifact_dir", artifact)
            object.__setattr__(self, "panel_path", panel)
            object.__setattr__(self, "manifest_path", manifest)
        object.__setattr__(self, "feature_names", features)
        object.__setattr__(self, "warnings", warnings)

    @classmethod
    def disabled(cls) -> ModelingPanelPipelineResult:
        return cls(enabled=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source_mode": self.source_mode,
            "artifact_dir": (
                None if self.artifact_dir is None else str(self.artifact_dir)
            ),
            "panel_path": None if self.panel_path is None else str(self.panel_path),
            "manifest_path": (
                None if self.manifest_path is None else str(self.manifest_path)
            ),
            "feature_names": list(self.feature_names),
            "label_column": self.label_column,
            "input_factor_rows": self.input_factor_rows,
            "input_return_rows": self.input_return_rows,
            "output_rows": self.output_rows,
            "warnings": list(self.warnings),
        }


class ModelingPanelPipelineExecutor:
    """Read two explicit Parquet inputs, build a panel, and publish its Artifact."""

    def __init__(
        self,
        config: ModelingPanelPipelineConfig,
        *,
        project_root: str | Path | None = None,
    ) -> None:
        if not isinstance(config, ModelingPanelPipelineConfig):
            raise ModelingPanelPipelineExecutionError(
                "config must be ModelingPanelPipelineConfig."
            )
        if project_root is not None and not isinstance(project_root, (str, Path)):
            raise ModelingPanelPipelineExecutionError(
                "project_root must be a str, pathlib.Path, or None."
            )
        self.config = config
        self.project_root = _absolute(
            Path(__file__).parents[2] if project_root is None else project_root
        )

    def execute(
        self,
        run_dir: str | Path,
        *,
        factor_research_result: FactorResearchExecutionResult | None = None,
    ) -> ModelingPanelPipelineResult:
        if not self.config.enabled:
            return ModelingPanelPipelineResult.disabled()
        run_path = self._run_dir(run_dir)
        factor_path, returns_path, published = self._source_paths(
            factor_research_result
        )
        factor_panel = self._read_parquet(factor_path, "factor_panel")
        forward_returns = self._read_parquet(
            returns_path, "forward_returns"
        )
        try:
            panel_result = ModelingPanelBuilder(self.config.builder).build(
                factor_panel, forward_returns
            )
        except ModelingPanelError as exc:
            raise ModelingPanelPipelineExecutionError(
                f"Modeling Panel build failed: {type(exc).__name__}."
            ) from exc
        if published is not None:
            self._validate_published_result(panel_result, published)
        artifact_dir = _absolute(
            run_path / self.config.output.artifact_subdir
        )
        if artifact_dir.parent != run_path:
            raise ModelingPanelPipelineExecutionError(
                "artifact_dir must be a direct child of run_dir."
            )
        try:
            written = ModelingPanelArtifactStore().write(
                panel_result,
                ModelingPanelArtifactConfig(
                    artifact_dir=artifact_dir,
                    parquet_compression=self.config.output.parquet_compression,
                    verify_after_write=self.config.output.verify_after_write,
                ),
            )
        except ModelingPanelArtifactError as exc:
            raise ModelingPanelPipelineExecutionError(
                f"Modeling Panel Artifact write failed: {type(exc).__name__}."
            ) from exc
        if not written.validation.is_valid:
            raise ModelingPanelPipelineExecutionError(
                "Modeling Panel Artifact validation is invalid."
            )
        return ModelingPanelPipelineResult(
            enabled=True,
            source_mode=self.config.source.mode,
            artifact_dir=written.artifact_dir,
            panel_path=written.panel_path,
            manifest_path=written.manifest_path,
            feature_names=panel_result.feature_names,
            label_column=panel_result.label_column,
            input_factor_rows=panel_result.audit.factor_input_rows,
            input_return_rows=panel_result.audit.return_input_rows,
            output_rows=panel_result.audit.output_rows,
            warnings=panel_result.audit.warnings,
        )

    def _run_dir(self, value: str | Path) -> Path:
        if not isinstance(value, (str, Path)):
            raise ModelingPanelPipelineExecutionError(
                "run_dir must be a str or pathlib.Path."
            )
        configured = Path(value)
        if configured.is_symlink():
            raise ModelingPanelPipelineExecutionError(
                "run_dir must not be a symlink."
            )
        path = _absolute(configured)
        if not path.exists() or not path.is_dir():
            raise ModelingPanelPipelineExecutionError(
                "run_dir must be an existing directory."
            )
        return path

    def _source_paths(
        self,
        factor_research_result: FactorResearchExecutionResult | None,
    ) -> tuple[Path, Path, object | None]:
        source = self.config.source
        if source.mode == "files":
            if factor_research_result is not None:
                raise ModelingPanelPipelineExecutionError(
                    "files mode does not accept factor_research_result."
                )
            if (
                source.factor_panel_path is None
                or source.forward_returns_path is None
            ):
                raise ModelingPanelPipelineExecutionError(
                    "files source paths are incomplete."
                )
            factor_path = self._input_path(
                source.factor_panel_path, "factor_panel"
            )
            returns_path = self._input_path(
                source.forward_returns_path, "forward_returns"
            )
            if factor_path == returns_path:
                raise ModelingPanelPipelineExecutionError(
                    "files source paths must resolve to different files."
                )
            return factor_path, returns_path, None
        if not isinstance(factor_research_result, FactorResearchExecutionResult):
            raise ModelingPanelPipelineExecutionError(
                "factor_research mode requires FactorResearchExecutionResult."
            )
        if (
            not factor_research_result.enabled
            or factor_research_result.published_outputs is None
        ):
            raise ModelingPanelPipelineExecutionError(
                "factor_research result has no successful published outputs."
            )
        published = factor_research_result.published_outputs
        if self.config.builder.label_column != published.label_column:
            raise ModelingPanelPipelineExecutionError(
                "builder label_column differs from published label_column."
            )
        include = self.config.builder.include_features
        exclude = self.config.builder.exclude_features
        if include is not None and include != published.feature_names:
            raise ModelingPanelPipelineExecutionError(
                "include_features differs from published feature_names."
            )
        missing_excluded = [
            name for name in exclude if name not in published.feature_names
        ]
        if missing_excluded:
            raise ModelingPanelPipelineExecutionError(
                "exclude_features contains names absent from published outputs."
            )
        return (
            self._existing_input(
                published.final_factor_panel_path, "final_factor_panel"
            ),
            self._existing_input(
                published.forward_returns_path, "forward_returns"
            ),
            published,
        )

    def _input_path(self, value: Path, role: str) -> Path:
        if not self.project_root.exists() or not self.project_root.is_dir():
            raise ModelingPanelPipelineExecutionError(
                "project_root must be an existing directory."
            )
        candidate = value if value.is_absolute() else self.project_root / value
        return self._existing_input(candidate, role)

    @staticmethod
    def _existing_input(value: Path, role: str) -> Path:
        if value.is_symlink():
            raise ModelingPanelPipelineExecutionError(
                f"{role} input must not be a symlink."
            )
        path = _absolute(value)
        if (
            path.suffix.lower() != ".parquet"
            or not path.exists()
            or not path.is_file()
        ):
            raise ModelingPanelPipelineExecutionError(
                f"{role} input must be an existing Parquet regular file."
            )
        return path

    @staticmethod
    def _read_parquet(path: Path, role: str) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            raise ModelingPanelPipelineExecutionError(
                f"Failed to read {role} Parquet input."
            ) from exc

    def _validate_published_result(self, panel_result: Any, published: Any) -> None:
        if panel_result.label_column != published.label_column:
            raise ModelingPanelPipelineExecutionError(
                "built label differs from published label."
            )
        include = self.config.builder.include_features
        exclude = self.config.builder.exclude_features
        if include is not None:
            expected = include
        else:
            excluded = set(exclude)
            expected = tuple(
                name for name in published.feature_names if name not in excluded
            )
        if panel_result.feature_names != expected:
            raise ModelingPanelPipelineExecutionError(
                "built features differ from published metadata."
            )
