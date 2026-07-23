"""Execute configured V2 factor research inside one existing pipeline run."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.factors.examples import register_example_factors
from src.factors.financial_factors import register_financial_factors
from src.factors.price_volume import register_price_volume_factors
from src.factors.registry import FactorRegistry
from src.factors.research_artifacts import FactorResearchArtifactStore
from src.factors.research_pipeline import FactorResearchRunner
from src.factors.valuation import register_valuation_factors
from src.pipeline.research_config import FactorResearchPipelineConfig


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

    @classmethod
    def disabled(cls) -> "FactorResearchExecutionResult":
        """Return the stable no-op result used when research is disabled."""
        return cls(enabled=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe summary without full input or result data."""
        return _json_safe(
            {
                "enabled": self.enabled,
                "artifact_dir": self.artifact_dir,
                "manifest": self.manifest,
                "table_shapes": self.table_shapes,
                "input_shapes": self.input_shapes,
                "factor_names": self.factor_names,
                "composition_method": self.composition_method,
            }
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

        return FactorResearchExecutionResult(
            enabled=True,
            artifact_dir=str(artifact_dir),
            manifest=manifest,
            table_shapes=result.table_shapes(),
            input_shapes=input_shapes,
            factor_names=research_config.factor_names,
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
