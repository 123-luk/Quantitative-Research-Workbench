"""Independent execution boundary for the optional V5 Holdings stage."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import pandas as pd

from src.portfolio_construction import PortfolioConstructionEngine

from src.holdings import (
    HOLDINGS_ARTIFACT_SCHEMA_VERSION,
    HoldingsArtifactConfig,
    HoldingsArtifactError,
    HoldingsArtifactStore,
    HoldingsBuilder,
    HoldingsDataError,
    SignalArtifactProvenance,
)
from src.pipeline.holdings_config import HoldingsPipelineConfig
from src.pipeline.signal_execution import SignalPipelineResult
from src.signals import SIGNAL_PARQUET_FILENAME, SignalArtifactError, SignalArtifactStore


class HoldingsPipelineExecutionError(Exception):
    """Raised when Holdings execution or its Signal handoff fails."""


def _absolute(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(value))


@dataclass(frozen=True)
class HoldingsPipelineResult:
    """Compact immutable Holdings stage summary without DataFrames."""

    enabled: bool
    source_signal_artifact_dir: Path | None = None
    artifact_dir: Path | None = None
    holdings_path: Path | None = None
    manifest_path: Path | None = None
    rows: int = 0
    trade_date_count: int = 0
    requested_top_n: int | None = None
    insufficient_universe_policy: str | None = None
    weighting: str | None = None
    schema_version: str | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise HoldingsPipelineExecutionError("enabled must be a bool.")
        for name in ("rows", "trade_date_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise HoldingsPipelineExecutionError(
                    f"{name} must be a non-negative integer."
                )
        if not self.enabled:
            if (
                self.source_signal_artifact_dir is not None
                or self.artifact_dir is not None
                or self.holdings_path is not None
                or self.manifest_path is not None
                or self.rows
                or self.trade_date_count
                or self.requested_top_n is not None
                or self.insufficient_universe_policy is not None
                or self.weighting is not None
                or self.schema_version is not None
            ):
                raise HoldingsPipelineExecutionError(
                    "disabled result fields must use empty defaults."
                )
        else:
            if self.rows <= 0 or self.trade_date_count <= 0:
                raise HoldingsPipelineExecutionError(
                    "enabled result requires positive row/date counts."
                )
            if (
                type(self.requested_top_n) is not int
                or self.requested_top_n < 1
                or self.insufficient_universe_policy not in {"error", "allow_partial"}
                or self.weighting != "equal_weight"
                or self.schema_version != HOLDINGS_ARTIFACT_SCHEMA_VERSION
            ):
                raise HoldingsPipelineExecutionError(
                    "enabled result metadata is invalid."
                )
            if (
                self.source_signal_artifact_dir is None
                or self.artifact_dir is None
                or self.holdings_path is None
                or self.manifest_path is None
            ):
                raise HoldingsPipelineExecutionError(
                    "enabled result requires source and Artifact paths."
                )
            source = _absolute(self.source_signal_artifact_dir)
            artifact = _absolute(self.artifact_dir)
            holdings = _absolute(self.holdings_path)
            manifest = _absolute(self.manifest_path)
            if (
                not source.is_dir()
                or source.is_symlink()
                or not artifact.is_dir()
                or artifact.is_symlink()
                or not holdings.is_file()
                or holdings.is_symlink()
                or not manifest.is_file()
                or manifest.is_symlink()
                or holdings.parent != artifact
                or manifest.parent != artifact
                or holdings.name != "holdings.parquet"
                or manifest.name != "manifest.json"
            ):
                raise HoldingsPipelineExecutionError(
                    "enabled result Artifact paths are invalid."
                )
            object.__setattr__(self, "source_signal_artifact_dir", source)
            object.__setattr__(self, "artifact_dir", artifact)
            object.__setattr__(self, "holdings_path", holdings)
            object.__setattr__(self, "manifest_path", manifest)
        json.dumps(self.as_dict(), allow_nan=False)

    @classmethod
    def disabled(cls) -> HoldingsPipelineResult:
        return cls(enabled=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "source_signal_artifact_dir": (
                None
                if self.source_signal_artifact_dir is None
                else str(self.source_signal_artifact_dir)
            ),
            "artifact_dir": None if self.artifact_dir is None else str(self.artifact_dir),
            "holdings_path": None if self.holdings_path is None else str(self.holdings_path),
            "manifest_path": None if self.manifest_path is None else str(self.manifest_path),
            "rows": self.rows,
            "trade_date_count": self.trade_date_count,
            "requested_top_n": self.requested_top_n,
            "insufficient_universe_policy": self.insufficient_universe_policy,
            "weighting": self.weighting,
            "schema_version": self.schema_version,
        }


class HoldingsPipelineExecutor:
    """Build and persist Holdings from one explicit Signal Artifact."""

    def __init__(
        self,
        config: HoldingsPipelineConfig,
        engine: PortfolioConstructionEngine | None = None,
    ) -> None:
        if not isinstance(config, HoldingsPipelineConfig):
            raise HoldingsPipelineExecutionError(
                "config must be HoldingsPipelineConfig."
            )
        self.config = config
        if engine is not None and not isinstance(
            engine, PortfolioConstructionEngine
        ):
            raise HoldingsPipelineExecutionError(
                "engine must be PortfolioConstructionEngine."
            )
        self._engine = engine or PortfolioConstructionEngine()

    def execute(
        self,
        run_dir: str | Path,
        *,
        signal_result: SignalPipelineResult | None = None,
    ) -> HoldingsPipelineResult:
        if not self.config.enabled:
            return HoldingsPipelineResult.disabled()
        run_path = self._run_dir(run_dir)
        source_dir, signal_path = self._source(signal_result)
        try:
            source_store = SignalArtifactStore()
            validation = source_store.validate(source_dir)
            if not validation.is_valid or validation.manifest is None:
                raise HoldingsPipelineExecutionError(
                    "source Signal Artifact validation is invalid."
                )
            manifest = validation.manifest
            record = next(
                (
                    item
                    for item in manifest.files
                    if item.relative_path == SIGNAL_PARQUET_FILENAME
                ),
                None,
            )
            if record is None:
                raise HoldingsPipelineExecutionError(
                    "source Signal Artifact has no Signal Parquet record."
                )
            signals = pd.read_parquet(signal_path, engine="pyarrow")
            built = HoldingsBuilder(self._engine).build(
                signals,
                top_n=self.config.top_n,
                insufficient_universe_policy=self.config.insufficient_universe_policy,
                weighting=self.config.weighting,
                portfolio_construction=self.config.portfolio_construction,
            )
            provenance = SignalArtifactProvenance(
                signal_artifact_dir=source_dir,
                signal_path=signal_path,
                signal_schema_version=manifest.signal_schema_version,
                signal_sha256=record.sha256,
            )
            artifact_dir = _absolute(run_path / self.config.artifact_subdir)
            if artifact_dir.parent != run_path:
                raise HoldingsPipelineExecutionError(
                    "artifact_dir must be a direct child of run_dir."
                )
            written = HoldingsArtifactStore().write(
                built,
                provenance,
                HoldingsArtifactConfig(artifact_dir),
                portfolio_construction=self.config.portfolio_construction,
            )
        except HoldingsPipelineExecutionError:
            raise
        except SignalArtifactError as exc:
            raise HoldingsPipelineExecutionError(
                f"Signal Artifact validation failed: {type(exc).__name__}."
            ) from exc
        except HoldingsDataError as exc:
            raise HoldingsPipelineExecutionError(
                f"Holdings build failed: {type(exc).__name__}."
            ) from exc
        except HoldingsArtifactError as exc:
            raise HoldingsPipelineExecutionError(
                f"Holdings Artifact write failed: {type(exc).__name__}."
            ) from exc
        except (OSError, ValueError, ImportError) as exc:
            raise HoldingsPipelineExecutionError(
                f"Signal payload read failed: {type(exc).__name__}."
            ) from exc
        if not written.validation.is_valid:
            raise HoldingsPipelineExecutionError(
                "Holdings Artifact validation is invalid."
            )
        return HoldingsPipelineResult(
            enabled=True,
            source_signal_artifact_dir=source_dir,
            artifact_dir=written.artifact_dir,
            holdings_path=written.holdings_path,
            manifest_path=written.manifest_path,
            rows=built.audit.output_rows,
            trade_date_count=built.audit.trade_date_count,
            requested_top_n=built.audit.requested_top_n,
            insufficient_universe_policy=built.audit.insufficient_universe_policy,
            weighting=built.audit.weighting,
            schema_version=written.schema_version,
        )

    @staticmethod
    def _run_dir(value: object) -> Path:
        if not isinstance(value, (str, os.PathLike)):
            raise HoldingsPipelineExecutionError(
                "run_dir must be a str or os.PathLike."
            )
        path = Path(os.fspath(value))
        if path.is_symlink():
            raise HoldingsPipelineExecutionError("run_dir must not be a symlink.")
        resolved = _absolute(path)
        if not resolved.exists() or not resolved.is_dir():
            raise HoldingsPipelineExecutionError(
                "run_dir must be an existing directory."
            )
        return resolved

    @staticmethod
    def _source(
        signal_result: SignalPipelineResult | None,
    ) -> tuple[Path, Path]:
        if not isinstance(signal_result, SignalPipelineResult):
            raise HoldingsPipelineExecutionError(
                "Holdings requires SignalPipelineResult."
            )
        if (
            not signal_result.enabled
            or signal_result.artifact_dir is None
            or signal_result.signal_path is None
        ):
            raise HoldingsPipelineExecutionError(
                "Holdings requires an enabled Signal result with artifact_dir."
            )
        directory = _absolute(signal_result.artifact_dir)
        signal_path = _absolute(signal_result.signal_path)
        if signal_path != directory / SIGNAL_PARQUET_FILENAME:
            raise HoldingsPipelineExecutionError(
                "Signal result does not identify its fixed Signal payload."
            )
        return directory, signal_path
