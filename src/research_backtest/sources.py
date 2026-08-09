"""Explicit native Holdings sources for the V6 research backtest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.holdings import (
    HOLDINGS_PARQUET_FILENAME,
    HoldingsArtifactError,
    HoldingsArtifactStore,
)

if TYPE_CHECKING:
    from src.pipeline.holdings_execution import HoldingsPipelineResult
    from src.pipeline.research_backtest_config import BacktestSourceConfig


class ResearchBacktestHoldingsSourceError(ValueError):
    """Raised when an exact native Holdings source cannot be resolved."""


def _absolute(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(value))


class ResearchBacktestHoldingsSourceResult:
    """Defensively expose validated Holdings data and compact source facts."""

    __slots__ = (
        "_holdings",
        "_artifact_dir",
        "_holdings_path",
        "_schema_version",
        "_rows",
        "_date_count",
    )

    def __init__(
        self,
        holdings: pd.DataFrame,
        *,
        artifact_dir: Path,
        holdings_path: Path,
        schema_version: str,
        rows: int,
        date_count: int,
    ) -> None:
        if not isinstance(holdings, pd.DataFrame) or holdings.empty:
            raise ResearchBacktestHoldingsSourceError(
                "validated Holdings payload must be a non-empty DataFrame."
            )
        if rows != len(holdings) or date_count != holdings["trade_date"].nunique():
            raise ResearchBacktestHoldingsSourceError(
                "Holdings source counts differ from the validated payload."
            )
        self._holdings = holdings.copy(deep=True)
        self._artifact_dir = artifact_dir
        self._holdings_path = holdings_path
        self._schema_version = schema_version
        self._rows = rows
        self._date_count = date_count

    @property
    def holdings(self) -> pd.DataFrame:
        return self._holdings.copy(deep=True)

    @property
    def artifact_dir(self) -> Path:
        return self._artifact_dir

    @property
    def holdings_path(self) -> Path:
        return self._holdings_path

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def date_count(self) -> int:
        return self._date_count


class ResearchBacktestHoldingsSourceAdapter:
    """Resolve one explicit native Holdings Artifact without discovery."""

    def load(
        self,
        source: BacktestSourceConfig,
        *,
        holdings_result: HoldingsPipelineResult | None = None,
    ) -> ResearchBacktestHoldingsSourceResult:
        from src.pipeline.holdings_execution import HoldingsPipelineResult
        from src.pipeline.research_backtest_config import BacktestSourceConfig

        if not isinstance(source, BacktestSourceConfig):
            raise ResearchBacktestHoldingsSourceError(
                "source must be a BacktestSourceConfig."
            )
        if source.mode == "pipeline":
            if not isinstance(holdings_result, HoldingsPipelineResult):
                raise ResearchBacktestHoldingsSourceError(
                    "pipeline source requires the current HoldingsPipelineResult."
                )
            if not holdings_result.enabled or holdings_result.artifact_dir is None:
                raise ResearchBacktestHoldingsSourceError(
                    "pipeline source requires an enabled Holdings result."
                )
            artifact_dir = _absolute(holdings_result.artifact_dir)
        else:
            if holdings_result is not None:
                raise ResearchBacktestHoldingsSourceError(
                    "files source does not accept a HoldingsPipelineResult."
                )
            if source.artifact_dir is None:
                raise ResearchBacktestHoldingsSourceError(
                    "files source requires an explicit Holdings Artifact directory."
                )
            artifact_dir = _absolute(source.artifact_dir)

        store = HoldingsArtifactStore()
        try:
            validation = store.validate(artifact_dir)
        except HoldingsArtifactError as exc:
            raise ResearchBacktestHoldingsSourceError(
                "Holdings Artifact validation failed."
            ) from exc
        if not validation.is_valid or validation.manifest is None:
            raise ResearchBacktestHoldingsSourceError(
                "Holdings Artifact validation failed."
            )
        holdings_path = artifact_dir / HOLDINGS_PARQUET_FILENAME
        if (
            not holdings_path.is_file()
            or holdings_path.is_symlink()
            or holdings_path.parent != artifact_dir
        ):
            raise ResearchBacktestHoldingsSourceError(
                "Holdings source does not identify its fixed native payload."
            )
        if source.mode == "pipeline" and (
            holdings_result is None
            or holdings_result.holdings_path is None
            or _absolute(holdings_result.holdings_path) != holdings_path
        ):
            raise ResearchBacktestHoldingsSourceError(
                "current Holdings result does not identify the validated payload."
            )
        try:
            holdings = pd.read_parquet(holdings_path, engine="pyarrow")
        except (OSError, ValueError, ImportError) as exc:
            raise ResearchBacktestHoldingsSourceError(
                "validated Holdings payload read failed."
            ) from exc
        return ResearchBacktestHoldingsSourceResult(
            holdings,
            artifact_dir=artifact_dir,
            holdings_path=holdings_path,
            schema_version=validation.manifest.holdings_schema_version,
            rows=len(holdings),
            date_count=int(holdings["trade_date"].nunique()),
        )
