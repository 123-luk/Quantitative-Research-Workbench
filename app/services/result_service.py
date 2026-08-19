"""Exact-run, validator-backed, read-only result views."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pandas as pd
import yaml

from src.holdings.artifacts import HoldingsArtifactStore
from src.ml.artifacts import MLExperimentArtifactStore
from src.pipeline.config import PipelineConfig
from src.pipeline.experiment import ExperimentManager
from src.research_backtest.analytics import PERFORMANCE_METRIC_KEYS
from src.research_backtest.artifacts import ResearchBacktestArtifactStore
from src.signals.artifacts import SignalArtifactStore


class ResultServiceError(ValueError):
    """Raised when an exact run or one of its present artifacts is invalid."""


@dataclass(frozen=True)
class ArtifactLineageView:
    artifact_type: str
    relative_path: str
    schema_version: str | None
    status: str
    upstream: Mapping[str, object]


@dataclass(frozen=True)
class ResultBundle:
    run_id: str
    status: str | None
    created_at: str | None
    raw_config: Mapping[str, object] | None
    config_summary: Mapping[str, object]
    metrics: Mapping[str, int | float | None]
    holdings: pd.DataFrame
    daily_returns: pd.DataFrame
    benchmark: pd.DataFrame
    rebalances: pd.DataFrame
    nav: pd.DataFrame
    drawdown: pd.DataFrame
    monthly_returns: pd.DataFrame
    drawdown_matches_metric: bool | None
    artifacts: tuple[ArtifactLineageView, ...]
    signal_available: bool
    holdings_available: bool
    research_backtest_available: bool


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ResultServiceError(f"Canonical JSON path is invalid: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except Exception as exc:
        raise ResultServiceError(f"Cannot read canonical JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ResultServiceError(f"Canonical JSON must be an object: {path.name}")
    return value


def _read_run_config(path: Path) -> tuple[dict[str, object], PipelineConfig] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ResultServiceError("Run config snapshot path is invalid.")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
        if not isinstance(raw, dict):
            raise TypeError("config is not a mapping")
        return deepcopy(raw), PipelineConfig.from_dict(raw)
    except Exception as exc:
        raise ResultServiceError("Run config snapshot is invalid.") from exc


def _config_summary(raw: Mapping[str, object] | None) -> dict[str, object]:
    if raw is None:
        return {}
    result: dict[str, object] = {}
    for label, key in (
        ("Start Date", "backtest_start"),
        ("End Date", "backtest_end"),
        ("Stock Pool", "stock_pool"),
        ("Factors", "selected_factors"),
        ("Benchmark", "benchmark"),
    ):
        if key in raw and raw[key] is not None:
            result[label] = deepcopy(raw[key])
    ml = raw.get("ml_experiment")
    if isinstance(ml, Mapping):
        experiment = ml.get("experiment")
        if isinstance(experiment, Mapping):
            training = experiment.get("training")
            if isinstance(training, Mapping) and training.get("model_name") is not None:
                result["Model"] = training["model_name"]
    signal = raw.get("signal")
    if isinstance(signal, Mapping) and signal.get("signal_direction") is not None:
        result["Signal Direction"] = signal["signal_direction"]
    holdings = raw.get("holdings")
    if isinstance(holdings, Mapping):
        if holdings.get("top_n") is not None:
            result["Top N"] = holdings["top_n"]
        portfolio = holdings.get("portfolio_construction")
        if isinstance(portfolio, Mapping):
            if portfolio.get("method") is not None:
                result["Portfolio Method"] = portfolio["method"]
            params = portfolio.get("params")
            if isinstance(params, Mapping):
                risk = params.get("risk_model")
                if isinstance(risk, Mapping):
                    for label, key in (
                        ("Risk Estimator", "estimator"),
                        ("Lookback Trading Days", "lookback_trading_days"),
                        ("Minimum Observations", "min_observations"),
                    ):
                        if risk.get(key) is not None:
                            result[label] = risk[key]
                else:
                    for label, key in (
                        ("Lookback Trading Days", "lookback_trading_days"),
                        ("Minimum Observations", "min_observations"),
                    ):
                        if params.get(key) is not None:
                            result[label] = params[key]
            constraints = portfolio.get("constraints")
            if isinstance(constraints, list):
                for item in constraints:
                    if isinstance(item, Mapping) and item.get("type") == "max_weight":
                        values = item.get("params")
                        if isinstance(values, Mapping) and values.get("max_weight") is not None:
                            result["Max Weight"] = values["max_weight"]
    backtest = raw.get("research_backtest")
    if isinstance(backtest, Mapping) and backtest.get("enabled") is True:
        benchmark = backtest.get("benchmark")
        costs = backtest.get("transaction_cost")
        performance = backtest.get("performance")
        if isinstance(benchmark, Mapping) and benchmark.get("benchmark_code") is not None:
            result["Backtest Benchmark"] = benchmark["benchmark_code"]
        if isinstance(costs, Mapping) and costs.get("cost_bps") is not None:
            result["Transaction Cost (bps)"] = costs["cost_bps"]
        if isinstance(performance, Mapping):
            if performance.get("annual_risk_free_rate") is not None:
                result["Annual Risk-Free Rate"] = performance["annual_risk_free_rate"]
            if performance.get("annualization_days") is not None:
                result["Annualization Days"] = performance["annualization_days"]
    return result


class ResultService:
    """Load only the exact selected run and validate every present artifact."""

    def __init__(self, output_root: str | Path) -> None:
        self._experiments = ExperimentManager(output_root)

    def load(self, run_id: str) -> ResultBundle:
        try:
            run_dir = self._experiments.resolve_run_dir(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise ResultServiceError(str(exc)) from exc
        config_result = _read_run_config(run_dir / "config_snapshot.yaml")
        raw_config = None if config_result is None else config_result[0]
        config = None if config_result is None else config_result[1]
        run_info = _read_json(run_dir / "run_info.json") if (run_dir / "run_info.json").exists() else {}

        artifacts: list[ArtifactLineageView] = []
        signal_available = False
        holdings_available = False
        rb_available = False
        holdings = pd.DataFrame(columns=["trade_date", "ts_code", "target_weight", "score", "rank"])
        daily = pd.DataFrame()
        benchmark = pd.DataFrame()
        rebalances = pd.DataFrame()
        metrics: dict[str, int | float | None] = {}

        if config is not None and config.ml_experiment.enabled and config.ml_experiment.save_artifacts:
            experiment_id = config.ml_experiment.experiment_id
            if experiment_id:
                ml_dir = run_dir / config.ml_experiment.artifact_root / experiment_id
                if ml_dir.exists():
                    try:
                        report = MLExperimentArtifactStore().validate(ml_dir)
                        manifest = MLExperimentArtifactStore().read_manifest(ml_dir)
                    except Exception as exc:
                        raise ResultServiceError("ML Artifact validation failed.") from exc
                    artifacts.append(ArtifactLineageView(
                        "ml", ml_dir.relative_to(run_dir).as_posix(), manifest.schema_version,
                        "valid", MappingProxyType({"experiment_id": report.experiment_id, "model_name": manifest.model_name}),
                    ))

        stage_specs = (
            ("signal", None if config is None else config.signal.enabled, None if config is None else config.signal.artifact_subdir),
            ("holdings", None if config is None else config.holdings.enabled, None if config is None else config.holdings.artifact_subdir),
            ("research_backtest", None if config is None else config.research_backtest.enabled, None if config is None else config.research_backtest.artifact_subdir),
        )
        for artifact_type, enabled, subdir in stage_specs:
            if not enabled or not subdir:
                continue
            artifact_dir = run_dir / subdir
            if not artifact_dir.exists():
                continue
            if artifact_type == "signal":
                try:
                    report = SignalArtifactStore().validate(artifact_dir)
                except Exception as exc:
                    raise ResultServiceError("Signal Artifact validation failed.") from exc
                if not report.is_valid or report.manifest is None:
                    raise ResultServiceError("Signal Artifact validation failed.")
                signal_available = True
                manifest_dict = report.manifest.as_dict()
                artifacts.append(ArtifactLineageView(
                    "signal", subdir, report.manifest.artifact_schema_version, "valid",
                    MappingProxyType(dict(manifest_dict["source_provenance"])),
                ))
            elif artifact_type == "holdings":
                try:
                    report = HoldingsArtifactStore().validate(artifact_dir)
                except Exception as exc:
                    raise ResultServiceError("Holdings Artifact validation failed.") from exc
                if not report.is_valid or report.manifest is None:
                    raise ResultServiceError("Holdings Artifact validation failed.")
                holdings_available = True
                try:
                    holdings = pd.read_parquet(
                        artifact_dir / "holdings.parquet", engine="pyarrow"
                    ).copy(deep=True)
                except Exception as exc:
                    raise ResultServiceError("Holdings Artifact read failed.") from exc
                artifacts.append(ArtifactLineageView(
                    "holdings", subdir, report.manifest.artifact_schema_version, "valid",
                    MappingProxyType(dict(report.manifest.source_signal_provenance)),
                ))
            else:
                try:
                    report = ResearchBacktestArtifactStore().validate(artifact_dir)
                except Exception as exc:
                    raise ResultServiceError(
                        "Research Backtest Artifact validation failed."
                    ) from exc
                if not report.is_valid or report.manifest is None:
                    raise ResultServiceError("Research Backtest Artifact validation failed.")
                rb_available = True
                metrics = _read_json(artifact_dir / "metrics.json")  # type: ignore[assignment]
                if set(metrics) != set(PERFORMANCE_METRIC_KEYS):
                    raise ResultServiceError("Research Backtest metric keys are invalid.")
                try:
                    daily = pd.read_parquet(
                        artifact_dir / "daily_portfolio.parquet", engine="pyarrow"
                    ).copy(deep=True)
                    benchmark = pd.read_parquet(
                        artifact_dir / "benchmark.parquet", engine="pyarrow"
                    ).copy(deep=True)
                    rebalances = pd.read_parquet(
                        artifact_dir / "rebalances.parquet", engine="pyarrow"
                    ).copy(deep=True)
                except Exception as exc:
                    raise ResultServiceError(
                        "Research Backtest Artifact read failed."
                    ) from exc
                audit = _read_json(artifact_dir / "audit.json")
                upstream = audit.get("upstream_holdings")
                artifacts.append(ArtifactLineageView(
                    "research_backtest", subdir, report.manifest.artifact_schema_version, "valid",
                    MappingProxyType(deepcopy(dict(upstream)) if isinstance(upstream, Mapping) else {}),
                ))

        nav = pd.DataFrame(columns=["trade_date", "Portfolio Net NAV", "Benchmark NAV"])
        drawdown = pd.DataFrame(columns=["trade_date", "drawdown"])
        monthly = pd.DataFrame(columns=["month", "net_return"])
        drawdown_matches: bool | None = None
        if rb_available:
            left = daily.loc[:, ["trade_date", "net_nav"]].rename(columns={"net_nav": "Portfolio Net NAV"})
            right = benchmark.loc[:, ["trade_date", "benchmark_nav"]].rename(columns={"benchmark_nav": "Benchmark NAV"})
            nav = left.merge(right, on="trade_date", how="inner", validate="one_to_one").sort_values("trade_date", kind="stable").reset_index(drop=True)
            if len(nav) != len(left) or len(nav) != len(right):
                raise ResultServiceError("NAV and benchmark Artifact dates do not align exactly.")
            nav_values = nav["Portfolio Net NAV"]
            drawdown = pd.DataFrame({
                "trade_date": nav["trade_date"],
                "drawdown": nav_values / nav_values.cummax() - 1.0,
            })
            canonical = metrics.get("net_max_drawdown")
            if isinstance(canonical, (int, float)) and not isinstance(canonical, bool) and math.isfinite(float(canonical)):
                drawdown_matches = math.isclose(float(drawdown["drawdown"].min()), float(canonical), rel_tol=1e-9, abs_tol=1e-9)
            month_values = daily.assign(month=daily["trade_date"].dt.to_period("M")).groupby("month", sort=True)["net_return"].agg(lambda values: (1.0 + values).prod() - 1.0)
            monthly = month_values.rename("net_return").reset_index()
            monthly["month"] = monthly["month"].astype(str)

        return ResultBundle(
            run_id=run_id,
            status=run_info.get("status") if isinstance(run_info.get("status"), str) else None,
            created_at=run_info.get("created_at") if isinstance(run_info.get("created_at"), str) else None,
            raw_config=None if raw_config is None else MappingProxyType(deepcopy(raw_config)),
            config_summary=MappingProxyType(_config_summary(raw_config)),
            metrics=MappingProxyType(deepcopy(metrics)),
            holdings=holdings.copy(deep=True),
            daily_returns=daily.copy(deep=True),
            benchmark=benchmark.copy(deep=True),
            rebalances=rebalances.copy(deep=True),
            nav=nav.copy(deep=True),
            drawdown=drawdown.copy(deep=True),
            monthly_returns=monthly.copy(deep=True),
            drawdown_matches_metric=drawdown_matches,
            artifacts=tuple(artifacts),
            signal_available=signal_available,
            holdings_available=holdings_available,
            research_backtest_available=rb_available,
        )
