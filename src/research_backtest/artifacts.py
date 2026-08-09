"""Native persistence and independent validation for V6 backtest results."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pandas as pd

from src.holdings.artifacts import (
    HOLDINGS_MANIFEST_FILENAME,
    HOLDINGS_PARQUET_FILENAME,
    HoldingsArtifactStore,
)
from src.holdings.contracts import HOLDINGS_OUTPUT_COLUMNS
from src.research_backtest.analytics import (
    BENCHMARK_DAILY_COLUMNS,
    PERFORMANCE_METRIC_KEYS,
    PerformanceAnalyticsResult,
)
from src.research_backtest.portfolio import (
    DAILY_PORTFOLIO_COLUMNS,
    PortfolioDailyAccountingResult,
)
from src.research_backtest.rebalance import (
    REBALANCE_OUTPUT_COLUMNS,
    WEIGHT_TOLERANCE,
    RebalanceAccountingResult,
)
from src.research_backtest.returns import (
    BENCHMARK_RETURN_SOURCE_NAME,
    RAW_RETURN_FIELD,
    RETURN_UNIT,
    SECURITY_RETURN_SOURCE_NAME,
)

if TYPE_CHECKING:
    from src.pipeline.research_backtest_config import ResearchBacktestPipelineConfig


RESEARCH_BACKTEST_ARTIFACT_SCHEMA_VERSION = "1.0"
RESEARCH_BACKTEST_ARTIFACT_TYPE = "research_backtest"
REBALANCES_FILENAME = "rebalances.parquet"
DAILY_PORTFOLIO_FILENAME = "daily_portfolio.parquet"
BENCHMARK_FILENAME = "benchmark.parquet"
METRICS_FILENAME = "metrics.json"
RESEARCH_BACKTEST_CONFIG_FILENAME = "config.json"
RESEARCH_BACKTEST_AUDIT_FILENAME = "audit.json"
RESEARCH_BACKTEST_MANIFEST_FILENAME = "manifest.json"
RESEARCH_BACKTEST_ARTIFACT_FILENAMES = (
    REBALANCES_FILENAME,
    DAILY_PORTFOLIO_FILENAME,
    BENCHMARK_FILENAME,
    METRICS_FILENAME,
    RESEARCH_BACKTEST_CONFIG_FILENAME,
    RESEARCH_BACKTEST_AUDIT_FILENAME,
    RESEARCH_BACKTEST_MANIFEST_FILENAME,
)
_PAYLOAD_FILENAMES = RESEARCH_BACKTEST_ARTIFACT_FILENAMES[:-1]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 1024 * 1024
_AUDIT_FIELDS = {
    "schema_version",
    "artifact_type",
    "start_date",
    "end_date",
    "observation_count",
    "rebalance_count",
    "schedule_mode",
    "effective_rule",
    "return_convention",
    "turnover_definition",
    "cost_bps",
    "cost_rate_basis",
    "benchmark_code",
    "benchmark_alignment_policy",
    "annualization_days",
    "annual_risk_free_rate",
    "initial_nav",
    "security_return_source",
    "benchmark_return_source",
    "return_unit",
    "suspension_missing_return_policy",
    "unknown_missing_policy",
    "upstream_holdings",
    "rebalance_rows",
    "daily_portfolio_rows",
    "benchmark_rows",
    "metrics_count",
    "timing_convention",
    "first_effective_day_strategy_gross_return_zero",
    "first_effective_day_benchmark_return_zero",
}
_LINEAGE_FIELDS = {
    "holdings_artifact_dir",
    "holdings_data_path",
    "holdings_schema_version",
    "holdings_manifest_path",
    "holdings_manifest_sha256",
    "holdings_data_sha256",
    "holdings_rows",
    "holdings_date_count",
}


class ResearchBacktestArtifactError(ValueError):
    """Base error for native research-backtest Artifact operations."""


class ResearchBacktestArtifactExistsError(ResearchBacktestArtifactError):
    """Raised when no-overwrite prevents publication."""


class ResearchBacktestArtifactWriteError(ResearchBacktestArtifactError):
    """Raised when a complete Artifact cannot be safely published."""


class ResearchBacktestArtifactValidationError(ResearchBacktestArtifactError):
    """Raised for invalid Artifact API inputs or strict metadata."""


def _strict_keys(value: object, expected: set[str], context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ResearchBacktestArtifactValidationError(
            f"{context} must be a mapping with string keys."
        )
    if set(value) != expected:
        raise ResearchBacktestArtifactValidationError(f"{context} fields are invalid.")
    return dict(value)


def _path(value: object, *, field_name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ResearchBacktestArtifactValidationError(
            f"{field_name} must be str or os.PathLike."
        )
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw.strip() or raw != raw.strip() or "\x00" in raw:
        raise ResearchBacktestArtifactValidationError(
            f"{field_name} must identify a non-empty trimmed directory."
        )
    path = Path(raw)
    if raw in {".", ".."} or path == Path(path.anchor):
        raise ResearchBacktestArtifactValidationError(
            f"{field_name} must identify an explicit child directory."
        )
    return Path(os.path.abspath(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ResearchBacktestArtifactValidationError("checksum read failed.") from exc
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _read_json(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResearchBacktestArtifactValidationError(
            "strict JSON read failed."
        ) from exc
    if not isinstance(value, dict):
        raise ResearchBacktestArtifactValidationError(
            "JSON top-level value must be an object."
        )
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise ResearchBacktestArtifactWriteError("strict JSON write failed.") from exc


@dataclass(frozen=True)
class ResearchBacktestArtifactFileRecord:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.relative_path not in _PAYLOAD_FILENAMES:
            raise ResearchBacktestArtifactValidationError("file record path is invalid.")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ResearchBacktestArtifactValidationError("file size must be positive.")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ResearchBacktestArtifactValidationError("file sha256 is invalid.")

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> ResearchBacktestArtifactFileRecord:
        data = _strict_keys(
            value, {"relative_path", "size_bytes", "sha256"}, "file record"
        )
        return cls(**data)  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ResearchBacktestArtifactManifest:
    artifact_type: str
    artifact_schema_version: str
    created_at_utc: str
    files: tuple[ResearchBacktestArtifactFileRecord, ...]

    def __post_init__(self) -> None:
        if self.artifact_type != RESEARCH_BACKTEST_ARTIFACT_TYPE:
            raise ResearchBacktestArtifactValidationError("artifact_type is invalid.")
        if self.artifact_schema_version != RESEARCH_BACKTEST_ARTIFACT_SCHEMA_VERSION:
            raise ResearchBacktestArtifactValidationError(
                "artifact_schema_version is invalid."
            )
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc.endswith("Z"):
            raise ResearchBacktestArtifactValidationError("created_at_utc is invalid.")
        try:
            created = datetime.fromisoformat(self.created_at_utc[:-1] + "+00:00")
        except ValueError as exc:
            raise ResearchBacktestArtifactValidationError(
                "created_at_utc is invalid."
            ) from exc
        if created.utcoffset() != timezone.utc.utcoffset(created):
            raise ResearchBacktestArtifactValidationError("created_at_utc must be UTC.")
        if (
            len(self.files) != len(_PAYLOAD_FILENAMES)
            or tuple(item.relative_path for item in self.files) != _PAYLOAD_FILENAMES
        ):
            raise ResearchBacktestArtifactValidationError("manifest files are invalid.")

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> ResearchBacktestArtifactManifest:
        data = _strict_keys(
            value,
            {"artifact_type", "artifact_schema_version", "created_at_utc", "files"},
            "manifest",
        )
        if not isinstance(data["files"], list):
            raise ResearchBacktestArtifactValidationError("manifest files must be a list.")
        return cls(
            artifact_type=data["artifact_type"],  # type: ignore[arg-type]
            artifact_schema_version=data["artifact_schema_version"],  # type: ignore[arg-type]
            created_at_utc=data["created_at_utc"],  # type: ignore[arg-type]
            files=tuple(
                ResearchBacktestArtifactFileRecord.from_dict(item)
                for item in data["files"]  # type: ignore[union-attr]
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_type": self.artifact_type,
            "artifact_schema_version": self.artifact_schema_version,
            "created_at_utc": self.created_at_utc,
            "files": [item.as_dict() for item in self.files],
        }


@dataclass(frozen=True)
class ResearchBacktestArtifactValidationIssue:
    code: str
    message: str
    relative_path: str | None = None


@dataclass(frozen=True)
class ResearchBacktestArtifactValidationResult:
    artifact_dir: Path
    is_valid: bool
    issues: tuple[ResearchBacktestArtifactValidationIssue, ...]
    manifest: ResearchBacktestArtifactManifest | None


@dataclass(frozen=True)
class ResearchBacktestArtifactResult:
    artifact_dir: Path
    rebalances_path: Path
    daily_portfolio_path: Path
    benchmark_path: Path
    metrics_path: Path
    config_path: Path
    audit_path: Path
    manifest_path: Path
    schema_version: str
    observation_count: int
    rebalance_count: int
    benchmark_code: str
    validation: ResearchBacktestArtifactValidationResult


def _issue(
    code: str, message: str, relative_path: str | None = None
) -> ResearchBacktestArtifactValidationIssue:
    return ResearchBacktestArtifactValidationIssue(code, message, relative_path)


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...], context: str) -> None:
    for column in columns:
        values = frame[column]
        if (
            pd.api.types.is_bool_dtype(values.dtype)
            or not pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_complex_dtype(values.dtype)
        ):
            raise ResearchBacktestArtifactValidationError(
                f"{context} {column} dtype is invalid."
            )
        if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
            raise ResearchBacktestArtifactValidationError(
                f"{context} {column} must be finite."
            )


def _dates(frame: pd.DataFrame, column: str, context: str) -> None:
    values = frame[column]
    if (
        not pd.api.types.is_datetime64_any_dtype(values.dtype)
        or getattr(values.dt, "tz", None) is not None
        or values.isna().any()
        or not values.eq(values.dt.normalize()).all()
    ):
        raise ResearchBacktestArtifactValidationError(
            f"{context} {column} is invalid."
        )


def _validate_rebalances(frame: pd.DataFrame) -> None:
    if frame.empty or tuple(frame.columns) != REBALANCE_OUTPUT_COLUMNS:
        raise ResearchBacktestArtifactValidationError("rebalances schema is invalid.")
    _dates(frame, "holdings_trade_date", "rebalances")
    _dates(frame, "effective_date", "rebalances")
    _numeric(
        frame,
        (
            "pre_rebalance_weight",
            "target_weight",
            "weight_change",
            "pre_cash_weight",
            "target_cash_weight",
            "cash_weight_change",
            "turnover",
        ),
        "rebalances",
    )
    if frame.duplicated(["effective_date", "ts_code"]).any():
        raise ResearchBacktestArtifactValidationError("rebalance keys are duplicated.")
    expected = frame.sort_values(
        ["effective_date", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)
    if not frame.reset_index(drop=True).equals(expected):
        raise ResearchBacktestArtifactValidationError("rebalances order is invalid.")
    if not frame["ts_code"].map(
        lambda value: isinstance(value, str) and value == value.strip() and bool(value)
    ).all():
        raise ResearchBacktestArtifactValidationError("rebalance codes are invalid.")
    for _, group in frame.groupby("effective_date", sort=False):
        if group["holdings_trade_date"].nunique() != 1:
            raise ResearchBacktestArtifactValidationError(
                "rebalance event holdings date is inconsistent."
            )
        for column in (
            "pre_cash_weight",
            "target_cash_weight",
            "cash_weight_change",
            "turnover",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise ResearchBacktestArtifactValidationError(
                    f"rebalance event {column} is inconsistent."
                )
        pre_cash = float(group["pre_cash_weight"].iloc[0])
        target_cash = float(group["target_cash_weight"].iloc[0])
        if not (
            group["holdings_trade_date"].iloc[0]
            < group["effective_date"].iloc[0]
        ):
            raise ResearchBacktestArtifactValidationError(
                "rebalance effective date must follow its holdings date."
            )
        if min(pre_cash, target_cash, float(group["turnover"].iloc[0])) < 0.0:
            raise ResearchBacktestArtifactValidationError(
                "rebalance cash or turnover is negative."
            )
        if (group[["pre_rebalance_weight", "target_weight"]] < 0.0).any().any():
            raise ResearchBacktestArtifactValidationError("rebalance weight is negative.")
        if not np.isclose(
            group["pre_rebalance_weight"].sum() + pre_cash,
            1.0,
            rtol=0.0,
            atol=WEIGHT_TOLERANCE,
        ):
            raise ResearchBacktestArtifactValidationError("rebalance pre-state is invalid.")
        if not np.isclose(
            group["target_weight"].sum() + target_cash,
            1.0,
            rtol=0.0,
            atol=WEIGHT_TOLERANCE,
        ):
            raise ResearchBacktestArtifactValidationError(
                "rebalance target state is invalid."
            )
        if not np.isclose(
            group["weight_change"],
            group["target_weight"] - group["pre_rebalance_weight"],
            rtol=0.0,
            atol=WEIGHT_TOLERANCE,
        ).all():
            raise ResearchBacktestArtifactValidationError(
                "rebalance weight changes are invalid."
            )


def _validate_daily(frame: pd.DataFrame) -> None:
    if frame.empty or tuple(frame.columns) != DAILY_PORTFOLIO_COLUMNS:
        raise ResearchBacktestArtifactValidationError("daily portfolio schema is invalid.")
    _dates(frame, "trade_date", "daily portfolio")
    _numeric(
        frame,
        (
            "gross_return",
            "transaction_cost",
            "net_return",
            "gross_nav",
            "net_nav",
            "turnover",
            "traded_notional",
        ),
        "daily portfolio",
    )
    if frame["trade_date"].duplicated().any() or not frame["trade_date"].is_monotonic_increasing:
        raise ResearchBacktestArtifactValidationError("daily dates are invalid.")
    if any(type(item) not in (bool, np.bool_) for item in frame["is_rebalance"]):
        raise ResearchBacktestArtifactValidationError("is_rebalance is invalid.")
    if (frame[["gross_nav", "net_nav"]] <= 0.0).any().any():
        raise ResearchBacktestArtifactValidationError("daily NAV is invalid.")
    quiet = frame.loc[~frame["is_rebalance"]]
    if not np.isclose(
        quiet[["transaction_cost", "turnover", "traded_notional"]],
        0.0,
        rtol=0.0,
        atol=WEIGHT_TOLERANCE,
    ).all():
        raise ResearchBacktestArtifactValidationError(
            "non-rebalance accounting fields must be zero."
        )
    if not np.isclose(
        frame["gross_return"].iloc[0], 0.0, rtol=0.0, atol=WEIGHT_TOLERANCE
    ):
        raise ResearchBacktestArtifactValidationError(
            "first strategy gross return must be zero."
        )


def _validate_benchmark(frame: pd.DataFrame, code: str) -> None:
    if frame.empty or tuple(frame.columns) != BENCHMARK_DAILY_COLUMNS:
        raise ResearchBacktestArtifactValidationError("benchmark schema is invalid.")
    _dates(frame, "trade_date", "benchmark")
    _numeric(frame, ("benchmark_return", "benchmark_nav"), "benchmark")
    if frame["trade_date"].duplicated().any() or not frame["trade_date"].is_monotonic_increasing:
        raise ResearchBacktestArtifactValidationError("benchmark dates are invalid.")
    if tuple(sorted(set(frame["benchmark_code"]))) != (code,):
        raise ResearchBacktestArtifactValidationError("benchmark code is invalid.")
    if (frame["benchmark_nav"] <= 0.0).any():
        raise ResearchBacktestArtifactValidationError("benchmark NAV is invalid.")
    if not np.isclose(
        frame["benchmark_return"].iloc[0], 0.0, rtol=0.0, atol=WEIGHT_TOLERANCE
    ):
        raise ResearchBacktestArtifactValidationError(
            "first benchmark return must be zero."
        )


def _validate_metrics(value: object) -> dict[str, int | float | None]:
    if not isinstance(value, Mapping) or set(value) != set(PERFORMANCE_METRIC_KEYS):
        raise ResearchBacktestArtifactValidationError("metrics keys are invalid.")
    result: dict[str, int | float | None] = {}
    for key in PERFORMANCE_METRIC_KEYS:
        item = value[key]
        if item is not None and type(item) not in (int, float):
            raise ResearchBacktestArtifactValidationError("metric value is invalid.")
        if isinstance(item, float) and not math.isfinite(item):
            raise ResearchBacktestArtifactValidationError("metric value is non-finite.")
        result[key] = item
    json.dumps(result, allow_nan=False)
    return result


def _config(value: object) -> tuple[dict[str, object], ResearchBacktestPipelineConfig]:
    from src.pipeline.research_backtest_config import ResearchBacktestPipelineConfig

    try:
        config = ResearchBacktestPipelineConfig.from_dict(value)
    except Exception as exc:
        raise ResearchBacktestArtifactValidationError(
            "research backtest config is invalid."
        ) from exc
    if (
        not config.enabled
        or config.transaction_cost is None
        or config.benchmark is None
        or config.performance is None
    ):
        raise ResearchBacktestArtifactValidationError(
            "artifact config must be enabled and complete."
        )
    snapshot = config.to_dict()
    json.dumps(snapshot, allow_nan=False)
    return snapshot, config


def _lineage(artifact_dir: object) -> tuple[dict[str, object], pd.DataFrame]:
    directory = _path(artifact_dir, field_name="holdings_artifact_dir")
    report = HoldingsArtifactStore().validate(directory)
    if not report.is_valid or report.manifest is None:
        raise ResearchBacktestArtifactValidationError(
            "upstream Holdings Artifact validation failed."
        )
    manifest_path = directory / HOLDINGS_MANIFEST_FILENAME
    data_path = directory / HOLDINGS_PARQUET_FILENAME
    try:
        frame = pd.read_parquet(data_path, engine="pyarrow")
    except Exception as exc:
        raise ResearchBacktestArtifactValidationError(
            "upstream Holdings parquet read failed."
        ) from exc
    record = next(
        item for item in report.manifest.files if item.relative_path == HOLDINGS_PARQUET_FILENAME
    )
    value = {
        "holdings_artifact_dir": str(directory),
        "holdings_data_path": str(data_path),
        "holdings_schema_version": report.manifest.holdings_schema_version,
        "holdings_manifest_path": str(manifest_path),
        "holdings_manifest_sha256": _sha256(manifest_path),
        "holdings_data_sha256": record.sha256,
        "holdings_rows": int(len(frame)),
        "holdings_date_count": int(frame["trade_date"].nunique()),
    }
    return value, frame


def _validate_lineage(value: object) -> tuple[dict[str, object], pd.DataFrame]:
    data = _strict_keys(value, _LINEAGE_FIELDS, "upstream Holdings lineage")
    expected, frame = _lineage(data["holdings_artifact_dir"])
    if data != expected:
        raise ResearchBacktestArtifactValidationError(
            "upstream Holdings lineage identity differs from validated files."
        )
    return data, frame


def _holdings_match(rebalances: pd.DataFrame, holdings: pd.DataFrame) -> None:
    target = rebalances.loc[
        rebalances["target_weight"].gt(WEIGHT_TOLERANCE),
        ["holdings_trade_date", "ts_code", "target_weight"],
    ].rename(columns={"holdings_trade_date": "trade_date"})
    target = target.sort_values(
        ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
    )
    upstream = holdings.loc[:, ["trade_date", "ts_code", "target_weight"]]
    upstream = upstream.sort_values(
        ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
    )
    target_keys = tuple(zip(target["trade_date"], target["ts_code"]))
    upstream_keys = tuple(zip(upstream["trade_date"], upstream["ts_code"]))
    if target_keys != upstream_keys or not np.allclose(
        target["target_weight"],
        upstream["target_weight"],
        rtol=0.0,
        atol=WEIGHT_TOLERANCE,
    ):
        raise ResearchBacktestArtifactValidationError(
            "rebalance targets differ from upstream Holdings Artifact."
        )


def _cross_validate(
    rebalances: pd.DataFrame,
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    metrics: Mapping[str, int | float | None],
    initial_nav: float,
) -> None:
    event_dates = tuple(rebalances["effective_date"].drop_duplicates())
    daily_events = tuple(daily.loc[daily["is_rebalance"], "trade_date"])
    if event_dates != daily_events:
        raise ResearchBacktestArtifactValidationError(
            "rebalance effective dates differ from daily events."
        )
    for date, group in rebalances.groupby("effective_date", sort=False):
        turnover = float(group["turnover"].iloc[0])
        daily_turnover = float(
            daily.loc[daily["trade_date"].eq(date), "turnover"].iloc[0]
        )
        if not np.isclose(
            turnover, daily_turnover, rtol=0.0, atol=WEIGHT_TOLERANCE
        ):
            raise ResearchBacktestArtifactValidationError(
                "rebalance turnover differs from daily portfolio."
            )
    if tuple(daily["trade_date"]) != tuple(benchmark["trade_date"]):
        raise ResearchBacktestArtifactValidationError(
            "daily portfolio and benchmark calendars differ."
        )
    observations = len(daily)
    rebalances_count = int(daily["is_rebalance"].sum())
    if (
        metrics["observation_count"] != observations
        or metrics["rebalance_count"] != rebalances_count
        or len(benchmark) != observations
    ):
        raise ResearchBacktestArtifactValidationError(
            "metrics observation or rebalance counts differ."
        )
    expected_returns = {
        "gross_total_return": float(daily["gross_nav"].iloc[-1] / initial_nav - 1.0),
        "net_total_return": float(daily["net_nav"].iloc[-1] / initial_nav - 1.0),
        "benchmark_total_return": float(
            benchmark["benchmark_nav"].iloc[-1] / initial_nav - 1.0
        ),
    }
    for name, expected in expected_returns.items():
        if not np.isclose(
            float(metrics[name]), expected, rtol=0.0, atol=WEIGHT_TOLERANCE
        ):
            raise ResearchBacktestArtifactValidationError(
                f"metric {name} differs from final NAV."
            )


def _audit(
    value: object,
    *,
    config: ResearchBacktestPipelineConfig,
    rebalances: pd.DataFrame,
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    metrics: Mapping[str, int | float | None],
    lineage: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if value is not None:
        data = _strict_keys(value, _AUDIT_FIELDS, "audit")
    else:
        assert config.transaction_cost is not None
        assert config.benchmark is not None
        assert config.performance is not None
        data = {
            "schema_version": RESEARCH_BACKTEST_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": RESEARCH_BACKTEST_ARTIFACT_TYPE,
            "start_date": daily["trade_date"].iloc[0].date().isoformat(),
            "end_date": daily["trade_date"].iloc[-1].date().isoformat(),
            "observation_count": int(len(daily)),
            "rebalance_count": int(daily["is_rebalance"].sum()),
            "schedule_mode": config.schedule.mode,
            "effective_rule": config.return_alignment.effective_rule,
            "return_convention": config.return_alignment.return_convention,
            "turnover_definition": config.portfolio.turnover_definition,
            "cost_bps": config.transaction_cost.cost_bps,
            "cost_rate_basis": config.transaction_cost.rate_basis,
            "benchmark_code": config.benchmark.benchmark_code,
            "benchmark_alignment_policy": config.benchmark.alignment_policy,
            "annualization_days": config.performance.annualization_days,
            "annual_risk_free_rate": config.performance.annual_risk_free_rate,
            "initial_nav": config.portfolio.initial_nav,
            "security_return_source": f"{SECURITY_RETURN_SOURCE_NAME}.{RAW_RETURN_FIELD}",
            "benchmark_return_source": f"{BENCHMARK_RETURN_SOURCE_NAME}.{RAW_RETURN_FIELD}",
            "return_unit": RETURN_UNIT,
            "suspension_missing_return_policy": "proven_full_day_suspension_zero_only",
            "unknown_missing_policy": "error",
            "upstream_holdings": dict(lineage or {}),
            "rebalance_rows": int(len(rebalances)),
            "daily_portfolio_rows": int(len(daily)),
            "benchmark_rows": int(len(benchmark)),
            "metrics_count": int(len(metrics)),
            "timing_convention": "post_close_rebalance_accounting",
            "first_effective_day_strategy_gross_return_zero": True,
            "first_effective_day_benchmark_return_zero": True,
        }
    if (
        data["schema_version"] != RESEARCH_BACKTEST_ARTIFACT_SCHEMA_VERSION
        or data["artifact_type"] != RESEARCH_BACKTEST_ARTIFACT_TYPE
        or data["start_date"] != daily["trade_date"].iloc[0].date().isoformat()
        or data["end_date"] != daily["trade_date"].iloc[-1].date().isoformat()
        or data["observation_count"] != len(daily)
        or data["rebalance_count"] != int(daily["is_rebalance"].sum())
        or data["rebalance_rows"] != len(rebalances)
        or data["daily_portfolio_rows"] != len(daily)
        or data["benchmark_rows"] != len(benchmark)
        or data["metrics_count"] != len(metrics)
    ):
        raise ResearchBacktestArtifactValidationError("audit counts or identity differ.")
    assumptions = {
        "schedule_mode": config.schedule.mode,
        "effective_rule": config.return_alignment.effective_rule,
        "return_convention": config.return_alignment.return_convention,
        "turnover_definition": config.portfolio.turnover_definition,
        "cost_bps": config.transaction_cost.cost_bps,  # type: ignore[union-attr]
        "cost_rate_basis": config.transaction_cost.rate_basis,  # type: ignore[union-attr]
        "benchmark_code": config.benchmark.benchmark_code,  # type: ignore[union-attr]
        "benchmark_alignment_policy": (
            config.benchmark.alignment_policy  # type: ignore[union-attr]
        ),
        "annualization_days": config.performance.annualization_days,  # type: ignore[union-attr]
        "annual_risk_free_rate": (
            config.performance.annual_risk_free_rate  # type: ignore[union-attr]
        ),
        "initial_nav": config.portfolio.initial_nav,
    }
    if any(
        type(data[name]) is not type(expected) or data[name] != expected
        for name, expected in assumptions.items()
    ):
        raise ResearchBacktestArtifactValidationError("audit assumptions differ.")
    if (
        data["security_return_source"] != f"{SECURITY_RETURN_SOURCE_NAME}.{RAW_RETURN_FIELD}"
        or data["benchmark_return_source"] != f"{BENCHMARK_RETURN_SOURCE_NAME}.{RAW_RETURN_FIELD}"
        or data["return_unit"] != RETURN_UNIT
        or data["suspension_missing_return_policy"]
        != "proven_full_day_suspension_zero_only"
        or data["unknown_missing_policy"] != "error"
        or data["timing_convention"] != "post_close_rebalance_accounting"
        or data["first_effective_day_strategy_gross_return_zero"] is not True
        or data["first_effective_day_benchmark_return_zero"] is not True
    ):
        raise ResearchBacktestArtifactValidationError("audit semantics differ.")
    _strict_keys(data["upstream_holdings"], _LINEAGE_FIELDS, "upstream lineage")
    return data


def _record(path: Path) -> ResearchBacktestArtifactFileRecord:
    try:
        return ResearchBacktestArtifactFileRecord(
            path.name, path.stat().st_size, _sha256(path)
        )
    except (OSError, ResearchBacktestArtifactError) as exc:
        raise ResearchBacktestArtifactWriteError("payload metadata failed.") from exc


class ResearchBacktestArtifactStore:
    """Publish and independently validate one explicit native Artifact."""

    def read_manifest(self, artifact_dir: object) -> ResearchBacktestArtifactManifest:
        directory = _path(artifact_dir, field_name="artifact_dir")
        return ResearchBacktestArtifactManifest.from_dict(
            _read_json(directory / RESEARCH_BACKTEST_MANIFEST_FILENAME)
        )

    def validate(self, artifact_dir: object) -> ResearchBacktestArtifactValidationResult:
        directory = _path(artifact_dir, field_name="artifact_dir")
        issues: list[ResearchBacktestArtifactValidationIssue] = []
        if not directory.exists() or not directory.is_dir() or directory.is_symlink():
            return ResearchBacktestArtifactValidationResult(
                directory,
                False,
                (_issue("artifact_dir_invalid", "Artifact directory is invalid."),),
                None,
            )
        entries = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
        actual = {item.name for item in entries}
        expected = set(RESEARCH_BACKTEST_ARTIFACT_FILENAMES)
        for name in sorted(expected - actual):
            issues.append(_issue("missing_file", "Required file is missing.", name))
        for name in sorted(actual - expected):
            issues.append(_issue("unexpected_entry", "Unexpected entry.", name))
        safe = {
            item.name: item
            for item in entries
            if item.name in expected and item.is_file() and not item.is_symlink()
        }
        if set(safe) != expected:
            issues.append(_issue("unsafe_file", "Artifact files must be regular files."))
        manifest: ResearchBacktestArtifactManifest | None = None
        if RESEARCH_BACKTEST_MANIFEST_FILENAME in safe:
            try:
                manifest = self.read_manifest(directory)
            except ResearchBacktestArtifactError:
                issues.append(_issue("invalid_manifest", "Manifest is invalid."))
        if manifest is not None:
            for record in manifest.files:
                path = safe.get(record.relative_path)
                if path is None:
                    continue
                try:
                    if path.stat().st_size != record.size_bytes:
                        issues.append(
                            _issue(
                                "file_size_mismatch",
                                "File size differs.",
                                record.relative_path,
                            )
                        )
                    if _sha256(path) != record.sha256:
                        issues.append(
                            _issue(
                                "checksum_mismatch",
                                "Checksum differs.",
                                record.relative_path,
                            )
                        )
                except (OSError, ResearchBacktestArtifactError) as exc:
                    issues.append(
                        _issue("payload_read_error", str(exc), record.relative_path)
                    )
        try:
            config_data = _read_json(safe[RESEARCH_BACKTEST_CONFIG_FILENAME])
            _, config = _config(config_data)
            metrics = _validate_metrics(_read_json(safe[METRICS_FILENAME]))
            rebalances = pd.read_parquet(safe[REBALANCES_FILENAME], engine="pyarrow")
            daily = pd.read_parquet(safe[DAILY_PORTFOLIO_FILENAME], engine="pyarrow")
            benchmark = pd.read_parquet(safe[BENCHMARK_FILENAME], engine="pyarrow")
            _validate_rebalances(rebalances)
            _validate_daily(daily)
            _validate_benchmark(
                benchmark,
                config.benchmark.benchmark_code,  # type: ignore[union-attr]
            )
            audit_data = _read_json(safe[RESEARCH_BACKTEST_AUDIT_FILENAME])
            _audit(
                audit_data,
                config=config,
                rebalances=rebalances,
                daily=daily,
                benchmark=benchmark,
                metrics=metrics,
            )
            lineage, holdings = _validate_lineage(audit_data["upstream_holdings"])
            if lineage != audit_data["upstream_holdings"]:
                raise ResearchBacktestArtifactValidationError("lineage differs.")
            _holdings_match(rebalances, holdings)
            _cross_validate(
                rebalances,
                daily,
                benchmark,
                metrics,
                config.portfolio.initial_nav,
            )
        except Exception as exc:
            issues.append(_issue("content_validation_error", str(exc)))
        unique = tuple(dict.fromkeys(issues))
        return ResearchBacktestArtifactValidationResult(
            directory, not unique, unique, manifest
        )

    def publish(
        self,
        *,
        artifact_dir: object,
        rebalances: RebalanceAccountingResult,
        portfolio: PortfolioDailyAccountingResult,
        analytics: PerformanceAnalyticsResult,
        config: ResearchBacktestPipelineConfig,
        holdings_artifact_dir: object,
        parquet_compression: str = "zstd",
    ) -> ResearchBacktestArtifactResult:
        target = _path(artifact_dir, field_name="artifact_dir")
        if target.exists() or target.is_symlink():
            raise ResearchBacktestArtifactExistsError(
                "target artifact_dir already exists."
            )
        if parquet_compression not in {"zstd", "snappy"}:
            raise ResearchBacktestArtifactValidationError(
                "parquet_compression must be zstd or snappy."
            )
        if not isinstance(rebalances, RebalanceAccountingResult):
            raise ResearchBacktestArtifactValidationError(
                "rebalances must be a RebalanceAccountingResult."
            )
        if not isinstance(portfolio, PortfolioDailyAccountingResult):
            raise ResearchBacktestArtifactValidationError(
                "portfolio must be a PortfolioDailyAccountingResult."
            )
        if not isinstance(analytics, PerformanceAnalyticsResult):
            raise ResearchBacktestArtifactValidationError(
                "analytics must be a PerformanceAnalyticsResult."
            )
        _, config_value = _config(config)
        config_snapshot = config_value.to_dict()
        rebalance_frame = rebalances.rebalances
        daily_frame = portfolio.daily_portfolio
        benchmark_frame = analytics.benchmark_daily
        metrics = analytics.metrics
        _validate_rebalances(rebalance_frame)
        _validate_daily(daily_frame)
        _validate_benchmark(
            benchmark_frame, config_value.benchmark.benchmark_code  # type: ignore[union-attr]
        )
        _validate_metrics(metrics)
        if (
            portfolio.initial_nav != config_value.portfolio.initial_nav
            or portfolio.cost_bps
            != config_value.transaction_cost.cost_bps  # type: ignore[union-attr]
            or portfolio.start_date != daily_frame["trade_date"].iloc[0]
            or portfolio.end_date != daily_frame["trade_date"].iloc[-1]
            or portfolio.row_count != len(daily_frame)
            or portfolio.rebalance_count != int(daily_frame["is_rebalance"].sum())
            or rebalances.event_count != portfolio.rebalance_count
            or analytics.start_date != daily_frame["trade_date"].iloc[0]
            or analytics.end_date != daily_frame["trade_date"].iloc[-1]
            or analytics.observation_count != len(daily_frame)
            or analytics.benchmark_code
            != config_value.benchmark.benchmark_code  # type: ignore[union-attr]
        ):
            raise ResearchBacktestArtifactValidationError(
                "result metadata differs from config."
            )
        lineage, holdings = _lineage(holdings_artifact_dir)
        _holdings_match(rebalance_frame, holdings)
        _cross_validate(
            rebalance_frame,
            daily_frame,
            benchmark_frame,
            metrics,
            config_value.portfolio.initial_nav,
        )
        audit = _audit(
            None,
            config=config_value,
            rebalances=rebalance_frame,
            daily=daily_frame,
            benchmark=benchmark_frame,
            metrics=metrics,
            lineage=lineage,
        )
        parent = target.parent
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ResearchBacktestArtifactWriteError("artifact parent is invalid.")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ResearchBacktestArtifactWriteError(
                "artifact parent creation failed."
            ) from exc
        staging = parent / f".tmp-{target.name}-{uuid4().hex}"
        published = False
        try:
            staging.mkdir()
            rebalance_frame.to_parquet(
                staging / REBALANCES_FILENAME,
                engine="pyarrow",
                compression=parquet_compression,
                index=False,
            )
            daily_frame.to_parquet(
                staging / DAILY_PORTFOLIO_FILENAME,
                engine="pyarrow",
                compression=parquet_compression,
                index=False,
            )
            benchmark_frame.to_parquet(
                staging / BENCHMARK_FILENAME,
                engine="pyarrow",
                compression=parquet_compression,
                index=False,
            )
            _write_json(staging / METRICS_FILENAME, metrics)
            _write_json(staging / RESEARCH_BACKTEST_CONFIG_FILENAME, config_snapshot)
            _write_json(staging / RESEARCH_BACKTEST_AUDIT_FILENAME, audit)
            records = tuple(_record(staging / name) for name in _PAYLOAD_FILENAMES)
            manifest = ResearchBacktestArtifactManifest(
                artifact_type=RESEARCH_BACKTEST_ARTIFACT_TYPE,
                artifact_schema_version=RESEARCH_BACKTEST_ARTIFACT_SCHEMA_VERSION,
                created_at_utc=datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                files=records,
            )
            _write_json(
                staging / RESEARCH_BACKTEST_MANIFEST_FILENAME,
                manifest.as_dict(),
            )
            pre = self.validate(staging)
            if not pre.is_valid:
                raise ResearchBacktestArtifactWriteError(
                    "pre-publish validation failed."
                )
            if target.exists() or target.is_symlink():
                raise ResearchBacktestArtifactExistsError(
                    "target appeared before publication."
                )
            os.replace(staging, target)
            published = True
            validation = self.validate(target)
            if not validation.is_valid:
                shutil.rmtree(target, ignore_errors=True)
                raise ResearchBacktestArtifactWriteError(
                    "post-publish validation failed."
                )
            paths = [target / name for name in RESEARCH_BACKTEST_ARTIFACT_FILENAMES]
            return ResearchBacktestArtifactResult(
                artifact_dir=target,
                rebalances_path=paths[0],
                daily_portfolio_path=paths[1],
                benchmark_path=paths[2],
                metrics_path=paths[3],
                config_path=paths[4],
                audit_path=paths[5],
                manifest_path=paths[6],
                schema_version=RESEARCH_BACKTEST_ARTIFACT_SCHEMA_VERSION,
                observation_count=portfolio.row_count,
                rebalance_count=portfolio.rebalance_count,
                benchmark_code=analytics.benchmark_code,
                validation=validation,
            )
        except ResearchBacktestArtifactError:
            if staging.exists() and not published:
                shutil.rmtree(staging, ignore_errors=True)
            raise
        except Exception as exc:
            if staging.exists() and not published:
                shutil.rmtree(staging, ignore_errors=True)
            raise ResearchBacktestArtifactWriteError("artifact write failed.") from exc
