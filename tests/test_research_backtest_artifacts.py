from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import src.research_backtest.artifacts as artifact_module
from src.holdings.artifacts import (
    HoldingsArtifactConfig,
    HoldingsArtifactStore,
    SignalArtifactProvenance,
)
from src.holdings.builder import HoldingsBuilder
from src.research_backtest import (
    BENCHMARK_DAILY_COLUMNS,
    DAILY_PORTFOLIO_COLUMNS,
    PERFORMANCE_METRIC_KEYS,
    REBALANCE_OUTPUT_COLUMNS,
    RESEARCH_BACKTEST_ARTIFACT_FILENAMES,
    RESEARCH_BACKTEST_ARTIFACT_SCHEMA_VERSION,
    RESEARCH_BACKTEST_ARTIFACT_TYPE,
    PerformanceAnalyticsEngine,
    PortfolioDailyAccountingResult,
    RebalanceAccountingResult,
    ResearchBacktestArtifactExistsError,
    ResearchBacktestArtifactStore,
    ResearchBacktestArtifactValidationResult,
    ResearchBacktestArtifactWriteError,
)
from src.pipeline.research_backtest_config import (
    BenchmarkConfig,
    PerformanceConfig,
    ResearchBacktestPipelineConfig,
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _refresh(directory: Path, filename: str) -> None:
    manifest = _json(directory / "manifest.json")
    path = directory / filename
    for record in manifest["files"]:  # type: ignore[index]
        if record["relative_path"] == filename:
            record["size_bytes"] = path.stat().st_size
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(directory / "manifest.json", manifest)


def _signals(codes: tuple[str, str] = ("A.SZ", "B.SZ")) -> pd.DataFrame:
    rows = []
    for date in ("2024-01-02", "2024-01-04"):
        for rank, code in enumerate(codes, start=1):
            rows.append(
                {
                    "trade_date": pd.Timestamp(date),
                    "ts_code": code,
                    "score": float(3 - rank),
                    "rank": rank,
                }
            )
    frame = pd.DataFrame(rows)
    frame["trade_date"] = frame["trade_date"].astype("datetime64[ns]")
    frame["ts_code"] = frame["ts_code"].astype("string")
    frame["rank"] = frame["rank"].astype(np.int64)
    return frame


def _holdings_artifact(
    tmp_path: Path,
    *,
    name: str = "holdings",
    codes: tuple[str, str] = ("A.SZ", "B.SZ"),
):
    result = HoldingsBuilder().build(
        _signals(codes),
        top_n=2,
        insufficient_universe_policy="error",
        weighting="equal_weight",
    )
    source = tmp_path / f"{name}-signal"
    source.mkdir()
    signal_path = source / "signals.parquet"
    signal_path.write_bytes(b"validated-signal")
    provenance = SignalArtifactProvenance(
        source,
        signal_path,
        "1.0",
        hashlib.sha256(signal_path.read_bytes()).hexdigest(),
    )
    return HoldingsArtifactStore().write(
        result,
        provenance,
        HoldingsArtifactConfig(tmp_path / name),
    )


def _rebalances(codes: tuple[str, str] = ("A.SZ", "B.SZ")):
    rows = []
    for index, (holdings_date, effective_date) in enumerate(
        (("2024-01-02", "2024-01-03"), ("2024-01-04", "2024-01-05"))
    ):
        for code in codes:
            rows.append(
                {
                    "holdings_trade_date": pd.Timestamp(holdings_date),
                    "effective_date": pd.Timestamp(effective_date),
                    "ts_code": code,
                    "pre_rebalance_weight": 0.0 if index == 0 else 0.5,
                    "target_weight": 0.5,
                    "weight_change": 0.5 if index == 0 else 0.0,
                    "pre_cash_weight": 1.0 if index == 0 else 0.0,
                    "target_cash_weight": 0.0,
                    "cash_weight_change": -1.0 if index == 0 else 0.0,
                    "turnover": 1.0 if index == 0 else 0.0,
                }
            )
    return RebalanceAccountingResult(
        pd.DataFrame(rows, columns=list(REBALANCE_OUTPUT_COLUMNS))
    )


def _portfolio() -> PortfolioDailyAccountingResult:
    frame = pd.DataFrame(
        [
            ["2024-01-03", 0.0, 0.001, -0.001, 1.0, 0.999, True, 1.0, 1.0],
            ["2024-01-04", 0.0, 0.0, 0.0, 1.0, 0.999, False, 0.0, 0.0],
            ["2024-01-05", 0.0, 0.0, 0.0, 1.0, 0.999, True, 0.0, 0.0],
        ],
        columns=list(DAILY_PORTFOLIO_COLUMNS),
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return PortfolioDailyAccountingResult(
        frame,
        start_date=pd.Timestamp("2024-01-03"),
        end_date=pd.Timestamp("2024-01-05"),
        rebalance_count=2,
        initial_nav=1.0,
        cost_bps=10.0,
    )


def _analytics(portfolio: PortfolioDailyAccountingResult | None = None):
    portfolio = _portfolio() if portfolio is None else portfolio
    benchmark = pd.DataFrame(
        [
            ["2024-01-03", "TEST.IDX", 0.5],
            ["2024-01-04", "TEST.IDX", 0.0],
            ["2024-01-05", "TEST.IDX", 0.0],
        ],
        columns=["trade_date", "benchmark_code", "return"],
    )
    return PerformanceAnalyticsEngine(
        BenchmarkConfig("TEST.IDX"), PerformanceConfig(0.0)
    ).run(portfolio=portfolio, benchmark_returns=benchmark)


def _config() -> ResearchBacktestPipelineConfig:
    return ResearchBacktestPipelineConfig.from_dict(
        {
            "enabled": True,
            "source": {"mode": "pipeline", "artifact_dir": None},
            "schedule": {"mode": "holdings_dates"},
            "return_alignment": {
                "effective_rule": "next_trading_day",
                "return_convention": "adjusted_close_to_close",
            },
            "portfolio": {
                "initial_nav": 1.0,
                "turnover_definition": "half_l1_pre_to_target",
            },
            "transaction_cost": {
                "cost_bps": 10.0,
                "rate_basis": "one_way_traded_notional",
            },
            "benchmark": {
                "benchmark_code": "TEST.IDX",
                "alignment_policy": "strict_common_calendar",
            },
            "performance": {
                "annualization_days": 252,
                "annual_risk_free_rate": 0.0,
            },
            "artifact_subdir": "research_backtest",
        }
    )


def _publish(tmp_path: Path, name: str = "backtest"):
    holdings = _holdings_artifact(tmp_path)
    portfolio = _portfolio()
    result = ResearchBacktestArtifactStore().publish(
        artifact_dir=tmp_path / name,
        rebalances=_rebalances(),
        portfolio=portfolio,
        analytics=_analytics(portfolio),
        config=_config(),
        holdings_artifact_dir=holdings.artifact_dir,
    )
    return holdings, result


def test_contract_constants_are_exact() -> None:
    assert RESEARCH_BACKTEST_ARTIFACT_SCHEMA_VERSION == "1.0"
    assert RESEARCH_BACKTEST_ARTIFACT_TYPE == "research_backtest"
    assert RESEARCH_BACKTEST_ARTIFACT_FILENAMES == (
        "rebalances.parquet",
        "daily_portfolio.parquet",
        "benchmark.parquet",
        "metrics.json",
        "config.json",
        "audit.json",
        "manifest.json",
    )


def test_publish_validate_and_roundtrip(tmp_path: Path) -> None:
    holdings, result = _publish(tmp_path)
    assert {item.name for item in result.artifact_dir.iterdir()} == set(
        RESEARCH_BACKTEST_ARTIFACT_FILENAMES
    )
    assert result.validation.is_valid
    assert ResearchBacktestArtifactStore().validate(result.artifact_dir).is_valid
    assert result.schema_version == "1.0"
    assert result.observation_count == 3
    assert result.rebalance_count == 2
    assert result.benchmark_code == "TEST.IDX"
    assert result.rebalances_path.name == "rebalances.parquet"
    assert result.daily_portfolio_path.name == "daily_portfolio.parquet"
    assert result.benchmark_path.name == "benchmark.parquet"
    assert tuple(pd.read_parquet(result.rebalances_path).columns) == REBALANCE_OUTPUT_COLUMNS
    assert tuple(pd.read_parquet(result.daily_portfolio_path).columns) == DAILY_PORTFOLIO_COLUMNS
    assert tuple(pd.read_parquet(result.benchmark_path).columns) == BENCHMARK_DAILY_COLUMNS
    assert tuple(_json(result.metrics_path)) == tuple(sorted(PERFORMANCE_METRIC_KEYS))
    assert _json(result.config_path) == _config().to_dict()
    assert holdings.validation.is_valid


def test_audit_contains_assumptions_sources_timing_and_lineage(tmp_path: Path) -> None:
    holdings, result = _publish(tmp_path)
    audit = _json(result.audit_path)
    assert audit["schema_version"] == "1.0"
    assert audit["artifact_type"] == "research_backtest"
    assert audit["start_date"] == "2024-01-03"
    assert audit["end_date"] == "2024-01-05"
    assert audit["observation_count"] == 3
    assert audit["rebalance_count"] == 2
    assert audit["cost_bps"] == 10.0
    assert audit["benchmark_code"] == "TEST.IDX"
    assert audit["security_return_source"] == "tushare.daily.pct_chg"
    assert audit["benchmark_return_source"] == "tushare.index_daily.pct_chg"
    assert audit["timing_convention"] == "post_close_rebalance_accounting"
    assert audit["first_effective_day_strategy_gross_return_zero"] is True
    assert audit["first_effective_day_benchmark_return_zero"] is True
    lineage = audit["upstream_holdings"]
    assert lineage["holdings_artifact_dir"] == str(holdings.artifact_dir)
    assert lineage["holdings_rows"] == 4
    assert lineage["holdings_date_count"] == 2
    assert lineage["holdings_manifest_sha256"] == hashlib.sha256(
        holdings.manifest_path.read_bytes()
    ).hexdigest()
    assert lineage["holdings_data_sha256"] == hashlib.sha256(
        holdings.holdings_path.read_bytes()
    ).hexdigest()


def test_manifest_records_six_payloads_and_excludes_self(tmp_path: Path) -> None:
    _, result = _publish(tmp_path)
    manifest = _json(result.manifest_path)
    assert manifest["artifact_schema_version"] == "1.0"
    assert manifest["artifact_type"] == "research_backtest"
    assert [item["relative_path"] for item in manifest["files"]] == list(
        RESEARCH_BACKTEST_ARTIFACT_FILENAMES[:-1]
    )
    assert "manifest.json" not in {
        item["relative_path"] for item in manifest["files"]
    }
    for record in manifest["files"]:
        path = result.artifact_dir / record["relative_path"]
        assert record["size_bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_inputs_are_not_mutated(tmp_path: Path) -> None:
    holdings = _holdings_artifact(tmp_path)
    rebalances = _rebalances()
    portfolio = _portfolio()
    analytics = _analytics(portfolio)
    before = (
        rebalances.rebalances,
        portfolio.daily_portfolio,
        analytics.benchmark_daily,
        analytics.metrics,
        _config().to_dict(),
    )
    ResearchBacktestArtifactStore().publish(
        artifact_dir=tmp_path / "backtest",
        rebalances=rebalances,
        portfolio=portfolio,
        analytics=analytics,
        config=_config(),
        holdings_artifact_dir=holdings.artifact_dir,
    )
    pdt.assert_frame_equal(rebalances.rebalances, before[0])
    pdt.assert_frame_equal(portfolio.daily_portfolio, before[1])
    pdt.assert_frame_equal(analytics.benchmark_daily, before[2])
    assert analytics.metrics == before[3]
    assert _config().to_dict() == before[4]


@pytest.mark.parametrize("target_kind", ["empty", "valid", "corrupt"])
def test_no_overwrite_preserves_existing_target(
    tmp_path: Path, target_kind: str
) -> None:
    holdings = _holdings_artifact(tmp_path)
    target = tmp_path / "backtest"
    if target_kind == "valid":
        published = ResearchBacktestArtifactStore().publish(
            artifact_dir=target,
            rebalances=_rebalances(),
            portfolio=_portfolio(),
            analytics=_analytics(),
            config=_config(),
            holdings_artifact_dir=holdings.artifact_dir,
        )
        before = {p.name: p.read_bytes() for p in published.artifact_dir.iterdir()}
    else:
        target.mkdir()
        (target / "owned.txt").write_text(target_kind, encoding="utf-8")
        before = {"owned.txt": (target / "owned.txt").read_bytes()}
    with pytest.raises(ResearchBacktestArtifactExistsError):
        ResearchBacktestArtifactStore().publish(
            artifact_dir=target,
            rebalances=_rebalances(),
            portfolio=_portfolio(),
            analytics=_analytics(),
            config=_config(),
            holdings_artifact_dir=holdings.artifact_dir,
        )
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before


def test_manifest_is_last_and_publish_is_one_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holdings = _holdings_artifact(tmp_path)
    writes: list[str] = []
    replaces: list[tuple[object, object]] = []
    original_write = artifact_module._write_json
    original_replace = artifact_module.os.replace

    def tracked_write(path: Path, value: object) -> None:
        writes.append(path.name)
        original_write(path, value)  # type: ignore[arg-type]

    def tracked_replace(source: object, target: object) -> None:
        replaces.append((source, target))
        original_replace(source, target)

    monkeypatch.setattr(artifact_module, "_write_json", tracked_write)
    monkeypatch.setattr(artifact_module.os, "replace", tracked_replace)
    ResearchBacktestArtifactStore().publish(
        artifact_dir=tmp_path / "backtest",
        rebalances=_rebalances(),
        portfolio=_portfolio(),
        analytics=_analytics(),
        config=_config(),
        holdings_artifact_dir=holdings.artifact_dir,
    )
    assert writes == ["metrics.json", "config.json", "audit.json", "manifest.json"]
    assert len(replaces) == 1
    assert ".tmp-backtest-" in str(replaces[0][0])


@pytest.mark.parametrize("failure", ["parquet", "json", "manifest", "rename"])
def test_failures_leave_no_final_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    holdings = _holdings_artifact(tmp_path)
    if failure == "parquet":
        monkeypatch.setattr(
            pd.DataFrame,
            "to_parquet",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("failed")),
        )
    elif failure == "json":
        monkeypatch.setattr(
            artifact_module,
            "_write_json",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("failed")),
        )
    elif failure == "manifest":
        monkeypatch.setattr(
            artifact_module,
            "_record",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("failed")),
        )
    else:
        monkeypatch.setattr(
            artifact_module.os,
            "replace",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("failed")),
        )
    with pytest.raises(ResearchBacktestArtifactWriteError):
        ResearchBacktestArtifactStore().publish(
            artifact_dir=tmp_path / "backtest",
            rebalances=_rebalances(),
            portfolio=_portfolio(),
            analytics=_analytics(),
            config=_config(),
            holdings_artifact_dir=holdings.artifact_dir,
        )
    assert not (tmp_path / "backtest").exists()
    assert not tuple(tmp_path.glob(".tmp-backtest-*"))


@pytest.mark.parametrize("filename", RESEARCH_BACKTEST_ARTIFACT_FILENAMES)
def test_validator_rejects_each_missing_file(tmp_path: Path, filename: str) -> None:
    _, result = _publish(tmp_path)
    (result.artifact_dir / filename).unlink()
    assert not ResearchBacktestArtifactStore().validate(result.artifact_dir).is_valid


def test_validator_rejects_extra_file(tmp_path: Path) -> None:
    _, result = _publish(tmp_path)
    (result.artifact_dir / "extra.txt").write_text("bad", encoding="utf-8")
    assert not ResearchBacktestArtifactStore().validate(result.artifact_dir).is_valid


@pytest.mark.parametrize(
    "filename",
    [
        "rebalances.parquet",
        "daily_portfolio.parquet",
        "benchmark.parquet",
        "metrics.json",
        "config.json",
        "audit.json",
    ],
)
def test_payload_byte_tamper_detects_size_and_hash(
    tmp_path: Path, filename: str
) -> None:
    _, result = _publish(tmp_path)
    path = result.artifact_dir / filename
    path.write_bytes(path.read_bytes() + b"tamper")
    codes = {
        issue.code
        for issue in ResearchBacktestArtifactStore().validate(result.artifact_dir).issues
    }
    assert {"file_size_mismatch", "checksum_mismatch"} <= codes


@pytest.mark.parametrize(
    "filename", ["metrics.json", "config.json", "audit.json", "manifest.json"]
)
def test_malformed_json_fails(tmp_path: Path, filename: str) -> None:
    _, result = _publish(tmp_path)
    (result.artifact_dir / filename).write_text("{bad", encoding="utf-8")
    assert not ResearchBacktestArtifactStore().validate(result.artifact_dir).is_valid


@pytest.mark.parametrize("field", ["artifact_schema_version", "artifact_type"])
def test_manifest_identity_tamper_fails(tmp_path: Path, field: str) -> None:
    _, result = _publish(tmp_path)
    manifest = _json(result.manifest_path)
    manifest[field] = "wrong"
    _write_json(result.manifest_path, manifest)
    assert not ResearchBacktestArtifactStore().validate(result.artifact_dir).is_valid


@pytest.mark.parametrize(
    ("field", "value", "issue_code"),
    [
        ("sha256", "0" * 64, "checksum_mismatch"),
        ("size_bytes", 1, "file_size_mismatch"),
    ],
)
def test_manifest_payload_metadata_tamper_fails(
    tmp_path: Path, field: str, value: object, issue_code: str
) -> None:
    _, result = _publish(tmp_path)
    manifest = _json(result.manifest_path)
    manifest["files"][0][field] = value  # type: ignore[index]
    _write_json(result.manifest_path, manifest)
    report = ResearchBacktestArtifactStore().validate(result.artifact_dir)
    assert not report.is_valid
    assert issue_code in {issue.code for issue in report.issues}


@pytest.mark.parametrize("mutation", ["missing", "extra", "nan"])
def test_metrics_contract_tamper_fails(tmp_path: Path, mutation: str) -> None:
    _, result = _publish(tmp_path)
    metrics = _json(result.metrics_path)
    if mutation == "missing":
        metrics.pop("tracking_error")
    elif mutation == "extra":
        metrics["extra"] = 1
    else:
        metrics["tracking_error"] = float("nan")
    _write_json(result.metrics_path, metrics)
    _refresh(result.artifact_dir, "metrics.json")
    assert not ResearchBacktestArtifactStore().validate(result.artifact_dir).is_valid


def test_disabled_config_tamper_fails(tmp_path: Path) -> None:
    _, result = _publish(tmp_path)
    config = _json(result.config_path)
    config["enabled"] = False
    _write_json(result.config_path, config)
    _refresh(result.artifact_dir, "config.json")
    assert not ResearchBacktestArtifactStore().validate(result.artifact_dir).is_valid


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        ("daily_portfolio.parquet", "calendar"),
        ("daily_portfolio.parquet", "count"),
        ("benchmark.parquet", "calendar"),
        ("benchmark.parquet", "code"),
        ("rebalances.parquet", "turnover"),
        ("rebalances.parquet", "schema"),
    ],
)
def test_semantic_parquet_tamper_fails(
    tmp_path: Path, filename: str, mutation: str
) -> None:
    _, result = _publish(tmp_path)
    path = result.artifact_dir / filename
    frame = pd.read_parquet(path)
    if mutation == "calendar":
        frame.loc[1, "trade_date"] = pd.Timestamp("2024-01-10")
    elif mutation == "count":
        frame = frame.iloc[:-1]
    elif mutation == "code":
        frame.loc[:, "benchmark_code"] = "WRONG.IDX"
    elif mutation == "turnover":
        frame.loc[0, "turnover"] = 0.5
        frame.loc[1, "turnover"] = 0.5
    else:
        frame = frame.iloc[:, ::-1]
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    _refresh(result.artifact_dir, filename)
    assert not ResearchBacktestArtifactStore().validate(result.artifact_dir).is_valid


def test_upstream_holdings_tamper_invalidates_backtest(tmp_path: Path) -> None:
    holdings, result = _publish(tmp_path)
    holdings.holdings_path.write_bytes(holdings.holdings_path.read_bytes() + b"bad")
    assert not ResearchBacktestArtifactStore().validate(result.artifact_dir).is_valid


def test_bare_or_nonexistent_holdings_lineage_is_rejected(tmp_path: Path) -> None:
    holdings = _holdings_artifact(tmp_path)
    for value in (holdings.holdings_path, tmp_path / "missing"):
        with pytest.raises(Exception):
            ResearchBacktestArtifactStore().publish(
                artifact_dir=tmp_path / f"bad-{value.name}",
                rebalances=_rebalances(),
                portfolio=_portfolio(),
                analytics=_analytics(),
                config=_config(),
                holdings_artifact_dir=value,
            )


def test_holdings_lineage_mismatch_rejects_publication(tmp_path: Path) -> None:
    holdings = _holdings_artifact(
        tmp_path, name="other-holdings", codes=("C.SZ", "D.SZ")
    )
    with pytest.raises(Exception, match="differ"):
        ResearchBacktestArtifactStore().publish(
            artifact_dir=tmp_path / "backtest",
            rebalances=_rebalances(),
            portfolio=_portfolio(),
            analytics=_analytics(),
            config=_config(),
            holdings_artifact_dir=holdings.artifact_dir,
        )
    assert not (tmp_path / "backtest").exists()


def test_pre_publish_validation_failure_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holdings = _holdings_artifact(tmp_path)
    store = ResearchBacktestArtifactStore()
    report = ResearchBacktestArtifactValidationResult(
        tmp_path / "placeholder", False, (), None
    )
    monkeypatch.setattr(store, "validate", lambda path: report)
    with pytest.raises(ResearchBacktestArtifactWriteError, match="pre-publish"):
        store.publish(
            artifact_dir=tmp_path / "backtest",
            rebalances=_rebalances(),
            portfolio=_portfolio(),
            analytics=_analytics(),
            config=_config(),
            holdings_artifact_dir=holdings.artifact_dir,
        )
    assert not (tmp_path / "backtest").exists()
    assert not tuple(tmp_path.glob(".tmp-backtest-*"))
