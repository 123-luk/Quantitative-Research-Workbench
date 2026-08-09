from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.holdings import (
    HoldingsArtifactConfig,
    HoldingsArtifactStore,
    HoldingsBuilder,
    SignalArtifactProvenance,
)
from src.pipeline import BacktestSourceConfig, HoldingsPipelineResult
from src.research_backtest import (
    ResearchBacktestHoldingsSourceAdapter,
    ResearchBacktestHoldingsSourceError,
)


def _native_holdings(tmp_path: Path, name: str = "holdings"):
    rows = []
    for trade_date in ("2024-01-02", "2024-01-04"):
        for rank, code in enumerate(("A.SZ", "B.SZ"), start=1):
            rows.append(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "ts_code": code,
                    "score": float(3 - rank),
                    "rank": rank,
                }
            )
    signals = pd.DataFrame(rows)
    signals["trade_date"] = signals["trade_date"].astype("datetime64[ns]")
    signals["ts_code"] = signals["ts_code"].astype("string")
    signals["rank"] = signals["rank"].astype(np.int64)
    built = HoldingsBuilder().build(
        signals,
        top_n=2,
        insufficient_universe_policy="error",
        weighting="equal_weight",
    )
    source = tmp_path / f"{name}-signal"
    source.mkdir()
    signal_path = source / "signals.parquet"
    signal_path.write_bytes(b"source")
    provenance = SignalArtifactProvenance(
        source,
        signal_path,
        "1.0",
        hashlib.sha256(signal_path.read_bytes()).hexdigest(),
    )
    written = HoldingsArtifactStore().write(
        built,
        provenance,
        HoldingsArtifactConfig(tmp_path / name),
    )
    result = HoldingsPipelineResult(
        enabled=True,
        source_signal_artifact_dir=source,
        artifact_dir=written.artifact_dir,
        holdings_path=written.holdings_path,
        manifest_path=written.manifest_path,
        rows=built.audit.output_rows,
        trade_date_count=built.audit.trade_date_count,
        requested_top_n=2,
        insufficient_universe_policy="error",
        weighting="equal_weight",
        schema_version=written.schema_version,
    )
    return written, result


def test_pipeline_source_uses_exact_current_result(tmp_path: Path) -> None:
    written, pipeline_result = _native_holdings(tmp_path)
    loaded = ResearchBacktestHoldingsSourceAdapter().load(
        BacktestSourceConfig(mode="pipeline"),
        holdings_result=pipeline_result,
    )
    assert loaded.artifact_dir == written.artifact_dir
    assert loaded.holdings_path == written.holdings_path
    assert loaded.schema_version == "1.0"
    assert loaded.rows == 4 and loaded.date_count == 2
    pdt.assert_frame_equal(loaded.holdings, pd.read_parquet(written.holdings_path))


def test_files_source_uses_explicit_native_artifact(tmp_path: Path) -> None:
    written, _ = _native_holdings(tmp_path)
    loaded = ResearchBacktestHoldingsSourceAdapter().load(
        BacktestSourceConfig(mode="files", artifact_dir=written.artifact_dir)
    )
    assert loaded.artifact_dir == written.artifact_dir


def test_source_result_is_defensive(tmp_path: Path) -> None:
    written, _ = _native_holdings(tmp_path)
    loaded = ResearchBacktestHoldingsSourceAdapter().load(
        BacktestSourceConfig(mode="files", artifact_dir=written.artifact_dir)
    )
    first = loaded.holdings
    first.loc[:, "target_weight"] = 0.0
    assert not loaded.holdings["target_weight"].eq(0.0).all()


def test_pipeline_source_requires_current_result() -> None:
    with pytest.raises(ResearchBacktestHoldingsSourceError, match="current"):
        ResearchBacktestHoldingsSourceAdapter().load(
            BacktestSourceConfig(mode="pipeline")
        )


def test_files_source_rejects_pipeline_result(tmp_path: Path) -> None:
    written, pipeline_result = _native_holdings(tmp_path)
    with pytest.raises(ResearchBacktestHoldingsSourceError, match="does not accept"):
        ResearchBacktestHoldingsSourceAdapter().load(
            BacktestSourceConfig(mode="files", artifact_dir=written.artifact_dir),
            holdings_result=pipeline_result,
        )


@pytest.mark.parametrize("kind", ["bare", "missing", "tampered"])
def test_invalid_native_source_fails_closed(tmp_path: Path, kind: str) -> None:
    written, _ = _native_holdings(tmp_path)
    if kind == "bare":
        path = written.holdings_path
    elif kind == "missing":
        path = tmp_path / "missing"
    else:
        written.holdings_path.write_bytes(
            written.holdings_path.read_bytes() + b"tamper"
        )
        path = written.artifact_dir
    with pytest.raises(ResearchBacktestHoldingsSourceError, match="validation"):
        ResearchBacktestHoldingsSourceAdapter().load(
            BacktestSourceConfig(mode="files", artifact_dir=path)
        )


def test_pipeline_result_payload_identity_must_match(tmp_path: Path) -> None:
    written, pipeline_result = _native_holdings(tmp_path)
    other = tmp_path / "other.parquet"
    other.write_bytes(b"other")
    object.__setattr__(pipeline_result, "holdings_path", other)
    with pytest.raises(ResearchBacktestHoldingsSourceError, match="payload"):
        ResearchBacktestHoldingsSourceAdapter().load(
            BacktestSourceConfig(mode="pipeline"),
            holdings_result=pipeline_result,
        )
    assert written.holdings_path.is_file()


def test_adapter_exposes_no_discovery_methods() -> None:
    public = {
        name
        for name in dir(ResearchBacktestHoldingsSourceAdapter)
        if not name.startswith("_")
    }
    assert public == {"load"}
