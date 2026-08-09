"""Tests for the read-only V6 Streamlit Research Backtest helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pandas.testing as pdt
import pytest

import app.services.research_backtest_ui_service as service
def _result(artifact_dir: Path) -> dict[str, object]:
    return {
        "enabled": True,
        "artifact_dir": str(artifact_dir),
        "schema_version": "1.0",
        "observation_count": 2,
        "rebalance_count": 1,
        "start_date": "2024-01-02",
        "end_date": "2024-01-03",
        "benchmark_code": "000300.SH",
        "metrics": {
            "net_total_return": 0.25,
            "net_sharpe_ratio": None,
            "tracking_error": 0.1,
        },
    }


class _Store:
    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid
        self.seen: list[Path] = []

    def validate(self, artifact_dir: object) -> SimpleNamespace:
        self.seen.append(Path(artifact_dir))
        issues = () if self.valid else (SimpleNamespace(code="checksum_mismatch"),)
        return SimpleNamespace(is_valid=self.valid, issues=issues)


def test_exact_validated_artifact_nav_and_pipeline_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = tmp_path / "exact-artifact"
    store = _Store()
    result = _result(exact)
    before = deepcopy(result)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    daily = pd.DataFrame(
        {"trade_date": dates, "gross_nav": [1.0, 1.3], "net_nav": [1.0, 1.25]}
    )
    benchmark = pd.DataFrame(
        {"trade_date": dates, "benchmark_nav": [1.0, 1.1]}
    )
    reads: list[Path] = []

    def fake_read(path: object, *, engine: str) -> pd.DataFrame:
        resolved = Path(path)
        reads.append(resolved)
        assert engine == "pyarrow"
        return daily.copy() if resolved.name == "daily_portfolio.parquet" else benchmark.copy()

    monkeypatch.setattr(service.pd, "read_parquet", fake_read)
    payload = service.load_research_backtest_dashboard(
        result, store=store  # type: ignore[arg-type]
    )
    assert store.seen == [exact]
    assert reads == [exact / "daily_portfolio.parquet", exact / "benchmark.parquet"]
    assert payload.artifact_dir == exact
    assert payload.metrics == result["metrics"]
    assert payload.metrics is not result["metrics"]
    expected = pd.DataFrame(
        {
            "trade_date": dates,
            "gross_nav": [1.0, 1.3],
            "net_nav": [1.0, 1.25],
            "benchmark_nav": [1.0, 1.1],
        }
    )
    pdt.assert_frame_equal(payload.nav, expected)
    assert result == before


def test_tampered_artifact_fails_before_any_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        service.pd,
        "read_parquet",
        lambda *args, **kwargs: pytest.fail("payload must not be read"),
    )
    with pytest.raises(service.ResearchBacktestDashboardError, match="checksum_mismatch"):
        service.load_research_backtest_dashboard(
            _result(tmp_path / "tampered"), store=_Store(valid=False)  # type: ignore[arg-type]
        )


def test_mismatched_dates_fail_without_fill_intersection_or_interpolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daily = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2024-01-02"]), "gross_nav": [1.0], "net_nav": [1.0]}
    )
    benchmark = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2024-01-03"]), "benchmark_nav": [1.0]}
    )
    monkeypatch.setattr(
        service.pd,
        "read_parquet",
        lambda path, **kwargs: daily if Path(path).name.startswith("daily") else benchmark,
    )
    with pytest.raises(service.ResearchBacktestDashboardError, match="match exactly"):
        service.load_research_backtest_dashboard(
            _result(tmp_path / "exact"), store=_Store()  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("value", "percent", "expected"),
    [
        (None, False, "N/A"),
        (float("nan"), False, "N/A"),
        (float("inf"), False, "N/A"),
        (0.1234, True, "12.34%"),
        (1.234, False, "1.23"),
    ],
)
def test_metric_formatting_is_presentation_only(
    value: object, percent: bool, expected: str
) -> None:
    assert service.format_research_backtest_metric(value, percent=percent) == expected


def test_service_source_has_no_discovery_or_business_recalculation() -> None:
    source = Path(service.__file__).read_text(encoding="utf-8")
    for forbidden in (
        ".glob(",
        ".rglob(",
        "mtime",
        "latest",
        "cumprod(",
        "pct_change(",
        ".ffill(",
        "interpolate(",
    ):
        assert forbidden not in source.lower()
