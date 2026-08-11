from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from src.research_data import AdjustedPriceDataUnavailable, AdjustedPriceRequest, AdjustedPriceService, CanonicalMarketSlice


def daily(rows: list[tuple[str, str, float]], *, future_scale: float = 1.0) -> pd.DataFrame:
    result = []
    for code, day, close in rows:
        scale = future_scale if day > "2024-01-03" else 1.0
        result.append({"ts_code": code, "trade_date": day, "open": close - 1, "high": close + 1, "low": close - 2, "close": close * scale, "vol": 100.0, "amount": 200.0})
    return pd.DataFrame(result)


def factors(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"ts_code": code, "trade_date": day, "adj_factor": value} for code, day, value in rows],
        columns=("ts_code", "trade_date", "adj_factor"),
    )


@dataclass
class FakeSource:
    daily_frame: pd.DataFrame
    factor_frame: pd.DataFrame

    def daily(self, dates: tuple[str, ...]) -> CanonicalMarketSlice:
        return CanonicalMarketSlice(self.daily_frame, "daily", "1.0", "daily:fixture")

    def adj_factor(self, dates: tuple[str, ...]) -> CanonicalMarketSlice:
        return CanonicalMarketSlice(self.factor_frame, "adj_factor", "1.0", "factor:fixture")


def request(*dates: str) -> AdjustedPriceRequest:
    return AdjustedPriceRequest(("600001.SH", "000001.SZ"), dates)


def test_exact_raw_times_factor_all_ohlc_and_raw_volume_amount() -> None:
    source = FakeSource(daily([("600001.SH", "2024-01-02", 10.0), ("000001.SZ", "2024-01-02", 20.0)]), factors([("600001.SH", "2024-01-02", 2.0), ("000001.SZ", "2024-01-02", 0.5)]))
    result = AdjustedPriceService(source).compute(request("2024-01-02"))
    frame = result.frame
    assert tuple(frame["ts_code"]) == ("000001.SZ", "600001.SH")
    first = frame.iloc[0]
    assert (first["adj_open"], first["adj_high"], first["adj_low"], first["adj_close"]) == (9.5, 10.5, 9.0, 10.0)
    assert (first["vol"], first["amount"]) == (100.0, 200.0)
    assert result.diagnostics["raw_volume_amount_adjusted"] is False
    changed = result.frame
    changed.loc[0, "adj_close"] = -1
    assert result.frame.loc[0, "adj_close"] == 10.0
    with pytest.raises(TypeError):
        result.diagnostics["x"] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        result.source_identity = "changed"  # type: ignore[misc]


def test_missing_and_duplicate_exact_keys_fail_closed() -> None:
    one_daily = daily([("600001.SH", "2024-01-02", 10.0)])
    service = AdjustedPriceService(FakeSource(one_daily, factors([])))
    with pytest.raises(AdjustedPriceDataUnavailable, match="lack exact"):
        service.compute(request("2024-01-02"))
    duplicate_daily = pd.concat([one_daily, one_daily], ignore_index=True)
    with pytest.raises(AdjustedPriceDataUnavailable, match="daily contains duplicate"):
        AdjustedPriceService(FakeSource(duplicate_daily, factors([("600001.SH", "2024-01-02", 1.0)]))).compute(request("2024-01-02"))
    duplicate_factor = pd.concat([factors([("600001.SH", "2024-01-02", 1.0)]), factors([("600001.SH", "2024-01-02", 1.0)])], ignore_index=True)
    with pytest.raises(AdjustedPriceDataUnavailable, match="adj_factor contains duplicate"):
        AdjustedPriceService(FakeSource(one_daily, duplicate_factor)).compute(request("2024-01-02"))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_adjustment_factor_fails_closed(bad: object) -> None:
    source = FakeSource(daily([("600001.SH", "2024-01-02", 10.0)]), factors([("600001.SH", "2024-01-02", bad)]))  # type: ignore[list-item]
    with pytest.raises(AdjustedPriceDataUnavailable):
        AdjustedPriceService(source).compute(request("2024-01-02"))


def test_future_rows_end_extension_and_wild_perturbation_do_not_reanchor_history() -> None:
    base_daily = daily([("600001.SH", "2024-01-02", 10.0), ("600001.SH", "2024-01-03", 11.0)])
    base_factor = factors([("600001.SH", "2024-01-02", 2.0), ("600001.SH", "2024-01-03", 2.1)])
    through_t = AdjustedPriceService(FakeSource(base_daily, base_factor)).compute(request("2024-01-02", "2024-01-03")).frame
    extended_daily = pd.concat([base_daily, daily([("600001.SH", "2024-01-04", 999999.0), ("600001.SH", "2024-01-05", 0.0001)])], ignore_index=True)
    extended_factor = pd.concat([base_factor, factors([("600001.SH", "2024-01-04", 9999.0), ("600001.SH", "2024-01-05", 0.00001)])], ignore_index=True)
    same_request = AdjustedPriceService(FakeSource(extended_daily, extended_factor)).compute(request("2024-01-02", "2024-01-03")).frame
    longer = AdjustedPriceService(FakeSource(extended_daily, extended_factor)).compute(request("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")).frame
    pd.testing.assert_frame_equal(through_t, same_request, check_exact=True)
    pd.testing.assert_frame_equal(through_t, longer.loc[longer["trade_date"].le("2024-01-03")].reset_index(drop=True), check_exact=True)


def test_orphan_adjustment_does_not_manufacture_daily_observation() -> None:
    source = FakeSource(daily([("600001.SH", "2024-01-02", 10.0)]), factors([("600001.SH", "2024-01-02", 1.0), ("000001.SZ", "2024-01-02", 7.0)]))
    result = AdjustedPriceService(source).compute(request("2024-01-02"))
    assert tuple(result.frame["ts_code"]) == ("600001.SH",)
    assert result.diagnostics["orphan_adjustment_rows"] == 1


def test_requirements_are_exactly_daily_and_adj_factor() -> None:
    requirements = AdjustedPriceService.requirements(start_date="2024-01-02", end_date="2024-01-31", scope="CN_A")
    assert tuple(item.dataset_id for item in requirements) == ("adj_factor", "daily")
    assert {dict(item.scope)["scope"] for item in requirements} == {"CN_A"}
    assert not ({"daily_basic", "trade_cal", "suspend_d"} & {item.dataset_id for item in requirements})


def test_backtest_return_source_remains_pct_chg_not_price_change() -> None:
    from src.research_backtest.returns import build_benchmark_daily_returns, build_security_daily_returns
    security = pd.DataFrame([{"trade_date": "2024-01-02", "ts_code": "600001.SH", "close": 1000.0, "pct_chg": 2.5}])
    benchmark = pd.DataFrame([{"trade_date": "2024-01-02", "ts_code": "000300.SH", "close": 0.01, "pct_chg": -3.0}])
    assert build_security_daily_returns(security).iloc[0]["return"] == 0.025
    assert build_benchmark_daily_returns(benchmark, benchmark_code="000300.SH").iloc[0]["return"] == -0.03
