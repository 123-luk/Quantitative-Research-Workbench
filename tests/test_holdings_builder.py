"""Tests for deterministic Top-N equal-weight Holdings construction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.holdings import HOLDINGS_OUTPUT_COLUMNS
from src.holdings.builder import (
    WEIGHT_SUM_ABSOLUTE_TOLERANCE,
    HoldingsBuilder,
    HoldingsDataError,
)


def _signal(counts: tuple[int, ...] = (30, 30)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, count in enumerate(counts, start=1):
        trade_date = pd.Timestamp(f"2024-01-{date_number:02d}")
        for rank in range(1, count + 1):
            rows.append({
                "trade_date": trade_date,
                "ts_code": f"S{rank:03d}",
                "score": float(count - rank),
                "rank": rank,
            })
    frame = pd.DataFrame(rows)
    frame["trade_date"] = frame["trade_date"].astype("datetime64[ns]")
    frame["ts_code"] = frame["ts_code"].astype("string")
    frame["rank"] = frame["rank"].astype(np.int64)
    return frame


def _build(
    frame: pd.DataFrame,
    top_n: int,
    policy: str = "error",
    weighting: str = "equal_weight",
):
    return HoldingsBuilder().build(
        frame,
        top_n=top_n,
        insufficient_universe_policy=policy,
        weighting=weighting,
    )


@pytest.mark.parametrize(
    "top_n,expected_weight",
    [(1, 1.0), (10, 0.1), (20, 0.05)],
)
def test_configurable_top_n_and_equal_weight(
    top_n: int, expected_weight: float
) -> None:
    result = _build(_signal(), top_n)
    holdings = result.holdings
    assert tuple(holdings.columns) == HOLDINGS_OUTPUT_COLUMNS
    assert holdings.groupby("trade_date").size().tolist() == [top_n, top_n]
    assert holdings.groupby("trade_date")["rank"].apply(list).tolist() == [
        list(range(1, top_n + 1)), list(range(1, top_n + 1))
    ]
    assert np.allclose(holdings["target_weight"], expected_weight, rtol=0, atol=0)
    assert np.allclose(
        holdings.groupby("trade_date")["target_weight"].sum(),
        1.0,
        rtol=0,
        atol=WEIGHT_SUM_ABSOLUTE_TOLERANCE,
    )
    assert result.audit.requested_top_n == top_n
    assert result.audit.weighting == "equal_weight"


def test_changing_n_preserves_signal_score_rank_and_prefix() -> None:
    signal = _signal((15, 15))
    before = signal.copy(deep=True)
    five = _build(signal, 5).holdings
    ten = _build(signal, 10).holdings
    pdt.assert_frame_equal(signal, before)
    for trade_date in signal["trade_date"].unique():
        source = signal.loc[signal["trade_date"] == trade_date]
        left = five.loc[five["trade_date"] == trade_date]
        right = ten.loc[ten["trade_date"] == trade_date]
        assert left["ts_code"].tolist() == right.head(5)["ts_code"].tolist()
        pdt.assert_series_equal(left["score"].reset_index(drop=True), source.head(5)["score"].reset_index(drop=True), check_names=False)
        pdt.assert_series_equal(left["rank"].reset_index(drop=True), source.head(5)["rank"].reset_index(drop=True), check_names=False)
    assert len(five) == 10
    assert len(ten) == 20
    assert np.allclose(five["target_weight"], 0.2)
    assert np.allclose(ten["target_weight"], 0.1)


def test_insufficient_error_is_explicit_and_has_no_partial_result() -> None:
    with pytest.raises(
        HoldingsDataError,
        match=r"2024-01-02.*requested top_n=10.*available_count=7",
    ):
        _build(_signal((12, 7)), 10, "error")


def test_allow_partial_uses_actual_count_and_audits_partial_date() -> None:
    result = _build(_signal((12, 7)), 10, "allow_partial")
    holdings = result.holdings
    assert holdings.groupby("trade_date").size().tolist() == [10, 7]
    first = holdings.loc[holdings["trade_date"] == pd.Timestamp("2024-01-01")]
    second = holdings.loc[holdings["trade_date"] == pd.Timestamp("2024-01-02")]
    assert np.allclose(first["target_weight"], 0.1, rtol=0, atol=0)
    assert np.allclose(second["target_weight"], 1 / 7, rtol=0, atol=1e-15)
    assert result.audit.requested_top_n == 10
    assert result.audit.partial_dates == (pd.Timestamp("2024-01-02"),)
    assert [item.as_dict() for item in result.audit.per_date_counts] == [
        {
            "trade_date": "2024-01-01",
            "available_count": 12,
            "selected_count": 10,
            "partial": False,
        },
        {
            "trade_date": "2024-01-02",
            "available_count": 7,
            "selected_count": 7,
            "partial": True,
        },
    ]
    assert result.audit.warnings == ("partial universe on 2024-01-02",)


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "10", np.int64(10)])
def test_top_n_runtime_contract_matches_pipeline_config(value: object) -> None:
    with pytest.raises(HoldingsDataError):
        _build(_signal((3,)), value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["partial", "ERRORS", "", None, 1])
def test_unsupported_policy_is_rejected(value: object) -> None:
    with pytest.raises(HoldingsDataError):
        _build(_signal((3,)), 1, value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value", ["score_weight", "rank_weight", "risk_parity", "", None, 1]
)
def test_only_equal_weight_is_supported(value: object) -> None:
    with pytest.raises(HoldingsDataError):
        _build(_signal((3,)), 1, weighting=value)  # type: ignore[arg-type]


def test_policy_and_weighting_are_normalized_like_config() -> None:
    result = _build(_signal((2,)), 1, " ERROR ", " EQUAL_WEIGHT ")
    assert result.audit.insufficient_universe_policy == "error"
    assert result.audit.weighting == "equal_weight"


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate_columns"])
def test_exact_signal_columns_are_required(mutation: str) -> None:
    frame = _signal((3,))
    if mutation == "missing":
        frame = frame.drop(columns="score")
    elif mutation == "extra":
        frame["selected"] = True
    else:
        frame.columns = ["trade_date", "ts_code", "score", "score"]
    with pytest.raises(HoldingsDataError, match="exactly"):
        _build(frame, 1)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_key", "bad_date", "timezone_date", "empty_code",
        "whitespace_code", "nonnumeric_score", "nan_score", "inf_score",
        "float_rank", "zero_rank", "duplicate_rank", "skipped_rank",
        "noncanonical_order",
    ],
)
def test_invalid_signal_semantics_are_rejected(mutation: str) -> None:
    frame = _signal((4,))
    if mutation == "duplicate_key":
        frame.loc[1, ["trade_date", "ts_code"]] = frame.loc[0, ["trade_date", "ts_code"]]
    elif mutation == "bad_date":
        frame["trade_date"] = ["bad"] * len(frame)
    elif mutation == "timezone_date":
        frame["trade_date"] = frame["trade_date"].dt.tz_localize("UTC")
    elif mutation == "empty_code":
        frame.loc[0, "ts_code"] = ""
    elif mutation == "whitespace_code":
        frame.loc[0, "ts_code"] = " S001 "
    elif mutation == "nonnumeric_score":
        frame["score"] = "bad"
    elif mutation == "nan_score":
        frame.loc[0, "score"] = np.nan
    elif mutation == "inf_score":
        frame.loc[0, "score"] = np.inf
    elif mutation == "float_rank":
        frame["rank"] = frame["rank"].astype(float)
    elif mutation == "zero_rank":
        frame.loc[0, "rank"] = 0
    elif mutation == "duplicate_rank":
        frame.loc[1, "rank"] = frame.loc[0, "rank"]
    elif mutation == "skipped_rank":
        frame.loc[1, "rank"] = 9
    elif mutation == "noncanonical_order":
        frame = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(HoldingsDataError):
        _build(frame, 1)


def test_empty_or_non_dataframe_is_rejected() -> None:
    with pytest.raises(HoldingsDataError):
        _build(pd.DataFrame(columns=["trade_date", "ts_code", "score", "rank"]), 1)
    with pytest.raises(HoldingsDataError):
        _build([] , 1)  # type: ignore[arg-type]


def test_output_is_deterministic_defensive_and_has_no_leakage() -> None:
    signal = _signal((8, 8))
    before = signal.copy(deep=True)
    result = _build(signal, 5)
    first = result.holdings
    second = _build(signal, 5).holdings
    pdt.assert_frame_equal(first, second)
    changed = result.holdings
    changed.loc[0, "target_weight"] = -1.0
    pdt.assert_frame_equal(result.holdings, first)
    pdt.assert_frame_equal(signal, before)
    assert tuple(first.columns) == HOLDINGS_OUTPUT_COLUMNS
    assert not {
        "selected", "top_n", "weighting", "insufficient_universe_policy",
        "target", "y_true", "fold_id", "entry_trade_date", "exit_trade_date",
    } & set(first.columns)
    assert bool((first["target_weight"] > 0).all())


def test_build_audit_is_frozen_and_json_safe() -> None:
    audit = _build(_signal((3,)), 2).audit
    assert audit.as_dict()["requested_top_n"] == 2
    with pytest.raises(FrozenInstanceError):
        audit.requested_top_n = 3  # type: ignore[misc]


def test_builder_has_no_business_defaults_or_selection_flags() -> None:
    signature = inspect.signature(HoldingsBuilder.build)
    for name in ("top_n", "insufficient_universe_policy", "weighting"):
        assert signature.parameters[name].default is inspect.Parameter.empty
    builder = HoldingsBuilder()
    assert not hasattr(builder, "rank")
    assert not hasattr(builder, "optimize")
