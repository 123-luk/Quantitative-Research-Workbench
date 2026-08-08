"""Tests for pure canonical Signal construction and deterministic ranking."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.signals import SIGNAL_OUTPUT_COLUMNS, SignalBuilder, SignalDataError


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-02"]
            ),
            "ts_code": ["B", "B", "A", "A"],
            "entry_trade_date": pd.to_datetime(
                ["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-03"]
            ),
            "exit_trade_date": pd.to_datetime(
                ["2024-01-04", "2024-01-03", "2024-01-03", "2024-01-04"]
            ),
            "target": [0.1, 0.2, 0.3, 0.4],
            "prediction": [2.0, 1.0, 1.0, 3.0],
            "fold_id": [1, 0, 0, 1],
            "feature_a": [10.0, 20.0, 30.0, 40.0],
        }
    )


def _build(
    frame: pd.DataFrame,
    direction: str = "descending",
    column: str = "prediction",
):
    return SignalBuilder().build(
        frame,
        prediction_column=column,
        signal_direction=direction,
    )


def test_happy_path_outputs_only_canonical_columns_and_maps_score() -> None:
    result = _build(_predictions())
    signals = result.signals
    assert tuple(signals.columns) == SIGNAL_OUTPUT_COLUMNS
    assert list(signals.columns) == ["trade_date", "ts_code", "score", "rank"]
    assert signals["score"].tolist() == [1.0, 1.0, 3.0, 2.0]
    assert not {
        "target",
        "fold_id",
        "entry_trade_date",
        "exit_trade_date",
        "feature_a",
    }.intersection(signals.columns)
    assert result.audit.input_rows == result.audit.output_rows == 4
    assert result.audit.trade_date_count == 2
    assert result.audit.prediction_column == "prediction"
    assert result.audit.signal_direction == "descending"


def test_descending_ascending_and_tie_break_are_deterministic() -> None:
    frame = _predictions()
    descending = _build(frame, "descending").signals
    ascending = _build(frame, "ascending").signals
    assert descending[["trade_date", "ts_code"]].values.tolist() == [
        [pd.Timestamp("2024-01-01"), "A"],
        [pd.Timestamp("2024-01-01"), "B"],
        [pd.Timestamp("2024-01-02"), "A"],
        [pd.Timestamp("2024-01-02"), "B"],
    ]
    assert ascending[["trade_date", "ts_code"]].values.tolist() == [
        [pd.Timestamp("2024-01-01"), "A"],
        [pd.Timestamp("2024-01-01"), "B"],
        [pd.Timestamp("2024-01-02"), "B"],
        [pd.Timestamp("2024-01-02"), "A"],
    ]
    assert descending.groupby("trade_date")["rank"].apply(list).tolist() == [
        [1, 2],
        [1, 2],
    ]
    left = descending.sort_values(["trade_date", "ts_code"])["score"].tolist()
    right = ascending.sort_values(["trade_date", "ts_code"])["score"].tolist()
    assert left == right


@pytest.mark.parametrize("order", [[3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]])
@pytest.mark.parametrize("direction", ["descending", "ascending"])
def test_shuffled_and_reversed_inputs_are_invariant(
    order: list[int], direction: str
) -> None:
    baseline = _build(_predictions(), direction).signals
    shuffled = _predictions().iloc[order].copy()
    pdt.assert_frame_equal(_build(shuffled, direction).signals, baseline)


def test_dates_are_independent_and_repeated_builds_match() -> None:
    frame = _predictions()
    baseline = _build(frame).signals
    changed = frame.copy()
    mask = changed["trade_date"].eq(pd.Timestamp("2024-01-02"))
    changed.loc[mask, "prediction"] += 100
    result = _build(changed).signals
    expected_date = baseline.loc[
        baseline["trade_date"].eq(pd.Timestamp("2024-01-01"))
    ].reset_index(drop=True)
    actual_date = result.loc[
        result["trade_date"].eq(pd.Timestamp("2024-01-01"))
    ].reset_index(drop=True)
    pdt.assert_frame_equal(expected_date, actual_date)
    pdt.assert_frame_equal(_build(frame).signals, _build(frame).signals)


def test_advanced_prediction_column_maps_without_coercion() -> None:
    frame = _predictions().assign(alpha_prediction=[0.4, 0.3, 0.2, 0.1])
    result = _build(frame, column=" alpha_prediction ")
    assert result.audit.prediction_column == "alpha_prediction"
    assert set(result.signals["score"]) == {0.1, 0.2, 0.3, 0.4}


@pytest.mark.parametrize("missing", ["trade_date", "ts_code", "prediction"])
def test_missing_required_columns_are_rejected(missing: str) -> None:
    with pytest.raises(SignalDataError, match="missing required"):
        _build(_predictions().drop(columns=[missing]))


def test_duplicate_keys_are_rejected_without_repair() -> None:
    frame = pd.concat([_predictions(), _predictions().iloc[[0]]], ignore_index=True)
    with pytest.raises(SignalDataError, match="unique"):
        _build(frame)


@pytest.mark.parametrize(
    "bad_date",
    [
        None,
        "",
        "not-a-date",
        "2024-01-01T12:00:00",
        "2024-01-01T00:00:00+08:00",
        20240101,
    ],
)
def test_bad_dates_are_rejected(bad_date: object) -> None:
    frame = _predictions().astype({"trade_date": "object"})
    frame.loc[0, "trade_date"] = bad_date
    with pytest.raises(SignalDataError, match="trade_date"):
        _build(frame)


def test_iso_dates_and_whitespace_codes_follow_project_canonicalization() -> None:
    frame = _predictions()
    frame["trade_date"] = frame["trade_date"].dt.strftime(" %Y-%m-%d ")
    frame["ts_code"] = frame["ts_code"].map(lambda value: f" {value} ")
    signals = _build(frame).signals
    assert str(signals["trade_date"].dtype) == "datetime64[ns]"
    assert str(signals["ts_code"].dtype) == "string"
    assert set(signals["ts_code"]) == {"A", "B"}


@pytest.mark.parametrize("bad_code", [None, "", "   ", 1])
def test_bad_codes_are_rejected(bad_code: object) -> None:
    frame = _predictions().astype({"ts_code": "object"})
    frame.loc[0, "ts_code"] = bad_code
    with pytest.raises(SignalDataError, match="ts_code"):
        _build(frame)


def test_code_normalization_cannot_hide_duplicate_keys() -> None:
    frame = _predictions()
    extra = frame.iloc[[0]].copy()
    extra["ts_code"] = " B "
    with pytest.raises(SignalDataError, match="unique"):
        _build(pd.concat([frame, extra], ignore_index=True))


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_nonfinite_numeric_scores_are_rejected(bad_score: float) -> None:
    frame = _predictions()
    frame.loc[0, "prediction"] = bad_score
    with pytest.raises(SignalDataError, match="finite"):
        _build(frame)


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(["1.0", "2.0", "3.0", "4.0"]),
        pd.Series([True, False, True, False]),
        pd.Series([1 + 2j, 2 + 0j, 3 + 0j, 4 + 0j]),
    ],
)
def test_nonnumeric_bool_or_complex_scores_are_rejected(series: pd.Series) -> None:
    frame = _predictions()
    frame["prediction"] = series
    with pytest.raises(SignalDataError, match="real numeric"):
        _build(frame)


@pytest.mark.parametrize(
    "column",
    [
        "target",
        "y_true",
        "fold_id",
        "trade_date",
        "ts_code",
        "entry_trade_date",
        "exit_trade_date",
        "rank",
        "selected",
        "target_weight",
    ],
)
def test_protected_columns_cannot_be_mapped_to_score(column: str) -> None:
    frame = _predictions()
    if column not in frame:
        frame[column] = 1.0
    with pytest.raises(SignalDataError, match="protected"):
        _build(frame, column=column)


def test_input_and_result_are_defensive() -> None:
    frame = _predictions()
    before = frame.copy(deep=True)
    result = _build(frame)
    pdt.assert_frame_equal(frame, before)
    original = result.signals
    exposed = result.signals
    exposed.loc[0, "score"] = 999.0
    pdt.assert_frame_equal(result.signals, original)


def test_builder_requires_explicit_semantics_and_has_no_selection_api() -> None:
    signature = inspect.signature(SignalBuilder.build)
    assert signature.parameters["prediction_column"].default is inspect.Parameter.empty
    assert signature.parameters["signal_direction"].default is inspect.Parameter.empty
    builder = SignalBuilder()
    assert not hasattr(builder, "select")
    assert not hasattr(builder, "build_holdings")