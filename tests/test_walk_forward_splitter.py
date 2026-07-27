"""Tests for strict date-level walk-forward splitting and label availability."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.ml import MLDataset, MLDatasetBuilder
from src.ml.splitting import (
    WalkForwardConfig,
    WalkForwardConfigError,
    WalkForwardDataError,
    WalkForwardError,
    WalkForwardInsufficientHistoryError,
    WalkForwardIntegrityError,
    WalkForwardPlan,
    WalkForwardSplit,
    WalkForwardSplitter,
    _CandidateHistoryError,
)


def _dataset(
    periods: int = 16,
    *,
    stocks_per_date: int | tuple[int, ...] = 2,
    exit_lags: int | tuple[int, ...] = 2,
) -> MLDataset:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    counts = (
        (stocks_per_date,) * periods
        if isinstance(stocks_per_date, int)
        else stocks_per_date
    )
    lags = (exit_lags,) * periods if isinstance(exit_lags, int) else exit_lags
    assert len(counts) == periods
    assert len(lags) == periods
    factor_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(dates):
        for stock_index in range(counts[date_index]):
            ts_code = f"S{stock_index:02d}"
            factor_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": ts_code,
                    "factor": float(date_index * 10 + stock_index),
                }
            )
            label_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": ts_code,
                    "entry_trade_date": trade_date + pd.Timedelta(days=1),
                    "exit_trade_date": trade_date
                    + pd.Timedelta(days=lags[date_index]),
                    "forward_return": float(date_index + 1) / 100.0,
                }
            )
    return MLDatasetBuilder().build(
        pd.DataFrame(factor_rows),
        pd.DataFrame(label_rows),
        ["factor"],
    )


def _config(**overrides: object) -> WalkForwardConfig:
    values: dict[str, object] = {
        "train_window_periods": 2,
        "validation_periods": 2,
        "window_type": "rolling",
        "retrain_frequency": 3,
        "embargo_periods": 1,
    }
    values.update(overrides)
    return WalkForwardConfig(**values)  # type: ignore[arg-type]


def _plan(dataset: MLDataset | None = None, **config: object) -> WalkForwardPlan:
    return WalkForwardSplitter(_config(**config)).build(dataset or _dataset())


def _forged_dataset(metadata: pd.DataFrame) -> MLDataset:
    forged = object.__new__(MLDataset)
    forged._metadata = metadata.copy(deep=True)  # type: ignore[attr-defined]
    forged._labels = pd.Series(np.zeros(len(metadata)))  # type: ignore[attr-defined]
    return forged


def test_config_normalizes_window_type_and_is_frozen_json_safe() -> None:
    rolling = _config(window_type=" Rolling ")
    expanding = _config(window_type=" ExPANDing ")
    assert rolling.window_type == "rolling"
    assert expanding.window_type == "expanding"
    json.dumps(rolling.as_dict())
    with pytest.raises(FrozenInstanceError):
        rolling.window_type = "expanding"  # type: ignore[misc]


@pytest.mark.parametrize("window_type", ["", "fixed", None, 1])
def test_invalid_window_type_raises(window_type: object) -> None:
    with pytest.raises(WalkForwardConfigError, match="window_type"):
        _config(window_type=window_type)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_window_periods", 0),
        ("validation_periods", 0),
        ("retrain_frequency", 0),
        ("embargo_periods", -1),
        ("train_window_periods", True),
        ("validation_periods", False),
        ("retrain_frequency", 1.5),
        ("embargo_periods", "1"),
    ],
)
def test_period_fields_validate_integer_ranges(field: str, value: object) -> None:
    with pytest.raises(WalkForwardConfigError, match=field):
        _config(**{field: value})


def test_basic_rolling_windows_and_audit_fields() -> None:
    plan = _plan()
    first = plan.splits[0]
    dates = plan.all_score_dates

    assert plan.n_splits == 3
    assert first.retrain_id == 1
    assert first.train_dates == dates[0:2]
    assert first.train_validation_purged_dates == dates[2:4]
    assert first.validation_dates == dates[4:6]
    assert first.embargo_dates == dates[6:7]
    assert first.label_unavailable_dates == dates[7:9]
    assert first.prediction_dates == dates[9:12]
    assert first.max_train_exit_date == dates[3]
    assert first.max_validation_exit_date == dates[7]
    assert first.max_train_exit_date < first.validation_start_date
    assert first.max_validation_exit_date < first.prediction_start_date


def test_rolling_fixed_windows_and_earlier_history_is_not_purge() -> None:
    plan = _plan()
    assert all(len(split.train_dates) == 2 for split in plan.splits)
    second = plan.splits[1]
    eligible_but_older = set(plan.all_score_dates[:5]) - set(second.train_dates)
    assert eligible_but_older
    assert not eligible_but_older & set(second.train_validation_purged_dates)


def test_prediction_blocks_are_gap_free_nonoverlapping_and_keep_short_final() -> None:
    plan = _plan()
    combined = tuple(
        date for split in plan.splits for date in split.prediction_dates
    )
    assert [split.retrain_id for split in plan.splits] == [1, 2, 3]
    assert [len(split.prediction_dates) for split in plan.splits] == [3, 3, 1]
    assert len(set(combined)) == len(combined)
    assert combined == plan.all_score_dates[plan.n_skipped_initial_dates :]
    assert plan.first_prediction_date == combined[0]
    assert plan.last_prediction_date == combined[-1]


def test_expanding_uses_all_eligible_history_and_never_shrinks() -> None:
    plan = _plan(window_type="expanding")
    lengths = [len(split.train_dates) for split in plan.splits]
    assert lengths == sorted(lengths)
    assert lengths[0] == 2
    assert lengths[1] > lengths[0]
    assert plan.splits[0].train_dates[0] == plan.all_score_dates[0]
    assert plan.splits[1].train_dates[0] == plan.all_score_dates[0]
    for split in plan.splits:
        expected = tuple(
            date
            for date in plan.all_score_dates
            if date < split.validation_start_date
            and date + pd.Timedelta(days=2) < split.validation_start_date
        )
        assert split.train_dates == expected


def test_exit_equal_prediction_cutoff_is_unavailable_for_entire_date() -> None:
    plan = _plan()
    first = plan.splits[0]
    equal_cutoff_date = first.prediction_start_date - pd.Timedelta(days=2)
    assert equal_cutoff_date in first.label_unavailable_dates
    assert equal_cutoff_date not in first.validation_dates
    assert equal_cutoff_date not in first.train_dates


def test_partial_stock_maturity_uses_date_level_max_exit() -> None:
    dataset = _dataset()
    metadata = dataset.metadata
    target_date = pd.Timestamp("2024-01-10")
    target_rows = metadata["trade_date"].eq(target_date)
    metadata.loc[target_rows, "exit_trade_date"] = [
        pd.Timestamp("2024-01-12"),
        pd.Timestamp("2024-01-13"),
    ]
    guarded = _forged_dataset(metadata)
    plan = WalkForwardSplitter(_config()).build(guarded)
    second = plan.splits[1]

    assert second.prediction_start_date == pd.Timestamp("2024-01-13")
    assert target_date in second.label_unavailable_dates
    target_indices = tuple(metadata.index[target_rows])
    assert not set(target_indices) & set(second.train_indices)
    assert not set(target_indices) & set(second.validation_indices)


def test_unavailable_dates_are_sorted_unique_and_historical() -> None:
    for split in _plan().splits:
        assert split.label_unavailable_dates == tuple(
            sorted(set(split.label_unavailable_dates))
        )
        assert all(
            date < split.prediction_start_date
            for date in split.label_unavailable_dates
        )


def test_validation_is_latest_fixed_window_after_embargo() -> None:
    plan = _plan()
    first = plan.splits[0]
    mature = tuple(
        date
        for date in plan.all_score_dates
        if date < first.prediction_start_date
        and date + pd.Timedelta(days=2) < first.prediction_start_date
    )
    assert first.embargo_dates == mature[-1:]
    assert first.validation_dates == mature[:-1][-2:]


def test_train_validation_equal_cutoff_is_purged_but_prediction_mature() -> None:
    first = _plan().splits[0]
    equal_validation_date = first.validation_start_date - pd.Timedelta(days=2)
    assert equal_validation_date in first.train_validation_purged_dates
    assert equal_validation_date not in first.train_dates
    assert equal_validation_date not in first.validation_dates
    assert equal_validation_date not in first.label_unavailable_dates


@pytest.mark.parametrize("embargo_periods", [0, 1, 2])
def test_embargo_removes_only_recent_mature_dates(embargo_periods: int) -> None:
    plan = _plan(embargo_periods=embargo_periods)
    first = plan.splits[0]
    assert len(first.embargo_dates) == embargo_periods
    if embargo_periods == 0:
        assert first.embargo_dates == ()
    assert not set(first.embargo_dates) & set(first.validation_dates)
    assert not set(first.embargo_dates) & set(first.train_dates)
    assert not set(first.embargo_dates) & set(first.label_unavailable_dates)


def test_embargo_history_shortage_is_skipped_until_valid() -> None:
    without = _plan(embargo_periods=0)
    with_embargo = _plan(embargo_periods=2)
    assert with_embargo.first_prediction_date > without.first_prediction_date
    assert with_embargo.n_skipped_initial_dates > without.n_skipped_initial_dates


def test_initial_dates_are_skipped_as_exact_prefix() -> None:
    plan = _plan()
    position = plan.all_score_dates.index(plan.first_prediction_date)
    assert plan.skipped_initial_prediction_dates == plan.all_score_dates[:position]
    assert all(
        date < plan.first_prediction_date
        for date in plan.skipped_initial_prediction_dates
    )


def test_no_valid_split_raises_detailed_insufficient_history() -> None:
    with pytest.raises(
        WalkForwardInsufficientHistoryError,
        match=(
            r"n_score_dates=6.*train_window_periods=10.*validation_periods=3"
            r".*embargo_periods=2.*window_type='rolling'"
            r".*retrain_frequency=4.*max_mature_history_periods="
        ),
    ):
        WalkForwardSplitter(
            _config(
                train_window_periods=10,
                validation_periods=3,
                embargo_periods=2,
                retrain_frequency=4,
            )
        ).build(_dataset(periods=6))


def test_failure_after_first_valid_split_is_integrity_error() -> None:
    dataset = _dataset()
    baseline = _plan(dataset)
    first_start = baseline.first_prediction_date

    class InterruptingSplitter(WalkForwardSplitter):
        def _window_for_prediction(self, prediction_start, all_score_dates, date_max_exit):
            if prediction_start > first_start:
                raise _CandidateHistoryError("simulated later interruption")
            return super()._window_for_prediction(
                prediction_start, all_score_dates, date_max_exit
            )

    with pytest.raises(
        WalkForwardIntegrityError,
        match="after the first split.*simulated later interruption",
    ):
        InterruptingSplitter(_config()).build(dataset)


def test_cross_sections_map_whole_dates_and_original_indices() -> None:
    counts = (1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1)
    dataset = _dataset(stocks_per_date=counts)
    metadata = dataset.metadata
    plan = _plan(dataset)
    first = plan.splits[0]

    for dates, indices in (
        (first.train_dates, first.train_indices),
        (first.validation_dates, first.validation_indices),
        (first.prediction_dates, first.prediction_indices),
    ):
        expected = tuple(metadata.index[metadata["trade_date"].isin(dates)])
        assert indices == expected
        for date in dates:
            expected_date_indices = set(
                metadata.index[metadata["trade_date"].eq(date)]
            )
            assert expected_date_indices <= set(indices)
    assert first.n_train_rows == len(first.train_indices)
    assert first.n_validation_rows == len(first.validation_indices)
    assert first.n_prediction_rows == len(first.prediction_indices)


def test_partition_indices_are_sorted_unique_and_disjoint() -> None:
    for split in _plan().splits:
        for indices in (
            split.train_indices,
            split.validation_indices,
            split.prediction_indices,
        ):
            assert indices == tuple(sorted(set(indices)))
        assert not set(split.train_indices) & set(split.validation_indices)
        assert not set(split.train_indices) & set(split.prediction_indices)
        assert not set(split.validation_indices) & set(split.prediction_indices)


class GuardedDataset(MLDataset):
    @property
    def features(self):
        raise AssertionError("splitter must not read features")

    @property
    def labels(self):
        raise AssertionError("splitter must not read labels")


def test_splitter_uses_only_public_metadata_and_sample_count() -> None:
    source = _dataset()
    guarded = GuardedDataset(
        source.features,
        source.labels,
        source.metadata,
        source.feature_names,
        source.label_name,
        source.audit,
    )
    metadata_before = guarded.metadata
    summary_before = guarded.summary()

    plan = WalkForwardSplitter(_config()).build(guarded)

    assert plan.n_splits > 0
    pdt.assert_frame_equal(guarded.metadata, metadata_before)
    assert guarded.summary() == summary_before


def test_non_dataset_input_raises_data_error() -> None:
    with pytest.raises(WalkForwardDataError, match="must be an MLDataset"):
        WalkForwardSplitter(_config()).build(object())  # type: ignore[arg-type]


def test_empty_metadata_and_non_range_index_are_rejected() -> None:
    metadata = _dataset().metadata
    with pytest.raises(WalkForwardDataError, match="contains 0 rows"):
        WalkForwardSplitter(_config()).build(_forged_dataset(metadata.iloc[0:0]))
    bad_index = metadata.copy().set_axis(range(1, len(metadata) + 1))
    with pytest.raises(WalkForwardDataError, match="RangeIndex"):
        WalkForwardSplitter(_config()).build(_forged_dataset(bad_index))


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("trade_date", None, "trade_date.*invalid_count=1"),
        ("entry_trade_date", None, "entry_trade_date.*invalid_count=1"),
        ("exit_trade_date", None, "exit_trade_date.*invalid_count=1"),
        ("ts_code", None, "ts_code.*invalid_count=1"),
        ("ts_code", "   ", "ts_code.*invalid_count=1"),
    ],
)
def test_missing_metadata_values_are_rejected(
    column: str, value: object, match: str
) -> None:
    metadata = _dataset().metadata
    metadata.loc[0, column] = value
    with pytest.raises(WalkForwardDataError, match=match):
        WalkForwardSplitter(_config()).build(_forged_dataset(metadata))


def test_duplicate_unsorted_and_wrong_schema_metadata_are_rejected() -> None:
    metadata = _dataset().metadata
    duplicate = pd.concat([metadata, metadata.iloc[[0]]], ignore_index=True)
    with pytest.raises(WalkForwardDataError, match="duplicate.*count=1"):
        WalkForwardSplitter(_config()).build(_forged_dataset(duplicate))

    unsorted = metadata.iloc[::-1].reset_index(drop=True)
    with pytest.raises(WalkForwardDataError, match="stably sorted"):
        WalkForwardSplitter(_config()).build(_forged_dataset(unsorted))

    wrong_columns = metadata.assign(extra=1)
    with pytest.raises(WalkForwardDataError, match="columns must exactly"):
        WalkForwardSplitter(_config()).build(_forged_dataset(wrong_columns))


def test_invalid_metadata_date_order_is_rejected() -> None:
    metadata = _dataset().metadata
    metadata.loc[0, "entry_trade_date"] = (
        metadata.loc[0, "trade_date"] - pd.Timedelta(days=1)
    )
    with pytest.raises(WalkForwardDataError, match="trade_date <= entry_trade_date"):
        WalkForwardSplitter(_config()).build(_forged_dataset(metadata))

    metadata = _dataset().metadata
    metadata.loc[0, "exit_trade_date"] = metadata.loc[0, "entry_trade_date"]
    with pytest.raises(
        WalkForwardDataError, match="entry_trade_date < exit_trade_date"
    ):
        WalkForwardSplitter(_config()).build(_forged_dataset(metadata))


def test_split_and_plan_summaries_are_json_safe_and_hide_indices() -> None:
    plan = _plan()
    split_summary = plan.splits[0].summary()
    plan_summary = plan.summary()
    json.dumps(split_summary, allow_nan=False)
    json.dumps(plan_summary, allow_nan=False)
    text = json.dumps(plan_summary)
    assert "train_indices" not in text
    assert "validation_indices" not in text
    assert "prediction_indices" not in text
    assert not any(
        isinstance(value, pd.Timestamp)
        for summary in (split_summary, plan_summary)
        for value in summary.values()
    )
    assert plan_summary["n_splits"] == plan.n_splits
    assert plan_summary["n_prediction_dates"] == plan.n_prediction_dates
    assert plan_summary["n_prediction_rows"] == sum(
        split.n_prediction_rows for split in plan.splits
    )


def test_all_score_dates_and_plan_properties_are_consistent() -> None:
    plan = _plan()
    assert plan.all_score_dates == tuple(sorted(set(plan.all_score_dates)))
    assert plan.n_score_dates == 16
    assert plan.n_splits == len(plan.splits)
    assert plan.n_prediction_dates == (
        plan.n_score_dates - plan.n_skipped_initial_dates
    )
    assert plan.first_prediction_date == plan.splits[0].prediction_dates[0]
    assert plan.last_prediction_date == plan.splits[-1].prediction_dates[-1]


def test_split_integrity_validation_rejects_overlap_and_bad_exit_cutoff() -> None:
    split = _plan().splits[0]
    with pytest.raises(WalkForwardIntegrityError, match="overlapping partition indices"):
        replace(split, validation_indices=split.train_indices)
    with pytest.raises(WalkForwardIntegrityError, match="max_train_exit_date"):
        replace(split, max_train_exit_date=split.validation_start_date)


def test_plan_integrity_rejects_prediction_gap_and_bad_ids() -> None:
    plan = _plan()
    with pytest.raises(
        WalkForwardIntegrityError,
        match="last_prediction_date must equal the final global score date",
    ):
        replace(
            plan,
            splits=plan.splits[:-1],
            last_prediction_date=plan.splits[-2].prediction_end_date,
        )
    with pytest.raises(WalkForwardIntegrityError, match="retrain_id sequence"):
        replace(
            plan,
            splits=(replace(plan.splits[0], retrain_id=2), *plan.splits[1:]),
        )
    shortened_first = replace(
        plan.splits[0],
        prediction_indices=plan.splits[0].prediction_indices[:-2],
        prediction_dates=plan.splits[0].prediction_dates[:-1],
        prediction_end_date=plan.splits[0].prediction_dates[-2],
        n_prediction_rows=len(plan.splits[0].prediction_indices) - 2,
    )
    with pytest.raises(WalkForwardIntegrityError, match="without gaps"):
        replace(plan, splits=(shortened_first, *plan.splits[1:]))


def test_error_hierarchy_and_public_types() -> None:
    assert issubclass(WalkForwardConfigError, WalkForwardError)
    assert issubclass(WalkForwardDataError, WalkForwardError)
    assert issubclass(WalkForwardInsufficientHistoryError, WalkForwardError)
    assert issubclass(WalkForwardIntegrityError, WalkForwardError)
    assert WalkForwardConfig is not None
    assert WalkForwardSplit is not None
    assert WalkForwardPlan is not None
    assert WalkForwardSplitter is not None
