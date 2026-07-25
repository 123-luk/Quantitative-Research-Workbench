"""Tests for the strict, model-free V3-A ML dataset contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pandas.testing as pdt
import pytest

from src.ml import (
    MLDataset,
    MLDatasetAlignmentError,
    MLDatasetAudit,
    MLDatasetBuilder,
    MLDatasetConfig,
    MLDatasetDuplicateKeyError,
    MLDatasetError,
    MLDatasetSchemaError,
    MLDatasetValueError,
)


FEATURE_NAMES = ("factor_a", "factor_b")
METADATA_COLUMNS = [
    "trade_date",
    "ts_code",
    "entry_trade_date",
    "exit_trade_date",
]


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = pd.DataFrame(
        {
            "trade_date": ["2024-01-03", "2024-01-02", "2024-01-02"],
            "ts_code": [" B ", "B", "A"],
            "factor_a": [3.0, 2.0, 1.0],
            "factor_b": [30.0, 20.0, 10.0],
            "ignored": [300.0, 200.0, 100.0],
        }
    )
    labels = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-02"],
            "ts_code": ["A", "B", " B "],
            "entry_trade_date": ["2024-01-03", "2024-01-04", "2024-01-03"],
            "exit_trade_date": ["2024-01-04", "2024-01-05", "2024-01-04"],
            "entry_price": [10.0, 30.0, 20.0],
            "exit_price": [11.0, 33.0, 22.0],
            "forward_return": [0.1, 0.3, 0.2],
        }
    )
    return factors, labels


def _build(
    factors: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    config: MLDatasetConfig | None = None,
) -> MLDataset:
    default_factors, default_labels = _panels()
    return MLDatasetBuilder(config).build(
        default_factors if factors is None else factors,
        default_labels if labels is None else labels,
        feature_names,
    )


def _audit(**overrides: object) -> MLDatasetAudit:
    values: dict[str, object] = {
        "input_feature_rows": 1,
        "input_label_rows": 1,
        "aligned_rows": 1,
        "output_rows": 1,
        "dropped_label_rows": 0,
        "missing_label_rows": 0,
        "nonfinite_label_rows": 0,
        "label_coverage": 1.0,
        "feature_count": 1,
        "feature_missing_counts": (("factor", 0),),
        "feature_missing_rates": (("factor", 0.0),),
        "feature_nonfinite_counts": (("factor", 0),),
        "min_trade_date": pd.Timestamp("2024-01-02"),
        "max_trade_date": pd.Timestamp("2024-01-02"),
    }
    values.update(overrides)
    return MLDatasetAudit(**values)  # type: ignore[arg-type]


def test_normal_build_schema_sorting_types_and_public_properties() -> None:
    dataset = _build()

    assert dataset.feature_names == FEATURE_NAMES
    assert dataset.label_name == "forward_return"
    assert dataset.n_samples == 3
    assert dataset.n_features == 2
    assert list(dataset.features.columns) == list(FEATURE_NAMES)
    assert dataset.labels.name == "forward_return"
    assert list(dataset.metadata.columns) == METADATA_COLUMNS
    assert dataset.metadata["ts_code"].tolist() == ["A", "B", "B"]
    assert dataset.metadata["trade_date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert all(
        ptypes.is_datetime64_ns_dtype(dataset.metadata[column])
        for column in ("trade_date", "entry_trade_date", "exit_trade_date")
    )
    assert str(dataset.metadata["ts_code"].dtype) == "string"
    assert all(dtype == np.dtype("float64") for dtype in dataset.features.dtypes)
    assert dataset.labels.dtype == np.dtype("float64")
    assert isinstance(dataset.features.index, pd.RangeIndex)
    assert dataset.features.index.equals(dataset.labels.index)
    assert dataset.features.index.equals(dataset.metadata.index)


def test_feature_order_is_caller_controlled_without_changing_row_alignment() -> None:
    normal = _build(feature_names=("factor_a", "factor_b"))
    reversed_features = _build(feature_names=("factor_b", "factor_a"))

    assert list(reversed_features.features.columns) == ["factor_b", "factor_a"]
    pdt.assert_frame_equal(normal.metadata, reversed_features.metadata)
    pdt.assert_series_equal(normal.labels, reversed_features.labels)
    pdt.assert_series_equal(
        normal.features["factor_a"], reversed_features.features["factor_a"]
    )


def test_inputs_are_deep_copied_and_never_modified() -> None:
    factors, labels = _panels()
    factors_before = factors.copy(deep=True)
    labels_before = labels.copy(deep=True)

    _build(factors, labels)

    pdt.assert_frame_equal(factors, factors_before)
    pdt.assert_frame_equal(labels, labels_before)


def test_dataset_properties_return_defensive_copies() -> None:
    dataset = _build()
    features = dataset.features
    labels = dataset.labels
    metadata = dataset.metadata
    features.loc[0, "factor_a"] = -999.0
    labels.loc[0] = -999.0
    metadata.loc[0, "ts_code"] = "MUTATED"

    assert dataset.features.loc[0, "factor_a"] == 1.0
    assert dataset.labels.loc[0] == pytest.approx(0.1)
    assert dataset.metadata.loc[0, "ts_code"] == "A"


@pytest.mark.parametrize("names", [(), [], "", ("",), ("  ",)])
def test_feature_names_must_be_nonempty_sequence(names: object) -> None:
    with pytest.raises(MLDatasetSchemaError, match="feature_names"):
        MLDatasetBuilder().build(*_panels(), names)  # type: ignore[arg-type]


def test_feature_names_are_stripped_and_duplicates_rejected() -> None:
    dataset = _build(feature_names=(" factor_a ", "factor_b"))
    assert dataset.feature_names == FEATURE_NAMES
    with pytest.raises(MLDatasetSchemaError, match="duplicates.*factor_a"):
        _build(feature_names=("factor_a", " factor_a "))


@pytest.mark.parametrize(
    "reserved",
    [
        "trade_date",
        "ts_code",
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
        "forward_return",
    ],
)
def test_reserved_fields_cannot_be_features(reserved: str) -> None:
    with pytest.raises(MLDatasetSchemaError, match="reserved"):
        _build(feature_names=(reserved,))


def test_missing_feature_and_required_columns_raise_schema_errors() -> None:
    factors, labels = _panels()
    with pytest.raises(MLDatasetSchemaError, match="factor_c"):
        _build(factors, labels, feature_names=("factor_c",))
    with pytest.raises(MLDatasetSchemaError, match="factor_panel.*trade_date"):
        _build(factors.drop(columns="trade_date"), labels)
    with pytest.raises(MLDatasetSchemaError, match="forward_returns.*entry_trade_date"):
        _build(factors, labels.drop(columns="entry_trade_date"))


@pytest.mark.parametrize("panel_name", ["factor", "label"])
def test_invalid_trade_dates_raise_value_error(panel_name: str) -> None:
    factors, labels = _panels()
    target = factors if panel_name == "factor" else labels
    target.loc[0, "trade_date"] = "not-a-date"
    with pytest.raises(MLDatasetValueError, match="trade_date.*unparseable=1"):
        _build(factors, labels)


@pytest.mark.parametrize("panel_name", ["factor", "label"])
@pytest.mark.parametrize("bad_code", [None, "", "   "])
def test_missing_or_empty_stock_codes_raise(
    panel_name: str, bad_code: object
) -> None:
    factors, labels = _panels()
    target = factors if panel_name == "factor" else labels
    target.loc[0, "ts_code"] = bad_code
    with pytest.raises(MLDatasetValueError, match="ts_code.*invalid_count=1"):
        _build(factors, labels)


@pytest.mark.parametrize("panel_name", ["factor", "label"])
def test_duplicate_keys_raise_with_count_and_examples(panel_name: str) -> None:
    factors, labels = _panels()
    target = factors if panel_name == "factor" else labels
    target = pd.concat([target, target.iloc[[0]].copy()], ignore_index=True)
    if panel_name == "factor":
        factors = target
    else:
        labels = target
    with pytest.raises(
        MLDatasetDuplicateKeyError,
        match=r"duplicate_key_count=1.*examples=",
    ):
        _build(factors, labels)


def test_feature_only_key_raises_alignment_error() -> None:
    factors, labels = _panels()
    labels = labels.iloc[1:].reset_index(drop=True)
    with pytest.raises(
        MLDatasetAlignmentError,
        match="feature_only_key_count=1, label_only_key_count=0",
    ):
        _build(factors, labels)


def test_label_only_key_raises_alignment_error() -> None:
    factors, labels = _panels()
    factors = factors.iloc[1:].reset_index(drop=True)
    with pytest.raises(
        MLDatasetAlignmentError,
        match="feature_only_key_count=0, label_only_key_count=1",
    ):
        _build(factors, labels)


def test_both_key_directions_are_reported_without_silent_inner_join() -> None:
    factors, labels = _panels()
    labels.loc[0, ["trade_date", "ts_code"]] = ["2024-02-01", "Z"]
    with pytest.raises(
        MLDatasetAlignmentError,
        match="feature_only_key_count=1, label_only_key_count=1",
    ):
        _build(factors, labels)


def test_shuffled_inputs_still_use_strict_one_to_one_key_alignment() -> None:
    factors, labels = _panels()
    first = _build(factors, labels)
    second = _build(
        factors.sample(frac=1.0, random_state=1).reset_index(drop=True),
        labels.sample(frac=1.0, random_state=2).reset_index(drop=True),
    )
    pdt.assert_frame_equal(first.features, second.features)
    pdt.assert_series_equal(first.labels, second.labels)
    pdt.assert_frame_equal(first.metadata, second.metadata)


def test_feature_nan_is_retained_and_missing_audit_is_ordered() -> None:
    factors, labels = _panels()
    factors.loc[0, "factor_a"] = np.nan
    factors.loc[1, "factor_b"] = np.nan
    dataset = _build(factors, labels)

    assert dataset.features.isna().sum().to_dict() == {
        "factor_a": 1,
        "factor_b": 1,
    }
    assert dataset.audit.feature_missing_counts == (
        ("factor_a", 1),
        ("factor_b", 1),
    )
    assert dict(dataset.audit.feature_missing_rates) == pytest.approx(
        {"factor_a": 1 / 3, "factor_b": 1 / 3}
    )


def test_feature_infinities_become_nan_and_are_audited() -> None:
    factors, labels = _panels()
    factors.loc[0, "factor_a"] = np.inf
    factors.loc[1, "factor_b"] = -np.inf
    dataset = _build(factors, labels)

    assert np.isnan(dataset.features.loc[2, "factor_a"])
    assert np.isnan(dataset.features.loc[1, "factor_b"])
    assert dataset.audit.feature_nonfinite_counts == (
        ("factor_a", 1),
        ("factor_b", 1),
    )


def test_non_numeric_feature_and_all_missing_feature_raise() -> None:
    factors, labels = _panels()
    factors["factor_a"] = factors["factor_a"].astype(object)
    factors.loc[0, "factor_a"] = "bad"
    with pytest.raises(
        MLDatasetValueError, match="feature column 'factor_a'.*invalid_count=1"
    ):
        _build(factors, labels)

    factors, labels = _panels()
    factors["factor_a"] = np.nan
    with pytest.raises(
        MLDatasetValueError, match="'factor_a'.*entirely missing"
    ):
        _build(factors, labels)


def test_missing_and_infinite_labels_are_dropped_with_exclusive_counts() -> None:
    factors, labels = _panels()
    labels.loc[0, "forward_return"] = np.nan
    labels.loc[1, "forward_return"] = np.inf
    dataset = _build(factors, labels)

    assert dataset.n_samples == 1
    assert dataset.labels.tolist() == [0.2]
    assert dataset.audit.aligned_rows == 3
    assert dataset.audit.output_rows == 1
    assert dataset.audit.missing_label_rows == 1
    assert dataset.audit.nonfinite_label_rows == 1
    assert dataset.audit.dropped_label_rows == 2
    assert dataset.audit.label_coverage == pytest.approx(1 / 3)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_labels_are_dropped(value: float) -> None:
    factors, labels = _panels()
    labels.loc[0, "forward_return"] = value
    dataset = _build(factors, labels)
    assert dataset.audit.nonfinite_label_rows == 1
    assert dataset.audit.missing_label_rows == 0
    assert not np.isinf(dataset.labels).any()


def test_all_invalid_labels_and_non_numeric_label_raise() -> None:
    factors, labels = _panels()
    labels["forward_return"] = [np.nan, np.inf, -np.inf]
    with pytest.raises(MLDatasetValueError, match="All aligned labels were dropped"):
        _build(factors, labels)

    factors, labels = _panels()
    labels["forward_return"] = labels["forward_return"].astype(object)
    labels.loc[0, "forward_return"] = "bad"
    with pytest.raises(
        MLDatasetValueError, match="label column 'forward_return'.*invalid_count=1"
    ):
        _build(factors, labels)


@pytest.mark.parametrize("date_col", ["entry_trade_date", "exit_trade_date"])
def test_audit_dates_may_be_missing_only_for_dropped_labels(date_col: str) -> None:
    factors, labels = _panels()
    labels.loc[0, date_col] = None
    with pytest.raises(MLDatasetValueError, match=f"{date_col}.*missing_count=1"):
        _build(factors, labels)

    labels.loc[0, "forward_return"] = np.nan
    dataset = _build(factors, labels)
    assert dataset.n_samples == 2


def test_date_order_is_strictly_validated() -> None:
    factors, labels = _panels()
    labels.loc[0, "entry_trade_date"] = "2024-01-01"
    with pytest.raises(
        MLDatasetValueError, match="trade_date <= entry_trade_date"
    ):
        _build(factors, labels)

    factors, labels = _panels()
    labels.loc[0, "exit_trade_date"] = labels.loc[0, "entry_trade_date"]
    with pytest.raises(
        MLDatasetValueError, match="entry_trade_date < exit_trade_date"
    ):
        _build(factors, labels)


def test_prices_label_keys_and_dates_never_enter_features() -> None:
    dataset = _build()
    assert list(dataset.features.columns) == list(FEATURE_NAMES)
    assert not {
        "trade_date",
        "ts_code",
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
        "forward_return",
    } & set(dataset.features.columns)
    assert list(dataset.metadata.columns) == METADATA_COLUMNS


def test_custom_label_column_and_config_validation() -> None:
    factors, labels = _panels()
    labels = labels.rename(columns={"forward_return": "target"})
    config = MLDatasetConfig(label_col=" target ")
    dataset = _build(factors, labels, config=config)
    assert dataset.label_name == "target"
    assert dataset.labels.name == "target"

    with pytest.raises(MLDatasetSchemaError, match="non-empty"):
        MLDatasetConfig(label_col=" ")
    for reserved in ("trade_date", "ts_code", "entry_trade_date", "exit_trade_date"):
        with pytest.raises(MLDatasetSchemaError, match="reserved"):
            MLDatasetConfig(label_col=reserved)


def test_audit_and_summary_are_json_safe_compact_and_frozen() -> None:
    dataset = _build()
    audit_payload = dataset.audit.as_dict()
    summary = dataset.summary()
    json.dumps(audit_payload, allow_nan=False)
    json.dumps(summary, allow_nan=False)
    assert audit_payload["min_trade_date"] == "2024-01-02T00:00:00"
    assert summary["feature_names"] == list(FEATURE_NAMES)
    assert "features" not in summary
    with pytest.raises(FrozenInstanceError):
        dataset.audit.output_rows = 99  # type: ignore[misc]
    detached = dataset.audit.as_dict()
    detached["feature_missing_counts"]["factor_a"] = 99
    assert dict(dataset.audit.feature_missing_counts)["factor_a"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_feature_rows", -1),
        ("input_label_rows", -1),
        ("aligned_rows", -1),
        ("output_rows", -1),
        ("dropped_label_rows", -1),
        ("missing_label_rows", -1),
        ("nonfinite_label_rows", -1),
        ("feature_count", -1),
        ("label_coverage", -0.1),
        ("label_coverage", 1.1),
    ],
)
def test_audit_rejects_negative_counts_and_invalid_coverage(
    field: str, value: object
) -> None:
    with pytest.raises(MLDatasetValueError, match=field):
        _audit(**{field: value})


def test_audit_validates_cross_field_consistency_and_stat_order() -> None:
    with pytest.raises(MLDatasetValueError, match="dropped_label_rows"):
        _audit(dropped_label_rows=1)
    with pytest.raises(MLDatasetSchemaError, match="order"):
        _audit(feature_missing_rates=(("other", 0.0),))


def test_empty_panels_raise_clear_value_errors() -> None:
    factors, labels = _panels()
    with pytest.raises(MLDatasetValueError, match="factor_panel contains 0 rows"):
        _build(factors.iloc[0:0], labels)
    with pytest.raises(MLDatasetValueError, match="forward_returns contains 0 rows"):
        _build(factors, labels.iloc[0:0])


def test_public_error_hierarchy_and_imports() -> None:
    assert issubclass(MLDatasetSchemaError, MLDatasetError)
    assert issubclass(MLDatasetDuplicateKeyError, MLDatasetError)
    assert issubclass(MLDatasetAlignmentError, MLDatasetError)
    assert issubclass(MLDatasetValueError, MLDatasetError)
    assert MLDataset is not None
    assert MLDatasetAudit is not None
    assert MLDatasetBuilder is not None
    assert MLDatasetConfig is not None


def test_mldataset_constructor_rejects_schema_and_index_mismatches() -> None:
    features = pd.DataFrame({"factor": [1.0]})
    labels = pd.Series([0.1], name="forward_return")
    metadata = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")],
            "ts_code": pd.Series(["A"], dtype="string"),
            "entry_trade_date": [pd.Timestamp("2024-01-03")],
            "exit_trade_date": [pd.Timestamp("2024-01-04")],
        }
    )
    MLDataset(
        features,
        labels,
        metadata,
        ("factor",),
        "forward_return",
        _audit(),
    )

    with pytest.raises(MLDatasetSchemaError, match="features columns"):
        MLDataset(
            features,
            labels,
            metadata,
            ("other",),
            "forward_return",
            _audit(),
        )
    with pytest.raises(MLDatasetAlignmentError, match="RangeIndex"):
        MLDataset(
            features.set_axis([5]),
            labels,
            metadata,
            ("factor",),
            "forward_return",
            _audit(),
        )
