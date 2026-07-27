"""Tests for the pure in-memory V4-B Modeling Panel Builder."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pandas.testing as pdt
import pytest

from src.modeling_panel import (
    MODELING_PANEL_AUDIT_COLUMNS,
    MODELING_PANEL_KEY_COLUMNS,
    MODELING_PANEL_SCHEMA_VERSION,
    ModelingPanelAlignmentError,
    ModelingPanelBuilder,
    ModelingPanelConfig,
    ModelingPanelConfigError,
    ModelingPanelDataError,
    ModelingPanelLeakageError,
    ModelingPanelResult,
)


FEATURE_NAMES = ("factor_a", "factor_b")


def _factor_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [
                "2024-01-02",
                "2024-01-01",
                "2024-01-02",
                "2024-01-01",
            ],
            "ts_code": [" B ", "A", "A", "B"],
            "factor_a": [3.0, 1.0, 4.0, 2.0],
            "factor_b": [30.0, 10.0, 40.0, np.nan],
        }
    )


def _forward_returns(label: str = "forward_return") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-01",
                "2024-01-02",
            ],
            "ts_code": ["B", "A", "A", " B "],
            "entry_trade_date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-02",
                "2024-01-03",
            ],
            "exit_trade_date": [
                "2024-01-03",
                "2024-01-04",
                "2024-01-03",
                "2024-01-04",
            ],
            "entry_price": [20.0, 40.0, 10.0, 30.0],
            "exit_price": [18.0, 44.0, 11.0, 33.0],
            label: [-0.1, 0.1, 0.1, 0.1],
        }
    )


def _builder(**config: object) -> ModelingPanelBuilder:
    return ModelingPanelBuilder(ModelingPanelConfig(**config))  # type: ignore[arg-type]


def _build_result(
    factors: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    builder: ModelingPanelBuilder | None = None,
) -> ModelingPanelResult:
    return (builder or ModelingPanelBuilder()).build(
        _factor_panel() if factors is None else factors,
        _forward_returns() if returns is None else returns,
    )


def test_default_and_explicit_config_api() -> None:
    default = ModelingPanelBuilder()
    config = ModelingPanelConfig(include_features=("factor_b", "factor_a"))
    explicit = ModelingPanelBuilder(config)
    assert default.config == ModelingPanelConfig()
    assert explicit.config is config
    assert explicit.build(
        _factor_panel(), _forward_returns()
    ).feature_names == ("factor_b", "factor_a")


@pytest.mark.parametrize("value", [{}, "config", 1, object()])
def test_builder_rejects_non_config(value: object) -> None:
    with pytest.raises(ModelingPanelConfigError, match="config"):
        ModelingPanelBuilder(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("factor", None),
        ("factor", []),
        ("returns", None),
        ("returns", []),
    ],
)
def test_build_requires_dataframes(argument: str, value: object) -> None:
    factors: object = _factor_panel()
    returns: object = _forward_returns()
    if argument == "factor":
        factors = value
    else:
        returns = value
    with pytest.raises(ModelingPanelDataError, match="DataFrame"):
        ModelingPanelBuilder().build(factors, returns)  # type: ignore[arg-type]


@pytest.mark.parametrize("argument", ["factor", "returns"])
def test_build_rejects_empty_inputs(argument: str) -> None:
    factors = _factor_panel()
    returns = _forward_returns()
    if argument == "factor":
        factors = factors.iloc[0:0]
    else:
        returns = returns.iloc[0:0]
    with pytest.raises(ModelingPanelDataError, match="empty"):
        ModelingPanelBuilder().build(factors, returns)


def test_inputs_are_deep_copied_and_build_is_deterministic() -> None:
    factors = _factor_panel()
    returns = _forward_returns()
    factors_before = factors.copy(deep=True)
    returns_before = returns.copy(deep=True)
    builder = ModelingPanelBuilder()

    first = builder.build(factors, returns)
    second = builder.build(factors, returns)

    pdt.assert_frame_equal(factors, factors_before)
    pdt.assert_frame_equal(returns, returns_before)
    pdt.assert_frame_equal(first.panel, second.panel)
    assert first.audit == second.audit
    factors.loc[0, "factor_a"] = -999.0
    returns.loc[0, "forward_return"] = -999.0
    assert first.panel.loc[3, "factor_a"] == 3.0
    assert first.panel.loc[0, "forward_return"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("panel_name", "column"),
    [
        ("factor", "trade_date"),
        ("factor", "ts_code"),
        ("returns", "trade_date"),
        ("returns", "ts_code"),
        ("returns", "entry_trade_date"),
        ("returns", "exit_trade_date"),
        ("returns", "entry_price"),
        ("returns", "exit_price"),
        ("returns", "forward_return"),
    ],
)
def test_required_columns_are_enforced(panel_name: str, column: str) -> None:
    factors = _factor_panel()
    returns = _forward_returns()
    if panel_name == "factor":
        factors = factors.drop(columns=column)
    else:
        returns = returns.drop(columns=column)
    with pytest.raises(ModelingPanelDataError, match=column):
        ModelingPanelBuilder().build(factors, returns)


@pytest.mark.parametrize("panel_name", ["factor", "returns"])
def test_duplicate_and_multiindex_columns_are_rejected(panel_name: str) -> None:
    factors = _factor_panel()
    returns = _forward_returns()
    target = factors if panel_name == "factor" else returns
    target.columns = [
        *target.columns[:-1],
        target.columns[-2],
    ]
    with pytest.raises(ModelingPanelDataError, match="unique"):
        ModelingPanelBuilder().build(factors, returns)

    factors = _factor_panel()
    returns = _forward_returns()
    target = factors if panel_name == "factor" else returns
    target.columns = pd.MultiIndex.from_tuples(
        [(str(name), "") for name in target.columns]
    )
    with pytest.raises(ModelingPanelDataError, match="MultiIndex"):
        ModelingPanelBuilder().build(factors, returns)


@pytest.mark.parametrize(
    "column",
    [
        "forward_return",
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
    ],
)
def test_factor_panel_reserved_columns_raise_leakage(column: str) -> None:
    factors = _factor_panel()
    factors[column] = 1.0
    with pytest.raises(ModelingPanelLeakageError, match=column):
        _build_result(factors=factors)


def test_custom_label_and_known_forward_return_are_both_forbidden_in_factors() -> None:
    config = ModelingPanelConfig(label_column="target")
    builder = ModelingPanelBuilder(config)
    returns = _forward_returns("target")
    for column in ("target", "forward_return"):
        factors = _factor_panel()
        factors[column] = 0.1
        with pytest.raises(ModelingPanelLeakageError, match=column):
            builder.build(factors, returns)


def test_extra_return_columns_are_ignored_warned_and_not_features() -> None:
    returns = _forward_returns()
    returns["holding_period"] = 2
    returns["other_return"] = 0.5
    result = _build_result(returns=returns)
    assert "holding_period" not in result.panel
    assert "other_return" not in result.panel
    assert result.audit.warnings[0] == (
        "Ignored extra forward-return columns: "
        "['holding_period', 'other_return']."
    )


def test_non_key_name_collision_is_rejected_before_merge() -> None:
    returns = _forward_returns()
    returns["factor_a"] = 99.0
    with pytest.raises(ModelingPanelDataError, match="conflicting.*factor_a"):
        _build_result(returns=returns)


def test_dates_and_codes_are_normalized_without_changing_calendar_dates() -> None:
    result = _build_result()
    panel = result.panel
    assert panel["trade_date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
    ]
    assert panel["ts_code"].tolist() == ["A", "B", "A", "B"]
    assert str(panel["ts_code"].dtype) == "string"
    for column in ("trade_date", "entry_trade_date", "exit_trade_date"):
        assert ptypes.is_datetime64_ns_dtype(panel[column])


def test_python_date_and_datetime_values_are_accepted() -> None:
    factors = _factor_panel()
    returns = _forward_returns()
    factors["trade_date"] = [
        date(2024, 1, 2),
        datetime(2024, 1, 1),
        date(2024, 1, 2),
        datetime(2024, 1, 1),
    ]
    returns["entry_trade_date"] = [
        date(2024, 1, 2),
        datetime(2024, 1, 3),
        date(2024, 1, 2),
        datetime(2024, 1, 3),
    ]
    assert len(_build_result(factors=factors, returns=returns).panel) == 4


@pytest.mark.parametrize(
    ("panel_name", "column", "value"),
    [
        ("factor", "trade_date", "not-a-date"),
        ("returns", "entry_trade_date", "2024/01/02"),
        ("returns", "entry_trade_date", ""),
        ("returns", "exit_trade_date", "tomorrow"),
        ("factor", "trade_date", pd.NaT),
        ("returns", "trade_date", None),
        ("factor", "trade_date", pd.Timestamp("2024-01-01 12:00")),
        ("returns", "entry_trade_date", datetime(2024, 1, 2, 0, 1)),
        (
            "factor",
            "trade_date",
            pd.Timestamp("2024-01-01", tz="UTC"),
        ),
        (
            "returns",
            "exit_trade_date",
            datetime(2024, 1, 3, tzinfo=timezone.utc),
        ),
    ],
)
def test_invalid_timezone_or_non_daily_dates_are_rejected(
    panel_name: str, column: str, value: object
) -> None:
    factors = _factor_panel()
    returns = _forward_returns()
    target = factors if panel_name == "factor" else returns
    target[column] = target[column].astype(object)
    target.at[0, column] = value
    with pytest.raises(ModelingPanelDataError, match=column):
        _build_result(factors=factors, returns=returns)


def test_missing_entry_and_exit_dates_are_allowed_for_missing_label_rows() -> None:
    returns = _forward_returns()
    returns.loc[0, ["entry_trade_date", "exit_trade_date"]] = pd.NaT
    returns.loc[0, ["entry_price", "exit_price", "forward_return"]] = np.nan
    result = _build_result(returns=returns)
    assert result.audit.label_missing_count == 1
    assert pd.isna(result.panel.loc[1, "entry_trade_date"])


@pytest.mark.parametrize("panel_name", ["factor", "returns"])
@pytest.mark.parametrize("bad_code", [None, "", "   ", 123])
def test_invalid_or_numeric_codes_are_rejected(
    panel_name: str, bad_code: object
) -> None:
    factors = _factor_panel()
    returns = _forward_returns()
    target = factors if panel_name == "factor" else returns
    target["ts_code"] = target["ts_code"].astype(object)
    target.at[0, "ts_code"] = bad_code
    with pytest.raises(ModelingPanelDataError, match="ts_code"):
        _build_result(factors=factors, returns=returns)


@pytest.mark.parametrize("panel_name", ["factor", "returns"])
def test_duplicate_keys_are_rejected_without_deduplication(panel_name: str) -> None:
    factors = _factor_panel()
    returns = _forward_returns()
    target = factors if panel_name == "factor" else returns
    target.loc[1, ["trade_date", "ts_code"]] = target.loc[
        0, ["trade_date", "ts_code"]
    ].to_numpy()
    before = target.copy(deep=True)
    with pytest.raises(
        ModelingPanelAlignmentError,
        match="duplicate_rows=2, duplicate_unique_keys=1",
    ):
        _build_result(factors=factors, returns=returns)
    pdt.assert_frame_equal(target, before)


def test_duplicate_samples_are_sorted_bounded_and_deterministic() -> None:
    rows: list[dict[str, object]] = []
    for index in reversed(range(25)):
        for repeat in range(2):
            rows.append(
                {
                    "trade_date": "2024-01-01",
                    "ts_code": f"S{index:02d}",
                    "factor_a": float(index + repeat),
                }
            )
    factors = pd.DataFrame(rows)
    with pytest.raises(ModelingPanelAlignmentError) as first:
        ModelingPanelBuilder().build(factors, _forward_returns())
    with pytest.raises(ModelingPanelAlignmentError) as second:
        ModelingPanelBuilder().build(factors, _forward_returns())
    message = str(first.value)
    assert message == str(second.value)
    assert "duplicate_rows=50" in message
    assert "duplicate_unique_keys=25" in message
    assert "S00" in message and "S19" in message and "S20" not in message


def test_default_features_preserve_factor_column_order() -> None:
    factors = _factor_panel()[
        ["trade_date", "factor_b", "ts_code", "factor_a"]
    ]
    result = _build_result(factors=factors)
    assert result.feature_names == ("factor_b", "factor_a")


def test_include_features_are_exact_and_ignored_columns_are_warned() -> None:
    result = _build_result(
        builder=_builder(include_features=("factor_b",)),
    )
    assert result.feature_names == ("factor_b",)
    assert "factor_a" not in result.panel
    assert result.audit.warnings[0] == "Ignored factor columns: ['factor_a']."


@pytest.mark.parametrize("field", ["include_features", "exclude_features"])
def test_configured_features_must_exist_in_input(field: str) -> None:
    with pytest.raises(ModelingPanelDataError, match=field):
        _build_result(builder=_builder(**{field: ("missing",)}))


def test_exclude_preserves_source_order_and_empty_resolution_fails() -> None:
    result = _build_result(builder=_builder(exclude_features=("factor_a",)))
    assert result.feature_names == ("factor_b",)
    with pytest.raises(ModelingPanelDataError, match="no Modeling Panel features"):
        _build_result(
            builder=_builder(exclude_features=("factor_a", "factor_b"))
        )


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("bool", pd.Series([True, False, True, False])),
        ("object", pd.Series(["1", "2", "3", "4"], dtype=object)),
        ("string", pd.Series(["1", "2", "3", "4"], dtype="string")),
    ],
)
def test_non_numeric_feature_dtypes_are_rejected(
    kind: str, value: pd.Series
) -> None:
    factors = _factor_panel()
    factors["factor_a"] = value
    with pytest.raises(ModelingPanelDataError, match="factor_a.*numeric"):
        _build_result(factors=factors)


def test_feature_infinity_is_rejected_but_nan_is_retained_and_audited() -> None:
    factors = _factor_panel()
    factors.at[0, "factor_a"] = np.inf
    with pytest.raises(ModelingPanelDataError, match="factor_a.*infinite_count=1"):
        _build_result(factors=factors)

    result = _build_result()
    assert result.audit.feature_missing_counts == (
        ("factor_a", 0),
        ("factor_b", 1),
    )
    assert result.audit.feature_missing_rates == (
        ("factor_a", 0.0),
        ("factor_b", 0.25),
    )
    assert pd.isna(result.panel.loc[1, "factor_b"])


def test_all_missing_output_or_labeled_feature_is_rejected() -> None:
    factors = _factor_panel()
    factors["factor_a"] = np.nan
    with pytest.raises(ModelingPanelDataError, match="entirely missing"):
        _build_result(factors=factors)

    factors = _factor_panel()
    returns = _forward_returns()
    returns.loc[0, ["entry_trade_date", "exit_trade_date"]] = pd.NaT
    returns.loc[0, ["entry_price", "exit_price", "forward_return"]] = np.nan
    factors["factor_a"] = np.nan
    factors.loc[
        (factors["trade_date"] == "2024-01-01")
        & (factors["ts_code"].str.strip() == "B"),
        "factor_a",
    ] = 9.0
    with pytest.raises(ModelingPanelDataError, match="non-missing-label"):
        _build_result(factors=factors, returns=returns)


def test_constant_high_missing_and_suspicious_warnings_are_ordered() -> None:
    factors = _factor_panel()
    factors["factor_a"] = 1.0
    factors["factor_b"] = [np.nan, 2.0, np.nan, 3.0]
    factors["next_year_growth_forecast"] = [1.0, 2.0, 3.0, 4.0]
    result = _build_result(factors=factors)
    assert result.audit.constant_features == ("factor_a",)
    assert result.audit.suspicious_feature_names == (
        "next_year_growth_forecast",
    )
    assert result.audit.warnings[:3] == (
        "Suspicious feature names require provenance review: "
        "['next_year_growth_forecast'].",
        "Constant features were retained: ['factor_a'].",
        "Features with missing rate >= 0.50 were retained: ['factor_b'].",
    )


def test_forward_pe_is_not_suspicious() -> None:
    factors = _factor_panel()
    factors["forward_pe"] = [10.0, 11.0, 12.0, 13.0]
    result = _build_result(factors=factors)
    assert result.audit.suspicious_feature_names == ()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("forward_return", pd.Series([True, False, True, False])),
        ("forward_return", pd.Series(["-.1", ".1", ".1", ".1"])),
        ("entry_price", pd.Series([True, True, True, True])),
        ("entry_price", pd.Series(["20", "40", "10", "30"])),
        ("exit_price", pd.Series([True, True, True, True])),
        ("exit_price", pd.Series(["18", "44", "11", "33"])),
    ],
)
def test_label_and_price_require_numeric_non_boolean_dtype(
    column: str, value: pd.Series
) -> None:
    returns = _forward_returns()
    returns[column] = value
    with pytest.raises(ModelingPanelDataError, match=f"{column}.*numeric"):
        _build_result(returns=returns)


@pytest.mark.parametrize("column", ["forward_return", "entry_price", "exit_price"])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_label_and_prices_reject_infinity(column: str, value: float) -> None:
    returns = _forward_returns()
    returns.at[0, column] = value
    with pytest.raises(ModelingPanelDataError, match=f"{column}.*infinite_count"):
        _build_result(returns=returns)


@pytest.mark.parametrize("column", ["entry_price", "exit_price"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_prices_must_be_strictly_positive(column: str, value: float) -> None:
    returns = _forward_returns()
    returns.at[0, column] = value
    with pytest.raises(ModelingPanelDataError, match=f"{column}.*positive"):
        _build_result(returns=returns)


def test_missing_label_policy_and_warning() -> None:
    returns = _forward_returns()
    returns.loc[0, ["entry_trade_date", "exit_trade_date"]] = pd.NaT
    returns.loc[0, ["entry_price", "exit_price", "forward_return"]] = np.nan
    allowed = _build_result(returns=returns)
    assert allowed.audit.label_missing_count == 1
    assert "Missing labels were retained: count=1." in allowed.audit.warnings
    with pytest.raises(ModelingPanelDataError, match="missing_count=1"):
        _build_result(
            returns=returns,
            builder=_builder(allow_missing_labels=False),
        )


@pytest.mark.parametrize(
    "missing_column",
    [
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
    ],
)
def test_nonmissing_label_requires_complete_audit_values(
    missing_column: str,
) -> None:
    returns = _forward_returns()
    returns.at[0, missing_column] = (
        pd.NaT if "date" in missing_column else np.nan
    )
    with pytest.raises(ModelingPanelDataError, match=missing_column):
        _build_result(returns=returns)


def test_exit_and_prices_require_corresponding_dates_even_without_label() -> None:
    cases = (
        (
            "entry_price",
            {
                "entry_trade_date": pd.NaT,
                "exit_trade_date": pd.NaT,
                "entry_price": 20.0,
                "exit_price": np.nan,
            },
        ),
        (
            "exit_trade_date",
            {
                "entry_trade_date": pd.NaT,
                "exit_trade_date": "2024-01-03",
                "entry_price": np.nan,
                "exit_price": np.nan,
            },
        ),
        (
            "exit_price",
            {"exit_trade_date": pd.NaT, "exit_price": 20.0},
        ),
    )
    for dependent_column, updates in cases:
        returns = _forward_returns()
        returns.at[0, "forward_return"] = np.nan
        for column, value in updates.items():
            returns.at[0, column] = value
        with pytest.raises(ModelingPanelDataError, match=dependent_column):
            _build_result(returns=returns)


def test_bidirectional_unmatched_audit_and_warning() -> None:
    factors = _factor_panel()
    returns = _forward_returns()
    factors.loc[0, "ts_code"] = "FACTOR_ONLY"
    returns.loc[3, "ts_code"] = "RETURN_ONLY"
    result = _build_result(factors=factors, returns=returns)
    assert result.audit.factor_only.row_count == 1
    assert result.audit.return_only.row_count == 1
    assert result.audit.factor_only.sampled_keys == (
        (pd.Timestamp("2024-01-02"), "FACTOR_ONLY"),
    )
    assert result.audit.return_only.sampled_keys == (
        (pd.Timestamp("2024-01-02"), "RETURN_ONLY"),
    )
    assert result.audit.warnings[0] == (
        "Unmatched keys were dropped: factor_only=1, return_only=1."
    )
    assert result.audit.matched_rows == result.audit.output_rows == 3


def test_unmatched_error_and_no_intersection_raise_alignment_errors() -> None:
    factors = _factor_panel()
    factors.loc[0, "ts_code"] = "ONLY"
    with pytest.raises(ModelingPanelAlignmentError, match="factor_only=1"):
        _build_result(
            factors=factors,
            builder=_builder(unmatched_policy="error"),
        )

    factors = _factor_panel()
    factors["ts_code"] = ["X1", "X2", "X3", "X4"]
    with pytest.raises(ModelingPanelAlignmentError, match="no matched keys"):
        _build_result(factors=factors)


def test_unmatched_samples_are_bounded_sorted_and_deterministic() -> None:
    factors = pd.DataFrame(
        {
            "trade_date": ["2024-01-01"] * 25 + ["2024-01-02"],
            "ts_code": [f"X{index:02d}" for index in reversed(range(25))] + ["A"],
            "factor_a": np.arange(26, dtype=float),
        }
    )
    returns = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"],
            "ts_code": ["A"],
            "entry_trade_date": ["2024-01-03"],
            "exit_trade_date": ["2024-01-04"],
            "entry_price": [10.0],
            "exit_price": [11.0],
            "forward_return": [0.1],
        }
    )
    result = _build_result(factors=factors, returns=returns)
    samples = result.audit.factor_only.sampled_keys
    assert len(samples) == 20
    assert samples == tuple(sorted(samples))
    assert samples[0][1] == "X00"
    assert samples[-1][1] == "X19"


def test_time_order_normal_and_non_strict_equal_entry() -> None:
    normal = _build_result()
    assert normal.audit.entry_before_signal_count == 0
    assert normal.audit.entry_equal_signal_count == 0
    assert normal.audit.exit_not_after_entry_count == 0

    returns = _forward_returns()
    returns["entry_trade_date"] = returns["trade_date"]
    relaxed = _build_result(
        returns=returns,
        builder=_builder(require_entry_after_signal=False),
    )
    assert relaxed.audit.entry_equal_signal_count == 4


@pytest.mark.parametrize(
    ("entry_date", "exit_date", "message"),
    [
        ("2023-12-31", "2024-01-03", "precedes"),
        ("2024-01-01", "2024-01-03", "equals"),
        ("2024-01-02", "2024-01-02", "later"),
        ("2024-01-03", "2024-01-02", "later"),
    ],
)
def test_time_order_violations_raise_bounded_leakage_errors(
    entry_date: str, exit_date: str, message: str
) -> None:
    returns = _forward_returns()
    returns.at[2, "entry_trade_date"] = entry_date
    returns.at[2, "exit_trade_date"] = exit_date
    with pytest.raises(ModelingPanelLeakageError, match=message) as error:
        _build_result(returns=returns)
    assert "violation_count=1" in str(error.value)
    assert "2024-01-01" in str(error.value)
    assert len(str(error.value)) < 350


def test_formula_exact_and_within_tolerance_pass_without_recalculation() -> None:
    exact = _build_result()
    assert exact.audit.label_formula_mismatch_count == 0
    returns = _forward_returns()
    original = returns.at[0, "forward_return"]
    returns.at[0, "forward_return"] = original + 5e-12
    result = _build_result(returns=returns)
    assert result.panel.loc[1, "forward_return"] == pytest.approx(
        original + 5e-12
    )


def test_formula_mismatch_raises_and_missing_label_does_not_participate() -> None:
    returns = _forward_returns()
    returns.at[0, "forward_return"] = 0.5
    with pytest.raises(
        ModelingPanelLeakageError, match="label_formula_mismatch_count=1"
    ) as error:
        _build_result(returns=returns)
    assert "2024-01-01" in str(error.value)
    assert len(str(error.value)) < 350

    returns.at[0, "forward_return"] = np.nan
    result = _build_result(returns=returns)
    assert result.audit.label_formula_mismatch_count == 0


def test_audit_coverage_distributions_and_fixed_output_shape() -> None:
    result = _build_result()
    audit = result.audit
    assert audit.schema_version == MODELING_PANEL_SCHEMA_VERSION
    assert audit.factor_input_rows == 4
    assert audit.return_input_rows == 4
    assert audit.matched_rows == audit.output_rows == 4
    assert audit.date_count == 2
    assert audit.security_count == 2
    assert audit.first_trade_date == pd.Timestamp("2024-01-01")
    assert audit.last_trade_date == pd.Timestamp("2024-01-02")
    assert audit.first_entry_trade_date == pd.Timestamp("2024-01-02")
    assert audit.last_entry_trade_date == pd.Timestamp("2024-01-03")
    assert audit.first_exit_trade_date == pd.Timestamp("2024-01-03")
    assert audit.last_exit_trade_date == pd.Timestamp("2024-01-04")
    assert audit.feature_count == 2
    assert audit.feature_names == FEATURE_NAMES
    assert audit.feature_non_finite_counts == (
        ("factor_a", 0),
        ("factor_b", 0),
    )
    assert audit.duplicate_factor_key_count == 0
    assert audit.duplicate_return_key_count == 0
    assert (
        audit.per_date_security_count_min,
        audit.per_date_security_count_median,
        audit.per_date_security_count_max,
    ) == (2, 2.0, 2)
    assert (
        audit.per_security_observation_count_min,
        audit.per_security_observation_count_median,
        audit.per_security_observation_count_max,
    ) == (2, 2.0, 2)
    assert audit.warnings == ()
    assert tuple(result.panel.columns) == (
        *MODELING_PANEL_KEY_COLUMNS,
        *FEATURE_NAMES,
        *MODELING_PANEL_AUDIT_COLUMNS,
        "forward_return",
    )
    assert isinstance(result.panel.index, pd.RangeIndex)
    assert result.feature_names == audit.feature_names
    assert result.label_column == audit.label_column == audit.config.label_column


def test_small_cross_section_and_short_history_warnings_have_fixed_order() -> None:
    factors = _factor_panel().iloc[[0, 1, 2]].copy()
    returns = _forward_returns().iloc[[3, 2, 1]].copy()
    result = _build_result(factors=factors, returns=returns)
    assert result.audit.warnings == (
        "At least one trade_date has fewer than 2 securities: minimum=1.",
        "At least one security has fewer than 2 observations: minimum=1.",
    )


def test_result_defensive_copy_and_v3_feature_inference_shape() -> None:
    result = _build_result()
    first = result.panel
    second = result.panel
    assert first is not second
    first.loc[0, "factor_a"] = -999.0
    assert result.panel.loc[0, "factor_a"] == 1.0
    reserved = {
        "trade_date",
        "ts_code",
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
        result.label_column,
    }
    inferred = tuple(
        column for column in result.panel.columns if column not in reserved
    )
    assert inferred == result.feature_names
    assert "_merge" not in result.panel
