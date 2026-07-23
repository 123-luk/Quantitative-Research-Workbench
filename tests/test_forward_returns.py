"""Tests for unified-calendar, exact-date forward-return label construction."""

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pandas.testing as pdt
import pytest

from src.factors.evaluation import FactorEvaluationConfig, FactorEvaluator
from src.factors.forward_returns import ForwardReturnBuilder, ForwardReturnConfig
from src.factors.quantile_evaluation import (
    FactorQuantileEvaluator,
    QuantileEvaluationConfig,
)


def _config(**overrides: object) -> ForwardReturnConfig:
    defaults = {"entry_lag_periods": 1, "holding_periods": 2}
    defaults.update(overrides)
    return ForwardReturnConfig(**defaults)


def _builder(**config: object) -> ForwardReturnBuilder:
    return ForwardReturnBuilder(_config(**config))


def _prices(
    dates: list[str] | None = None,
    codes: tuple[str, ...] = ("A", "B"),
) -> pd.DataFrame:
    dates = dates or [
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
    ]
    rows = []
    for date_index, date in enumerate(dates):
        for code_index, code in enumerate(codes):
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "close": 100.0 + 10.0 * code_index + 5.0 * date_index,
                    "open": 90.0 + 10.0 * code_index + 4.0 * date_index,
                }
            )
    return pd.DataFrame(rows)


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-08", "2024-01-05", "2024-01-05"],
            "ts_code": [" B ", "B", "A"],
            "ignored_factor": [1.0, 2.0, 3.0],
        }
    )


def test_default_config_is_serializable_frozen_and_describable() -> None:
    default = ForwardReturnConfig()
    assert default.to_dict() == {
        "price_col": "close",
        "return_col": "forward_return",
        "entry_lag_periods": 1,
        "holding_periods": 20,
        "require_positive_prices": True,
    }
    json.dumps(default.to_dict())
    assert ForwardReturnBuilder().describe_config() == default.to_dict()
    with pytest.raises(FrozenInstanceError):
        default.price_col = "open"  # type: ignore[misc]


@pytest.mark.parametrize("field", ["price_col", "return_col"])
@pytest.mark.parametrize("value", ["", " ", None])
def test_empty_column_names_raise(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        ForwardReturnConfig(**{field: value})


@pytest.mark.parametrize("value", ["trade_date", "ts_code"])
def test_price_column_cannot_conflict_with_keys(value: str) -> None:
    with pytest.raises(ValueError):
        ForwardReturnConfig(price_col=value)


@pytest.mark.parametrize(
    "value",
    [
        "trade_date",
        "ts_code",
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
    ],
)
def test_return_column_cannot_conflict_with_output_fields(value: str) -> None:
    with pytest.raises(ValueError):
        ForwardReturnConfig(return_col=value)


@pytest.mark.parametrize("value", [-1, 1.5, True, "1"])
def test_invalid_entry_lag_raises(value: object) -> None:
    with pytest.raises(ValueError):
        ForwardReturnConfig(entry_lag_periods=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "1"])
def test_invalid_holding_period_raises(value: object) -> None:
    with pytest.raises(ValueError):
        ForwardReturnConfig(holding_periods=value)  # type: ignore[arg-type]


def test_require_positive_prices_must_be_bool() -> None:
    with pytest.raises(ValueError):
        ForwardReturnConfig(require_positive_prices=1)  # type: ignore[arg-type]


def test_constructor_rejects_invalid_config() -> None:
    with pytest.raises(TypeError):
        ForwardReturnBuilder(object())  # type: ignore[arg-type]


def test_inputs_must_be_dataframes() -> None:
    with pytest.raises(TypeError):
        _builder().build([], _prices())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _builder().build(_scores(), [])  # type: ignore[arg-type]


@pytest.mark.parametrize("column", ["trade_date", "ts_code"])
def test_missing_score_columns_raise(column: str) -> None:
    with pytest.raises(ValueError):
        _builder().build(_scores().drop(columns=column), _prices())


@pytest.mark.parametrize("column", ["trade_date", "ts_code", "close"])
def test_missing_price_columns_raise(column: str) -> None:
    with pytest.raises(ValueError):
        _builder().build(_scores(), _prices().drop(columns=column))


def test_duplicate_keys_and_empty_codes_raise() -> None:
    scores = pd.concat([_scores(), _scores().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        _builder().build(scores, _prices())
    prices = pd.concat([_prices(), _prices().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        _builder().build(_scores(), prices)
    bad_scores = _scores()
    bad_scores.loc[0, "ts_code"] = " "
    with pytest.raises(ValueError):
        _builder().build(bad_scores, _prices())
    bad_prices = _prices()
    bad_prices.loc[0, "ts_code"] = " "
    with pytest.raises(ValueError):
        _builder().build(_scores(), bad_prices)


def test_invalid_dates_raise() -> None:
    scores = _scores()
    scores.loc[0, "trade_date"] = "bad"
    with pytest.raises(ValueError):
        _builder().build(scores, _prices())
    prices = _prices()
    prices["trade_date"] = prices["trade_date"].astype(object)
    prices.loc[0, "trade_date"] = "bad"
    with pytest.raises(ValueError):
        _builder().build(_scores(), prices)


def test_score_date_must_exist_in_market_calendar() -> None:
    scores = pd.DataFrame({"trade_date": ["2024-01-06"], "ts_code": ["A"]})
    with pytest.raises(ValueError, match="market calendar"):
        _builder().build(scores, _prices())


def test_inputs_are_not_mutated() -> None:
    scores = _scores()
    prices = _prices()
    scores_before = scores.copy(deep=True)
    prices_before = prices.copy(deep=True)
    _builder().build(scores, prices)
    pdt.assert_frame_equal(scores, scores_before)
    pdt.assert_frame_equal(prices, prices_before)


def test_entry_lag_zero_uses_score_date_and_holding_period_calendar_position() -> None:
    scores = pd.DataFrame({"trade_date": ["2024-01-05"], "ts_code": ["A"]})
    result = _builder(entry_lag_periods=0, holding_periods=2).build(
        scores, _prices()
    )
    assert result.loc[0, "entry_trade_date"] == pd.Timestamp("2024-01-05")
    assert result.loc[0, "exit_trade_date"] == pd.Timestamp("2024-01-09")
    assert result.loc[0, "entry_price"] == 100.0
    assert result.loc[0, "exit_price"] == 110.0


def test_default_style_lag_uses_next_market_date_not_weekend() -> None:
    scores = pd.DataFrame({"trade_date": ["2024-01-05"], "ts_code": ["A"]})
    result = _builder(entry_lag_periods=1, holding_periods=1).build(
        scores, _prices()
    )
    assert result.loc[0, "entry_trade_date"] == pd.Timestamp("2024-01-08")
    assert result.loc[0, "exit_trade_date"] == pd.Timestamp("2024-01-09")


def test_input_order_does_not_change_calendar_or_output() -> None:
    baseline = _builder().build(_scores(), _prices())
    shuffled = _builder().build(
        _scores().sample(frac=1.0, random_state=2),
        _prices().sample(frac=1.0, random_state=3),
    )
    pdt.assert_frame_equal(baseline, shuffled)


def test_unified_market_calendar_does_not_use_stock_available_rows() -> None:
    prices = _prices()
    prices = prices.loc[
        ~(
            prices["ts_code"].eq("A")
            & prices["trade_date"].eq("2024-01-09")
        )
    ]
    scores = pd.DataFrame({"trade_date": ["2024-01-05"], "ts_code": ["A"]})
    result = _builder(entry_lag_periods=1, holding_periods=1).build(scores, prices)
    assert result.loc[0, "entry_trade_date"] == pd.Timestamp("2024-01-08")
    assert result.loc[0, "exit_trade_date"] == pd.Timestamp("2024-01-09")
    assert np.isnan(result.loc[0, "exit_price"])
    assert np.isnan(result.loc[0, "forward_return"])


def test_entry_out_of_range_leaves_all_future_fields_missing() -> None:
    scores = pd.DataFrame({"trade_date": ["2024-01-12"], "ts_code": ["A"]})
    result = _builder(entry_lag_periods=1, holding_periods=1).build(
        scores, _prices()
    )
    assert pd.isna(result.loc[0, "entry_trade_date"])
    assert pd.isna(result.loc[0, "exit_trade_date"])
    assert np.isnan(result.loc[0, "entry_price"])
    assert np.isnan(result.loc[0, "exit_price"])
    assert np.isnan(result.loc[0, "forward_return"])


def test_exit_out_of_range_preserves_available_entry_audit() -> None:
    scores = pd.DataFrame({"trade_date": ["2024-01-11"], "ts_code": ["A"]})
    result = _builder(entry_lag_periods=1, holding_periods=1).build(
        scores, _prices()
    )
    assert result.loc[0, "entry_trade_date"] == pd.Timestamp("2024-01-12")
    assert result.loc[0, "entry_price"] == 125.0
    assert pd.isna(result.loc[0, "exit_trade_date"])
    assert np.isnan(result.loc[0, "exit_price"])
    assert np.isnan(result.loc[0, "forward_return"])


def test_manual_return_formula_and_stock_specific_prices() -> None:
    scores = pd.DataFrame(
        {
            "trade_date": ["2024-01-05", "2024-01-05"],
            "ts_code": ["A", "B"],
        }
    )
    result = _builder(entry_lag_periods=1, holding_periods=2).build(
        scores, _prices()
    )
    expected_a = 115.0 / 105.0 - 1.0
    expected_b = 125.0 / 115.0 - 1.0
    assert result.loc[result["ts_code"] == "A", "forward_return"].iloc[0] == pytest.approx(
        expected_a
    )
    assert result.loc[result["ts_code"] == "B", "forward_return"].iloc[0] == pytest.approx(
        expected_b
    )


def test_custom_price_and_return_columns() -> None:
    scores = pd.DataFrame({"trade_date": ["2024-01-05"], "ts_code": ["A"]})
    result = _builder(
        price_col="open",
        return_col="label",
        entry_lag_periods=1,
        holding_periods=1,
    ).build(scores, _prices())
    assert "label" in result.columns
    assert "forward_return" not in result.columns
    assert result.loc[0, "entry_price"] == 94.0
    assert result.loc[0, "exit_price"] == 98.0
    assert result.loc[0, "label"] == pytest.approx(98.0 / 94.0 - 1.0)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("entry", np.nan),
        ("exit", np.nan),
        ("entry", "bad"),
        ("exit", np.inf),
    ],
)
def test_invalid_entry_or_exit_price_keeps_return_missing(
    target: str, value: object
) -> None:
    prices = _prices()
    prices["close"] = prices["close"].astype(object)
    target_date = "2024-01-08" if target == "entry" else "2024-01-09"
    prices.loc[
        prices["ts_code"].eq("A") & prices["trade_date"].eq(target_date), "close"
    ] = value
    scores = pd.DataFrame({"trade_date": ["2024-01-05"], "ts_code": ["A"]})
    result = _builder(entry_lag_periods=1, holding_periods=1).build(scores, prices)
    assert np.isnan(result.loc[0, "forward_return"])
    assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()


@pytest.mark.parametrize(("target", "value"), [("entry", 0.0), ("exit", -1.0)])
def test_nonpositive_prices_are_invalid_by_default(
    target: str, value: float
) -> None:
    prices = _prices()
    target_date = "2024-01-08" if target == "entry" else "2024-01-09"
    prices.loc[
        prices["ts_code"].eq("A") & prices["trade_date"].eq(target_date), "close"
    ] = value
    scores = pd.DataFrame({"trade_date": ["2024-01-05"], "ts_code": ["A"]})
    result = _builder(entry_lag_periods=1, holding_periods=1).build(scores, prices)
    assert np.isnan(result.loc[0, "forward_return"])


def test_negative_prices_can_be_used_only_when_policy_disabled() -> None:
    prices = _prices()
    prices.loc[
        prices["ts_code"].eq("A")
        & prices["trade_date"].isin(["2024-01-08", "2024-01-09"]),
        "close",
    ] = [-100.0, -110.0]
    scores = pd.DataFrame({"trade_date": ["2024-01-05"], "ts_code": ["A"]})
    result = _builder(
        entry_lag_periods=1,
        holding_periods=1,
        require_positive_prices=False,
    ).build(scores, prices)
    assert result.loc[0, "forward_return"] == pytest.approx(0.1)


def test_zero_entry_cannot_produce_infinity_when_positive_policy_disabled() -> None:
    prices = _prices()
    prices.loc[
        prices["ts_code"].eq("A") & prices["trade_date"].eq("2024-01-08"),
        "close",
    ] = 0.0
    scores = pd.DataFrame({"trade_date": ["2024-01-05"], "ts_code": ["A"]})
    result = _builder(
        entry_lag_periods=1,
        holding_periods=1,
        require_positive_prices=False,
    ).build(scores, prices)
    assert np.isnan(result.loc[0, "forward_return"])


def test_missing_price_is_not_filled_or_shifted() -> None:
    prices = _prices()
    prices = prices.loc[
        ~(
            prices["ts_code"].eq("A")
            & prices["trade_date"].isin(["2024-01-08", "2024-01-09"])
        )
    ]
    scores = pd.DataFrame({"trade_date": ["2024-01-05"], "ts_code": ["A"]})
    result = _builder(entry_lag_periods=1, holding_periods=1).build(scores, prices)
    assert result.loc[0, "entry_trade_date"] == pd.Timestamp("2024-01-08")
    assert result.loc[0, "exit_trade_date"] == pd.Timestamp("2024-01-09")
    assert np.isnan(result.loc[0, "entry_price"])
    assert np.isnan(result.loc[0, "exit_price"])
    assert np.isnan(result.loc[0, "forward_return"])


def test_one_stock_missing_price_does_not_affect_another() -> None:
    scores = pd.DataFrame(
        {"trade_date": ["2024-01-05", "2024-01-05"], "ts_code": ["A", "B"]}
    )
    baseline = _builder(entry_lag_periods=1, holding_periods=1).build(
        scores, _prices()
    )
    prices = _prices()
    prices.loc[
        prices["ts_code"].eq("A") & prices["trade_date"].eq("2024-01-08"),
        "close",
    ] = np.nan
    result = _builder(entry_lag_periods=1, holding_periods=1).build(scores, prices)
    assert np.isnan(result.loc[result["ts_code"] == "A", "forward_return"]).all()
    pdt.assert_series_equal(
        result.loc[result["ts_code"] == "B", "forward_return"].reset_index(drop=True),
        baseline.loc[baseline["ts_code"] == "B", "forward_return"].reset_index(
            drop=True
        ),
    )


def test_output_contract_row_basis_sorting_and_dtypes() -> None:
    scores = _scores()
    result = _builder().build(scores, _prices())
    assert list(result.columns) == [
        "trade_date",
        "ts_code",
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
        "forward_return",
    ]
    assert len(result) == len(scores)
    assert list(result["ts_code"]) == ["A", "B", "B"]
    assert ptypes.is_datetime64_any_dtype(result["trade_date"])
    assert ptypes.is_datetime64_any_dtype(result["entry_trade_date"])
    assert ptypes.is_datetime64_any_dtype(result["exit_trade_date"])
    for column in ("entry_price", "exit_price", "forward_return"):
        assert ptypes.is_float_dtype(result[column])
    assert "ignored_factor" not in result.columns


def test_empty_score_panel_returns_typed_empty_output() -> None:
    result = _builder().build(_scores().iloc[:0], _prices())
    assert result.empty
    assert list(result.columns) == [
        "trade_date",
        "ts_code",
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
        "forward_return",
    ]
    assert ptypes.is_datetime64_any_dtype(result["trade_date"])
    assert ptypes.is_float_dtype(result["forward_return"])


def test_extra_price_records_never_add_output_rows() -> None:
    score = pd.DataFrame({"trade_date": ["2024-01-05"], "ts_code": ["A"]})
    result = _builder().build(score, _prices(codes=("A", "B", "C")))
    assert len(result) == 1
    assert result.loc[0, "ts_code"] == "A"


def _single_label(
    prices: pd.DataFrame,
    score_date: str = "2024-01-05",
) -> float:
    score = pd.DataFrame({"trade_date": [score_date], "ts_code": ["A"]})
    return float(
        _builder(entry_lag_periods=1, holding_periods=1)
        .build(score, prices)
        .loc[0, "forward_return"]
    )


def test_prices_before_score_and_after_exit_do_not_change_label() -> None:
    baseline = _single_label(_prices(), score_date="2024-01-08")
    changed = _prices()
    changed.loc[changed["trade_date"].eq("2024-01-05"), "close"] = 9999.0
    changed.loc[changed["trade_date"].gt("2024-01-10"), "close"] = -9999.0
    assert _single_label(changed, score_date="2024-01-08") == baseline


def test_entry_and_exit_price_changes_change_label() -> None:
    baseline = _single_label(_prices())
    entry_changed = _prices()
    entry_changed.loc[
        entry_changed["ts_code"].eq("A")
        & entry_changed["trade_date"].eq("2024-01-08"),
        "close",
    ] = 200.0
    exit_changed = _prices()
    exit_changed.loc[
        exit_changed["ts_code"].eq("A")
        & exit_changed["trade_date"].eq("2024-01-09"),
        "close",
    ] = 200.0
    assert _single_label(entry_changed) != baseline
    assert _single_label(exit_changed) != baseline


def test_other_stock_price_changes_do_not_change_label() -> None:
    baseline = _single_label(_prices())
    changed = _prices()
    changed.loc[changed["ts_code"].eq("B"), "close"] = 9999.0
    assert _single_label(changed) == baseline


def test_appending_dates_after_determined_exit_does_not_change_label() -> None:
    baseline = _single_label(_prices())
    extended = pd.concat(
        [_prices(), _prices(dates=["2024-01-15", "2024-01-16"])],
        ignore_index=True,
    )
    assert _single_label(extended) == baseline


def test_extending_calendar_can_make_terminal_label_available() -> None:
    score = pd.DataFrame({"trade_date": ["2024-01-11"], "ts_code": ["A"]})
    initial = _builder(entry_lag_periods=1, holding_periods=1).build(
        score, _prices()
    )
    assert np.isnan(initial.loc[0, "forward_return"])
    extended = pd.concat(
        [_prices(), _prices(dates=["2024-01-15"])],
        ignore_index=True,
    )
    result = _builder(entry_lag_periods=1, holding_periods=1).build(score, extended)
    assert result.loc[0, "entry_trade_date"] == pd.Timestamp("2024-01-12")
    assert result.loc[0, "exit_trade_date"] == pd.Timestamp("2024-01-15")
    assert np.isfinite(result.loc[0, "forward_return"])


def test_evaluation_and_quantile_integration() -> None:
    dates = pd.date_range("2024-02-01", periods=6)
    codes = [f"S{i:02d}" for i in range(10)]
    price_rows = []
    factor_rows = []
    for date_index, date in enumerate(dates):
        for stock_index, code in enumerate(codes):
            growth = 0.005 * (stock_index + 1)
            price_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "close": 100.0 * (1.0 + growth) ** date_index,
                }
            )
            if date_index < 4:
                factor_rows.append(
                    {
                        "trade_date": date,
                        "ts_code": code,
                        "factor": float(stock_index),
                    }
                )
    factors = pd.DataFrame(factor_rows)
    factor_before = factors.copy(deep=True)
    labels = ForwardReturnBuilder(
        ForwardReturnConfig(entry_lag_periods=1, holding_periods=1)
    ).build(factors[["trade_date", "ts_code"]], pd.DataFrame(price_rows))
    evaluator = FactorEvaluator(FactorEvaluationConfig(min_cross_section_size=5))
    ic = evaluator.evaluate_ic(factors, labels, ["factor"])
    quantile_evaluator = FactorQuantileEvaluator(
        QuantileEvaluationConfig(
            quantiles=5, min_cross_section_size=5, min_group_size=1
        )
    )
    quantiles = quantile_evaluator.evaluate_quantiles(
        factors, labels, ["factor"]
    )
    long_short = quantile_evaluator.evaluate_long_short(quantiles)
    pdt.assert_frame_equal(factors, factor_before)
    assert "forward_return" not in factors.columns
    assert labels["entry_trade_date"].notna().all()
    assert labels["exit_trade_date"].notna().all()
    assert np.isfinite(labels["forward_return"]).all()
    assert np.isfinite(ic["ic"]).all()
    assert np.isfinite(ic["rank_ic"]).all()
    assert np.isfinite(quantiles["group_return"].dropna()).all()
    assert np.isfinite(long_short["long_short_return"]).all()
    assert not np.isinf(
        labels.select_dtypes(include=[np.number])
    ).any().any()
