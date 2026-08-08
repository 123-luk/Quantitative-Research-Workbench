"""Tests for the separated V5 Signal and Holdings row contracts."""

from __future__ import annotations

import pytest

from src.holdings import (
    HOLDINGS_FORBIDDEN_OUTPUT_COLUMNS,
    HOLDINGS_KEY_COLUMNS,
    HOLDINGS_OUTPUT_COLUMNS,
    HOLDINGS_SCHEMA_VERSION,
    HoldingsContractError,
    validate_holdings_columns,
    validate_holdings_key_columns,
)
from src.signals import (
    SIGNAL_FORBIDDEN_OUTPUT_COLUMNS,
    SIGNAL_KEY_COLUMNS,
    SIGNAL_OUTPUT_COLUMNS,
    SIGNAL_SCHEMA_VERSION,
    SignalContractError,
    validate_signal_columns,
    validate_signal_key_columns,
)


def test_signal_canonical_contract_is_minimal_and_immutable() -> None:
    assert SIGNAL_SCHEMA_VERSION == "1.0"
    assert SIGNAL_KEY_COLUMNS == ("trade_date", "ts_code")
    assert SIGNAL_OUTPUT_COLUMNS == ("trade_date", "ts_code", "score", "rank")
    assert not SIGNAL_FORBIDDEN_OUTPUT_COLUMNS.intersection(SIGNAL_OUTPUT_COLUMNS)
    assert {"target", "y_true", "fold_id", "top_n", "selected"}.issubset(
        SIGNAL_FORBIDDEN_OUTPUT_COLUMNS
    )
    assert isinstance(SIGNAL_KEY_COLUMNS, tuple)
    assert isinstance(SIGNAL_OUTPUT_COLUMNS, tuple)
    assert isinstance(SIGNAL_FORBIDDEN_OUTPUT_COLUMNS, frozenset)


def test_holdings_canonical_contract_is_separate_and_immutable() -> None:
    assert HOLDINGS_SCHEMA_VERSION == "1.0"
    assert HOLDINGS_KEY_COLUMNS == ("trade_date", "ts_code")
    assert HOLDINGS_OUTPUT_COLUMNS == (
        "trade_date",
        "ts_code",
        "target_weight",
        "score",
        "rank",
    )
    assert "target_weight" in HOLDINGS_OUTPUT_COLUMNS
    assert "weight" not in HOLDINGS_OUTPUT_COLUMNS
    assert "selected" not in HOLDINGS_OUTPUT_COLUMNS
    assert HOLDINGS_FORBIDDEN_OUTPUT_COLUMNS == frozenset({"weight", "selected"})
    assert isinstance(HOLDINGS_OUTPUT_COLUMNS, tuple)


def test_contract_validators_return_exact_canonical_definitions() -> None:
    assert validate_signal_key_columns(SIGNAL_KEY_COLUMNS) == SIGNAL_KEY_COLUMNS
    assert validate_signal_columns(SIGNAL_OUTPUT_COLUMNS) == SIGNAL_OUTPUT_COLUMNS
    assert validate_holdings_key_columns(HOLDINGS_KEY_COLUMNS) == HOLDINGS_KEY_COLUMNS
    assert validate_holdings_columns(HOLDINGS_OUTPUT_COLUMNS) == HOLDINGS_OUTPUT_COLUMNS


@pytest.mark.parametrize(
    "columns",
    [
        ("trade_date", "ts_code", "score"),
        ("trade_date", "ts_code", "score", "rank", "selected"),
        ("trade_date", "ts_code", "score", "score"),
        "trade_date",
        ("trade_date", "", "score", "rank"),
    ],
)
def test_signal_validator_rejects_noncanonical_definitions(columns: object) -> None:
    with pytest.raises(SignalContractError):
        validate_signal_columns(columns)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "columns",
    [
        ("trade_date", "ts_code", "weight", "score", "rank"),
        ("trade_date", "ts_code", "target_weight", "score"),
        ("trade_date", "ts_code", "target_weight", "score", "selected"),
        ("trade_date", "ts_code", "target_weight", "score", "score"),
        None,
    ],
)
def test_holdings_validator_rejects_noncanonical_definitions(columns: object) -> None:
    with pytest.raises(HoldingsContractError):
        validate_holdings_columns(columns)  # type: ignore[arg-type]


def test_key_validators_reject_wrong_order_and_duplicates() -> None:
    with pytest.raises(SignalContractError):
        validate_signal_key_columns(("ts_code", "trade_date"))
    with pytest.raises(HoldingsContractError):
        validate_holdings_key_columns(("trade_date", "trade_date"))


def test_contract_packages_do_not_expose_business_operations() -> None:
    import src.holdings as holdings
    import src.signals as signals

    for module in (signals, holdings):
        for name in ("build", "rank", "select_top_n", "weight", "write", "load"):
            assert not hasattr(module, name)
