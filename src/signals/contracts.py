"""Canonical, persistence-independent contracts for V5 Signal rows."""

from __future__ import annotations

from collections.abc import Iterable


SIGNAL_SCHEMA_VERSION = "1.0"
SIGNAL_KEY_COLUMNS = ("trade_date", "ts_code")
SIGNAL_OUTPUT_COLUMNS = (*SIGNAL_KEY_COLUMNS, "score", "rank")
SIGNAL_FORBIDDEN_OUTPUT_COLUMNS = frozenset(
    {"target", "y_true", "fold_id", "top_n", "selected"}
)


class SignalContractError(ValueError):
    """Raised when a Signal schema declaration violates the contract."""


def _column_tuple(columns: Iterable[object], field_name: str) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes)):
        raise SignalContractError(f"{field_name} must be an iterable of strings.")
    try:
        values = tuple(columns)
    except TypeError as exc:
        raise SignalContractError(
            f"{field_name} must be an iterable of strings."
        ) from exc
    if any(not isinstance(value, str) or not value for value in values):
        raise SignalContractError(
            f"{field_name} must contain non-empty strings."
        )
    if len(values) != len(set(values)):
        raise SignalContractError(f"{field_name} must not contain duplicates.")
    return values


def validate_signal_key_columns(columns: Iterable[object]) -> tuple[str, ...]:
    """Return the canonical Signal key or reject a different declaration."""
    values = _column_tuple(columns, "Signal key columns")
    if values != SIGNAL_KEY_COLUMNS:
        raise SignalContractError(
            f"Signal key columns must be {SIGNAL_KEY_COLUMNS!r}."
        )
    return values


def validate_signal_columns(columns: Iterable[object]) -> tuple[str, ...]:
    """Return the canonical ordered Signal columns or reject the declaration."""
    values = _column_tuple(columns, "Signal output columns")
    if values != SIGNAL_OUTPUT_COLUMNS:
        raise SignalContractError(
            f"Signal output columns must be {SIGNAL_OUTPUT_COLUMNS!r}."
        )
    return values
