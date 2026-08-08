"""Canonical, persistence-independent contracts for V5 Holdings rows."""

from __future__ import annotations

from collections.abc import Iterable


HOLDINGS_SCHEMA_VERSION = "1.0"
HOLDINGS_KEY_COLUMNS = ("trade_date", "ts_code")
HOLDINGS_OUTPUT_COLUMNS = (
    *HOLDINGS_KEY_COLUMNS,
    "target_weight",
    "score",
    "rank",
)
HOLDINGS_FORBIDDEN_OUTPUT_COLUMNS = frozenset({"weight", "selected"})


class HoldingsContractError(ValueError):
    """Raised when a Holdings schema declaration violates the contract."""


class HoldingsDataError(HoldingsContractError):
    """Raised when canonical Signals cannot form valid Holdings."""


def _column_tuple(columns: Iterable[object], field_name: str) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes)):
        raise HoldingsContractError(
            f"{field_name} must be an iterable of strings."
        )
    try:
        values = tuple(columns)
    except TypeError as exc:
        raise HoldingsContractError(
            f"{field_name} must be an iterable of strings."
        ) from exc
    if any(not isinstance(value, str) or not value for value in values):
        raise HoldingsContractError(
            f"{field_name} must contain non-empty strings."
        )
    if len(values) != len(set(values)):
        raise HoldingsContractError(
            f"{field_name} must not contain duplicates."
        )
    return values


def validate_holdings_key_columns(columns: Iterable[object]) -> tuple[str, ...]:
    """Return the canonical Holdings key or reject a different declaration."""
    values = _column_tuple(columns, "Holdings key columns")
    if values != HOLDINGS_KEY_COLUMNS:
        raise HoldingsContractError(
            f"Holdings key columns must be {HOLDINGS_KEY_COLUMNS!r}."
        )
    return values


def validate_holdings_columns(columns: Iterable[object]) -> tuple[str, ...]:
    """Return canonical ordered Holdings columns or reject the declaration."""
    values = _column_tuple(columns, "Holdings output columns")
    if values != HOLDINGS_OUTPUT_COLUMNS:
        raise HoldingsContractError(
            f"Holdings output columns must be {HOLDINGS_OUTPUT_COLUMNS!r}."
        )
    return values
