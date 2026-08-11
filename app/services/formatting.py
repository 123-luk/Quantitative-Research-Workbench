"""Consistent presentation-only formatting for workbench views."""

from __future__ import annotations

import math


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def format_percentage(value: object, decimals: int = 2) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{number:.{decimals}%}"


def format_float(value: object, decimals: int = 2) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{number:.{decimals}f}"


def format_count(value: object) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{int(number):,}"


def format_bps(value: object, decimals: int = 1) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{number:.{decimals}f} bps"

