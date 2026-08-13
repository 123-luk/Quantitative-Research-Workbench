"""Provider-neutral canonical validation and cross-provider comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.canonical_store import normalize_frame
from src.data.contracts import DatasetSpec


@dataclass(frozen=True)
class QualityIssue:
    category: str
    field: str | None
    key: str | None


@dataclass(frozen=True)
class ProviderComparison:
    consistent: bool
    issues: tuple[QualityIssue, ...]
    left_rows: int
    right_rows: int
    absolute_tolerance: float


def validate_quality(spec: DatasetSpec, frame: pd.DataFrame) -> tuple[QualityIssue, ...]:
    rows = normalize_frame(spec, frame)
    issues: list[QualityIssue] = []
    if "ts_code" in rows and not rows["ts_code"].str.match(r"^\d{6}\.(SH|SZ|BJ)$", na=False).all():
        issues.append(QualityIssue("INVALID_SECURITY_CODE", "ts_code", None))
    if {"open", "high", "low", "close"}.issubset(rows):
        invalid = (rows["high"] < rows[["open", "close", "low"]].max(axis=1)) | (rows["low"] > rows[["open", "close", "high"]].min(axis=1))
        if invalid.any():
            issues.append(QualityIssue("OHLC_INVARIANT", "OHLC", None))
    for field in ("vol", "amount"):
        if field in rows and rows[field].dropna().lt(0).any():
            issues.append(QualityIssue("NEGATIVE_VALUE", field, None))
    if "adj_factor" in rows and rows["adj_factor"].dropna().le(0).any():
        issues.append(QualityIssue("NON_POSITIVE_ADJUSTMENT_FACTOR", "adj_factor", None))
    if spec.provider_row_limit is not None and len(rows) > spec.provider_row_limit:
        issues.append(QualityIssue("ROW_LIMIT_EXCEEDED", None, None))
    return tuple(issues)


def compare_providers(
    spec: DatasetSpec,
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    absolute_tolerance: float = 1e-8,
) -> ProviderComparison:
    if absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    issues: list[QualityIssue] = []
    if set(left.columns) != set(right.columns):
        issues.append(QualityIssue("FIELD_SET_MISMATCH", None, None))
    try:
        a, b = normalize_frame(spec, left), normalize_frame(spec, right)
    except Exception:
        return ProviderComparison(False, (QualityIssue("SCHEMA_INVALID", None, None),), len(left), len(right), absolute_tolerance)
    if len(a) != len(b):
        issues.append(QualityIssue("ROW_COUNT_MISMATCH", None, None))
    key = list(spec.primary_key)
    a_keys = set(map(tuple, a[key].itertuples(index=False, name=None)))
    b_keys = set(map(tuple, b[key].itertuples(index=False, name=None)))
    if a_keys != b_keys:
        issues.append(QualityIssue("PRIMARY_KEY_SET_MISMATCH", None, None))
    shared = a_keys & b_keys
    ai, bi = a.set_index(key), b.set_index(key)
    for item in sorted(shared):
        row_a = ai.loc[item]
        row_b = bi.loc[item]
        key_text = "|".join(map(str, item))
        for field in (name for name in spec.required_fields if name not in key):
            va, vb = row_a[field], row_b[field]
            if pd.isna(va) and pd.isna(vb):
                continue
            if pd.api.types.is_number(va) and pd.api.types.is_number(vb):
                equal = bool(np.isclose(float(va), float(vb), rtol=0.0, atol=absolute_tolerance, equal_nan=True))
            else:
                equal = va == vb
            if not equal:
                issues.append(QualityIssue("VALUE_MISMATCH", field, key_text))
    return ProviderComparison(not issues, tuple(issues), len(a), len(b), absolute_tolerance)
