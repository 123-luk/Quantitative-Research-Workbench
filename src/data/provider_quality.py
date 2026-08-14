"""Provider-neutral canonical validation and cross-provider comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.data.canonical_store import normalize_frame
from src.data.contracts import DatasetSpec


SECURITY_IDENTIFIER_RULE_ID = "TS_CODE_6_DIGIT_CN_EXCHANGE_SUFFIX"
SECURITY_IDENTIFIER_PATTERN = r"^[0-9]{6}\.(?:SH|SZ|BJ)$"
MAX_INVALID_IDENTIFIER_SAMPLES = 20
_IDENTIFIER_SAMPLE_FIELDS = (
    "ts_code", "symbol", "list_status", "market", "exchange",
    "raw_ts_code", "normalized_ts_code", "rule_id",
)


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


def invalid_security_code_mask(rows: pd.DataFrame) -> pd.Series:
    """Return the exact mask used by the canonical security identifier gate."""
    if "ts_code" not in rows:
        return pd.Series(False, index=rows.index, dtype=bool)
    return ~rows["ts_code"].astype("string").str.match(
        SECURITY_IDENTIFIER_PATTERN, na=False
    )


def _safe_identifier_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, int, float)):
        return None
    if pd.isna(value):
        return None
    return str(value)[:128]


def sanitize_identifier_evidence(value: object) -> dict[str, object]:
    """Whitelist and bound persisted quality evidence before storage/display."""
    if not isinstance(value, Mapping) or not value:
        return {}
    samples: list[dict[str, object]] = []
    raw_samples = value.get("samples")
    if isinstance(raw_samples, (list, tuple)):
        for item in raw_samples:
            if not isinstance(item, Mapping):
                continue
            sample = {
                field: _safe_identifier_text(item.get(field))
                for field in _IDENTIFIER_SAMPLE_FIELDS
            }
            if sample["rule_id"] != SECURITY_IDENTIFIER_RULE_ID:
                continue
            samples.append(sample)
    samples = sorted(
        {tuple(item[field] for field in _IDENTIFIER_SAMPLE_FIELDS): item for item in samples}.values(),
        key=lambda item: tuple(str(item[field] or "") for field in _IDENTIFIER_SAMPLE_FIELDS),
    )[:MAX_INVALID_IDENTIFIER_SAMPLES]

    def count(name: str) -> int:
        item = value.get(name)
        return item if type(item) is int and item >= 0 else 0

    raw_status_counts = value.get("status_row_counts")
    status_counts = {
        status: raw_status_counts.get(status, 0)
        if isinstance(raw_status_counts, Mapping)
        and type(raw_status_counts.get(status, 0)) is int
        and raw_status_counts.get(status, 0) >= 0
        else 0
        for status in ("L", "D", "P", "G")
    }
    invalid_count = count("invalid_count")
    return {
        "invalid_count": invalid_count,
        "reason_counts": {SECURITY_IDENTIFIER_RULE_ID: invalid_count},
        "samples": samples,
        "status_row_counts": status_counts,
        "pre_merge_rows": count("pre_merge_rows"),
        "merged_rows": count("merged_rows"),
        "deduplicated_rows": count("deduplicated_rows"),
    }


def invalid_security_identifier_evidence(
    rows: pd.DataFrame,
    *,
    raw_frames: Mapping[str, pd.DataFrame],
    status_row_counts: Mapping[str, int],
    pre_merge_rows: int,
) -> dict[str, object]:
    """Build bounded public identifier evidence for a merged stock_basic frame."""
    mask = invalid_security_code_mask(rows)
    invalid = rows.loc[mask].copy()
    raw_by_status_and_code: dict[tuple[str, str], str | None] = {}
    for status in ("L", "D", "P", "G"):
        frame = raw_frames.get(status)
        if not isinstance(frame, pd.DataFrame) or "ts_code" not in frame:
            continue
        ordered = sorted(
            (_safe_identifier_text(item) for item in frame["ts_code"].tolist()),
            key=lambda item: str(item or ""),
        )
        for raw_code in ordered:
            raw_by_status_and_code.setdefault((status, str(raw_code)), raw_code)

    samples: list[dict[str, object]] = []
    for row in invalid.itertuples(index=False):
        normalized_code = _safe_identifier_text(getattr(row, "ts_code", None))
        status = _safe_identifier_text(getattr(row, "list_status", None))
        samples.append({
            "ts_code": normalized_code,
            "symbol": _safe_identifier_text(getattr(row, "symbol", None)),
            "list_status": status,
            "market": _safe_identifier_text(getattr(row, "market", None)),
            "exchange": _safe_identifier_text(getattr(row, "exchange", None)),
            "raw_ts_code": raw_by_status_and_code.get(
                (str(status), str(normalized_code)), normalized_code
            ),
            "normalized_ts_code": normalized_code,
            "rule_id": SECURITY_IDENTIFIER_RULE_ID,
        })
    return sanitize_identifier_evidence({
        "invalid_count": int(mask.sum()),
        "samples": samples,
        "status_row_counts": dict(status_row_counts),
        "pre_merge_rows": pre_merge_rows,
        "merged_rows": len(rows),
        "deduplicated_rows": max(0, pre_merge_rows - len(rows)),
    })


def validate_quality(spec: DatasetSpec, frame: pd.DataFrame) -> tuple[QualityIssue, ...]:
    rows = normalize_frame(spec, frame)
    issues: list[QualityIssue] = []
    if spec.dataset_id not in {"index_daily"} and invalid_security_code_mask(rows).any():
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
