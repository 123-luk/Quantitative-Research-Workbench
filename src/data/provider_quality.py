"""Provider-neutral canonical validation and cross-provider comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.data.canonical_store import normalize_frame
from src.data.contracts import DatasetSpec, IdentifierContract
from src.data.security_identifiers import (
    CANONICAL_SECURITY_PATTERN,
    CANONICAL_SECURITY_RULE_ID,
    INVALID_REFERENCE_RULE_ID,
    LEGACY_REFERENCE_RULE_ID,
    UNSUPPORTED_LEGACY_RULE_ID,
    SecurityIdentifierClass,
    classify_provider_reference_identifier,
)


SECURITY_IDENTIFIER_RULE_ID = CANONICAL_SECURITY_RULE_ID
SECURITY_IDENTIFIER_PATTERN = CANONICAL_SECURITY_PATTERN.pattern
MAX_INVALID_IDENTIFIER_SAMPLES = 20
_IDENTIFIER_SAMPLE_FIELDS = (
    "ts_code", "symbol", "list_status", "market", "exchange",
    "raw_ts_code", "normalized_ts_code", "list_date", "delist_date",
    "classification", "decision_reason", "required_start", "required_end",
    "rule_id",
)
_IDENTIFIER_RULE_IDS = frozenset({
    CANONICAL_SECURITY_RULE_ID,
    LEGACY_REFERENCE_RULE_ID,
    UNSUPPORTED_LEGACY_RULE_ID,
    INVALID_REFERENCE_RULE_ID,
})


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
    return ~rows["ts_code"].astype("string").map(
        lambda value: bool(CANONICAL_SECURITY_PATTERN.fullmatch(str(value)))
        if not pd.isna(value) else False
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
                if field in item
            }
            if sample.get("rule_id") not in _IDENTIFIER_RULE_IDS:
                continue
            samples.append(sample)
    samples = sorted(
        {
            tuple(item.get(field) for field in _IDENTIFIER_SAMPLE_FIELDS): item
            for item in samples
        }.values(),
        key=lambda item: tuple(
            str(item.get(field) or "") for field in _IDENTIFIER_SAMPLE_FIELDS
        ),
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
    excluded_count = count("excluded_count")
    raw_reason_counts = value.get("reason_counts")
    reason_counts = {
        rule_id: raw_reason_counts.get(rule_id, 0)
        for rule_id in sorted(_IDENTIFIER_RULE_IDS)
        if isinstance(raw_reason_counts, Mapping)
        and type(raw_reason_counts.get(rule_id, 0)) is int
        and raw_reason_counts.get(rule_id, 0) > 0
    }
    if not reason_counts and invalid_count:
        reason_counts = {SECURITY_IDENTIFIER_RULE_ID: invalid_count}
    raw_classification_counts = value.get("classification_counts")
    classification_counts = {
        classification.value: raw_classification_counts.get(classification.value, 0)
        for classification in SecurityIdentifierClass
        if isinstance(raw_classification_counts, Mapping)
        and type(raw_classification_counts.get(classification.value, 0)) is int
        and raw_classification_counts.get(classification.value, 0) >= 0
    }
    return {
        "invalid_count": invalid_count,
        "excluded_count": excluded_count,
        "reason_counts": reason_counts,
        "classification_counts": classification_counts,
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
    required_start: str | None = None,
    required_end: str | None = None,
) -> dict[str, object]:
    """Build bounded public identifier evidence for a merged stock_basic frame."""
    mask = invalid_security_code_mask(rows)
    noncanonical = rows.loc[mask].copy()
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
    invalid_count = 0
    excluded_count = 0
    reason_counts: dict[str, int] = {}
    classification_counts = {
        SecurityIdentifierClass.CANONICAL_TRADABLE.value: int((~mask).sum()),
        SecurityIdentifierClass.LEGACY_REFERENCE.value: 0,
        SecurityIdentifierClass.INVALID.value: 0,
    }
    for row in noncanonical.itertuples(index=False):
        normalized_code = _safe_identifier_text(getattr(row, "ts_code", None))
        status = _safe_identifier_text(getattr(row, "list_status", None))
        decision = classify_provider_reference_identifier(
            ts_code=normalized_code,
            list_status=status,
            list_date=getattr(row, "list_date", None),
            delist_date=getattr(row, "delist_date", None),
            required_start=required_start,
            required_end=required_end,
        )
        classification_counts[decision.classification.value] += 1
        reason_counts[decision.rule_id] = reason_counts.get(decision.rule_id, 0) + 1
        if decision.classification is SecurityIdentifierClass.LEGACY_REFERENCE:
            excluded_count += 1
        elif decision.classification is SecurityIdentifierClass.INVALID:
            invalid_count += 1
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
            "list_date": _safe_identifier_text(getattr(row, "list_date", None)),
            "delist_date": _safe_identifier_text(getattr(row, "delist_date", None)),
            "classification": decision.classification.value,
            "decision_reason": decision.reason,
            "required_start": required_start,
            "required_end": required_end,
            "rule_id": decision.rule_id,
        })
    return sanitize_identifier_evidence({
        "invalid_count": invalid_count,
        "excluded_count": excluded_count,
        "reason_counts": reason_counts,
        "classification_counts": classification_counts,
        "samples": samples,
        "status_row_counts": dict(status_row_counts),
        "pre_merge_rows": pre_merge_rows,
        "merged_rows": len(rows),
        "deduplicated_rows": max(0, pre_merge_rows - len(rows)),
    })


def validate_quality(
    spec: DatasetSpec,
    frame: pd.DataFrame,
    *,
    required_start: str | None = None,
    required_end: str | None = None,
) -> tuple[QualityIssue, ...]:
    rows = normalize_frame(spec, frame)
    issues: list[QualityIssue] = []
    if spec.identifier_contract is IdentifierContract.CANONICAL_TRADABLE and invalid_security_code_mask(rows).any():
        issues.append(QualityIssue("INVALID_SECURITY_CODE", "ts_code", None))
    elif spec.identifier_contract is IdentifierContract.PROVIDER_REFERENCE:
        decisions = tuple(
            classify_provider_reference_identifier(
                ts_code=getattr(row, "ts_code", None),
                list_status=getattr(row, "list_status", None),
                list_date=getattr(row, "list_date", None),
                delist_date=getattr(row, "delist_date", None),
                required_start=required_start,
                required_end=required_end,
            )
            for row in rows.loc[invalid_security_code_mask(rows)].itertuples(index=False)
        )
        if any(item.rule_id == UNSUPPORTED_LEGACY_RULE_ID for item in decisions):
            issues.append(QualityIssue(UNSUPPORTED_LEGACY_RULE_ID, "ts_code", None))
        if any(
            item.classification is SecurityIdentifierClass.INVALID
            and item.rule_id != UNSUPPORTED_LEGACY_RULE_ID
            for item in decisions
        ):
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
