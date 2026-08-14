"""Central security identifier contracts for provider references and trading."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

import pandas as pd

from src.data.contracts import canonical_date


CANONICAL_SECURITY_PATTERN = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
CANONICAL_SECURITY_RULE_ID = "TS_CODE_6_DIGIT_CN_EXCHANGE_SUFFIX"
LEGACY_REFERENCE_RULE_ID = "LEGACY_REFERENCE_OUTSIDE_REQUIRED_INTERVAL"
UNSUPPORTED_LEGACY_RULE_ID = "UNSUPPORTED_LEGACY_SECURITY_IDENTIFIER"
INVALID_REFERENCE_RULE_ID = "NONCANONICAL_REFERENCE_IDENTITY_UNPROVEN"


class SecurityIdentifierClass(str, Enum):
    CANONICAL_TRADABLE = "CANONICAL_TRADABLE"
    LEGACY_REFERENCE = "LEGACY_REFERENCE"
    INVALID = "INVALID"


@dataclass(frozen=True)
class SecurityIdentifierDecision:
    classification: SecurityIdentifierClass
    rule_id: str
    reason: str


def classify_provider_reference_identifier(
    *,
    ts_code: object,
    list_status: object,
    list_date: object,
    delist_date: object,
    required_start: str | None,
    required_end: str | None,
) -> SecurityIdentifierDecision:
    """Classify without inventing a mapping from a provider reference identity."""
    code = str(ts_code) if ts_code is not None and not pd.isna(ts_code) else ""
    if CANONICAL_SECURITY_PATTERN.fullmatch(code):
        return SecurityIdentifierDecision(
            SecurityIdentifierClass.CANONICAL_TRADABLE,
            CANONICAL_SECURITY_RULE_ID,
            "canonical tradable security identifier",
        )
    status = str(list_status) if list_status is not None and not pd.isna(list_status) else ""
    try:
        listed = canonical_date(list_date)
        delisted = canonical_date(delist_date)
        start = canonical_date(required_start) if required_start is not None else None
        end = canonical_date(required_end) if required_end is not None else None
    except (TypeError, ValueError):
        listed = delisted = start = end = None
    if (
        status == "D"
        and listed is not None
        and delisted is not None
        and listed <= delisted
        and start is not None
        and end is not None
    ):
        if delisted <= start or listed > end:
            return SecurityIdentifierDecision(
                SecurityIdentifierClass.LEGACY_REFERENCE,
                LEGACY_REFERENCE_RULE_ID,
                "delisted provider reference does not overlap the complete required interval",
            )
        return SecurityIdentifierDecision(
            SecurityIdentifierClass.INVALID,
            UNSUPPORTED_LEGACY_RULE_ID,
            "legacy provider reference overlaps the complete required interval without a verified tradable mapping",
        )
    return SecurityIdentifierDecision(
        SecurityIdentifierClass.INVALID,
        INVALID_REFERENCE_RULE_ID,
        "non-canonical provider reference lacks D lifecycle dates proving it is outside the complete required interval",
    )
