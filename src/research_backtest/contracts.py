"""Canonical semantic vocabulary for the V6 research portfolio backtest.

These contracts describe research accounting only. They deliberately leave
market-data providers, adjusted-price construction, and runtime calculations
to later V6 stages.
"""

from __future__ import annotations


RESEARCH_BACKTEST_SOURCE_MODES = ("pipeline", "files")
RESEARCH_BACKTEST_SCHEDULE_MODES = ("holdings_dates",)
RESEARCH_BACKTEST_EFFECTIVE_DATE_RULES = ("next_trading_day",)
RESEARCH_BACKTEST_RETURN_CONVENTIONS = ("adjusted_close_to_close",)
RESEARCH_BACKTEST_TURNOVER_DEFINITIONS = ("half_l1_pre_to_target",)
RESEARCH_BACKTEST_COST_RATE_BASES = ("one_way_traded_notional",)
RESEARCH_BACKTEST_BENCHMARK_ALIGNMENT_POLICIES = ("strict_common_calendar",)


class ResearchBacktestContractError(ValueError):
    """Raised when a research-backtest semantic declaration is unsupported."""


def _validate_choice(
    value: object, *, field_name: str, allowed: tuple[str, ...]
) -> str:
    if not isinstance(value, str):
        raise ResearchBacktestContractError(f"{field_name} must be a string.")
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise ResearchBacktestContractError(
            f"{field_name} must be one of {allowed!r}."
        )
    return normalized


def validate_source_mode(value: object) -> str:
    """Return an exact supported source mode."""
    return _validate_choice(
        value, field_name="source mode", allowed=RESEARCH_BACKTEST_SOURCE_MODES
    )


def validate_schedule_mode(value: object) -> str:
    """Enforce that ordered Holdings snapshots own the rebalance schedule."""
    return _validate_choice(
        value,
        field_name="schedule mode",
        allowed=RESEARCH_BACKTEST_SCHEDULE_MODES,
    )


def validate_effective_date_rule(value: object) -> str:
    """Return the canonical post-information-date effective rule."""
    return _validate_choice(
        value,
        field_name="effective date rule",
        allowed=RESEARCH_BACKTEST_EFFECTIVE_DATE_RULES,
    )


def validate_return_convention(value: object) -> str:
    """Validate economic return intent without choosing an adjustment provider."""
    return _validate_choice(
        value,
        field_name="return convention",
        allowed=RESEARCH_BACKTEST_RETURN_CONVENTIONS,
    )


def validate_turnover_definition(value: object) -> str:
    """Validate drifted pre-rebalance to target portfolio accounting semantics.

    The complete portfolio state includes its cash leg. Consequently, moving
    from 100 percent cash to a fully invested target is a full initial purchase,
    not an erroneous half-position turnover result.
    """
    return _validate_choice(
        value,
        field_name="turnover definition",
        allowed=RESEARCH_BACKTEST_TURNOVER_DEFINITIONS,
    )


def validate_cost_rate_basis(value: object) -> str:
    """Return the proportional research-friction notional convention."""
    return _validate_choice(
        value,
        field_name="cost rate basis",
        allowed=RESEARCH_BACKTEST_COST_RATE_BASES,
    )


def validate_benchmark_alignment_policy(value: object) -> str:
    """Return the strict strategy/benchmark calendar alignment policy."""
    return _validate_choice(
        value,
        field_name="benchmark alignment policy",
        allowed=RESEARCH_BACKTEST_BENCHMARK_ALIGNMENT_POLICIES,
    )
