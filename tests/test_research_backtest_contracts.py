"""Tests for canonical V6 research-backtest semantic contracts."""

from __future__ import annotations

import pytest

import src.research_backtest as contracts
from src.research_backtest import (
    RESEARCH_BACKTEST_BENCHMARK_ALIGNMENT_POLICIES,
    RESEARCH_BACKTEST_COST_RATE_BASES,
    RESEARCH_BACKTEST_EFFECTIVE_DATE_RULES,
    RESEARCH_BACKTEST_RETURN_CONVENTIONS,
    RESEARCH_BACKTEST_SCHEDULE_MODES,
    RESEARCH_BACKTEST_SOURCE_MODES,
    RESEARCH_BACKTEST_TURNOVER_DEFINITIONS,
    ResearchBacktestContractError,
    validate_benchmark_alignment_policy,
    validate_cost_rate_basis,
    validate_effective_date_rule,
    validate_return_convention,
    validate_schedule_mode,
    validate_source_mode,
    validate_turnover_definition,
)


def test_contract_allowlists_are_exact_immutable_tuples() -> None:
    expected = {
        "source": (RESEARCH_BACKTEST_SOURCE_MODES, ("pipeline", "files")),
        "schedule": (RESEARCH_BACKTEST_SCHEDULE_MODES, ("holdings_dates",)),
        "effective": (
            RESEARCH_BACKTEST_EFFECTIVE_DATE_RULES,
            ("next_trading_day",),
        ),
        "return": (
            RESEARCH_BACKTEST_RETURN_CONVENTIONS,
            ("adjusted_close_to_close",),
        ),
        "turnover": (
            RESEARCH_BACKTEST_TURNOVER_DEFINITIONS,
            ("half_l1_pre_to_target",),
        ),
        "cost": (
            RESEARCH_BACKTEST_COST_RATE_BASES,
            ("one_way_traded_notional",),
        ),
        "benchmark": (
            RESEARCH_BACKTEST_BENCHMARK_ALIGNMENT_POLICIES,
            ("strict_common_calendar",),
        ),
    }
    for actual, frozen in expected.values():
        assert actual == frozen
        assert isinstance(actual, tuple)


@pytest.mark.parametrize(
    ("validator", "value", "expected"),
    [
        (validate_source_mode, " PIPELINE ", "pipeline"),
        (validate_source_mode, "files", "files"),
        (validate_schedule_mode, "HOLDINGS_DATES", "holdings_dates"),
        (validate_effective_date_rule, "NEXT_TRADING_DAY", "next_trading_day"),
        (
            validate_return_convention,
            "ADJUSTED_CLOSE_TO_CLOSE",
            "adjusted_close_to_close",
        ),
        (
            validate_turnover_definition,
            "HALF_L1_PRE_TO_TARGET",
            "half_l1_pre_to_target",
        ),
        (
            validate_cost_rate_basis,
            "ONE_WAY_TRADED_NOTIONAL",
            "one_way_traded_notional",
        ),
        (
            validate_benchmark_alignment_policy,
            "STRICT_COMMON_CALENDAR",
            "strict_common_calendar",
        ),
    ],
)
def test_contract_validators_return_exact_values(
    validator: object, value: str, expected: str
) -> None:
    assert validator(value) == expected  # type: ignore[operator]


@pytest.mark.parametrize("value", ["monthly", "weekly", "daily", "custom"])
def test_schedule_rejects_frequency_behavior(value: str) -> None:
    with pytest.raises(ResearchBacktestContractError):
        validate_schedule_mode(value)


@pytest.mark.parametrize(
    ("validator", "value"),
    [
        (validate_source_mode, "latest"),
        (validate_effective_date_rule, "same_day"),
        (validate_effective_date_rule, "next_open"),
        (validate_return_convention, "raw_close"),
        (validate_return_convention, "qfq"),
        (validate_return_convention, "hfq"),
        (validate_turnover_definition, "target_to_target"),
        (validate_cost_rate_basis, "round_trip"),
        (validate_benchmark_alignment_policy, "forward_fill"),
        (validate_source_mode, 1),
    ],
)
def test_contract_validators_reject_unsupported_values(
    validator: object, value: object
) -> None:
    with pytest.raises(ResearchBacktestContractError):
        validator(value)  # type: ignore[operator]


def test_package_has_no_frequency_contract() -> None:
    assert not hasattr(contracts, "ResearchFrequency")
    assert not hasattr(contracts, "BacktestFrequency")
    assert RESEARCH_BACKTEST_SCHEDULE_MODES == ("holdings_dates",)


def test_contract_package_has_no_engine_or_artifact_operations() -> None:
    for name in ("run", "calculate", "load", "write", "rebalance"):
        assert not hasattr(contracts, name)
