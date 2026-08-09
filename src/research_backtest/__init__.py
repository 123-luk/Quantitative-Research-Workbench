"""Public semantic contracts for the canonical V6 research backtest."""

from src.research_backtest.benchmark_calendar import (
    BenchmarkCalendarAlignmentError,
    BenchmarkCalendarError,
    validate_strict_common_calendar,
)
from src.research_backtest.calendar import (
    TradingCalendar,
    TradingCalendarCoverageError,
    TradingCalendarDataError,
    TradingCalendarError,
    TradingCalendarProviderError,
    TushareTradingCalendarAdapter,
)

from src.research_backtest.contracts import (
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

__all__ = [
    "BenchmarkCalendarAlignmentError",
    "BenchmarkCalendarError",
    "RESEARCH_BACKTEST_BENCHMARK_ALIGNMENT_POLICIES",
    "RESEARCH_BACKTEST_COST_RATE_BASES",
    "RESEARCH_BACKTEST_EFFECTIVE_DATE_RULES",
    "RESEARCH_BACKTEST_RETURN_CONVENTIONS",
    "RESEARCH_BACKTEST_SCHEDULE_MODES",
    "RESEARCH_BACKTEST_SOURCE_MODES",
    "RESEARCH_BACKTEST_TURNOVER_DEFINITIONS",
    "ResearchBacktestContractError",
    "TradingCalendar",
    "TradingCalendarCoverageError",
    "TradingCalendarDataError",
    "TradingCalendarError",
    "TradingCalendarProviderError",
    "TushareTradingCalendarAdapter",
    "validate_benchmark_alignment_policy",
    "validate_cost_rate_basis",
    "validate_effective_date_rule",
    "validate_return_convention",
    "validate_schedule_mode",
    "validate_source_mode",
    "validate_turnover_definition",
    "validate_strict_common_calendar",
]
