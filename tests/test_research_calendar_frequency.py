from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.contracts import ResearchFrequency
from src.factors import BP, DIVIDEND_YIELD_TTM, FactorDependencyPlanner, FactorFrequencyError, FactorFrequencySpec, FactorMetadata, FactorRegistry, FunctionFactor
from src.research_data import HistoryRequirement, ResearchCalendar, ResearchCalendarError


def calendar(start: str = "2023-12-20", end: str = "2024-03-05", *, holidays: tuple[str, ...] = ("2024-01-01", "2024-02-09")) -> ResearchCalendar:
    rows = []
    for day in pd.date_range(start, end, freq="D"):
        text = day.strftime("%Y-%m-%d")
        rows.append({"cal_date": text, "is_open": int(day.weekday() < 5 and text not in holidays)})
    return ResearchCalendar(pd.DataFrame(rows).sample(frac=1.0, random_state=7))


def test_daily_monthly_exact_open_dates_weekend_holiday_and_partial_months() -> None:
    research = calendar("2024-01-01", "2024-03-31")
    daily = research.formation_dates(ResearchFrequency.DAILY, "2024-01-30", "2024-02-12")
    assert daily == ("2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-12")
    monthly = research.formation_dates(ResearchFrequency.MONTHLY, "2024-01-02", "2024-03-03")
    assert monthly == ("2024-01-31", "2024-02-29")
    assert "2024-03-01" not in monthly  # A partial final month is not mislabeled month-end.


def test_calendar_coverage_duplicate_and_month_without_open_fail_closed() -> None:
    rows = pd.DataFrame([{"cal_date": "2024-01-01", "is_open": 0}, {"cal_date": "2024-01-02", "is_open": 1}])
    with pytest.raises(ResearchCalendarError, match="duplicate"):
        ResearchCalendar(pd.concat([rows, rows.iloc[[0]]], ignore_index=True))
    research = ResearchCalendar(rows)
    with pytest.raises(ResearchCalendarError, match="completely cover"):
        research.formation_dates(ResearchFrequency.DAILY, "2023-12-31", "2024-01-02")
    closed_month = ResearchCalendar(pd.DataFrame([{"cal_date": day.strftime("%Y-%m-%d"), "is_open": 0} for day in pd.date_range("2024-01-01", "2024-01-31")]))
    with pytest.raises(ResearchCalendarError, match="no open trading date"):
        closed_month.formation_dates(ResearchFrequency.MONTHLY, "2024-01-01", "2024-01-31")


def test_history_requirement_strict_parse_and_serialization() -> None:
    values = (HistoryRequirement.trading_days(3), HistoryRequirement.calendar_months(2), HistoryRequirement.latest_as_of())
    assert tuple(HistoryRequirement.from_dict(item.to_dict()) for item in values) == values
    for invalid in (0, -1, True):
        with pytest.raises(ValueError):
            HistoryRequirement.trading_days(invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown"):
        HistoryRequirement.from_dict({"kind": "CALENDAR_DAYS", "count": 3})


def test_trading_day_history_is_inclusive_and_uses_exact_open_calendar() -> None:
    research = calendar()
    window = research.resolve_history("2024-01-02", HistoryRequirement.trading_days(3))
    assert window.open_dates == ("2023-12-28", "2023-12-29", "2024-01-02")
    assert window.start_date == "2023-12-28" and window.end_date == "2024-01-02"
    with pytest.raises(ResearchCalendarError, match="open trading day"):
        research.resolve_history("2024-01-01", HistoryRequirement.trading_days(3))
    short = calendar("2024-01-01", "2024-01-03")
    with pytest.raises(ResearchCalendarError, match="insufficient"):
        short.resolve_history("2024-01-03", HistoryRequirement.trading_days(3))


def test_calendar_month_history_starts_at_first_open_of_earliest_included_month() -> None:
    research = calendar()
    window = research.resolve_history("2024-02-29", HistoryRequirement.calendar_months(2))
    assert window.start_date == "2024-01-02"
    assert window.end_date == "2024-02-29"
    assert research.resolve_history("2024-02-29", HistoryRequirement.latest_as_of()).open_dates == ("2024-02-29",)


def test_legacy_factor_is_daily_only_but_valuation_is_explicit_daily_and_monthly() -> None:
    legacy = FactorMetadata("legacy", "test", 1, required_datasets=("daily",), source_fields=("close",), lookback_days=2)
    assert legacy.frequency_spec(ResearchFrequency.DAILY).history_requirement == HistoryRequirement.trading_days(2)
    with pytest.raises(FactorFrequencyError, match="does not support"):
        legacy.frequency_spec(ResearchFrequency.MONTHLY)
    for frequency in ResearchFrequency:
        spec = BP.metadata.frequency_spec(frequency)
        assert spec.required_datasets == ("daily_basic",)
        assert spec.history_requirement == HistoryRequirement.latest_as_of()
        assert "no monthly averaging" in spec.observation_semantics
        assert FactorFrequencySpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(FactorFrequencyError, match="does not support"):
        DIVIDEND_YIELD_TTM.metadata.frequency_spec(ResearchFrequency.MONTHLY)


def test_plugin_factor_drives_monthly_adjusted_price_requirements_without_dispatch() -> None:
    frequency_spec = FactorFrequencySpec(
        ResearchFrequency.MONTHLY,
        ("daily", "adj_factor"),
        {"daily": ("ts_code", "trade_date", "close"), "adj_factor": ("ts_code", "trade_date", "adj_factor")},
        HistoryRequirement.trading_days(3),
        "monthly formation using trailing daily adjusted closes",
        "plugin_adjusted_momentum",
    )
    plugin = FunctionFactor(
        FactorMetadata("plugin_adjusted_momentum", "test", 1, required_datasets=("daily", "adj_factor"), source_fields=("adj_close",), lookback_days=3, frequency_specs=(frequency_spec,)),
        lambda frame: frame["adj_close"] / frame["adj_close"].shift(2) - 1,
    )
    registry = FactorRegistry()
    registry.register(plugin)
    requirements = FactorDependencyPlanner(registry, calendar()).requirements((plugin.metadata.name,), frequency=ResearchFrequency.MONTHLY, start_date="2024-01-01", end_date="2024-02-29", scope="CN_A")
    assert tuple(item.dataset_id for item in requirements) == ("adj_factor", "daily")
    assert {item.required_start for item in requirements} == {"2024-01-29"}
    assert {item.required_end for item in requirements} == {"2024-02-29"}
    source = Path("src/factors/frequency.py").read_text(encoding="utf-8")
    assert "if factor_name" not in source and '.resample("M").mean()' not in source


def test_future_factor_input_and_next_month_do_not_change_value_at_t() -> None:
    spec = FactorFrequencySpec(ResearchFrequency.MONTHLY, ("daily", "adj_factor"), {"daily": ("adj_close",), "adj_factor": ("adj_factor",)}, HistoryRequirement.trading_days(3), "trailing adjusted closes", "plugin")
    factor = FunctionFactor(FactorMetadata("plugin", "test", 1, required_datasets=("daily", "adj_factor"), source_fields=("adj_close",), frequency_specs=(spec,)), lambda frame: frame["adj_close"] / frame["adj_close"].shift(2) - 1)
    through_t = pd.DataFrame({"adj_close": [10.0, 11.0, 12.0]})
    extended = pd.DataFrame({"adj_close": [10.0, 11.0, 12.0, 99999.0, 0.01]})
    assert factor.compute(through_t).iloc[-1] == factor.compute(extended).iloc[2]
