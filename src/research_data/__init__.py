"""Research-time calendar, history, and canonical adjusted-price services."""

from src.research_data.adjusted_prices import (
    MARKET_SCOPE,
    RAW_PRICE_FIELDS,
    AdjustedPriceDataUnavailable,
    AdjustedPriceError,
    AdjustedPriceRequest,
    AdjustedPriceResult,
    AdjustedPriceService,
    CanonicalAdjustedPriceDataSource,
    CanonicalMarketSlice,
)
from src.research_data.calendar import (
    HistoryKind,
    HistoryRequirement,
    HistoryWindow,
    ResearchCalendar,
    ResearchCalendarError,
)

_LAZY_EXPORTS = {
    "ResearchDatasetSource": ("src.research_data.materialization", "ResearchDatasetSource"),
    "ResearchInputBuilder": ("src.research_data.materialization", "ResearchInputBuilder"),
    "ResearchInputMaterialization": ("src.research_data.materialization", "ResearchInputMaterialization"),
    "ResearchMaterializationStore": ("src.research_data.materialization", "ResearchMaterializationStore"),
    "ForwardReturnSpec": ("src.research_data.planning", "ForwardReturnSpec"),
    "ResearchInputDataUnavailable": ("src.research_data.planning", "ResearchInputDataUnavailable"),
    "ResearchInputError": ("src.research_data.planning", "ResearchInputError"),
    "ResearchInputPlan": ("src.research_data.planning", "ResearchInputPlan"),
    "ResearchInputPlanner": ("src.research_data.planning", "ResearchInputPlanner"),
    "TrainingLabelAvailabilityGuard": ("src.research_data.planning", "TrainingLabelAvailabilityGuard"),
    "compose_requirements": ("src.research_data.planning", "compose_requirements"),
}


def __getattr__(name: str):
    """Load factor-dependent P4C3 exports without a factors-package cycle."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "MARKET_SCOPE",
    "RAW_PRICE_FIELDS",
    "AdjustedPriceDataUnavailable",
    "AdjustedPriceError",
    "AdjustedPriceRequest",
    "AdjustedPriceResult",
    "AdjustedPriceService",
    "CanonicalAdjustedPriceDataSource",
    "CanonicalMarketSlice",
    "HistoryKind",
    "HistoryRequirement",
    "HistoryWindow",
    "ResearchCalendar",
    "ResearchCalendarError",
    "ResearchDatasetSource",
    "ResearchInputBuilder",
    "ResearchInputMaterialization",
    "ResearchMaterializationStore",
    "ForwardReturnSpec",
    "ResearchInputDataUnavailable",
    "ResearchInputError",
    "ResearchInputPlan",
    "ResearchInputPlanner",
    "TrainingLabelAvailabilityGuard",
    "compose_requirements",
]
