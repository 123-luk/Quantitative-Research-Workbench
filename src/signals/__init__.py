"""Public contracts and runtime APIs for the V5 Signal layer."""

from src.signals.contracts import (
    SIGNAL_FORBIDDEN_OUTPUT_COLUMNS,
    SIGNAL_KEY_COLUMNS,
    SIGNAL_OUTPUT_COLUMNS,
    SIGNAL_PROTECTED_SCORE_SOURCE_COLUMNS,
    SIGNAL_SCHEMA_VERSION,
    SignalContractError,
    SignalDataError,
    validate_signal_columns,
    validate_signal_key_columns,
)
from src.signals.builder import SignalBuildAudit, SignalBuilder, SignalBuildResult
from src.signals.sources import (
    NATIVE_ML_PREDICTIONS_FILENAME,
    PredictionSourceAdapter,
    PredictionSourceError,
    PredictionSourceProvenance,
    PredictionSourceResult,
)

__all__ = [
    "NATIVE_ML_PREDICTIONS_FILENAME",
    "PredictionSourceAdapter",
    "PredictionSourceError",
    "PredictionSourceProvenance",
    "PredictionSourceResult",
    "SIGNAL_FORBIDDEN_OUTPUT_COLUMNS",
    "SIGNAL_KEY_COLUMNS",
    "SIGNAL_OUTPUT_COLUMNS",
    "SIGNAL_PROTECTED_SCORE_SOURCE_COLUMNS",
    "SIGNAL_SCHEMA_VERSION",
    "SignalBuildAudit",
    "SignalBuilder",
    "SignalBuildResult",
    "SignalContractError",
    "SignalDataError",
    "validate_signal_columns",
    "validate_signal_key_columns",
]