"""Factor construction, metadata, registry, and testing package."""

from src.factors.base import Factor, FactorMetadata, FunctionFactor
from src.factors.examples import (
    MOMENTUM_20D,
    VOLATILITY_20D,
    register_example_factors,
)
from src.factors.registry import FactorRegistry, create_default_registry

__all__ = [
    "Factor",
    "FactorMetadata",
    "FactorRegistry",
    "FunctionFactor",
    "MOMENTUM_20D",
    "VOLATILITY_20D",
    "create_default_registry",
    "register_example_factors",
]

from src.factors.contracts import (
    normalize_factor_input,
    validate_factor_input,
    validate_required_fields,
)
from src.factors.factor_engine import FactorEngine

__all__.extend(
    [
        "FactorEngine",
        "normalize_factor_input",
        "validate_factor_input",
        "validate_required_fields",
    ]
)

from src.factors.price_volume import (
    AMIHUD_20D,
    MOMENTUM_60D,
    MOMENTUM_120D,
    MOMENTUM_252_20D,
    PRICE_52W_HIGH,
    PRICE_VOLUME_FACTORS,
    SHORT_TERM_REVERSAL_5D,
    TURNOVER_MEAN_20D,
    VOLATILITY_60D,
    register_price_volume_factors,
)

__all__.extend(
    [
        "AMIHUD_20D",
        "MOMENTUM_60D",
        "MOMENTUM_120D",
        "MOMENTUM_252_20D",
        "PRICE_52W_HIGH",
        "PRICE_VOLUME_FACTORS",
        "SHORT_TERM_REVERSAL_5D",
        "TURNOVER_MEAN_20D",
        "VOLATILITY_60D",
        "register_price_volume_factors",
    ]
)

from src.factors.valuation import (
    BP,
    DIVIDEND_YIELD_TTM,
    EP_TTM,
    LOG_CIRC_MV,
    LOG_TOTAL_MV,
    SP_TTM,
    VALUATION_FACTORS,
    register_valuation_factors,
)

__all__.extend(
    [
        "BP",
        "DIVIDEND_YIELD_TTM",
        "EP_TTM",
        "LOG_CIRC_MV",
        "LOG_TOTAL_MV",
        "SP_TTM",
        "VALUATION_FACTORS",
        "register_valuation_factors",
    ]
)

from src.factors.financial_alignment import (
    FinancialPointInTimeAligner,
    normalize_financial_data,
    normalize_trading_panel,
    normalize_value_columns,
)

__all__.extend(
    [
        "FinancialPointInTimeAligner",
        "normalize_financial_data",
        "normalize_trading_panel",
        "normalize_value_columns",
    ]
)

from src.factors.financial_factors import (
    DEBT_TO_ASSETS,
    FINANCIAL_FACTORS,
    GROSS_MARGIN_TTM,
    NET_MARGIN_TTM,
    NET_PROFIT_YOY,
    OPERATING_CF_TO_ASSETS,
    ROA_TTM,
    ROE_TTM,
    REVENUE_YOY,
    register_financial_factors,
)

__all__.extend(
    [
        "DEBT_TO_ASSETS",
        "FINANCIAL_FACTORS",
        "GROSS_MARGIN_TTM",
        "NET_MARGIN_TTM",
        "NET_PROFIT_YOY",
        "OPERATING_CF_TO_ASSETS",
        "ROA_TTM",
        "ROE_TTM",
        "REVENUE_YOY",
        "register_financial_factors",
    ]
)

from src.factors.preprocessing import FactorPreprocessor, PreprocessingConfig

__all__.extend(
    [
        "FactorPreprocessor",
        "PreprocessingConfig",
    ]
)

from src.factors.neutralization import FactorNeutralizer, NeutralizationConfig

__all__.extend(
    [
        "FactorNeutralizer",
        "NeutralizationConfig",
    ]
)
