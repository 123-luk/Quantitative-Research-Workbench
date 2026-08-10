"""Fail-closed risk-model error hierarchy."""


class RiskModelError(ValueError):
    """Base error for risk-model operations."""


class RiskModelConfigError(RiskModelError):
    """Raised for invalid public or estimator configuration."""


class RiskModelRegistryError(RiskModelError):
    """Raised for invalid estimator registry operations."""


class RiskModelDataError(RiskModelError):
    """Raised for invalid or insufficient historical data."""


class RiskModelValidationError(RiskModelError):
    """Raised when an estimator result violates the risk contract."""
