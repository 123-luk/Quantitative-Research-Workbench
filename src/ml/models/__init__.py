"""Public model contracts, linear adapters, and explicit registry."""

from src.ml.models.base import (
    ModelConfigError,
    ModelDataError,
    ModelError,
    ModelFeatureMismatchError,
    ModelFeatureImportanceUnavailableError,
    ModelFitAudit,
    ModelFitError,
    ModelNotFittedError,
    ModelParameterSpec,
    ModelPredictionError,
    ModelRegistryError,
    RegressionModelAdapter,
)
from src.ml.models.linear import (
    ElasticNetModelAdapter,
    ElasticNetModelConfig,
    RidgeModelAdapter,
    RidgeModelConfig,
)
from src.ml.models.registry import ModelRegistry, create_default_model_registry
from src.ml.models.tree import (
    HistGradientBoostingModelAdapter,
    HistGradientBoostingModelConfig,
)

__all__ = [
    "ElasticNetModelAdapter",
    "ElasticNetModelConfig",
    "ModelConfigError",
    "HistGradientBoostingModelAdapter",
    "HistGradientBoostingModelConfig",
    "ModelDataError",
    "ModelError",
    "ModelFeatureMismatchError",
    "ModelFitAudit",
    "ModelFeatureImportanceUnavailableError",
    "ModelFitError",
    "ModelNotFittedError",
    "ModelParameterSpec",
    "ModelPredictionError",
    "ModelRegistry",
    "ModelRegistryError",
    "RegressionModelAdapter",
    "RidgeModelAdapter",
    "RidgeModelConfig",
    "create_default_model_registry",
]
