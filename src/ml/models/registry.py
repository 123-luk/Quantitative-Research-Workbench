"""Explicit registry for project-owned model configurations and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.ml.models.base import (
    ModelConfigError,
    ModelParameterSpec,
    ModelRegistryError,
    RegressionModelAdapter,
)
from src.ml.models.linear import (
    ElasticNetModelAdapter,
    ElasticNetModelConfig,
    RidgeModelAdapter,
    RidgeModelConfig,
)
from src.ml.models.tree import (
    HistGradientBoostingModelAdapter,
    HistGradientBoostingModelConfig,
)


@dataclass(frozen=True)
class _ModelRegistration:
    config_class: (
        type[RidgeModelConfig]
        | type[ElasticNetModelConfig]
        | type[HistGradientBoostingModelConfig]
    )
    adapter_class: type[RegressionModelAdapter]


def _normalize_model_name(model_name: object) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ModelRegistryError("model registry: model_name must be non-empty")
    return model_name.strip().lower()


class ModelRegistry:
    """Small explicit registry supporting future project-owned model adapters."""

    def __init__(self, *, include_builtin_models: bool = True) -> None:
        if not isinstance(include_builtin_models, bool):
            raise ModelRegistryError(
                "model registry: include_builtin_models must be bool"
            )
        self._registrations: dict[str, _ModelRegistration] = {}
        if include_builtin_models:
            self.register("ridge", RidgeModelConfig, RidgeModelAdapter)
            self.register(
                "elastic_net",
                ElasticNetModelConfig,
                ElasticNetModelAdapter,
            )
            self.register(
                "hist_gradient_boosting",
                HistGradientBoostingModelConfig,
                HistGradientBoostingModelAdapter,
            )

    def register(
        self,
        model_name: str,
        config_class: (
            type[RidgeModelConfig]
            | type[ElasticNetModelConfig]
            | type[HistGradientBoostingModelConfig]
        ),
        adapter_class: type[RegressionModelAdapter],
    ) -> None:
        """Register a project-owned config class and adapter class explicitly."""
        name = _normalize_model_name(model_name)
        if name in self._registrations:
            raise ModelRegistryError(
                f"model registry: model {name!r} is already registered"
            )
        if not isinstance(config_class, type) or not all(
            callable(getattr(config_class, method, None))
            for method in ("from_dict", "parameter_schema")
        ):
            raise ModelRegistryError(
                f"model registry: config_class for {name!r} is invalid"
            )
        if not isinstance(adapter_class, type) or not issubclass(
            adapter_class, RegressionModelAdapter
        ):
            raise ModelRegistryError(
                f"model registry: adapter_class for {name!r} must implement "
                "RegressionModelAdapter"
            )
        self._registrations[name] = _ModelRegistration(
            config_class=config_class,
            adapter_class=adapter_class,
        )

    def list_models(self) -> tuple[str, ...]:
        """Return registered model names in stable sorted order."""
        return tuple(sorted(self._registrations))

    def _get(self, model_name: str) -> _ModelRegistration:
        name = _normalize_model_name(model_name)
        try:
            return self._registrations[name]
        except KeyError as exc:
            raise ModelRegistryError(
                f"model registry: unknown model {name!r}; "
                f"available models are {self.list_models()!r}"
            ) from exc

    def get_parameter_schema(
        self, model_name: str
    ) -> tuple[ModelParameterSpec, ...]:
        """Return immutable frozen parameter specifications for a model."""
        return tuple(self._get(model_name).config_class.parameter_schema())

    def get_default_parameters(self, model_name: str) -> dict[str, object]:
        """Return a fresh dictionary containing every model default."""
        config = self._get(model_name).config_class.from_dict(None)
        return dict(config.as_dict())

    def create(
        self,
        model_name: str,
        params: Mapping[str, object] | None = None,
    ) -> RegressionModelAdapter:
        """Create an adapter after resolving and validating all overrides."""
        registration = self._get(model_name)
        try:
            config = registration.config_class.from_dict(params)
        except ModelConfigError:
            raise
        return registration.adapter_class(config)


def create_default_model_registry() -> ModelRegistry:
    """Create an independent registry containing the supported core models."""
    return ModelRegistry()
