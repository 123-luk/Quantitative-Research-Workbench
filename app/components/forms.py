"""Generic schema-to-widget descriptors for the New Run form."""

from __future__ import annotations

from dataclasses import dataclass

from src.ml.models.base import ModelParameterSpec


@dataclass(frozen=True)
class ModelControlDescriptor:
    name: str
    label: str
    widget: str
    default: object
    advanced: bool
    minimum: int | float | None
    maximum: int | float | None
    step: int | float | None
    choices: tuple[object, ...] | None
    optional: bool
    help: str


def model_control_descriptor(spec: ModelParameterSpec) -> ModelControlDescriptor:
    if not isinstance(spec, ModelParameterSpec):
        raise TypeError("spec must be ModelParameterSpec.")
    widget = {
        "bool": "checkbox",
        "choice": "selectbox",
        "str": "text_input",
        "int": "number_input",
        "optional_int": "number_input",
        "float": "number_input",
        "optional_float": "number_input",
    }[spec.value_type]
    return ModelControlDescriptor(
        name=spec.name,
        label=spec.display_name,
        widget=widget,
        default=spec.default,
        advanced=spec.advanced,
        minimum=spec.minimum,
        maximum=spec.maximum,
        step=spec.step,
        choices=spec.choices,
        optional=spec.value_type.startswith("optional_"),
        help=spec.description,
    )


def split_model_parameter_schema(
    schema: tuple[ModelParameterSpec, ...],
) -> tuple[tuple[ModelControlDescriptor, ...], tuple[ModelControlDescriptor, ...]]:
    controls = tuple(model_control_descriptor(spec) for spec in schema)
    return (
        tuple(item for item in controls if not item.advanced),
        tuple(item for item in controls if item.advanced),
    )

