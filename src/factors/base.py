"""Core metadata and calculation interfaces for research factors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Tuple, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class FactorMetadata:
    """Describe a factor's identity, inputs, timing, and research semantics.

    ``direction`` is ``1`` when larger values are preferred and ``-1`` when
    smaller values are preferred. ``availability_lag_days`` records how long
    data must be delayed before it can be used in later research pipelines.
    """

    name: str
    category: str
    direction: int
    required_datasets: Tuple[str, ...] = field(default_factory=tuple)
    source_fields: Tuple[str, ...] = field(default_factory=tuple)
    lookback_days: int = 0
    frequency: str = "daily"
    availability_lag_days: int = 0
    description: str = ""
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Normalize immutable collections and validate metadata values."""
        object.__setattr__(self, "required_datasets", tuple(self.required_datasets))
        object.__setattr__(self, "source_fields", tuple(self.source_fields))

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Factor name must be a non-empty string.")
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("Factor category must be a non-empty string.")
        if isinstance(self.direction, bool) or self.direction not in (1, -1):
            raise ValueError("Factor direction must be either 1 or -1.")
        if self.lookback_days < 0:
            raise ValueError("Factor lookback_days must be greater than or equal to 0.")
        if self.availability_lag_days < 0:
            raise ValueError(
                "Factor availability_lag_days must be greater than or equal to 0."
            )
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.required_datasets
        ):
            raise ValueError("Factor required_datasets cannot contain empty values.")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.source_fields
        ):
            raise ValueError("Factor source_fields cannot contain empty values.")


@runtime_checkable
class Factor(Protocol):
    """Minimal interface implemented by every registered factor."""

    metadata: FactorMetadata

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute factor values aligned to the input DataFrame index."""


@dataclass(frozen=True)
class FunctionFactor:
    """A small factor implementation backed by a DataFrame-to-Series function."""

    metadata: FactorMetadata
    function: Callable[[pd.DataFrame], pd.Series] = field(repr=False, compare=False)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Validate required fields, calculate values, and enforce alignment."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Factor compute input must be a pandas DataFrame.")

        missing_fields = [
            field_name
            for field_name in self.metadata.source_fields
            if field_name not in data.columns
        ]
        if missing_fields:
            missing_text = ", ".join(missing_fields)
            raise ValueError(
                f"Factor '{self.metadata.name}' is missing required source fields: "
                f"{missing_text}."
            )

        result = self.function(data)
        if not isinstance(result, pd.Series):
            raise TypeError(
                f"Factor '{self.metadata.name}' compute must return a pandas Series."
            )
        if not result.index.equals(data.index):
            raise ValueError(
                f"Factor '{self.metadata.name}' output index must match the input index."
            )
        return result.rename(self.metadata.name)
