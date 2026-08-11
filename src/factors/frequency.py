"""Factor-owned research-frequency and dependency contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from src.data.contracts import DataRequirement, ResearchFrequency, canonical_date, coalesce_requirements
from src.research_data.calendar import HistoryRequirement, ResearchCalendar


class FactorFrequencyError(ValueError):
    """Raised for unsupported or internally inconsistent factor frequency metadata."""


@dataclass(frozen=True)
class FactorFrequencySpec:
    research_frequency: ResearchFrequency
    required_datasets: tuple[str, ...]
    required_fields: Mapping[str, tuple[str, ...]]
    history_requirement: HistoryRequirement
    observation_semantics: str
    calculator_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.research_frequency, ResearchFrequency):
            raise TypeError("research_frequency must be a ResearchFrequency.")
        datasets = tuple(self.required_datasets)
        if len(datasets) != len(set(datasets)) or any(not isinstance(item, str) or not item.strip() or item != item.strip() for item in datasets):
            raise FactorFrequencyError("required_datasets must contain unique trimmed dataset IDs.")
        if not isinstance(self.required_fields, Mapping) or set(self.required_fields) != set(datasets):
            raise FactorFrequencyError("required_fields keys must exactly match required_datasets.")
        fields: dict[str, tuple[str, ...]] = {}
        for dataset in datasets:
            raw_values = self.required_fields[dataset]
            if isinstance(raw_values, (str, bytes)):
                raise FactorFrequencyError(f"required_fields[{dataset!r}] must be a field collection.")
            values = tuple(raw_values)
            if len(values) != len(set(values)) or any(not isinstance(item, str) or not item.strip() or item != item.strip() for item in values):
                raise FactorFrequencyError(f"required_fields[{dataset!r}] must contain unique trimmed names.")
            fields[dataset] = values
        if not isinstance(self.history_requirement, HistoryRequirement):
            raise TypeError("history_requirement must be a HistoryRequirement.")
        for name in ("observation_semantics", "calculator_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise FactorFrequencyError(f"{name} must be a non-empty trimmed string.")
        object.__setattr__(self, "required_datasets", datasets)
        object.__setattr__(self, "required_fields", MappingProxyType(fields))

    def to_dict(self) -> dict[str, object]:
        return {
            "research_frequency": self.research_frequency.value,
            "required_datasets": list(self.required_datasets),
            "required_fields": {key: list(value) for key, value in self.required_fields.items()},
            "history_requirement": self.history_requirement.to_dict(),
            "observation_semantics": self.observation_semantics,
            "calculator_id": self.calculator_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FactorFrequencySpec":
        expected = {"research_frequency", "required_datasets", "required_fields", "history_requirement", "observation_semantics", "calculator_id"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise FactorFrequencyError("FactorFrequencySpec contains missing or extra fields.")
        try:
            frequency = ResearchFrequency(value["research_frequency"])
            history = HistoryRequirement.from_dict(value["history_requirement"])  # type: ignore[arg-type]
            datasets = tuple(value["required_datasets"])  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise FactorFrequencyError("FactorFrequencySpec serialization is invalid.") from exc
        return cls(frequency, datasets, value["required_fields"], history, value["observation_semantics"], value["calculator_id"])  # type: ignore[arg-type]


def point_in_time_frequency_specs(*, dataset: str, fields: tuple[str, ...], calculator_id: str) -> tuple[FactorFrequencySpec, ...]:
    """Declare identical as-of field semantics at daily and monthly formations."""
    return tuple(
        FactorFrequencySpec(
            frequency,
            (dataset,),
            {dataset: fields},
            HistoryRequirement.latest_as_of(),
            "formation-date as-of observation; no monthly averaging",
            calculator_id,
        )
        for frequency in (ResearchFrequency.DAILY, ResearchFrequency.MONTHLY)
    )


class FactorDependencyPlanner:
    """Build P4B requirements generically from registered factor metadata."""

    def __init__(self, registry: object, calendar: ResearchCalendar) -> None:
        if not callable(getattr(registry, "get", None)):
            raise TypeError("registry must provide get(name).")
        if not isinstance(calendar, ResearchCalendar):
            raise TypeError("calendar must be a ResearchCalendar.")
        self.registry = registry
        self.calendar = calendar

    def requirements(self, factor_names: Iterable[str], *, frequency: ResearchFrequency, start_date: object, end_date: object, scope: object) -> tuple[DataRequirement, ...]:
        names = tuple(factor_names)
        if not names or len(names) != len(set(names)) or any(not isinstance(item, str) or not item.strip() for item in names):
            raise FactorFrequencyError("factor_names must contain unique non-empty names.")
        formations = self.calendar.formation_dates(frequency, start_date, end_date)
        if not formations:
            raise FactorFrequencyError("requested interval contains no proven formation date.")
        requirements: list[DataRequirement] = []
        for name in names:
            factor = self.registry.get(name)
            metadata = getattr(factor, "metadata", None)
            method = getattr(metadata, "frequency_spec", None)
            if not callable(method):
                raise FactorFrequencyError(f"Factor {name!r} has no frequency metadata.")
            spec = method(frequency)
            history = self.calendar.resolve_history(formations[0], spec.history_requirement)
            for dataset in spec.required_datasets:
                requirements.append(
                    DataRequirement.create(
                        dataset,
                        scope=scope,
                        required_start=history.start_date,
                        required_end=formations[-1],
                        required_fields=spec.required_fields[dataset],
                        reason=f"factor:{name}:{frequency.value}:{spec.observation_semantics}",
                        as_of_cutoff=formations[-1],
                    )
                )
        return coalesce_requirements(requirements)
