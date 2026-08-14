"""Immutable Universe 1.0 membership contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
import re

from src.data.contracts import canonical_date
from src.data.security_identifiers import CANONICAL_SECURITY_PATTERN


CANONICAL_INDEX_PATTERN = re.compile(r"^[A-Z0-9]{1,20}\.[A-Z]{2,6}$")


class UniverseError(ValueError):
    pass


class UniverseConfigError(UniverseError):
    pass


class UniverseDataUnavailable(UniverseError):
    pass


class UnsupportedLegacySecurityIdentifier(UniverseDataUnavailable):
    pass


class UniverseType(str, Enum):
    CUSTOM = "CUSTOM"
    INDEX = "INDEX"
    ALL_A_SHARES = "ALL_A_SHARES"


def _type(value: object) -> UniverseType:
    if isinstance(value, UniverseType):
        return value
    if not isinstance(value, str):
        raise UniverseConfigError("universe_type must be a canonical string ID.")
    try:
        return UniverseType(value.strip().upper())
    except ValueError as exc:
        raise UniverseConfigError(f"Unknown universe_type: {value!r}.") from exc


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UniverseConfigError(f"{field_name} must be a non-empty string.")
    return value.strip().upper()


@dataclass(frozen=True)
class UniverseSpec:
    universe_type: UniverseType
    params: Mapping[str, object]

    def __post_init__(self) -> None:
        kind = _type(self.universe_type)
        if not isinstance(self.params, Mapping):
            raise UniverseConfigError("params must be a mapping.")
        raw = dict(self.params)
        if kind is UniverseType.CUSTOM:
            if set(raw) != {"securities"}:
                raise UniverseConfigError("CUSTOM params must contain only securities.")
            values = raw["securities"]
            if isinstance(values, (str, bytes)):
                raise UniverseConfigError("CUSTOM securities must be an ordered collection.")
            try:
                supplied = tuple(values)  # type: ignore[arg-type]
            except TypeError as exc:
                raise UniverseConfigError("CUSTOM securities must be an ordered collection.") from exc
            ordered: list[str] = []
            for value in supplied:
                code = _text(value, "CUSTOM security")
                if not (CANONICAL_SECURITY_PATTERN.fullmatch(code) or re.fullmatch(r"[0-9]{6}", code)):
                    raise UniverseConfigError(f"Invalid CUSTOM security code: {value!r}.")
                if code not in ordered:
                    ordered.append(code)
            if not ordered:
                raise UniverseConfigError("CUSTOM securities must not be empty.")
            normalized: dict[str, object] = {"securities": tuple(ordered)}
        elif kind is UniverseType.INDEX:
            if set(raw) != {"index_code"}:
                raise UniverseConfigError("INDEX params must contain only index_code.")
            code = _text(raw["index_code"], "index_code")
            if not CANONICAL_INDEX_PATTERN.fullmatch(code):
                raise UniverseConfigError("index_code must be a canonical provider identity.")
            normalized = {"index_code": code}
        else:
            if raw:
                raise UniverseConfigError("ALL_A_SHARES params must be empty.")
            normalized = {}
        object.__setattr__(self, "universe_type", kind)
        object.__setattr__(self, "params", MappingProxyType(normalized))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "UniverseSpec":
        if not isinstance(value, Mapping) or set(value) != {"universe_type", "params"}:
            raise UniverseConfigError("UniverseSpec requires universe_type and params only.")
        return cls(_type(value["universe_type"]), value["params"])  # type: ignore[arg-type]

    @classmethod
    def custom(cls, securities: object) -> "UniverseSpec":
        return cls(UniverseType.CUSTOM, {"securities": securities})

    @classmethod
    def index(cls, index_code: object) -> "UniverseSpec":
        return cls(UniverseType.INDEX, {"index_code": index_code})

    @classmethod
    def all_a_shares(cls) -> "UniverseSpec":
        return cls(UniverseType.ALL_A_SHARES, {})

    def to_dict(self) -> dict[str, object]:
        params = dict(self.params)
        if "securities" in params:
            params["securities"] = list(params["securities"])  # type: ignore[arg-type]
        return {"universe_type": self.universe_type.value, "params": params}


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return deepcopy(value)


@dataclass(frozen=True)
class UniverseSnapshot:
    formation_date: str
    securities: tuple[str, ...]
    universe_type: UniverseType
    source_identity: str
    source_as_of: str
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        formation = canonical_date(self.formation_date)
        source_as_of = canonical_date(self.source_as_of)
        securities = tuple(self.securities)
        if len(securities) != len(set(securities)) or any(not isinstance(item, str) or not CANONICAL_SECURITY_PATTERN.fullmatch(item) for item in securities):
            raise UniverseDataUnavailable("snapshot securities must be unique canonical ts_code values.")
        if not isinstance(self.source_identity, str) or not self.source_identity.strip():
            raise UniverseDataUnavailable("source_identity must not be empty.")
        if not isinstance(self.diagnostics, Mapping):
            raise UniverseDataUnavailable("diagnostics must be a mapping.")
        object.__setattr__(self, "formation_date", formation)
        object.__setattr__(self, "source_as_of", source_as_of)
        object.__setattr__(self, "securities", securities)
        object.__setattr__(self, "universe_type", _type(self.universe_type))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        def thaw(value: object) -> object:
            if isinstance(value, Mapping):
                return {str(key): thaw(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [thaw(item) for item in value]
            return deepcopy(value)
        return {"formation_date": self.formation_date, "securities": list(self.securities), "universe_type": self.universe_type.value, "source_identity": self.source_identity, "source_as_of": self.source_as_of, "diagnostics": thaw(self.diagnostics)}
