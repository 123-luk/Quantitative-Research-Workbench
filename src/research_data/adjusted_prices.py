"""Canonical, anchor-free adjusted OHLC research observations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np
import pandas as pd

from src.data.canonical_store import PartitionedParquetStore, content_hash, normalize_frame
from src.data.contracts import DataRequirement, canonical_date, coalesce_requirements, normalize_scope
from src.data.coverage_ledger import CoverageLedger
from src.data.coverage_planner import scope_key
from src.data.dataset_registry import DatasetRegistry
from src.universe.contracts import CANONICAL_SECURITY_PATTERN


MARKET_SCOPE = normalize_scope("CN_A")
RAW_PRICE_FIELDS = ("open", "high", "low", "close")


class AdjustedPriceError(ValueError):
    pass


class AdjustedPriceDataUnavailable(AdjustedPriceError):
    pass


@dataclass(frozen=True)
class CanonicalMarketSlice:
    frame: pd.DataFrame
    dataset_id: str
    schema_version: str
    source_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pd.DataFrame):
            raise AdjustedPriceDataUnavailable("canonical market slice must contain a DataFrame.")
        for name in ("dataset_id", "schema_version", "source_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise AdjustedPriceDataUnavailable(f"{name} must not be empty.")
        object.__setattr__(self, "frame", self.frame.copy(deep=True))


class AdjustedPriceDataSource(Protocol):
    def daily(self, dates: tuple[str, ...]) -> CanonicalMarketSlice: ...
    def adj_factor(self, dates: tuple[str, ...]) -> CanonicalMarketSlice: ...


class CanonicalAdjustedPriceDataSource:
    """Read only explicit ledger-proven daily market units from CURATED."""

    def __init__(self, *, registry: DatasetRegistry, ledger: CoverageLedger, store: PartitionedParquetStore, scope: object = "CN_A") -> None:
        self.registry = registry
        self.ledger = ledger
        self.store = store
        self.scope = normalize_scope(scope)

    def _read(self, dataset_id: str, dates: tuple[str, ...]) -> CanonicalMarketSlice:
        units = tuple(canonical_date(item) for item in dates)
        if not units or units != tuple(sorted(set(units))):
            raise AdjustedPriceDataUnavailable("canonical source dates must be ordered and unique.")
        spec = self.registry.get(dataset_id)
        records = {
            record.unit_key: record
            for record in self.ledger.records(dataset_id)
            if record.scope_key == scope_key(self.scope) and record.status == "COMPLETE"
        }
        missing = tuple(item for item in units if item not in records)
        if missing:
            raise AdjustedPriceDataUnavailable(f"Canonical {dataset_id} coverage is unavailable for {missing!r}.")
        frames: list[pd.DataFrame] = []
        identities: list[str] = []
        for unit in units:
            rows = self.store.rows_for_unit(spec, unit=unit, scope=self.scope)
            record = records[unit]
            if len(rows) != record.row_count or content_hash(spec, rows) != record.content_hash:
                raise AdjustedPriceDataUnavailable(f"Canonical {dataset_id} integrity check failed for {unit}.")
            frames.append(rows)
            identities.append(f"{unit}:{record.content_hash}")
        frame = normalize_frame(spec, pd.concat(frames, ignore_index=True))
        digest = sha256("|".join(identities).encode("utf-8")).hexdigest()
        return CanonicalMarketSlice(frame, dataset_id, spec.schema_version, f"{dataset_id}:{spec.schema_version}:{digest}")

    def daily(self, dates: tuple[str, ...]) -> CanonicalMarketSlice:
        return self._read("daily", dates)

    def adj_factor(self, dates: tuple[str, ...]) -> CanonicalMarketSlice:
        return self._read("adj_factor", dates)

    def load(self, dataset_id: str, dates: tuple[str, ...]) -> CanonicalMarketSlice:
        """Read another explicit market-snapshot dataset through the same proof boundary."""
        return self._read(dataset_id, dates)


@dataclass(frozen=True)
class AdjustedPriceRequest:
    securities: tuple[str, ...]
    dates: tuple[str, ...]
    price_fields: tuple[str, ...] = RAW_PRICE_FIELDS

    def __post_init__(self) -> None:
        securities = tuple(self.securities)
        if not securities or len(securities) != len(set(securities)) or any(not isinstance(item, str) or not CANONICAL_SECURITY_PATTERN.fullmatch(item) for item in securities):
            raise AdjustedPriceError("securities must contain ordered unique canonical ts_code values.")
        try:
            dates = tuple(sorted(set(canonical_date(item) for item in self.dates)))
        except ValueError as exc:
            raise AdjustedPriceError("dates must contain valid trading dates.") from exc
        if not dates:
            raise AdjustedPriceError("dates must not be empty.")
        fields = tuple(self.price_fields)
        if not fields or len(fields) != len(set(fields)) or not set(fields).issubset(RAW_PRICE_FIELDS):
            raise AdjustedPriceError(f"price_fields must be a unique non-empty subset of {RAW_PRICE_FIELDS!r}.")
        object.__setattr__(self, "securities", securities)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "price_fields", fields)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return deepcopy(value)


@dataclass(frozen=True, init=False)
class AdjustedPriceResult:
    _frame: pd.DataFrame
    source_identity: str
    source_as_of: str
    diagnostics: Mapping[str, object]

    def __init__(self, frame: pd.DataFrame, *, source_identity: str, source_as_of: str, diagnostics: Mapping[str, object]) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame.")
        if not isinstance(source_identity, str) or not source_identity.strip():
            raise AdjustedPriceError("source_identity must not be empty.")
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "source_identity", source_identity)
        object.__setattr__(self, "source_as_of", canonical_date(source_as_of))
        object.__setattr__(self, "diagnostics", _freeze(diagnostics))

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy(deep=True)


def _finite_numeric(frame: pd.DataFrame, columns: tuple[str, ...], *, context: str, positive: bool = False) -> pd.DataFrame:
    result = frame.copy(deep=True)
    for column in columns:
        if result[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise AdjustedPriceDataUnavailable(f"{context} {column} contains bool values.")
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all() or (positive and not values.gt(0).all()):
            qualifier = "finite and positive" if positive else "finite numeric"
            raise AdjustedPriceDataUnavailable(f"{context} {column} must be {qualifier}.")
        result[column] = values.astype(float)
    return result


class AdjustedPriceService:
    """Compute raw price multiplied by the same-key adjustment factor."""

    def __init__(self, source: AdjustedPriceDataSource) -> None:
        if not callable(getattr(source, "daily", None)) or not callable(getattr(source, "adj_factor", None)):
            raise TypeError("source must implement daily and adj_factor.")
        self.source = source

    @staticmethod
    def requirements(*, start_date: object, end_date: object, price_fields: tuple[str, ...] = RAW_PRICE_FIELDS, scope: object = "CN_A") -> tuple[DataRequirement, ...]:
        request_fields = tuple(price_fields)
        if not request_fields or len(request_fields) != len(set(request_fields)) or not set(request_fields).issubset(RAW_PRICE_FIELDS):
            raise AdjustedPriceError("price_fields must be a non-empty raw OHLC subset.")
        requirements = (
            DataRequirement.create("daily", scope=scope, required_start=start_date, required_end=end_date, required_fields=("ts_code", "trade_date", *request_fields), reason="canonical adjusted OHLC raw prices", as_of_cutoff=end_date),
            DataRequirement.create("adj_factor", scope=scope, required_start=start_date, required_end=end_date, required_fields=("ts_code", "trade_date", "adj_factor"), reason="canonical adjusted OHLC adjustment factors", as_of_cutoff=end_date),
        )
        return coalesce_requirements(requirements)

    def compute(self, request: AdjustedPriceRequest) -> AdjustedPriceResult:
        if not isinstance(request, AdjustedPriceRequest):
            raise TypeError("request must be an AdjustedPriceRequest.")
        daily_source = self.source.daily(request.dates)
        factor_source = self.source.adj_factor(request.dates)
        daily_required = {"ts_code", "trade_date", *request.price_fields}
        factor_required = {"ts_code", "trade_date", "adj_factor"}
        if not daily_required.issubset(daily_source.frame.columns) or not factor_required.issubset(factor_source.frame.columns):
            raise AdjustedPriceDataUnavailable("canonical adjusted-price inputs are missing required fields.")
        dates = set(request.dates)
        securities = set(request.securities)
        daily = daily_source.frame.loc[daily_source.frame["trade_date"].isin(dates) & daily_source.frame["ts_code"].isin(securities)].copy()
        factors = factor_source.frame.loc[factor_source.frame["trade_date"].isin(dates) & factor_source.frame["ts_code"].isin(securities)].copy()
        keys = ["ts_code", "trade_date"]
        if daily.duplicated(keys).any():
            raise AdjustedPriceDataUnavailable("daily contains duplicate requested primary keys.")
        if factors.duplicated(keys).any():
            raise AdjustedPriceDataUnavailable("adj_factor contains duplicate requested primary keys.")
        daily = _finite_numeric(daily, request.price_fields, context="daily")
        factors = _finite_numeric(factors, ("adj_factor",), context="adj_factor", positive=True)
        raw_passthrough = tuple(field for field in ("vol", "amount") if field in daily.columns)
        if raw_passthrough:
            daily = _finite_numeric(daily, raw_passthrough, context="daily")
        merged = daily.merge(factors.loc[:, keys + ["adj_factor"]], on=keys, how="left", validate="one_to_one", indicator=True)
        if not merged["_merge"].eq("both").all():
            missing = tuple(map(tuple, merged.loc[merged["_merge"].ne("both"), keys].itertuples(index=False, name=None)))
            raise AdjustedPriceDataUnavailable(f"daily rows lack exact adj_factor matches: {missing!r}.")
        merged = merged.drop(columns="_merge")
        for field in request.price_fields:
            merged[f"adj_{field}"] = merged[field] * merged["adj_factor"]
        output_columns = keys + list(request.price_fields) + ["adj_factor"] + [f"adj_{field}" for field in request.price_fields] + list(raw_passthrough)
        output = merged.loc[:, output_columns].sort_values(["trade_date", "ts_code"], kind="mergesort", ignore_index=True)
        daily_keys = set(map(tuple, daily.loc[:, keys].itertuples(index=False, name=None)))
        factor_keys = set(map(tuple, factors.loc[:, keys].itertuples(index=False, name=None)))
        identity = f"adjusted_price:raw_times_factor:{daily_source.source_identity}:{factor_source.source_identity}"
        return AdjustedPriceResult(output, source_identity=identity, source_as_of=request.dates[-1], diagnostics={"formula": "adj_OHLC = raw_OHLC * adj_factor", "dynamic_end_anchor": False, "daily_observation_rows": len(daily), "orphan_adjustment_rows": len(factor_keys - daily_keys), "raw_volume_amount_adjusted": False})
