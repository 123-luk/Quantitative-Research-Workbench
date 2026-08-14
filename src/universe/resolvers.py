"""Point-in-time Universe 1.0 membership resolvers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

import pandas as pd

from src.data.contracts import DataRequirement, ResearchFrequency, canonical_date, coalesce_requirements
from src.universe.contracts import CANONICAL_SECURITY_PATTERN, UniverseConfigError, UniverseDataUnavailable, UniverseSnapshot, UniverseSpec, UniverseType
from src.universe.data import STOCK_BASIC_SCOPE, CanonicalUniverseSlice, UniverseDataSource


class UniverseResolver(Protocol):
    universe_type: UniverseType
    def resolve(self, spec: UniverseSpec, formation_date: str, services: UniverseDataSource) -> UniverseSnapshot: ...
    def requirements(self, spec: UniverseSpec, start: str, end: str, frequency: ResearchFrequency) -> tuple[DataRequirement, ...]: ...


def _stock_requirement(start: str, end: str, reason: str, fields: tuple[str, ...]) -> DataRequirement:
    return DataRequirement.create("stock_basic", scope=dict(STOCK_BASIC_SCOPE), required_start=start, required_end=end, required_fields=fields, reason=reason)


def _stock_frame(source: CanonicalUniverseSlice) -> pd.DataFrame:
    required = {"ts_code", "symbol", "market", "exchange", "curr_type", "list_status", "list_date", "delist_date", "name"}
    missing = sorted(required - set(source.frame.columns))
    if missing or source.frame.empty:
        raise UniverseDataUnavailable(f"Canonical stock_basic is unavailable or missing fields: {missing!r}.")
    rows = source.frame.copy(deep=True)
    if rows["ts_code"].duplicated().any():
        raise UniverseDataUnavailable("Canonical stock_basic ts_code must be unique.")
    return rows


def _lifecycle(rows: pd.DataFrame, formation_date: str) -> pd.Series:
    formation = pd.Timestamp(canonical_date(formation_date))
    listed = pd.to_datetime(rows["list_date"], errors="coerce")
    delisted = pd.to_datetime(rows["delist_date"], errors="coerce")
    if listed.isna().any():
        raise UniverseDataUnavailable("stock_basic list_date contains invalid values.")
    invalid = delisted.notna() & delisted.lt(listed)
    if invalid.any():
        raise UniverseDataUnavailable("stock_basic delist_date precedes list_date.")
    return listed.le(formation) & (delisted.isna() | delisted.gt(formation))


def _canonical_custom(spec: UniverseSpec, rows: pd.DataFrame) -> tuple[str, ...]:
    by_symbol: dict[str, list[str]] = {}
    for row in rows.loc[:, ["symbol", "ts_code"]].itertuples(index=False):
        symbol = str(row.symbol).strip().upper()
        by_symbol.setdefault(symbol, []).append(str(row.ts_code).strip().upper())
    available = set(str(item).strip().upper() for item in rows["ts_code"])
    canonical: list[str] = []
    for raw in spec.params["securities"]:  # type: ignore[index]
        code = str(raw).upper()
        if CANONICAL_SECURITY_PATTERN.fullmatch(code):
            if code not in available:
                raise UniverseConfigError(f"CUSTOM security {code!r} is absent from canonical stock_basic.")
            resolved = code
        else:
            matches = tuple(sorted(set(by_symbol.get(code, ()))))
            if not matches:
                raise UniverseConfigError(f"Bare CUSTOM code {code!r} has no canonical stock_basic match.")
            if len(matches) != 1:
                raise UniverseConfigError(f"Bare CUSTOM code {code!r} is ambiguous.")
            resolved = matches[0]
        if resolved not in canonical:
            canonical.append(resolved)
    return tuple(canonical)


def _identity(*parts: str) -> str:
    return "|".join(parts)


@dataclass(frozen=True)
class CustomUniverseResolver:
    universe_type: UniverseType = UniverseType.CUSTOM

    def canonicalize(self, spec: UniverseSpec, services: UniverseDataSource) -> UniverseSpec:
        rows = _stock_frame(services.stock_basic())
        return UniverseSpec.custom(_canonical_custom(spec, rows))

    def resolve(self, spec: UniverseSpec, formation_date: str, services: UniverseDataSource) -> UniverseSnapshot:
        source = services.stock_basic()
        rows = _stock_frame(source)
        canonical = _canonical_custom(spec, rows)
        active = set(rows.loc[_lifecycle(rows, formation_date), "ts_code"].astype(str))
        securities = tuple(code for code in canonical if code in active)
        return UniverseSnapshot(formation_date, securities, self.universe_type, _identity("CUSTOM", ",".join(canonical), source.source_identity), source.source_as_of, {"input_securities": canonical, "lifecycle_filtered_count": len(canonical) - len(securities), "lifecycle_boundary": "list_date <= T < delist_date"})

    def requirements(self, spec: UniverseSpec, start: str, end: str, frequency: ResearchFrequency) -> tuple[DataRequirement, ...]:
        return (_stock_requirement(start, end, "CUSTOM code validation and point-in-time lifecycle", ("ts_code", "symbol", "list_date", "delist_date")),)


@dataclass(frozen=True)
class IndexUniverseResolver:
    universe_type: UniverseType = UniverseType.INDEX

    def resolve(self, spec: UniverseSpec, formation_date: str, services: UniverseDataSource) -> UniverseSnapshot:
        formation = canonical_date(formation_date)
        code = str(spec.params["index_code"])
        lifecycle_source = services.stock_basic()
        stocks = _stock_frame(lifecycle_source)
        weights_source = services.index_weight(code, formation)
        weights = weights_source.frame.copy(deep=True)
        required = {"index_code", "con_code", "trade_date", "weight"}
        if weights.empty or not required.issubset(weights.columns):
            raise UniverseDataUnavailable("Canonical index_weight is unavailable.")
        if set(weights["index_code"].astype(str)) != {code}:
            raise UniverseDataUnavailable("index_weight scope differs from UniverseSpec index_code.")
        dates = pd.to_datetime(weights["trade_date"], errors="coerce")
        if dates.isna().any():
            raise UniverseDataUnavailable("index_weight trade_date contains invalid values.")
        eligible_dates = dates.loc[dates.le(pd.Timestamp(formation))]
        if eligible_dates.empty:
            raise UniverseDataUnavailable("No index membership snapshot exists on or before formation_date.")
        selected_date = eligible_dates.max()
        selected = weights.loc[dates.eq(selected_date)].copy()
        if selected["con_code"].duplicated().any():
            raise UniverseDataUnavailable("Selected index snapshot has duplicate constituents.")
        constituents = set(selected["con_code"].astype(str))
        if any(not CANONICAL_SECURITY_PATTERN.fullmatch(item) for item in constituents):
            raise UniverseDataUnavailable("Index snapshot contains a non-canonical constituent identity.")
        unknown = tuple(sorted(constituents - set(stocks["ts_code"].astype(str))))
        if unknown:
            raise UniverseDataUnavailable(f"Index constituents lack stock_basic lifecycle: {unknown!r}.")
        active = set(stocks.loc[_lifecycle(stocks, formation), "ts_code"].astype(str))
        securities = tuple(sorted(constituents & active))
        source_as_of = selected_date.strftime("%Y-%m-%d")
        weight_sum = float(pd.to_numeric(selected["weight"], errors="raise").sum())
        selected_payload = selected.sort_values(["con_code", "trade_date"], kind="mergesort").to_csv(index=False, lineterminator="\n", float_format="%.17g")
        selected_hash = sha256(selected_payload.encode("utf-8")).hexdigest()
        index_source = f"{weights_source.dataset_id}:{weights_source.schema_version}"
        return UniverseSnapshot(formation, securities, self.universe_type, _identity("INDEX", code, source_as_of, index_source, selected_hash, lifecycle_source.source_identity), source_as_of, {"index_code": code, "provider_snapshot_date": source_as_of, "provider_weight_sum": weight_sum, "index_source": index_source, "selected_snapshot_hash": selected_hash, "source_semantics": "latest provider snapshot date <= formation date", "lifecycle_filtered_count": len(constituents) - len(securities)})

    def requirements(self, spec: UniverseSpec, start: str, end: str, frequency: ResearchFrequency) -> tuple[DataRequirement, ...]:
        prior_month = (pd.Timestamp(canonical_date(start)).to_period("M") - 1).start_time.date().isoformat()
        requirements = (
            _stock_requirement(start, end, "INDEX constituent lifecycle", ("ts_code", "list_date", "delist_date")),
            DataRequirement.create("index_weight", scope={"index_code": str(spec.params["index_code"])}, required_start=prior_month, required_end=end, required_fields=("index_code", "con_code", "trade_date", "weight"), reason="INDEX point-in-time membership snapshots", as_of_cutoff=end),
        )
        return coalesce_requirements(requirements)


_A_SHARE_MARKETS = frozenset({"主板", "创业板", "科创板", "北交所"})
_KNOWN_NON_A_MARKETS = frozenset({"B股", "CDR"})
_A_SHARE_EXCHANGES = frozenset({"SSE", "SZSE", "BSE"})


@dataclass(frozen=True)
class AllASharesUniverseResolver:
    universe_type: UniverseType = UniverseType.ALL_A_SHARES

    def resolve(self, spec: UniverseSpec, formation_date: str, services: UniverseDataSource) -> UniverseSnapshot:
        source = services.stock_basic()
        rows = _stock_frame(source)
        markets = set(rows["market"].dropna().astype(str))
        unknown_markets = tuple(sorted(markets - _A_SHARE_MARKETS - _KNOWN_NON_A_MARKETS))
        if unknown_markets:
            raise UniverseDataUnavailable(f"stock_basic contains unsupported market classifications: {unknown_markets!r}.")
        classification = rows["market"].isin(_A_SHARE_MARKETS) & rows["exchange"].isin(_A_SHARE_EXCHANGES) & rows["curr_type"].eq("CNY")
        active = _lifecycle(rows, formation_date)
        securities = tuple(sorted(rows.loc[classification & active, "ts_code"].astype(str)))
        return UniverseSnapshot(formation_date, securities, self.universe_type, _identity("ALL_A_SHARES", source.source_identity), source.source_as_of, {"classification": "stock_basic equity contract + market board + China exchange + CNY", "boards": tuple(sorted(_A_SHARE_MARKETS)), "classified_count": int(classification.sum()), "lifecycle_filtered_count": int((classification & ~active).sum()), "st_suspension_listing_age_filters": False})

    def requirements(self, spec: UniverseSpec, start: str, end: str, frequency: ResearchFrequency) -> tuple[DataRequirement, ...]:
        return (_stock_requirement(start, end, "ALL_A_SHARES classification and point-in-time lifecycle", ("ts_code", "name", "market", "exchange", "curr_type", "list_date", "delist_date")),)
