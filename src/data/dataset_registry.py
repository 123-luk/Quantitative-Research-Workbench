"""Fresh registry and the eight canonical Data Layer 2.0 dataset specs."""

from __future__ import annotations

from src.data.contracts import CoverageKind, DatasetSpec, FetchStrategy, NativeFrequency, RevisionPolicy, ScopeKind


class DatasetRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, DatasetSpec] = {}

    def register(self, spec: DatasetSpec) -> None:
        if not isinstance(spec, DatasetSpec):
            raise TypeError("spec must be a DatasetSpec.")
        if spec.dataset_id in self._specs:
            raise ValueError(f"Dataset {spec.dataset_id!r} is already registered.")
        self._specs[spec.dataset_id] = spec

    def get(self, dataset_id: str) -> DatasetSpec:
        try:
            return self._specs[dataset_id]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset: {dataset_id!r}.") from exc

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def list_specs(self) -> tuple[DatasetSpec, ...]:
        return tuple(self._specs[item] for item in self.list_ids())


def _spec(dataset_id: str, *, endpoint: str, native: NativeFrequency, scope: ScopeKind, key: tuple[str, ...], fields: tuple[str, ...], coverage: CoverageKind, partition: tuple[str, ...], fetch: FetchStrategy, completeness: str, availability: str, empty: bool = False, limit: int | None = None, revision: RevisionPolicy = RevisionPolicy.MISSING_ONLY) -> DatasetSpec:
    return DatasetSpec(dataset_id, "tushare", endpoint, native, scope, key, fields, coverage, partition, fetch, completeness, "1.0", revision, availability, empty, limit)


def create_default_dataset_registry() -> DatasetRegistry:
    registry = DatasetRegistry()
    specs = (
        _spec("trade_cal", endpoint="trade_cal", native=NativeFrequency.CALENDAR_DAY, scope=ScopeKind.ENTITY_SERIES, key=("exchange", "cal_date"), fields=("exchange", "cal_date", "is_open", "pretrade_date"), coverage=CoverageKind.CALENDAR_DATE, partition=("year",), fetch=FetchStrategy.ENTITY_DATE_RANGE, completeness="calendar_days", availability="same calendar observation", revision=RevisionPolicy.EXPLICIT_REFRESH),
        _spec("stock_basic", endpoint="stock_basic", native=NativeFrequency.REFERENCE_SNAPSHOT, scope=ScopeKind.REFERENCE_SNAPSHOT, key=("ts_code",), fields=("ts_code", "symbol", "name", "area", "industry", "market", "list_status", "list_date", "delist_date"), coverage=CoverageKind.REFERENCE_EFFECTIVE_THROUGH, partition=("snapshot",), fetch=FetchStrategy.REFERENCE_SNAPSHOT, completeness="reference_snapshot", availability="provider reference effective-through", revision=RevisionPolicy.EXPLICIT_REFRESH),
        _spec("daily", endpoint="daily", native=NativeFrequency.TRADING_DAY, scope=ScopeKind.MARKET_SNAPSHOT, key=("ts_code", "trade_date"), fields=("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"), coverage=CoverageKind.TRADE_DATE, partition=("year", "month"), fetch=FetchStrategy.MARKET_SNAPSHOT_BY_DATE, completeness="market_date_snapshot", availability="sparse observation; lifecycle resolved downstream", limit=6000),
        _spec("daily_basic", endpoint="daily_basic", native=NativeFrequency.TRADING_DAY, scope=ScopeKind.MARKET_SNAPSHOT, key=("ts_code", "trade_date"), fields=("ts_code", "trade_date", "close", "turnover_rate", "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "total_mv", "circ_mv"), coverage=CoverageKind.TRADE_DATE, partition=("year", "month"), fetch=FetchStrategy.MARKET_SNAPSHOT_BY_DATE, completeness="market_date_snapshot", availability="sparse provider observation", limit=6000),
        _spec("adj_factor", endpoint="adj_factor", native=NativeFrequency.TRADING_DAY, scope=ScopeKind.MARKET_SNAPSHOT, key=("ts_code", "trade_date"), fields=("ts_code", "trade_date", "adj_factor"), coverage=CoverageKind.TRADE_DATE, partition=("year", "month"), fetch=FetchStrategy.MARKET_SNAPSHOT_BY_DATE, completeness="market_date_snapshot", availability="raw adjustment factor", limit=6000),
        _spec("suspend_d", endpoint="suspend_d", native=NativeFrequency.TRADING_DAY, scope=ScopeKind.MARKET_SNAPSHOT, key=("ts_code", "trade_date"), fields=("ts_code", "trade_date", "suspend_timing", "suspend_type"), coverage=CoverageKind.TRADE_DATE, partition=("year", "month"), fetch=FetchStrategy.MARKET_SNAPSHOT_BY_DATE, completeness="event_date_snapshot", availability="zero rows is a proven empty event snapshot", empty=True, limit=6000),
        _spec("index_daily", endpoint="index_daily", native=NativeFrequency.TRADING_DAY, scope=ScopeKind.ENTITY_SERIES, key=("ts_code", "trade_date"), fields=("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"), coverage=CoverageKind.ENTITY_TRADE_DATE, partition=("entity", "year", "month"), fetch=FetchStrategy.ENTITY_DATE_RANGE, completeness="entity_dates", availability="explicit index observation", limit=6000),
        _spec("index_weight", endpoint="index_weight", native=NativeFrequency.MONTHLY_SNAPSHOT, scope=ScopeKind.ENTITY_MONTH_SNAPSHOT, key=("index_code", "con_code", "trade_date"), fields=("index_code", "con_code", "trade_date", "weight"), coverage=CoverageKind.ENTITY_MONTH, partition=("entity", "year"), fetch=FetchStrategy.ENTITY_MONTH_SNAPSHOT, completeness="entity_month_snapshot", availability="point-in-time constituent weights", limit=2000),
    )
    for spec in specs:
        registry.register(spec)
    return registry
