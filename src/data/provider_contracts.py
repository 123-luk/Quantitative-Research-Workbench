"""Machine-readable TuShare endpoint and dependency contracts.

Unknown official rules remain explicit.  Provider observations must never be
promoted into official facts by mutating this registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.data.provider_registry import ProviderId


UNKNOWN = "UNKNOWN"
OFFICIAL_NOT_STATED = "OFFICIAL_NOT_STATED"
PROXY_RULE_UNKNOWN = "PROXY_RULE_UNKNOWN"


class OfficialStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"
    DOCUMENTED_BUT_RULES_INCOMPLETE = "DOCUMENTED_BUT_RULES_INCOMPLETE"
    SEPARATE_PERMISSION = "SEPARATE_PERMISSION"


@dataclass(frozen=True)
class EndpointContract:
    provider_id: str
    dataset_id: str
    api_name: str
    official_url: str
    doc_id: int
    official_status: OfficialStatus
    minimum_points: int | str
    separate_permission: bool | str
    calls_per_minute: int | str
    calls_per_day: int | str
    max_rows: int | str
    update_time: str
    recommended_query: str
    required_parameters: tuple[str, ...]
    optional_parameters: tuple[str, ...]
    output_fields: tuple[str, ...]
    canonical_fields: tuple[str, ...]
    primary_key: tuple[str, ...]
    empty_semantics: str
    point_in_time_rule: str
    retryable_errors: tuple[str, ...]
    validators: tuple[str, ...]
    dependencies: tuple[str, ...]


_DOCS = "https://tushare.pro/document/2?doc_id={}"


def _official(
    dataset: str,
    doc: int,
    fields: tuple[str, ...],
    key: tuple[str, ...],
    *,
    points: int | str,
    per_minute: int | str = OFFICIAL_NOT_STATED,
    per_day: int | str = OFFICIAL_NOT_STATED,
    rows: int | str = OFFICIAL_NOT_STATED,
    update: str = OFFICIAL_NOT_STATED,
    query: str = OFFICIAL_NOT_STATED,
    required: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
    empty: str = "empty response is not automatically proof of coverage",
    pit: str = OFFICIAL_NOT_STATED,
    dependencies: tuple[str, ...] = (),
    status: OfficialStatus = OfficialStatus.AVAILABLE,
) -> EndpointContract:
    return EndpointContract(
        ProviderId.TUSHARE_OFFICIAL.value, dataset, dataset, _DOCS.format(doc), doc,
        status, points, False, per_minute, per_day, rows, update, query,
        required, optional, fields, fields, key, empty, pit,
        ("NETWORK_ERROR", "RATE_LIMITED"),
        ("schema", "types", "primary_key", "dates", "row_limit", "domain"),
        dependencies,
    )


_OFFICIAL = (
    _official("stock_basic", 25, ("ts_code", "symbol", "name", "area", "industry", "market", "exchange", "curr_type", "list_status", "list_date", "delist_date"), ("ts_code",), points=2000, per_minute=50, rows=5000, update=OFFICIAL_NOT_STATED, query="one reference snapshot per list_status", optional=("exchange", "list_status", "fields"), pit="reference snapshot; historical lifecycle fields prevent current-only universe filtering", dependencies=("all-A universe", "custom/index lifecycle", "research backtest lifecycle")),
    _official("trade_cal", 26, ("exchange", "cal_date", "is_open", "pretrade_date"), ("exchange", "cal_date"), points=2000, query="one exchange date range, incrementally cached", optional=("exchange", "start_date", "end_date", "is_open", "fields"), pit="calendar known for requested date", dependencies=("all research dates", "warm-up", "rebalance calendar")),
    _official("daily", 27, ("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"), ("ts_code", "trade_date"), points="BASIC_DAILY_PERMISSION", rows=6000, update="after market close; exact time OFFICIAL_NOT_STATED", query="one market trade_date", optional=("ts_code", "trade_date", "start_date", "end_date", "offset", "limit", "fields"), empty="suspended securities have no daily row; an empty market date is not suspension proof", pit="after-market-close data must not be used before availability", dependencies=("momentum", "volatility", "Amihud", "returns", "risk models", "backtest")),
    _official("adj_factor", 28, ("ts_code", "trade_date", "adj_factor"), ("ts_code", "trade_date"), points=2000, rows=OFFICIAL_NOT_STATED, update=OFFICIAL_NOT_STATED, query="one market trade_date", optional=("ts_code", "trade_date", "start_date", "end_date", "fields"), pit="same-date availability OFFICIAL_NOT_STATED", dependencies=("adjusted prices", "momentum", "volatility", "Amihud", "forward returns")),
    _official("daily_basic", 32, ("ts_code", "trade_date", "close", "turnover_rate", "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm", "total_mv", "circ_mv"), ("ts_code", "trade_date"), points=2000, rows=6000, update=OFFICIAL_NOT_STATED, query="one market trade_date", optional=("ts_code", "trade_date", "start_date", "end_date", "fields"), pit="after-market-close data must not be used before availability", dependencies=("EP", "BP", "SP", "TTM dividend yield", "size", "turnover")),
    _official("index_weight", 96, ("index_code", "con_code", "trade_date", "weight"), ("index_code", "con_code", "trade_date"), points=2000, rows=OFFICIAL_NOT_STATED, update=OFFICIAL_NOT_STATED, query="one index monthly snapshot window", required=("index_code",), optional=("trade_date", "start_date", "end_date"), pit="use only a constituent snapshot effective on or before formation", dependencies=("historical index universe",)),
    _official("stk_limit", 183, ("trade_date", "ts_code", "pre_close", "up_limit", "down_limit"), ("ts_code", "trade_date"), points=2000, rows=5800, update=OFFICIAL_NOT_STATED, query="one market trade_date", optional=("ts_code", "trade_date", "start_date", "end_date", "fields"), pit="same-date availability OFFICIAL_NOT_STATED", dependencies=("limit-price capability audit",)),
    _official("suspend_d", 214, ("ts_code", "trade_date", "suspend_timing", "suspend_type"), ("ts_code", "trade_date"), points=OFFICIAL_NOT_STATED, rows=OFFICIAL_NOT_STATED, update="irregular", query="one market trade_date", optional=("ts_code", "trade_date", "start_date", "end_date", "suspend_type", "fields"), empty="only a successful, schema-valid zero-row response proves no events for the exact scope", pit="event availability time OFFICIAL_NOT_STATED", dependencies=("strict suspension mode", "standard-mode precise event enrichment"), status=OfficialStatus.DOCUMENTED_BUT_RULES_INCOMPLETE),
    _official("index_daily", 95, ("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"), ("ts_code", "trade_date"), points=OFFICIAL_NOT_STATED, rows=OFFICIAL_NOT_STATED, update=OFFICIAL_NOT_STATED, query="one index date range", required=("ts_code",), optional=("trade_date", "start_date", "end_date", "fields"), pit="after-market-close data must not be used before availability", dependencies=("benchmark returns",), status=OfficialStatus.DOCUMENTED_BUT_RULES_INCOMPLETE),
    _official("monthly", 145, ("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"), ("ts_code", "trade_date"), points=OFFICIAL_NOT_STATED, rows=OFFICIAL_NOT_STATED, update=OFFICIAL_NOT_STATED, query="legacy wrapper only; no canonical consumer", optional=("ts_code", "trade_date", "start_date", "end_date", "fields"), dependencies=("legacy TushareClient method",), status=OfficialStatus.DOCUMENTED_BUT_RULES_INCOMPLETE),
)


class ProviderContractRegistry:
    def __init__(self) -> None:
        self._contracts: dict[tuple[str, str], EndpointContract] = {}
        for item in _OFFICIAL:
            self._contracts[(item.provider_id, item.dataset_id)] = item
            proxy = EndpointContract(
                ProviderId.TUSHARE_PROXY.value, item.dataset_id, item.api_name,
                item.official_url, item.doc_id, item.official_status,
                "PROXY_ADVERTISED_5000_LEVEL_NOT_OFFICIALLY_VERIFIED",
                "PROXY_PERMISSION_UNVERIFIED", PROXY_RULE_UNKNOWN,
                PROXY_RULE_UNKNOWN, PROXY_RULE_UNKNOWN, PROXY_RULE_UNKNOWN,
                item.recommended_query, item.required_parameters,
                item.optional_parameters, item.output_fields,
                item.canonical_fields, item.primary_key, item.empty_semantics,
                item.point_in_time_rule, item.retryable_errors,
                item.validators, item.dependencies,
            )
            self._contracts[(proxy.provider_id, proxy.dataset_id)] = proxy

    def get(self, provider_id: str, dataset_id: str) -> EndpointContract:
        try:
            return self._contracts[(ProviderId(provider_id).value, dataset_id)]
        except (KeyError, ValueError) as exc:
            raise KeyError(f"Unknown provider contract: {provider_id}/{dataset_id}") from exc

    def for_provider(self, provider_id: str) -> tuple[EndpointContract, ...]:
        provider = ProviderId(provider_id).value
        return tuple(sorted((item for (key, _), item in self._contracts.items() if key == provider), key=lambda item: item.dataset_id))
