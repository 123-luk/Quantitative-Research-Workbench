from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from src.data.contracts import ResearchFrequency
from src.universe import (
    CanonicalUniverseSlice,
    UniverseConfigError,
    UniverseDataUnavailable,
    UniverseResolverRegistry,
    UniverseService,
    UniverseSnapshot,
    UniverseSpec,
    UniverseType,
    create_default_universe_registry,
    universe_spec_from_legacy,
)


STOCK_COLUMNS = (
    "ts_code", "symbol", "name", "area", "industry", "market", "exchange",
    "curr_type", "list_status", "list_date", "delist_date",
)


def stock(code: str, *, symbol: str | None = None, name: str = "Normal", market: str = "主板", exchange: str = "SSE", currency: str = "CNY", listed: str = "2010-01-01", delisted: str | None = None, status: str = "L") -> dict[str, object]:
    return {
        "ts_code": code,
        "symbol": symbol or code[:6],
        "name": name,
        "area": "China",
        "industry": "Test",
        "market": market,
        "exchange": exchange,
        "curr_type": currency,
        "list_status": status,
        "list_date": listed,
        "delist_date": delisted,
    }


def stocks(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=STOCK_COLUMNS)


@dataclass
class FakeDataSource:
    stock_frame: pd.DataFrame
    weights_frame: pd.DataFrame
    stock_calls: int = 0
    weight_calls: int = 0

    def stock_basic(self) -> CanonicalUniverseSlice:
        self.stock_calls += 1
        return CanonicalUniverseSlice(self.stock_frame, "stock_basic", "1.1", "2026-12-31", "stock:fixture")

    def index_weight(self, index_code: str, through_date: str) -> CanonicalUniverseSlice:
        self.weight_calls += 1
        return CanonicalUniverseSlice(self.weights_frame, "index_weight", "1.0", through_date, "weights:fixture")


def weights(rows: list[tuple[str, str, float]], index: str = "000300.SH") -> pd.DataFrame:
    return pd.DataFrame(
        [{"index_code": index, "con_code": code, "trade_date": day, "weight": weight} for day, code, weight in rows],
        columns=("index_code", "con_code", "trade_date", "weight"),
    )


def empty_weights() -> pd.DataFrame:
    return pd.DataFrame(columns=("index_code", "con_code", "trade_date", "weight"))


def test_universe_spec_parse_serialize_and_strict_params() -> None:
    custom = UniverseSpec.custom(["600519.SH", "600519.SH", "000001.sz"])
    assert custom.to_dict() == {"universe_type": "CUSTOM", "params": {"securities": ["600519.SH", "000001.SZ"]}}
    assert UniverseSpec.from_dict(custom.to_dict()) == custom
    assert UniverseSpec.index("000300.sh").to_dict()["params"] == {"index_code": "000300.SH"}
    assert UniverseSpec.all_a_shares().to_dict() == {"universe_type": "ALL_A_SHARES", "params": {}}
    for invalid in (
        {"universe_type": "UNKNOWN", "params": {}},
        {"universe_type": "ALL_A_SHARES", "params": {"exclude_st": True}},
        {"universe_type": "INDEX", "params": {"index_code": "hs300"}},
    ):
        with pytest.raises(UniverseConfigError):
            UniverseSpec.from_dict(invalid)
    with pytest.raises(UniverseConfigError):
        UniverseSpec.custom([])


def test_default_registry_is_fresh_and_custom_plugin_is_supported() -> None:
    first = create_default_universe_registry()
    second = create_default_universe_registry()
    assert first is not second
    assert set(first.list_types()) == set(UniverseType)
    with pytest.raises(ValueError, match="already registered"):
        first.register(first.get(UniverseType.CUSTOM))
    empty = UniverseResolverRegistry()
    with pytest.raises(KeyError, match="No resolver"):
        empty.get(UniverseType.INDEX)

    class Plugin:
        universe_type = UniverseType.CUSTOM
        def resolve(self, spec, formation_date, services):
            return UniverseSnapshot(formation_date, (), UniverseType.CUSTOM, "plugin", formation_date, {})
        def requirements(self, spec, start, end, frequency):
            return ()
    empty.register(Plugin())
    assert UniverseService(empty).resolve(UniverseSpec.custom(["600519.SH"]), "2024-01-02", FakeDataSource(stocks(stock("600519.SH")), empty_weights())).source_identity == "plugin"


def test_custom_code_normalization_unique_bare_and_ambiguous() -> None:
    source = FakeDataSource(
        stocks(stock("600519.SH"), stock("000001.SZ"), stock("900001.SH", symbol="123456"), stock("800001.BJ", symbol="123456", exchange="BSE", market="北交所")),
        empty_weights(),
    )
    service = UniverseService()
    canonical = service.canonicalize_spec(UniverseSpec.custom(["000001", "600519.SH", "000001.SZ"]), source)
    assert canonical.params["securities"] == ("000001.SZ", "600519.SH")
    with pytest.raises(UniverseConfigError, match="no canonical"):
        service.canonicalize_spec(UniverseSpec.custom(["654321"]), source)
    with pytest.raises(UniverseConfigError, match="ambiguous"):
        service.canonicalize_spec(UniverseSpec.custom(["123456"]), source)


def test_custom_lifecycle_is_exclusive_on_delist_date_and_preserves_order() -> None:
    source = FakeDataSource(
        stocks(
            stock("600001.SH", listed="2010-01-01"),
            stock("000001.SZ", listed="2025-01-01"),
            stock("600002.SH", listed="2010-01-01", delisted="2024-01-10", status="D"),
            stock("600003.SH", listed="2010-01-01", delisted="2023-12-31", status="D"),
        ),
        empty_weights(),
    )
    spec = UniverseSpec.custom(["600002.SH", "600001.SH", "000001.SZ", "600003.SH"])
    before = UniverseService().resolve(spec, "2024-01-09", source)
    boundary = UniverseService().resolve(spec, "2024-01-10", source)
    assert before.securities == ("600002.SH", "600001.SH")
    assert boundary.securities == ("600001.SH",)
    assert boundary.diagnostics["lifecycle_boundary"] == "list_date <= T < delist_date"


def test_index_survivorship_and_future_snapshot_leakage_are_forbidden() -> None:
    all_codes = ["600001.SH", "600002.SH", "600003.SH", "000001.SZ", "000002.SZ", "000003.SZ"]
    source = FakeDataSource(
        stocks(*(stock(code, exchange="SZSE" if code.endswith("SZ") else "SSE") for code in all_codes)),
        weights(
            [("2019-01-31", code, 33.3) for code in all_codes[:3]]
            + [("2026-01-31", code, 33.3) for code in all_codes[3:]]
        ),
    )
    snapshot = UniverseService().resolve(UniverseSpec.index("000300.SH"), "2019-06-28", source)
    assert snapshot.securities == tuple(sorted(all_codes[:3]))
    assert not set(snapshot.securities) & set(all_codes[3:])
    assert snapshot.source_as_of == "2019-01-31"
    assert not hasattr(snapshot, "target_weights")

    source.weights_frame = weights([
        ("2024-01-31", "600001.SH", 100.0),
        ("2024-06-28", "000001.SZ", 100.0),
    ])
    april = UniverseService().resolve(UniverseSpec.index("000300.SH"), "2024-04-30", source)
    assert april.securities == ("600001.SH",)
    assert april.source_as_of == "2024-01-31"


def test_index_before_first_snapshot_fails_instead_of_backward_fill() -> None:
    source = FakeDataSource(stocks(stock("600001.SH")), weights([("2024-06-28", "600001.SH", 100.0)]))
    with pytest.raises(UniverseDataUnavailable, match="on or before"):
        UniverseService().resolve(UniverseSpec.index("000300.SH"), "2024-04-30", source)


def test_index_membership_keeps_suspended_or_missing_observation_constituent() -> None:
    source = FakeDataSource(stocks(stock("600001.SH")), weights([("2024-01-31", "600001.SH", 100.0)]))
    snapshot = UniverseService().resolve(UniverseSpec.index("000300.SH"), "2024-02-29", source)
    assert snapshot.securities == ("600001.SH",)
    assert source.stock_calls == 1 and source.weight_calls == 1


def test_all_a_shares_classification_lifecycle_and_no_hidden_eligibility() -> None:
    source = FakeDataSource(
        stocks(
            stock("600001.SH", name="ST Included"),
            stock("000001.SZ", market="主板", exchange="SZSE", listed="2024-01-26"),
            stock("300001.SZ", market="创业板", exchange="SZSE"),
            stock("688001.SH", market="科创板"),
            stock("830001.BJ", market="北交所", exchange="BSE"),
            stock("900901.SH", market="主板", currency="USD"),
            stock("600002.SH", market="CDR"),
            stock("600003.SH", delisted="2024-01-31", status="D"),
        ),
        empty_weights(),
    )
    snapshot = UniverseService().resolve(UniverseSpec.all_a_shares(), "2024-01-31", source)
    assert snapshot.securities == ("000001.SZ", "300001.SZ", "600001.SH", "688001.SH", "830001.BJ")
    assert "600001.SH" in snapshot.securities  # ST is membership, not eligibility.
    assert "000001.SZ" in snapshot.securities  # Newly listed remains a member.
    assert "600003.SH" not in snapshot.securities  # Exclusive delist boundary.
    assert snapshot.diagnostics["st_suspension_listing_age_filters"] is False


def test_all_a_shares_unknown_market_fails_closed() -> None:
    source = FakeDataSource(stocks(stock("600001.SH", market="UNKNOWN")), empty_weights())
    with pytest.raises(UniverseDataUnavailable, match="unsupported market"):
        UniverseService().resolve(UniverseSpec.all_a_shares(), "2024-01-31", source)


def test_daily_and_monthly_schedules_reuse_canonical_formation_helper() -> None:
    source = FakeDataSource(stocks(stock("600001.SH")), empty_weights())
    dates = ("2024-01-30", "2024-01-31", "2024-02-27", "2024-02-29")
    service = UniverseService()
    daily = service.resolve_schedule(UniverseSpec.all_a_shares(), frequency=ResearchFrequency.DAILY, open_dates=dates, services=source)
    monthly = service.resolve_schedule(UniverseSpec.all_a_shares(), frequency=ResearchFrequency.MONTHLY, open_dates=dates, services=source)
    assert tuple(item.formation_date for item in daily) == dates
    assert tuple(item.formation_date for item in monthly) == ("2024-01-31", "2024-02-29")


def test_exact_universe_requirements_have_no_unnecessary_market_datasets() -> None:
    service = UniverseService()
    custom = service.requirements(UniverseSpec.custom(["600001.SH"]), start="2024-01-01", end="2024-12-31", frequency=ResearchFrequency.DAILY)
    index = service.requirements(UniverseSpec.index("000300.SH"), start="2024-01-01", end="2024-12-31", frequency=ResearchFrequency.MONTHLY)
    all_a = service.requirements(UniverseSpec.all_a_shares(), start="2024-01-01", end="2024-12-31", frequency=ResearchFrequency.DAILY)
    assert tuple(item.dataset_id for item in custom) == ("stock_basic",)
    assert tuple(item.dataset_id for item in index) == ("index_weight", "stock_basic")
    assert tuple(item.dataset_id for item in all_a) == ("stock_basic",)
    weight = next(item for item in index if item.dataset_id == "index_weight")
    assert dict(weight.scope) == {"index_code": "000300.SH"}
    assert weight.required_start == "2023-12-01"
    for requirements in (custom, index, all_a):
        assert not ({"daily", "daily_basic", "adj_factor", "suspend_d"} & {item.dataset_id for item in requirements})


def test_snapshot_is_detached_immutable_and_has_no_business_payload() -> None:
    source = FakeDataSource(stocks(stock("600001.SH")), empty_weights())
    snapshot = UniverseService().resolve(UniverseSpec.all_a_shares(), "2024-01-31", source)
    source.stock_frame.loc[0, "ts_code"] = "000001.SZ"
    assert snapshot.securities == ("600001.SH",)
    with pytest.raises(TypeError):
        snapshot.diagnostics["new"] = True  # type: ignore[index]
    assert not hasattr(snapshot, "factor_values")
    assert not hasattr(snapshot, "target_weight")


def test_legacy_adapter_is_explicit_and_canonical() -> None:
    assert universe_spec_from_legacy("hs300") == UniverseSpec.index("000300.SH")
    assert universe_spec_from_legacy("all") == UniverseSpec.all_a_shares()
    with pytest.raises(UniverseConfigError):
        universe_spec_from_legacy("mystery")
