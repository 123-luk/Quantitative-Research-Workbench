from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.data.canonical_store import CanonicalDataError, PartitionedParquetStore, RawParquetStore, merge_canonical
from src.data.contracts import DataRequirement
from src.data.coverage_ledger import CoverageLedger
from src.data.dataset_registry import create_default_dataset_registry
from src.data.migration import LegacyCoverageMigrator
from src.data.preparation import CuratedTradingCalendarResolver, DataPreparationService, DataUnavailableError, MissingCredentialError


DAILY_FIELDS = ("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount")
INDEX_FIELDS = DAILY_FIELDS
SUSPEND_FIELDS = ("ts_code", "trade_date", "suspend_timing", "suspend_type")


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.fail = False

    def get_daily(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("daily", kwargs))
        if self.fail:
            raise RuntimeError("provider unavailable")
        day = str(kwargs["trade_date"])
        values = {name: 1.0 for name in DAILY_FIELDS}
        values.update(ts_code="000001.SZ", trade_date=day)
        return pd.DataFrame([values], columns=DAILY_FIELDS)

    def get_index_daily(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("index_daily", kwargs))
        if self.fail:
            raise RuntimeError("provider unavailable")
        start = pd.Timestamp(str(kwargs["start_date"]))
        end = pd.Timestamp(str(kwargs["end_date"]))
        rows = []
        for day in pd.date_range(start, end, freq="B"):
            values = {name: 1.0 for name in INDEX_FIELDS}
            values.update(ts_code=kwargs["ts_code"], trade_date=day.strftime("%Y%m%d"))
            rows.append(values)
        return pd.DataFrame(rows, columns=INDEX_FIELDS)

    def get_suspend_d(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(("suspend_d", kwargs))
        return pd.DataFrame(columns=SUSPEND_FIELDS)

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append(("trade_cal", {"start_date": start_date, "end_date": end_date}))
        return pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": day.strftime("%Y%m%d"), "is_open": int(day.weekday() < 5), "pretrade_date": None}
                for day in pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
            ]
        )


def service(tmp_path: Path, client: FakeClient, open_dates: tuple[str, ...]) -> DataPreparationService:
    return DataPreparationService(
        ledger=CoverageLedger(tmp_path / "metadata" / "catalog.sqlite"),
        curated_store=PartitionedParquetStore(tmp_path / "curated"),
        raw_store=RawParquetStore(tmp_path / "raw"),
        open_dates=lambda start, end: tuple(item for item in open_dates if start <= item <= end),
        client_factory=lambda _token: client,
    )


def daily_requirement(start: str, end: str) -> DataRequirement:
    return DataRequirement.create("daily", scope="CN_A", required_start=start, required_end=end, reason="test")


def test_empty_local_fetches_exact_dates_then_identical_call_is_offline(tmp_path) -> None:
    client = FakeClient()
    prepared = service(tmp_path, client, ("2024-01-02", "2024-01-03", "2024-01-04"))
    requirement = daily_requirement("2024-01-02", "2024-01-04")
    first = prepared.ensure((requirement,), credential="fake")
    assert first.provider_calls == 3
    assert [call[1]["trade_date"] for call in client.calls] == ["20240102", "20240103", "20240104"]
    client.calls.clear()
    second = prepared.ensure((requirement,))
    assert second.provider_calls == 0
    assert client.calls == []


def test_empty_local_bootstraps_calendar_before_daily_units(tmp_path) -> None:
    registry = create_default_dataset_registry()
    ledger = CoverageLedger(tmp_path / "metadata" / "catalog.sqlite")
    curated = PartitionedParquetStore(tmp_path / "curated")
    raw = RawParquetStore(tmp_path / "raw")
    resolver = CuratedTradingCalendarResolver(registry, ledger, curated, scope={"exchange": "SSE"})
    prepared = DataPreparationService(registry=registry, ledger=ledger, curated_store=curated, raw_store=raw, open_dates=resolver)
    client = FakeClient()
    calendar = DataRequirement.create("trade_cal", scope={"exchange": "SSE"}, required_start="2024-01-05", required_end="2024-01-08")
    daily = daily_requirement("2024-01-05", "2024-01-08")
    result = prepared.ensure((daily, calendar), client=client)
    assert result.status == "READY"
    assert client.calls[0][0] == "trade_cal"
    assert [item[1]["trade_date"] for item in client.calls if item[0] == "daily"] == ["20240105", "20240108"]


@pytest.mark.parametrize(
    "seed_start,seed_end,full_start,full_end,expected",
    [
        ("2024-01-02", "2024-01-03", "2024-01-02", "2024-01-05", ["20240104", "20240105"]),
        ("2024-01-04", "2024-01-05", "2024-01-02", "2024-01-05", ["20240102", "20240103"]),
    ],
)
def test_head_and_tail_gaps_fetch_only_missing(tmp_path, seed_start, seed_end, full_start, full_end, expected) -> None:
    dates = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")
    client = FakeClient()
    prepared = service(tmp_path, client, dates)
    prepared.ensure((daily_requirement(seed_start, seed_end),), client=client)
    client.calls.clear()
    prepared.ensure((daily_requirement(full_start, full_end),), client=client)
    assert [item[1]["trade_date"] for item in client.calls] == expected


def test_internal_and_two_disjoint_gaps_never_refetch_full_history(tmp_path) -> None:
    dates = tuple(f"2024-01-0{day}" for day in range(2, 7))
    client = FakeClient()
    prepared = service(tmp_path, client, dates)
    prepared.ensure((daily_requirement("2024-01-03", "2024-01-03"), daily_requirement("2024-01-05", "2024-01-05")), client=client)
    client.calls.clear()
    prepared.ensure((daily_requirement("2024-01-02", "2024-01-06"),), client=client)
    assert [item[1]["trade_date"] for item in client.calls] == ["20240102", "20240104", "20240106"]


def test_entity_a_complete_entity_b_fetches_only_b(tmp_path) -> None:
    dates = ("2024-01-02", "2024-01-03")
    client = FakeClient()
    prepared = service(tmp_path, client, dates)
    a = DataRequirement.create("index_daily", scope={"index_code": "A"}, required_start=dates[0], required_end=dates[-1])
    b = DataRequirement.create("index_daily", scope={"index_code": "B"}, required_start=dates[0], required_end=dates[-1])
    prepared.ensure((a,), client=client)
    client.calls.clear()
    result = prepared.ensure((a, b), client=client)
    assert result.provider_calls == 1
    assert client.calls[0][1]["ts_code"] == "B"


def test_zero_row_suspend_snapshot_is_complete_and_reused(tmp_path) -> None:
    client = FakeClient()
    prepared = service(tmp_path, client, ("2024-01-02",))
    requirement = DataRequirement.create("suspend_d", scope="CN_A", required_start="2024-01-02", required_end="2024-01-02")
    assert prepared.ensure((requirement,), client=client).provider_calls == 1
    client.calls.clear()
    assert prepared.ensure((requirement,)).provider_calls == 0
    assert prepared.verify_unit("suspend_d", scope="CN_A", unit="2024-01-02")


def test_nonempty_malformed_suspend_snapshot_remains_strict(tmp_path) -> None:
    class MalformedClient(FakeClient):
        def get_suspend_d(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame({"unexpected": ["value"]})

    client = MalformedClient()
    prepared = service(tmp_path, client, ("2024-01-02",))
    requirement = DataRequirement.create(
        "suspend_d", scope="CN_A", required_start="2024-01-02", required_end="2024-01-02"
    )
    with pytest.raises(DataUnavailableError):
        prepared.ensure((requirement,), client=client)
    assert not prepared.verify_unit("suspend_d", scope="CN_A", unit="2024-01-02")


def test_missing_credential_fails_before_provider(tmp_path) -> None:
    client = FakeClient()
    prepared = service(tmp_path, client, ("2024-01-02",))
    with pytest.raises(MissingCredentialError):
        prepared.ensure((daily_requirement("2024-01-02", "2024-01-02"),))
    assert client.calls == []


def test_provider_failure_keeps_existing_partition_and_no_false_coverage(tmp_path) -> None:
    client = FakeClient()
    prepared = service(tmp_path, client, ("2024-01-02", "2024-01-03"))
    prepared.ensure((daily_requirement("2024-01-02", "2024-01-02"),), client=client)
    spec = prepared.registry.get("daily")
    target = prepared.curated_store.partition_path(spec, unit="2024-01-02", scope=daily_requirement("2024-01-02", "2024-01-02").scope)
    before = target.read_bytes()
    client.fail = True
    with pytest.raises(DataUnavailableError):
        prepared.ensure((daily_requirement("2024-01-02", "2024-01-03"),), client=client)
    assert target.read_bytes() == before
    records = prepared.ledger.records("daily")
    assert [item.unit_key for item in records] == ["2024-01-02"]
    assert prepared.ledger.fetch_events()[-1]["status"] == "FAILED"


def test_write_failure_keeps_existing_and_coverage_unchanged(tmp_path, monkeypatch) -> None:
    client = FakeClient()
    prepared = service(tmp_path, client, ("2024-01-02", "2024-01-03"))
    prepared.ensure((daily_requirement("2024-01-02", "2024-01-02"),), client=client)
    spec = prepared.registry.get("daily")
    scope = daily_requirement("2024-01-02", "2024-01-03").scope
    target = prepared.curated_store.partition_path(spec, unit="2024-01-02", scope=scope)
    before = target.read_bytes()
    monkeypatch.setattr(prepared.curated_store, "_write_temp", lambda *_args: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(DataUnavailableError):
        prepared.ensure((daily_requirement("2024-01-02", "2024-01-03"),), client=client)
    assert target.read_bytes() == before
    assert [item.unit_key for item in prepared.ledger.records("daily")] == ["2024-01-02"]


def test_same_payload_is_idempotent_and_conflicting_payload_fails_closed() -> None:
    spec = create_default_dataset_registry().get("daily")
    row = {name: 1.0 for name in DAILY_FIELDS}
    row.update(ts_code="A", trade_date="20240102")
    frame = pd.DataFrame([row], columns=DAILY_FIELDS)
    assert len(merge_canonical(spec, frame, frame.copy())) == 1
    changed = frame.copy()
    changed.loc[0, "close"] = 99.0
    with pytest.raises(CanonicalDataError, match="conflicting"):
        merge_canonical(spec, frame, changed)


def test_content_hash_detects_manual_corruption(tmp_path) -> None:
    client = FakeClient()
    prepared = service(tmp_path, client, ("2024-01-02",))
    requirement = daily_requirement("2024-01-02", "2024-01-02")
    prepared.ensure((requirement,), client=client)
    assert prepared.verify_unit("daily", scope="CN_A", unit="2024-01-02")
    spec = prepared.registry.get("daily")
    target = prepared.curated_store.partition_path(spec, unit="2024-01-02", scope=requirement.scope)
    frame = pd.read_parquet(target)
    frame.loc[0, "close"] = 123.0
    frame.to_parquet(target, index=False)
    assert not prepared.verify_unit("daily", scope="CN_A", unit="2024-01-02")


def test_legacy_migration_imports_provable_entity_but_not_market_snapshot(tmp_path) -> None:
    registry = create_default_dataset_registry()
    ledger = CoverageLedger(tmp_path / "catalog.sqlite")
    migrator = LegacyCoverageMigrator(ledger, PartitionedParquetStore(tmp_path / "curated"))
    values = {name: 1.0 for name in INDEX_FIELDS}
    values.update(ts_code="IDX", trade_date="20240102")
    path = tmp_path / "index.parquet"
    pd.DataFrame([values], columns=INDEX_FIELDS).to_parquet(path, index=False)
    imported = migrator.import_file(registry.get("index_daily"), path, scope={"index_code": "IDX"})
    assert imported == ("2024-01-02",)
    market = tmp_path / "daily.parquet"
    pd.DataFrame([values], columns=DAILY_FIELDS).to_parquet(market, index=False)
    assert migrator.import_file(registry.get("daily"), market, scope="CN_A") == ()


def test_secret_never_persists_in_data_root(tmp_path) -> None:
    secret = "TEST_SECRET_TUSHARE_TOKEN_P4B_9F2A"
    client = FakeClient()
    prepared = service(tmp_path, client, ("2024-01-02",))
    prepared.ensure((daily_requirement("2024-01-02", "2024-01-02"),), credential=secret)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    with sqlite3.connect(tmp_path / "metadata" / "catalog.sqlite") as connection:
        assert secret not in str(connection.execute("SELECT * FROM fetch_events").fetchall())


def test_secret_is_redacted_if_client_initialization_fails(tmp_path) -> None:
    secret = "TEST_SECRET_TUSHARE_TOKEN_P4B_9F2A"
    prepared = DataPreparationService(
        ledger=CoverageLedger(tmp_path / "catalog.sqlite"),
        curated_store=PartitionedParquetStore(tmp_path / "curated"),
        raw_store=RawParquetStore(tmp_path / "raw"),
        open_dates=lambda _start, _end: ("2024-01-02",),
        client_factory=lambda token: (_ for _ in ()).throw(RuntimeError(token)),
    )
    with pytest.raises(DataUnavailableError) as captured:
        prepared.ensure((daily_requirement("2024-01-02", "2024-01-02"),), credential=secret)
    assert secret not in str(captured.value)
