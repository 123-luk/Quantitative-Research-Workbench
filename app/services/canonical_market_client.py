"""TuShare-shaped read adapter backed only by ledger-proven CURATED data."""

from __future__ import annotations

from datetime import date, timedelta
import pandas as pd

from src.data.canonical_store import PartitionedParquetStore, content_hash, normalize_frame
from src.data.contracts import canonical_date, normalize_scope
from src.data.coverage_ledger import CoverageLedger
from src.data.coverage_planner import scope_key
from src.data.dataset_registry import DatasetRegistry
from src.universe.data import STOCK_BASIC_SCOPE


class CanonicalPipelineDataUnavailable(RuntimeError):
    pass


def _calendar_units(start: str, end: str) -> tuple[str, ...]:
    first = date.fromisoformat(canonical_date(start))
    last = date.fromisoformat(canonical_date(end))
    return tuple((first + timedelta(days=index)).isoformat() for index in range((last - first).days + 1))


class CanonicalPipelineMarketClient:
    """Serve existing pipeline adapters without a provider or filesystem discovery."""

    def __init__(
        self,
        *,
        registry: DatasetRegistry,
        ledger: CoverageLedger,
        store: PartitionedParquetStore,
        stock_basic_as_of: str,
        market_scope: object = "CN_A",
        calendar_scope: object = (("exchange", "SSE"),),
    ) -> None:
        self.registry = registry
        self.ledger = ledger
        self.store = store
        self.stock_basic_as_of = canonical_date(stock_basic_as_of)
        self.market_scope = normalize_scope(market_scope)
        self.calendar_scope = normalize_scope(calendar_scope)

    def _read(self, dataset_id: str, scope: tuple[tuple[str, str], ...], units: tuple[str, ...]) -> pd.DataFrame:
        spec = self.registry.get(dataset_id)
        records = {
            record.unit_key: record
            for record in self.ledger.records(dataset_id)
            if record.scope_key == scope_key(scope) and record.status == "COMPLETE"
        }
        missing = tuple(unit for unit in units if unit not in records)
        if missing:
            raise CanonicalPipelineDataUnavailable(f"Canonical {dataset_id} coverage is incomplete.")
        frames: list[pd.DataFrame] = []
        for unit in units:
            rows = self.store.rows_for_unit(spec, unit=unit, scope=scope)
            record = records[unit]
            if len(rows) != record.row_count or content_hash(spec, rows) != record.content_hash:
                raise CanonicalPipelineDataUnavailable(f"Canonical {dataset_id} integrity validation failed.")
            if not rows.empty:
                frames.append(rows)
        if not frames:
            return pd.DataFrame(columns=spec.required_fields)
        return normalize_frame(spec, pd.concat(frames, ignore_index=True))

    def _open_units(self, start: object, end: object) -> tuple[str, ...]:
        start_date, end_date = canonical_date(start), canonical_date(end)
        calendar = self._read("trade_cal", self.calendar_scope, _calendar_units(start_date, end_date))
        return tuple(calendar.loc[calendar["is_open"].eq(1), "cal_date"].astype(str))

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self._read("trade_cal", self.calendar_scope, _calendar_units(start_date, end_date))

    def get_daily(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if trade_date is not None:
            units = (canonical_date(trade_date),)
        elif start_date is not None and end_date is not None:
            units = self._open_units(start_date, end_date)
        else:
            raise CanonicalPipelineDataUnavailable("daily requires an explicit date or range.")
        frame = self._read("daily", self.market_scope, units)
        return frame if ts_code is None else frame.loc[frame["ts_code"].eq(ts_code)].reset_index(drop=True)

    def get_index_daily(
        self,
        ts_code: str,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if trade_date is not None:
            units = (canonical_date(trade_date),)
        elif start_date is not None and end_date is not None:
            units = self._open_units(start_date, end_date)
        else:
            raise CanonicalPipelineDataUnavailable("index_daily requires an explicit date or range.")
        return self._read("index_daily", normalize_scope({"index_code": ts_code}), units)

    def get_stock_basic(self, list_status: str = "L") -> pd.DataFrame:
        frame = self._read("stock_basic", STOCK_BASIC_SCOPE, (self.stock_basic_as_of,))
        return frame.loc[frame["list_status"].eq(list_status)].reset_index(drop=True)

    def get_suspend_d(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        suspend_type: str | None = None,
    ) -> pd.DataFrame:
        del suspend_type
        if trade_date is not None:
            units = (canonical_date(trade_date),)
        elif start_date is not None and end_date is not None:
            units = self._open_units(start_date, end_date)
        else:
            raise CanonicalPipelineDataUnavailable("suspend_d requires an explicit date or range.")
        frame = self._read("suspend_d", self.market_scope, units)
        return frame if ts_code is None else frame.loc[frame["ts_code"].eq(ts_code)].reset_index(drop=True)
