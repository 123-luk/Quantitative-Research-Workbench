"""Post-close rebalance accounting for the canonical V6 research backtest."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from src.holdings import (
    HOLDINGS_OUTPUT_COLUMNS,
    WEIGHT_SUM_ABSOLUTE_TOLERANCE,
    validate_holdings_columns,
)
from src.research_backtest.availability import (
    SECURITY_STATUS_COLUMNS,
    resolve_security_return,
)
from src.research_backtest.calendar import (
    TradingCalendar,
    TradingCalendarDataError,
    _calendar_date,
)
from src.research_backtest.returns import SECURITY_DAILY_RETURN_COLUMNS


REBALANCE_OUTPUT_COLUMNS = (
    "holdings_trade_date",
    "effective_date",
    "ts_code",
    "pre_rebalance_weight",
    "target_weight",
    "weight_change",
    "pre_cash_weight",
    "target_cash_weight",
    "cash_weight_change",
    "turnover",
)
WEIGHT_TOLERANCE = WEIGHT_SUM_ABSOLUTE_TOLERANCE


class RebalanceAccountingError(ValueError):
    """Base error for rebalance accounting."""


class RebalanceInputError(RebalanceAccountingError):
    """Raised when an input violates a canonical schema or value contract."""


class RebalanceScheduleError(RebalanceAccountingError):
    """Raised when holdings dates cannot form an unambiguous schedule."""


class WeightDriftError(RebalanceAccountingError):
    """Raised when portfolio weights cannot be drifted safely."""


class RebalanceInvariantError(RebalanceAccountingError):
    """Raised when complete-weight or turnover invariants are violated."""


def _date(value: object, *, field_name: str) -> pd.Timestamp:
    try:
        return _calendar_date(value, field_name=field_name)
    except TradingCalendarDataError as exc:
        raise RebalanceInputError(str(exc)) from exc


def _code(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise RebalanceInputError(f"{field_name} must be a string.")
    code = value.strip()
    if not code:
        raise RebalanceInputError(f"{field_name} must be non-empty.")
    return code


def _real(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RebalanceInputError(f"{field_name} must be a finite real value.")
    result = float(value)
    if not np.isfinite(result):
        raise RebalanceInputError(f"{field_name} must be a finite real value.")
    return result


def _canonical_holdings(value: object) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise RebalanceInputError("holdings must be a pandas DataFrame.")
    if value.empty:
        raise RebalanceInputError("holdings must not be empty.")
    try:
        validate_holdings_columns(value.columns)
    except ValueError as exc:
        raise RebalanceInputError(str(exc)) from exc
    rows = value.loc[:, list(HOLDINGS_OUTPUT_COLUMNS)].copy(deep=True)
    rows["trade_date"] = [
        _date(item, field_name=f"trade_date[{index!r}]")
        for index, item in rows["trade_date"].items()
    ]
    rows["ts_code"] = [
        _code(item, field_name=f"ts_code[{index!r}]")
        for index, item in rows["ts_code"].items()
    ]
    rows["target_weight"] = [
        _real(item, field_name=f"target_weight[{index!r}]")
        for index, item in rows["target_weight"].items()
    ]
    if rows["target_weight"].lt(0.0).any():
        raise RebalanceInputError("target_weight must be non-negative.")
    if rows.duplicated(["trade_date", "ts_code"]).any():
        raise RebalanceInputError("holdings keys must be unique.")
    sums = rows.groupby("trade_date", sort=True)["target_weight"].sum()
    if not np.isclose(
        sums.to_numpy(), 1.0, rtol=0.0, atol=WEIGHT_TOLERANCE
    ).all():
        raise RebalanceInputError(
            "target_weight must sum to 1 on every holdings trade date."
        )
    return rows.sort_values(
        ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
    )


def _canonical_daily_frame(
    value: object,
    *,
    columns: tuple[str, str, str],
    context: str,
) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise RebalanceInputError(f"{context} must be a pandas DataFrame.")
    if tuple(value.columns) != columns:
        raise RebalanceInputError(f"{context} columns must be {columns!r}.")
    rows = value.copy(deep=True)
    rows["trade_date"] = [
        _date(item, field_name=f"{context}.trade_date[{index!r}]")
        for index, item in rows["trade_date"].items()
    ]
    rows["ts_code"] = [
        _code(item, field_name=f"{context}.ts_code[{index!r}]")
        for index, item in rows["ts_code"].items()
    ]
    if rows.duplicated(["trade_date", "ts_code"]).any():
        raise RebalanceInputError(f"{context} keys must be unique.")
    return rows


def _return_lookup(value: object) -> dict[tuple[pd.Timestamp, str], float]:
    rows = _canonical_daily_frame(
        value,
        columns=SECURITY_DAILY_RETURN_COLUMNS,
        context="security_returns",
    )
    output: dict[tuple[pd.Timestamp, str], float] = {}
    for row in rows.to_dict("records"):
        output[(row["trade_date"], row["ts_code"])] = _real(
            row["return"], field_name="security_returns.return"
        )
    return output


def _status_lookup(value: object) -> dict[tuple[pd.Timestamp, str], str]:
    rows = _canonical_daily_frame(
        value,
        columns=SECURITY_STATUS_COLUMNS,
        context="security_status",
    )
    output: dict[tuple[pd.Timestamp, str], str] = {}
    for row in rows.itertuples(index=False):
        if not isinstance(row.status, str):
            raise RebalanceInputError("security_status.status must be a string.")
        output[(row.trade_date, row.ts_code)] = row.status
    return output


def _complete_state(weights: dict[str, float], cash: float, *, context: str) -> None:
    values = tuple(weights.values()) + (cash,)
    if any(not np.isfinite(item) or item < -WEIGHT_TOLERANCE for item in values):
        raise RebalanceInvariantError(f"{context} weights must be finite and non-negative.")
    if not np.isclose(sum(values), 1.0, rtol=0.0, atol=WEIGHT_TOLERANCE):
        raise RebalanceInvariantError(f"{context} asset and cash weights must sum to 1.")


def _drift_one_day(
    weights: dict[str, float],
    cash: float,
    trade_date: pd.Timestamp,
    returns: dict[tuple[pd.Timestamp, str], float],
    statuses: dict[tuple[pd.Timestamp, str], str],
) -> tuple[dict[str, float], float]:
    values: dict[str, float] = {}
    for code, weight in weights.items():
        if weight <= WEIGHT_TOLERANCE:
            values[code] = weight
            continue
        key = (trade_date, code)
        if key not in statuses:
            raise RebalanceInputError(
                "security_status lacks a required positive-weight key: "
                f"{trade_date.strftime('%Y-%m-%d')}, {code!r}."
            )
        daily_return = resolve_security_return(
            trade_date=trade_date,
            ts_code=code,
            status=statuses[key],
            observed_return=returns.get(key),
        )
        if daily_return < -1.0:
            raise WeightDriftError(
                f"return cannot be below -1 for {trade_date:%Y-%m-%d}, {code!r}."
            )
        values[code] = weight * (1.0 + daily_return)
    total = sum(values.values()) + cash
    if not np.isfinite(total) or total <= 0.0:
        raise WeightDriftError("drifted portfolio total value must be strictly positive.")
    drifted = {code: value / total for code, value in values.items()}
    drifted_cash = cash / total
    _complete_state(drifted, drifted_cash, context="drifted pre-rebalance")
    return drifted, drifted_cash


class RebalanceAccountingResult:
    """Defensively expose one canonical rebalance ledger."""

    __slots__ = ("_ledger",)

    def __init__(self, ledger: pd.DataFrame) -> None:
        if not isinstance(ledger, pd.DataFrame) or ledger.empty:
            raise RebalanceInvariantError("rebalance ledger must be non-empty.")
        if tuple(ledger.columns) != REBALANCE_OUTPUT_COLUMNS:
            raise RebalanceInvariantError("rebalance ledger has an invalid schema.")
        self._ledger = ledger.copy(deep=True)

    @property
    def rebalances(self) -> pd.DataFrame:
        return self._ledger.copy(deep=True)

    @property
    def event_count(self) -> int:
        return int(self._ledger["effective_date"].nunique())

    @property
    def first_effective_date(self) -> pd.Timestamp:
        return self._ledger["effective_date"].min()

    @property
    def last_effective_date(self) -> pd.Timestamp:
        return self._ledger["effective_date"].max()


@dataclass(frozen=True)
class RebalanceAccountingEngine:
    """Drift holdings and account complete half-L1 turnover at each close."""

    trading_calendar: TradingCalendar

    def __post_init__(self) -> None:
        if not isinstance(self.trading_calendar, TradingCalendar):
            raise TypeError("trading_calendar must be a canonical TradingCalendar.")

    def run(
        self,
        *,
        holdings: pd.DataFrame,
        security_returns: pd.DataFrame,
        security_status: pd.DataFrame,
    ) -> RebalanceAccountingResult:
        rows = _canonical_holdings(holdings)
        returns = _return_lookup(security_returns)
        statuses = _status_lookup(security_status)

        events: list[tuple[pd.Timestamp, pd.Timestamp, dict[str, float]]] = []
        effective_dates: set[pd.Timestamp] = set()
        for trade_date, group in rows.groupby("trade_date", sort=True):
            effective_date = self.trading_calendar.next_trading_day(trade_date)
            if effective_date in effective_dates:
                raise RebalanceScheduleError(
                    "multiple holdings trade dates map to effective date "
                    f"{effective_date.strftime('%Y-%m-%d')}."
                )
            effective_dates.add(effective_date)
            targets = dict(zip(group["ts_code"], group["target_weight"]))
            events.append((trade_date, effective_date, targets))

        output: list[dict[str, object]] = []
        old_weights: dict[str, float] = {}
        old_cash = 1.0
        previous_effective: pd.Timestamp | None = None
        for trade_date, effective_date, targets in events:
            if previous_effective is not None:
                drift_dates = (
                    item
                    for item in self.trading_calendar.open_dates
                    if previous_effective < item <= effective_date
                )
                for drift_date in drift_dates:
                    old_weights, old_cash = _drift_one_day(
                        old_weights,
                        old_cash,
                        drift_date,
                        returns,
                        statuses,
                    )
            _complete_state(old_weights, old_cash, context="pre-rebalance")
            _complete_state(targets, 0.0, context="target")
            codes = sorted(set(old_weights) | set(targets))
            asset_changes = {
                code: targets.get(code, 0.0) - old_weights.get(code, 0.0)
                for code in codes
            }
            cash_change = -old_cash
            turnover = 0.5 * (
                sum(abs(item) for item in asset_changes.values()) + abs(cash_change)
            )
            if turnover < -WEIGHT_TOLERANCE or turnover > 1.0 + WEIGHT_TOLERANCE:
                raise RebalanceInvariantError("long-only turnover must be in [0, 1].")
            for code in codes:
                output.append(
                    {
                        "holdings_trade_date": trade_date,
                        "effective_date": effective_date,
                        "ts_code": code,
                        "pre_rebalance_weight": old_weights.get(code, 0.0),
                        "target_weight": targets.get(code, 0.0),
                        "weight_change": asset_changes[code],
                        "pre_cash_weight": old_cash,
                        "target_cash_weight": 0.0,
                        "cash_weight_change": cash_change,
                        "turnover": turnover,
                    }
                )
            old_weights = dict(targets)
            old_cash = 0.0
            previous_effective = effective_date

        ledger = pd.DataFrame(output, columns=list(REBALANCE_OUTPUT_COLUMNS))
        ledger = ledger.sort_values(
            ["effective_date", "ts_code"], kind="mergesort", ignore_index=True
        )
        return RebalanceAccountingResult(ledger)
