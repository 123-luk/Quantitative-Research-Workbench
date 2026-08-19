"""Daily return, proportional cost, and NAV accounting for V6 research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.research_backtest.calendar import (
    TradingCalendar,
    TradingCalendarCoverageError,
)
from src.research_backtest.rebalance import (
    REBALANCE_OUTPUT_COLUMNS,
    WEIGHT_TOLERANCE,
    RebalanceAccountingResult,
    RebalanceInputError,
    WeightDriftError,
    _code,
    _date,
    _drift_one_day_with_return,
    _real,
    _return_lookup,
    _status_lookup,
)

if TYPE_CHECKING:
    from src.pipeline.research_backtest_config import (
        PortfolioAccountingConfig,
        TransactionCostConfig,
    )


DAILY_PORTFOLIO_COLUMNS = (
    "trade_date",
    "gross_return",
    "transaction_cost",
    "net_return",
    "gross_nav",
    "net_nav",
    "is_rebalance",
    "turnover",
    "traded_notional",
)


class PortfolioDailyAccountingError(ValueError):
    """Base error for daily portfolio accounting."""


class PortfolioDailyInputError(PortfolioDailyAccountingError):
    """Raised when canonical inputs or the evaluation window are invalid."""


class PortfolioConsistencyError(PortfolioDailyAccountingError):
    """Raised when the daily state disagrees with V6-C event accounting."""


class PortfolioTransactionCostError(PortfolioDailyAccountingError):
    """Raised when a proportional cost would destroy portfolio value."""


class PortfolioValueError(PortfolioDailyAccountingError):
    """Raised when a gross or net portfolio factor is economically invalid."""


@dataclass(frozen=True)
class _RebalanceEvent:
    effective_date: pd.Timestamp
    pre_weights: dict[str, float]
    target_weights: dict[str, float]
    pre_cash: float
    target_cash: float
    turnover: float
    traded_notional: float


def _portfolio_date(value: object, *, field_name: str) -> pd.Timestamp:
    try:
        return _date(value, field_name=field_name)
    except RebalanceInputError as exc:
        raise PortfolioDailyInputError(str(exc)) from exc


def _portfolio_code(value: object, *, field_name: str) -> str:
    try:
        return _code(value, field_name=field_name)
    except RebalanceInputError as exc:
        raise PortfolioDailyInputError(str(exc)) from exc


def _portfolio_real(value: object, *, field_name: str) -> float:
    try:
        return _real(value, field_name=field_name)
    except RebalanceInputError as exc:
        raise PortfolioDailyInputError(str(exc)) from exc


def _canonical_rebalances(
    value: object,
    trading_calendar: TradingCalendar,
) -> tuple[_RebalanceEvent, ...]:
    if isinstance(value, RebalanceAccountingResult):
        rows = value.rebalances
    elif isinstance(value, pd.DataFrame):
        rows = value.copy(deep=True)
    else:
        raise PortfolioDailyInputError(
            "rebalances must be a RebalanceAccountingResult or pandas DataFrame."
        )
    if rows.empty or tuple(rows.columns) != REBALANCE_OUTPUT_COLUMNS:
        raise PortfolioDailyInputError(
            f"rebalances must be non-empty with columns {REBALANCE_OUTPUT_COLUMNS!r}."
        )
    for column in ("holdings_trade_date", "effective_date"):
        rows[column] = [
            _portfolio_date(item, field_name=f"{column}[{index!r}]")
            for index, item in rows[column].items()
        ]
    rows["ts_code"] = [
        _portfolio_code(item, field_name=f"ts_code[{index!r}]")
        for index, item in rows["ts_code"].items()
    ]
    numeric_columns = (
        "pre_rebalance_weight",
        "target_weight",
        "weight_change",
        "pre_cash_weight",
        "target_cash_weight",
        "cash_weight_change",
        "turnover",
    )
    for column in numeric_columns:
        rows[column] = [
            _portfolio_real(item, field_name=f"{column}[{index!r}]")
            for index, item in rows[column].items()
        ]
    if rows.duplicated(["effective_date", "ts_code"]).any():
        raise PortfolioDailyInputError(
            "rebalances must have unique (effective_date, ts_code) keys."
        )
    rows = rows.sort_values(
        ["effective_date", "ts_code"], kind="mergesort", ignore_index=True
    )

    events: list[_RebalanceEvent] = []
    for effective_date, group in rows.groupby("effective_date", sort=True):
        try:
            is_open = trading_calendar.is_trading_day(effective_date)
        except TradingCalendarCoverageError as exc:
            raise PortfolioDailyInputError(str(exc)) from exc
        if not is_open:
            raise PortfolioDailyInputError(
                f"effective date {effective_date:%Y-%m-%d} must be open."
            )
        if group["holdings_trade_date"].nunique() != 1:
            raise PortfolioDailyInputError(
                "event holdings_trade_date must be identical on every row."
            )
        if group["holdings_trade_date"].iloc[0] >= effective_date:
            raise PortfolioDailyInputError(
                "effective_date must be strictly later than holdings_trade_date."
            )
        repeated = (
            "pre_cash_weight",
            "target_cash_weight",
            "cash_weight_change",
            "turnover",
        )
        if any(group[column].nunique(dropna=False) != 1 for column in repeated):
            raise PortfolioDailyInputError(
                "event cash fields and turnover must be identical on every row."
            )
        pre_cash = float(group["pre_cash_weight"].iloc[0])
        target_cash = float(group["target_cash_weight"].iloc[0])
        cash_change = float(group["cash_weight_change"].iloc[0])
        turnover = float(group["turnover"].iloc[0])
        weight_values = group[
            ["pre_rebalance_weight", "target_weight"]
        ].to_numpy(dtype=float)
        if (weight_values < -WEIGHT_TOLERANCE).any() or min(
            pre_cash, target_cash, turnover
        ) < -WEIGHT_TOLERANCE:
            raise PortfolioDailyInputError(
                "rebalance weights, cash, and turnover must be non-negative."
            )
        if not np.isclose(
            group["pre_rebalance_weight"].sum() + pre_cash,
            1.0,
            rtol=0.0,
            atol=WEIGHT_TOLERANCE,
        ):
            raise PortfolioDailyInputError("event pre-state must sum to 1.")
        if not np.isclose(
            group["target_weight"].sum() + target_cash,
            1.0,
            rtol=0.0,
            atol=WEIGHT_TOLERANCE,
        ):
            raise PortfolioDailyInputError("event target state must sum to 1.")
        expected_change = group["target_weight"] - group["pre_rebalance_weight"]
        if not np.isclose(
            group["weight_change"],
            expected_change,
            rtol=0.0,
            atol=WEIGHT_TOLERANCE,
        ).all():
            raise PortfolioDailyInputError("event weight_change is inconsistent.")
        if not np.isclose(
            cash_change,
            target_cash - pre_cash,
            rtol=0.0,
            atol=WEIGHT_TOLERANCE,
        ):
            raise PortfolioDailyInputError("event cash_weight_change is inconsistent.")
        traded_notional = float(group["weight_change"].abs().sum())
        expected_turnover = 0.5 * (traded_notional + abs(cash_change))
        if not np.isclose(
            turnover,
            expected_turnover,
            rtol=0.0,
            atol=WEIGHT_TOLERANCE,
        ):
            raise PortfolioDailyInputError("event turnover is inconsistent.")
        events.append(
            _RebalanceEvent(
                effective_date=effective_date,
                pre_weights=dict(
                    zip(group["ts_code"], group["pre_rebalance_weight"])
                ),
                target_weights=dict(zip(group["ts_code"], group["target_weight"])),
                pre_cash=pre_cash,
                target_cash=target_cash,
                turnover=turnover,
                traded_notional=traded_notional,
            )
        )
    return tuple(events)


def _assert_pre_state(
    event: _RebalanceEvent,
    weights: dict[str, float],
    cash: float,
) -> None:
    codes = set(event.pre_weights) | set(weights)
    for code in codes:
        if not np.isclose(
            event.pre_weights.get(code, 0.0),
            weights.get(code, 0.0),
            rtol=0.0,
            atol=WEIGHT_TOLERANCE,
        ):
            raise PortfolioConsistencyError(
                "computed pre-rebalance security weights disagree with V6-C for "
                f"effective_date={event.effective_date:%Y-%m-%d}, ts_code={code!r}."
            )
    if not np.isclose(
        event.pre_cash,
        cash,
        rtol=0.0,
        atol=WEIGHT_TOLERANCE,
    ):
        raise PortfolioConsistencyError(
            "computed pre-rebalance cash weight disagrees with V6-C for "
            f"effective_date={event.effective_date:%Y-%m-%d}."
        )


class PortfolioDailyAccountingResult:
    """Defensively expose canonical daily portfolio accounting and audit facts."""

    __slots__ = (
        "_daily_portfolio",
        "_start_date",
        "_end_date",
        "_rebalance_count",
        "_initial_nav",
        "_cost_bps",
    )

    def __init__(
        self,
        daily_portfolio: pd.DataFrame,
        *,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        rebalance_count: int,
        initial_nav: float,
        cost_bps: float,
    ) -> None:
        if (
            not isinstance(daily_portfolio, pd.DataFrame)
            or daily_portfolio.empty
            or tuple(daily_portfolio.columns) != DAILY_PORTFOLIO_COLUMNS
        ):
            raise PortfolioDailyInputError("daily_portfolio has an invalid schema.")
        self._daily_portfolio = daily_portfolio.copy(deep=True)
        self._start_date = start_date
        self._end_date = end_date
        self._rebalance_count = rebalance_count
        self._initial_nav = initial_nav
        self._cost_bps = cost_bps

    @property
    def daily_portfolio(self) -> pd.DataFrame:
        return self._daily_portfolio.copy(deep=True)

    @property
    def start_date(self) -> pd.Timestamp:
        return self._start_date

    @property
    def end_date(self) -> pd.Timestamp:
        return self._end_date

    @property
    def row_count(self) -> int:
        return len(self._daily_portfolio)

    @property
    def rebalance_count(self) -> int:
        return self._rebalance_count

    @property
    def initial_nav(self) -> float:
        return self._initial_nav

    @property
    def cost_bps(self) -> float:
        return self._cost_bps


@dataclass(frozen=True)
class PortfolioDailyAccountingEngine:
    """Build daily research returns and NAV from canonical rebalance events."""

    trading_calendar: TradingCalendar
    portfolio_config: PortfolioAccountingConfig
    transaction_cost_config: TransactionCostConfig
    suspension_mode: str = "STRICT_EVENT"

    def __post_init__(self) -> None:
        from src.pipeline.research_backtest_config import (
            PortfolioAccountingConfig,
            TransactionCostConfig,
        )

        if not isinstance(self.trading_calendar, TradingCalendar):
            raise TypeError("trading_calendar must be a canonical TradingCalendar.")
        object.__setattr__(
            self,
            "portfolio_config",
            PortfolioAccountingConfig.from_dict(self.portfolio_config),
        )
        object.__setattr__(
            self,
            "transaction_cost_config",
            TransactionCostConfig.from_dict(self.transaction_cost_config),
        )
        if self.suspension_mode not in {"STRICT_EVENT", "STANDARD_ROBUST"}:
            raise ValueError("invalid suspension_mode")

    def run(
        self,
        *,
        rebalances: RebalanceAccountingResult | pd.DataFrame,
        security_returns: pd.DataFrame,
        security_status: pd.DataFrame,
        end_date: object,
    ) -> PortfolioDailyAccountingResult:
        events = _canonical_rebalances(rebalances, self.trading_calendar)
        start = events[0].effective_date
        end = _portfolio_date(end_date, field_name="end_date")
        try:
            end_is_open = self.trading_calendar.is_trading_day(end)
        except TradingCalendarCoverageError as exc:
            raise PortfolioDailyInputError(str(exc)) from exc
        if not end_is_open:
            raise PortfolioDailyInputError("end_date must be an open trading date.")
        if end < start:
            raise PortfolioDailyInputError(
                "end_date must be on or after the first effective date."
            )
        evaluation_dates = tuple(
            item
            for item in self.trading_calendar.open_dates
            if start <= item <= end
        )
        try:
            returns = _return_lookup(security_returns)
            statuses = _status_lookup(security_status)
        except RebalanceInputError as exc:
            raise PortfolioDailyInputError(str(exc)) from exc

        event_by_date = {
            event.effective_date: event
            for event in events
            if event.effective_date <= end
        }
        weights: dict[str, float] = {}
        cash = 1.0
        gross_nav = self.portfolio_config.initial_nav
        net_nav = self.portfolio_config.initial_nav
        cost_rate = self.transaction_cost_config.cost_bps / 10_000.0
        output: list[dict[str, object]] = []
        for trade_date in evaluation_dates:
            try:
                weights, cash, gross_return = _drift_one_day_with_return(
                    weights,
                    cash,
                    trade_date,
                    returns,
                    statuses,
                    self.suspension_mode,
                )
            except RebalanceInputError as exc:
                raise PortfolioDailyInputError(str(exc)) from exc
            except WeightDriftError as exc:
                raise PortfolioValueError(str(exc)) from exc

            event = event_by_date.get(trade_date)
            is_rebalance = event is not None
            turnover = 0.0
            traded_notional = 0.0
            transaction_cost = 0.0
            if event is not None:
                _assert_pre_state(event, weights, cash)
                turnover = event.turnover
                traded_notional = event.traded_notional
                transaction_cost = traded_notional * cost_rate
                if (
                    not np.isfinite(transaction_cost)
                    or transaction_cost < 0.0
                    or transaction_cost >= 1.0
                ):
                    raise PortfolioTransactionCostError(
                        "transaction_cost must be finite and in [0, 1)."
                    )
                weights = dict(event.target_weights)
                cash = event.target_cash

            gross_factor = 1.0 + gross_return
            net_factor = gross_factor * (1.0 - transaction_cost)
            net_return = net_factor - 1.0
            gross_nav *= gross_factor
            net_nav *= net_factor
            values = (gross_return, net_return, gross_nav, net_nav)
            if (
                not np.isfinite(values).all()
                or gross_factor <= 0.0
                or net_factor <= 0.0
                or gross_nav <= 0.0
                or net_nav <= 0.0
            ):
                raise PortfolioValueError(
                    "daily return factors and NAV values must be finite and positive."
                )
            output.append(
                {
                    "trade_date": trade_date,
                    "gross_return": gross_return,
                    "transaction_cost": transaction_cost,
                    "net_return": net_return,
                    "gross_nav": gross_nav,
                    "net_nav": net_nav,
                    "is_rebalance": bool(is_rebalance),
                    "turnover": turnover,
                    "traded_notional": traded_notional,
                }
            )
        daily = pd.DataFrame(output, columns=list(DAILY_PORTFOLIO_COLUMNS))
        return PortfolioDailyAccountingResult(
            daily,
            start_date=start,
            end_date=end,
            rebalance_count=sum(event.effective_date <= end for event in events),
            initial_nav=self.portfolio_config.initial_nav,
            cost_bps=self.transaction_cost_config.cost_bps,
        )
