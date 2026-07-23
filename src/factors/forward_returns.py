"""Auditable forward-return labels built on a unified market-date calendar.

Forward prices are intentionally used here because this module constructs
evaluation labels, not factor features. Labels must never be fed into factor
calculation, preprocessing, neutralization, or composition. Entry and exit
dates advance through the unique dates present in the complete price panel,
never through one stock's available-price rows. Missing prices therefore
remain missing and never delay either audit date.

The resulting forward return is not a realizable strategy return: transaction
costs, slippage, suspension execution, price limits, and portfolio mechanics
belong to later backtest stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Dict

import numpy as np
import pandas as pd


FORWARD_RETURN_AUDIT_COLUMNS = [
    "entry_trade_date",
    "exit_trade_date",
    "entry_price",
    "exit_price",
]


@dataclass(frozen=True)
class ForwardReturnConfig:
    """Configure exact market-period entry, exit, and price validation."""

    price_col: str = "close"
    return_col: str = "forward_return"
    entry_lag_periods: int = 1
    holding_periods: int = 20
    require_positive_prices: bool = True

    def __post_init__(self) -> None:
        """Validate names, market-period offsets, and price policy."""
        price_col = self._non_empty_name("price_col", self.price_col)
        return_col = self._non_empty_name("return_col", self.return_col)
        if price_col in {"trade_date", "ts_code"}:
            raise ValueError("price_col cannot conflict with trade_date or ts_code.")
        reserved_outputs = {
            "trade_date",
            "ts_code",
            *FORWARD_RETURN_AUDIT_COLUMNS,
        }
        if return_col in reserved_outputs:
            raise ValueError("return_col cannot conflict with output audit fields.")
        entry_lag = self._valid_integer(
            "entry_lag_periods", self.entry_lag_periods, minimum=0
        )
        holding = self._valid_integer(
            "holding_periods", self.holding_periods, minimum=1
        )
        if not isinstance(self.require_positive_prices, bool):
            raise ValueError("require_positive_prices must be a bool.")

        object.__setattr__(self, "price_col", price_col)
        object.__setattr__(self, "return_col", return_col)
        object.__setattr__(self, "entry_lag_periods", entry_lag)
        object.__setattr__(self, "holding_periods", holding)

    @staticmethod
    def _non_empty_name(field_name: str, value: object) -> str:
        """Return a stripped, non-empty column name."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _valid_integer(field_name: str, value: object, minimum: int) -> int:
        """Return an integer at or above a minimum while rejecting bool."""
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{field_name} must be an integer >= {minimum}.")
        normalized = int(value)
        if normalized < minimum:
            raise ValueError(f"{field_name} must be an integer >= {minimum}.")
        return normalized

    def to_dict(self) -> Dict[str, object]:
        """Return a serialization-friendly configuration dictionary."""
        return asdict(self)


class ForwardReturnBuilder:
    """Build exact-date future price labels for score-panel keys."""

    def __init__(self, config: ForwardReturnConfig | None = None) -> None:
        if config is not None and not isinstance(config, ForwardReturnConfig):
            raise TypeError("config must be a ForwardReturnConfig or None.")
        self.config = config if config is not None else ForwardReturnConfig()

    def describe_config(self) -> Dict[str, object]:
        """Return a serialization-friendly summary of the active configuration."""
        return self.config.to_dict()

    def build(
        self,
        score_panel: pd.DataFrame,
        price_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return one audited forward-return label per score-panel row."""
        scores = self._normalize_panel(
            score_panel,
            ["trade_date", "ts_code"],
            "score_panel",
        )
        prices = self._normalize_panel(
            price_panel,
            ["trade_date", "ts_code", self.config.price_col],
            "price_panel",
        )
        prices[self.config.price_col] = pd.to_numeric(
            prices[self.config.price_col], errors="coerce"
        ).astype(float).replace([np.inf, -np.inf], np.nan)

        output_columns = [
            "trade_date",
            "ts_code",
            *FORWARD_RETURN_AUDIT_COLUMNS,
            self.config.return_col,
        ]
        if scores.empty:
            return self._empty_output()

        market_calendar = pd.DatetimeIndex(
            prices["trade_date"].drop_duplicates().sort_values()
        )
        calendar_positions = {
            trade_date: position
            for position, trade_date in enumerate(market_calendar)
        }
        missing_score_dates = sorted(
            set(scores["trade_date"]) - set(market_calendar)
        )
        if missing_score_dates:
            missing_text = ", ".join(
                pd.Timestamp(date).isoformat() for date in missing_score_dates
            )
            raise ValueError(
                "Every score_panel trade_date must exist in the price-panel "
                f"market calendar; missing: {missing_text}."
            )

        price_lookup = prices.set_index(["trade_date", "ts_code"])[
            self.config.price_col
        ]
        rows = []
        for row in scores.itertuples(index=False):
            score_date = row.trade_date
            ts_code = row.ts_code
            score_index = calendar_positions[score_date]
            entry_index = score_index + self.config.entry_lag_periods
            exit_index = entry_index + self.config.holding_periods

            entry_date = pd.NaT
            exit_date = pd.NaT
            entry_price = np.nan
            exit_price = np.nan
            forward_return = np.nan

            if entry_index < len(market_calendar):
                entry_date = market_calendar[entry_index]
                entry_price = self._finite_price(
                    price_lookup.get((entry_date, ts_code), np.nan)
                )
                if exit_index < len(market_calendar):
                    exit_date = market_calendar[exit_index]
                    exit_price = self._finite_price(
                        price_lookup.get((exit_date, ts_code), np.nan)
                    )
                    if self._price_is_usable(entry_price) and self._price_is_usable(
                        exit_price
                    ):
                        calculated = float(exit_price / entry_price - 1.0)
                        if np.isfinite(calculated):
                            forward_return = calculated

            rows.append(
                {
                    "trade_date": score_date,
                    "ts_code": ts_code,
                    "entry_trade_date": entry_date,
                    "exit_trade_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    self.config.return_col: forward_return,
                }
            )

        result = pd.DataFrame(rows, columns=output_columns)
        for column in ("entry_price", "exit_price", self.config.return_col):
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(
                float
            ).replace([np.inf, -np.inf], np.nan)
        return result.sort_values(
            ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
        )

    def _price_is_usable(self, price: float) -> bool:
        """Return whether a finite price satisfies the configured sign policy."""
        if not np.isfinite(price):
            return False
        if self.config.require_positive_prices:
            return price > 0.0
        return price != 0.0

    @staticmethod
    def _finite_price(value: object) -> float:
        """Return a finite numeric price or NaN."""
        try:
            price = float(value)
        except (TypeError, ValueError):
            return np.nan
        return price if np.isfinite(price) else np.nan

    @staticmethod
    def _normalize_panel(
        panel: pd.DataFrame,
        required_columns: list[str],
        panel_name: str,
    ) -> pd.DataFrame:
        """Copy and validate dates, stripped stock codes, and unique keys."""
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(f"{panel_name} must be a pandas DataFrame.")
        missing = [
            column for column in required_columns if column not in panel.columns
        ]
        if missing:
            raise ValueError(
                f"{panel_name} is missing required columns: {', '.join(missing)}."
            )
        normalized = panel.loc[:, required_columns].copy(deep=True)
        dates = pd.to_datetime(normalized["trade_date"], errors="coerce")
        if dates.isna().any():
            raise ValueError(
                f"{panel_name} trade_date must contain valid, non-empty dates."
            )
        normalized["trade_date"] = dates
        codes = normalized["ts_code"].astype("string").str.strip()
        if codes.isna().any() or codes.eq("").any():
            raise ValueError(f"{panel_name} ts_code cannot contain empty values.")
        normalized["ts_code"] = codes
        if normalized.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(
                f"{panel_name} trade_date and ts_code combinations must be unique."
            )
        return normalized

    def _empty_output(self) -> pd.DataFrame:
        """Return an empty output with stable audit dtypes."""
        result = pd.DataFrame(
            {
                "trade_date": pd.Series(dtype="datetime64[ns]"),
                "ts_code": pd.Series(dtype="string"),
                "entry_trade_date": pd.Series(dtype="datetime64[ns]"),
                "exit_trade_date": pd.Series(dtype="datetime64[ns]"),
                "entry_price": pd.Series(dtype=float),
                "exit_price": pd.Series(dtype=float),
                self.config.return_col: pd.Series(dtype=float),
            }
        )
        return result
