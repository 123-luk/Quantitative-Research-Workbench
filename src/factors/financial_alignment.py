"""Point-in-time alignment of financial announcements to trading dates."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd

from src.factors.contracts import normalize_factor_input


FINANCIAL_KEY_FIELDS: Tuple[str, str, str] = ("ts_code", "ann_date", "end_date")
TRADING_KEY_FIELDS: Tuple[str, str] = ("trade_date", "ts_code")
RESERVED_OUTPUT_FIELDS = {
    "trade_date",
    "ts_code",
    "ann_date",
    "end_date",
    "source_ann_date",
    "source_end_date",
    "effective_trade_date",
}


def normalize_value_columns(value_columns: Sequence[str]) -> Tuple[str, ...]:
    """Validate requested financial value columns and return an immutable tuple."""
    if isinstance(value_columns, (str, bytes)):
        raise TypeError("value_columns must be a sequence of column names.")
    columns = tuple(value_columns)
    if not columns:
        raise ValueError("value_columns must contain at least one column name.")
    if any(not isinstance(column, str) or not column.strip() for column in columns):
        raise ValueError("value_columns cannot contain empty values.")
    if len(set(columns)) != len(columns):
        raise ValueError("value_columns cannot contain duplicate names.")
    reserved = sorted(set(columns) & RESERVED_OUTPUT_FIELDS)
    if reserved:
        raise ValueError(
            "value_columns cannot contain key or audit fields: "
            + ", ".join(reserved)
            + "."
        )
    return columns


def normalize_trading_panel(trading_panel: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy a non-empty trading panel with normalized keys."""
    if not isinstance(trading_panel, pd.DataFrame):
        raise TypeError("trading_panel must be a pandas DataFrame.")
    if trading_panel.empty:
        raise ValueError("trading_panel cannot be empty.")
    return normalize_factor_input(trading_panel)


def normalize_financial_data(
    financial_data: pd.DataFrame,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Validate and copy financial records with normalized dates and stock codes."""
    columns = normalize_value_columns(value_columns)
    if not isinstance(financial_data, pd.DataFrame):
        raise TypeError("financial_data must be a pandas DataFrame.")
    if financial_data.empty:
        raise ValueError("financial_data cannot be empty.")

    required_fields = set(FINANCIAL_KEY_FIELDS) | set(columns)
    missing_fields = sorted(required_fields - set(financial_data.columns))
    if missing_fields:
        raise ValueError(
            "financial_data is missing required fields: "
            + ", ".join(missing_fields)
            + "."
        )

    normalized = financial_data.copy(deep=True)
    ts_codes = normalized["ts_code"].astype("string").str.strip()
    if ts_codes.isna().any() or ts_codes.eq("").any():
        raise ValueError("financial_data ts_code cannot be empty.")
    normalized["ts_code"] = ts_codes

    for field_name in ("ann_date", "end_date"):
        if normalized[field_name].isna().any():
            raise ValueError(f"financial_data {field_name} cannot be empty.")
        dates = pd.to_datetime(normalized[field_name], errors="coerce")
        if dates.isna().any():
            raise ValueError(
                f"financial_data contains invalid or empty {field_name} values."
            )
        normalized[field_name] = dates

    duplicate_keys = normalized.duplicated(list(FINANCIAL_KEY_FIELDS), keep=False)
    if duplicate_keys.any():
        raise ValueError(
            "financial_data contains duplicate ts_code + ann_date + end_date rows."
        )
    return normalized


class FinancialPointInTimeAligner:
    """Align financial records by announcement availability for each stock.

    For each announcement, the first eligible trading date is the first date in
    that stock's input trading panel greater than or equal to ``ann_date``.
    ``availability_lag_trading_days`` advances from that position using actual
    trading rows, never calendar-day arithmetic. With lag 1, an announcement on
    a trading day becomes available on the next trading day; a weekend
    announcement becomes available one trading row after the first subsequent
    trading day.

    Records are matched independently by ``ts_code``. If multiple records share
    an effective trading date, the later ``ann_date`` wins, followed by the
    later ``end_date``. Output is stably sorted by ``trade_date`` then
    ``ts_code`` and retains ``source_ann_date`` and ``source_end_date``.
    """

    def align(
        self,
        trading_panel: pd.DataFrame,
        financial_data: pd.DataFrame,
        value_columns: Sequence[str],
        availability_lag_trading_days: int = 1,
    ) -> pd.DataFrame:
        """Return point-in-time financial values aligned to every trading row."""
        if isinstance(availability_lag_trading_days, bool) or not isinstance(
            availability_lag_trading_days,
            int,
        ):
            raise TypeError("availability_lag_trading_days must be an integer.")
        if availability_lag_trading_days < 0:
            raise ValueError(
                "availability_lag_trading_days must be greater than or equal to 0."
            )

        columns = normalize_value_columns(value_columns)
        trading = normalize_trading_panel(trading_panel)
        financial = normalize_financial_data(financial_data, columns)
        output_columns = [
            "trade_date",
            "ts_code",
            "source_ann_date",
            "source_end_date",
            *columns,
        ]

        aligned_frames: List[pd.DataFrame] = []
        for ts_code, stock_trading in trading.groupby("ts_code", sort=True):
            stock_trading = stock_trading.sort_values(
                "trade_date",
                kind="mergesort",
            ).loc[:, ["trade_date", "ts_code"]]
            stock_financial = financial.loc[financial["ts_code"] == ts_code].sort_values(
                ["ann_date", "end_date"],
                kind="mergesort",
            )

            effective_records = []
            trade_dates = stock_trading["trade_date"].reset_index(drop=True)
            for _, record in stock_financial.iterrows():
                first_position = int(trade_dates.searchsorted(record["ann_date"], side="left"))
                effective_position = first_position + availability_lag_trading_days
                if effective_position >= len(trade_dates):
                    continue
                effective_records.append(
                    {
                        "effective_trade_date": trade_dates.iloc[effective_position],
                        "source_ann_date": record["ann_date"],
                        "source_end_date": record["end_date"],
                        **{column: record[column] for column in columns},
                    }
                )

            if not effective_records:
                stock_output = stock_trading.copy()
                stock_output["source_ann_date"] = pd.NaT
                stock_output["source_end_date"] = pd.NaT
                for column in columns:
                    stock_output[column] = pd.NA
                aligned_frames.append(stock_output.loc[:, output_columns])
                continue

            effective = pd.DataFrame(effective_records).sort_values(
                ["effective_trade_date", "source_ann_date", "source_end_date"],
                kind="mergesort",
            )
            effective = effective.drop_duplicates(
                "effective_trade_date",
                keep="last",
            )
            stock_output = pd.merge_asof(
                stock_trading.sort_values("trade_date", kind="mergesort"),
                effective.sort_values("effective_trade_date", kind="mergesort"),
                left_on="trade_date",
                right_on="effective_trade_date",
                direction="backward",
                allow_exact_matches=True,
            ).drop(columns="effective_trade_date")
            aligned_frames.append(stock_output.loc[:, output_columns])

        result = pd.concat(aligned_frames, ignore_index=True)
        return result.sort_values(
            ["trade_date", "ts_code"],
            kind="mergesort",
            ignore_index=True,
        ).loc[:, output_columns]
