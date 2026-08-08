"""Pure in-memory normalization and deterministic ranking for V5 Signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.signals.contracts import (
    SIGNAL_KEY_COLUMNS,
    SIGNAL_OUTPUT_COLUMNS,
    SIGNAL_PROTECTED_SCORE_SOURCE_COLUMNS,
    SignalDataError,
    validate_signal_columns,
)


@dataclass(frozen=True)
class SignalBuildAudit:
    """Immutable summary of one deterministic in-memory Signal build."""

    input_rows: int
    output_rows: int
    trade_date_count: int
    first_trade_date: pd.Timestamp
    last_trade_date: pd.Timestamp
    prediction_column: str
    signal_direction: str

    def as_dict(self) -> dict[str, object]:
        """Return a detached JSON-safe audit mapping."""
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "trade_date_count": self.trade_date_count,
            "first_trade_date": self.first_trade_date.strftime("%Y-%m-%d"),
            "last_trade_date": self.last_trade_date.strftime("%Y-%m-%d"),
            "prediction_column": self.prediction_column,
            "signal_direction": self.signal_direction,
        }


class SignalBuildResult:
    """Defensively expose canonical Signals and their immutable build audit."""

    __slots__ = ("_signals", "_audit")

    def __init__(self, signals: pd.DataFrame, audit: SignalBuildAudit) -> None:
        if not isinstance(signals, pd.DataFrame) or signals.empty:
            raise SignalDataError("signals must be a non-empty DataFrame.")
        validate_signal_columns(signals.columns)
        if not isinstance(audit, SignalBuildAudit) or len(signals) != audit.output_rows:
            raise SignalDataError("Signal result and audit are inconsistent.")
        self._signals = signals.copy(deep=True)
        self._audit = audit

    @property
    def signals(self) -> pd.DataFrame:
        """Return a deep defensive copy of canonical Signal rows."""
        return self._signals.copy(deep=True)

    @property
    def audit(self) -> SignalBuildAudit:
        """Return the immutable build audit."""
        return self._audit


class SignalBuilder:
    """Map one validated prediction column to deterministic canonical Signals."""

    def build(
        self,
        predictions: pd.DataFrame,
        *,
        prediction_column: str,
        signal_direction: str,
    ) -> SignalBuildResult:
        """Build canonical per-date ranks using explicit effective semantics."""
        frame = self._copy_input(predictions)
        column = self._prediction_column(prediction_column)
        direction = self._direction(signal_direction)
        missing = [name for name in (*SIGNAL_KEY_COLUMNS, column) if name not in frame]
        if missing:
            raise SignalDataError(
                f"predictions are missing required columns: {missing!r}."
            )
        normalized = frame.loc[:, [*SIGNAL_KEY_COLUMNS, column]].copy(deep=True)
        normalized["trade_date"] = self._normalize_dates(
            normalized["trade_date"]
        )
        normalized["ts_code"] = self._normalize_codes(normalized["ts_code"])
        duplicate_mask = normalized.duplicated(list(SIGNAL_KEY_COLUMNS), keep=False)
        if bool(duplicate_mask.any()):
            raise SignalDataError(
                "prediction keys must be unique after canonicalization."
            )
        normalized["score"] = self._normalize_score(normalized[column])
        if column != "score":
            normalized = normalized.drop(columns=[column])
        ranked = normalized.sort_values(
            ["trade_date", "score", "ts_code"],
            ascending=[True, direction == "ascending", True],
            kind="mergesort",
        )
        ranked["rank"] = (
            ranked.groupby("trade_date", sort=False).cumcount() + 1
        ).astype(np.int64)
        output = ranked.loc[:, list(SIGNAL_OUTPUT_COLUMNS)].copy(deep=True)
        output = output.sort_values(
            ["trade_date", "rank", "ts_code"],
            ascending=[True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        audit = SignalBuildAudit(
            input_rows=len(frame),
            output_rows=len(output),
            trade_date_count=int(output["trade_date"].nunique()),
            first_trade_date=pd.Timestamp(output["trade_date"].min()),
            last_trade_date=pd.Timestamp(output["trade_date"].max()),
            prediction_column=column,
            signal_direction=direction,
        )
        return SignalBuildResult(output, audit)

    @staticmethod
    def _copy_input(value: object) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise SignalDataError("predictions must be a pandas DataFrame.")
        if value.empty:
            raise SignalDataError("predictions must not be empty.")
        if isinstance(value.columns, pd.MultiIndex) or not value.columns.is_unique:
            raise SignalDataError("prediction columns must be unique and flat.")
        if any(not isinstance(column, str) for column in value.columns):
            raise SignalDataError("prediction column names must be strings.")
        return value.copy(deep=True)

    @staticmethod
    def _prediction_column(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SignalDataError("prediction_column must be a non-empty string.")
        column = value.strip()
        if column in SIGNAL_PROTECTED_SCORE_SOURCE_COLUMNS:
            raise SignalDataError(
                f"prediction_column {column!r} is protected from score mapping."
            )
        return column

    @staticmethod
    def _direction(value: object) -> str:
        if not isinstance(value, str):
            raise SignalDataError("signal_direction must be a string.")
        direction = value.strip().lower()
        if direction not in {"descending", "ascending"}:
            raise SignalDataError(
                "signal_direction must be 'descending' or 'ascending'."
            )
        return direction

    @staticmethod
    def _normalize_dates(series: pd.Series) -> pd.Series:
        normalized: list[pd.Timestamp] = []
        for value in series.tolist():
            missing = (
                value is None
                or value is pd.NaT
                or value is pd.NA
                or (
                    isinstance(value, (float, np.floating))
                    and bool(np.isnan(value))
                )
                or (
                    isinstance(value, np.datetime64)
                    and bool(np.isnat(value))
                )
            )
            if missing:
                raise SignalDataError("trade_date cannot contain missing dates.")
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    raise SignalDataError("trade_date contains an invalid ISO date.")
                try:
                    parsed: object = datetime.fromisoformat(stripped)
                except ValueError:
                    try:
                        parsed = date.fromisoformat(stripped)
                    except ValueError as exc:
                        raise SignalDataError(
                            "trade_date contains an invalid ISO date."
                        ) from exc
            elif isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
                parsed = value
            else:
                raise SignalDataError(
                    "trade_date must contain dates or ISO date strings."
                )
            try:
                timestamp = pd.Timestamp(parsed)
            except (TypeError, ValueError) as exc:
                raise SignalDataError("trade_date contains an invalid date.") from exc
            if pd.isna(timestamp) or timestamp.tz is not None:
                raise SignalDataError(
                    "trade_date must contain timezone-naive valid dates."
                )
            if timestamp != timestamp.normalize():
                raise SignalDataError(
                    "trade_date must use day granularity with no time part."
                )
            normalized.append(timestamp)
        try:
            return pd.Series(
                normalized,
                index=series.index,
                name=series.name,
                dtype="datetime64[ns]",
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise SignalDataError(
                "trade_date contains a date outside datetime64[ns] range."
            ) from exc

    @staticmethod
    def _normalize_codes(series: pd.Series) -> pd.Series:
        values: list[str] = []
        for value in series.tolist():
            if not isinstance(value, (str, np.str_)):
                raise SignalDataError(
                    "ts_code must contain only non-empty strings."
                )
            code = str(value).strip()
            if not code:
                raise SignalDataError(
                    "ts_code must contain only non-empty strings."
                )
            values.append(code)
        return pd.Series(values, index=series.index, name=series.name, dtype="string")

    @staticmethod
    def _normalize_score(series: pd.Series) -> pd.Series:
        if (
            pd.api.types.is_bool_dtype(series.dtype)
            or not pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype)
        ):
            raise SignalDataError("score source must be real numeric data.")
        try:
            values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise SignalDataError("score source must be real numeric data.") from exc
        if not np.isfinite(values).all():
            raise SignalDataError("score source must contain only finite values.")
        return pd.Series(values, index=series.index, name="score", dtype=np.float64)