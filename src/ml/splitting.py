"""Deterministic walk-forward splitting with label maturity, purge, and embargo."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any

import pandas as pd

from src.ml.contracts import METADATA_COLUMNS, MLDataset


class WalkForwardError(ValueError):
    """Base error for walk-forward configuration, data, and integrity failures."""


class WalkForwardConfigError(WalkForwardError):
    """Raised when a walk-forward configuration value is invalid."""


class WalkForwardDataError(WalkForwardError):
    """Raised when an input dataset violates the public metadata contract."""


class WalkForwardInsufficientHistoryError(WalkForwardError):
    """Raised when no prediction date can produce one complete split."""


class WalkForwardIntegrityError(WalkForwardError):
    """Raised when a valid plan would become discontinuous or inconsistent."""


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configure score-date periods for walk-forward window construction."""

    train_window_periods: int
    validation_periods: int
    window_type: str = "rolling"
    retrain_frequency: int = 1
    embargo_periods: int = 0

    def __post_init__(self) -> None:
        for field_name, minimum in (
            ("train_window_periods", 1),
            ("validation_periods", 1),
            ("retrain_frequency", 1),
            ("embargo_periods", 0),
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise WalkForwardConfigError(
                    f"{field_name} must be an integer >= {minimum}; "
                    f"received {value!r}."
                )
            object.__setattr__(self, field_name, int(value))

        if not isinstance(self.window_type, str):
            raise WalkForwardConfigError(
                "window_type must be 'rolling' or 'expanding'; "
                f"received {self.window_type!r}."
            )
        window_type = self.window_type.strip().lower()
        if window_type not in {"rolling", "expanding"}:
            raise WalkForwardConfigError(
                "window_type must be 'rolling' or 'expanding'; "
                f"received {self.window_type!r}."
            )
        object.__setattr__(self, "window_type", window_type)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe configuration dictionary."""
        return asdict(self)


def _timestamp_tuple(
    field_name: str, values: object, *, allow_empty: bool
) -> tuple[pd.Timestamp, ...]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise WalkForwardIntegrityError(
            f"{field_name} must be a tuple of timestamps."
        ) from exc
    normalized: list[pd.Timestamp] = []
    for value in raw:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise WalkForwardIntegrityError(
                f"{field_name} contains an invalid timestamp {value!r}."
            ) from exc
        if pd.isna(timestamp):
            raise WalkForwardIntegrityError(
                f"{field_name} contains a missing timestamp."
            )
        normalized.append(timestamp)
    result = tuple(normalized)
    if not allow_empty and not result:
        raise WalkForwardIntegrityError(f"{field_name} must not be empty.")
    if result != tuple(sorted(set(result))):
        raise WalkForwardIntegrityError(
            f"{field_name} must be strictly increasing and unique."
        )
    return result


def _index_tuple(field_name: str, values: object) -> tuple[int, ...]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise WalkForwardIntegrityError(
            f"{field_name} must be a tuple of row indices."
        ) from exc
    normalized: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise WalkForwardIntegrityError(
                f"{field_name} contains invalid row index {value!r}."
            )
        normalized.append(int(value))
    result = tuple(normalized)
    if not result:
        raise WalkForwardIntegrityError(f"{field_name} must not be empty.")
    if result != tuple(sorted(set(result))):
        raise WalkForwardIntegrityError(
            f"{field_name} must be strictly increasing and unique."
        )
    return result


def _iso(value: pd.Timestamp) -> str:
    return value.isoformat()


def _date_range(values: tuple[pd.Timestamp, ...]) -> dict[str, Any]:
    return {
        "count": len(values),
        "start": _iso(values[0]) if values else None,
        "end": _iso(values[-1]) if values else None,
    }


@dataclass(frozen=True)
class WalkForwardSplit:
    """Hold row positions and date-level audit evidence for one retraining block."""

    retrain_id: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    prediction_indices: tuple[int, ...]
    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    prediction_dates: tuple[pd.Timestamp, ...]
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    validation_start_date: pd.Timestamp
    validation_end_date: pd.Timestamp
    prediction_start_date: pd.Timestamp
    prediction_end_date: pd.Timestamp
    n_train_rows: int
    n_validation_rows: int
    n_prediction_rows: int
    max_train_exit_date: pd.Timestamp
    max_validation_exit_date: pd.Timestamp
    embargo_dates: tuple[pd.Timestamp, ...] = ()
    train_validation_purged_dates: tuple[pd.Timestamp, ...] = ()
    label_unavailable_dates: tuple[pd.Timestamp, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.retrain_id, bool)
            or not isinstance(self.retrain_id, Integral)
            or self.retrain_id < 1
        ):
            raise WalkForwardIntegrityError(
                f"retrain_id must be an integer >= 1; received {self.retrain_id!r}."
            )
        object.__setattr__(self, "retrain_id", int(self.retrain_id))

        index_fields = (
            "train_indices",
            "validation_indices",
            "prediction_indices",
        )
        for field_name in index_fields:
            object.__setattr__(
                self, field_name, _index_tuple(field_name, getattr(self, field_name))
            )
        date_fields = ("train_dates", "validation_dates", "prediction_dates")
        for field_name in date_fields:
            object.__setattr__(
                self,
                field_name,
                _timestamp_tuple(field_name, getattr(self, field_name), allow_empty=False),
            )
        audit_date_fields = (
            "embargo_dates",
            "train_validation_purged_dates",
            "label_unavailable_dates",
        )
        for field_name in audit_date_fields:
            object.__setattr__(
                self,
                field_name,
                _timestamp_tuple(field_name, getattr(self, field_name), allow_empty=True),
            )

        if (
            set(self.train_indices) & set(self.validation_indices)
            or set(self.train_indices) & set(self.prediction_indices)
            or set(self.validation_indices) & set(self.prediction_indices)
        ):
            raise WalkForwardIntegrityError(
                f"retrain_id={self.retrain_id} has overlapping partition indices."
            )
        if (
            set(self.train_dates) & set(self.validation_dates)
            or set(self.train_dates) & set(self.prediction_dates)
            or set(self.validation_dates) & set(self.prediction_dates)
        ):
            raise WalkForwardIntegrityError(
                f"retrain_id={self.retrain_id} has overlapping partition dates."
            )

        for field_name, indices in (
            ("n_train_rows", self.train_indices),
            ("n_validation_rows", self.validation_indices),
            ("n_prediction_rows", self.prediction_indices),
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value != len(indices)
            ):
                raise WalkForwardIntegrityError(
                    f"{field_name} must equal its index count: "
                    f"value={value!r}, index_count={len(indices)}."
                )
            object.__setattr__(self, field_name, int(value))

        boundary_pairs = (
            ("train_start_date", self.train_dates[0]),
            ("train_end_date", self.train_dates[-1]),
            ("validation_start_date", self.validation_dates[0]),
            ("validation_end_date", self.validation_dates[-1]),
            ("prediction_start_date", self.prediction_dates[0]),
            ("prediction_end_date", self.prediction_dates[-1]),
        )
        for field_name, expected in boundary_pairs:
            actual = pd.Timestamp(getattr(self, field_name))
            if pd.isna(actual) or actual != expected:
                raise WalkForwardIntegrityError(
                    f"{field_name} must match its date tuple boundary: "
                    f"actual={actual!r}, expected={expected!r}."
                )
            object.__setattr__(self, field_name, actual)

        max_train_exit = pd.Timestamp(self.max_train_exit_date)
        max_validation_exit = pd.Timestamp(self.max_validation_exit_date)
        if pd.isna(max_train_exit) or pd.isna(max_validation_exit):
            raise WalkForwardIntegrityError(
                "max_train_exit_date and max_validation_exit_date must be valid."
            )
        object.__setattr__(self, "max_train_exit_date", max_train_exit)
        object.__setattr__(self, "max_validation_exit_date", max_validation_exit)

        if not self.train_end_date < self.validation_start_date:
            raise WalkForwardIntegrityError(
                "train_end_date must be strictly earlier than "
                "validation_start_date."
            )
        if not self.validation_end_date < self.prediction_start_date:
            raise WalkForwardIntegrityError(
                "validation_end_date must be strictly earlier than "
                "prediction_start_date."
            )
        if not self.max_train_exit_date < self.validation_start_date:
            raise WalkForwardIntegrityError(
                "max_train_exit_date must be strictly earlier than "
                "validation_start_date."
            )
        if not self.max_validation_exit_date < self.prediction_start_date:
            raise WalkForwardIntegrityError(
                "max_validation_exit_date must be strictly earlier than "
                "prediction_start_date."
            )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe compact split summary without row indices."""
        return {
            "retrain_id": self.retrain_id,
            "train": {
                **_date_range(self.train_dates),
                "rows": self.n_train_rows,
                "max_exit_date": _iso(self.max_train_exit_date),
            },
            "validation": {
                **_date_range(self.validation_dates),
                "rows": self.n_validation_rows,
                "max_exit_date": _iso(self.max_validation_exit_date),
            },
            "prediction": {
                **_date_range(self.prediction_dates),
                "rows": self.n_prediction_rows,
            },
            "embargo": _date_range(self.embargo_dates),
            "train_validation_purged": _date_range(
                self.train_validation_purged_dates
            ),
            "label_unavailable": _date_range(self.label_unavailable_dates),
        }


@dataclass(frozen=True)
class WalkForwardPlan:
    """Hold a complete, gap-free sequence of walk-forward prediction blocks."""

    config: WalkForwardConfig
    splits: tuple[WalkForwardSplit, ...]
    all_score_dates: tuple[pd.Timestamp, ...]
    skipped_initial_prediction_dates: tuple[pd.Timestamp, ...]
    first_prediction_date: pd.Timestamp
    last_prediction_date: pd.Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.config, WalkForwardConfig):
            raise WalkForwardIntegrityError(
                "config must be a WalkForwardConfig."
            )
        try:
            splits = tuple(self.splits)
        except TypeError as exc:
            raise WalkForwardIntegrityError(
                "splits must be a non-empty tuple."
            ) from exc
        if not splits or any(not isinstance(item, WalkForwardSplit) for item in splits):
            raise WalkForwardIntegrityError(
                "splits must contain at least one WalkForwardSplit."
            )
        object.__setattr__(self, "splits", splits)

        all_dates = _timestamp_tuple(
            "all_score_dates", self.all_score_dates, allow_empty=False
        )
        skipped = _timestamp_tuple(
            "skipped_initial_prediction_dates",
            self.skipped_initial_prediction_dates,
            allow_empty=True,
        )
        object.__setattr__(self, "all_score_dates", all_dates)
        object.__setattr__(self, "skipped_initial_prediction_dates", skipped)

        expected_ids = tuple(range(1, len(splits) + 1))
        actual_ids = tuple(split.retrain_id for split in splits)
        if actual_ids != expected_ids:
            raise WalkForwardIntegrityError(
                f"retrain_id sequence must be {expected_ids!r}; "
                f"received {actual_ids!r}."
            )

        prediction_dates = tuple(
            date for split in splits for date in split.prediction_dates
        )
        if len(set(prediction_dates)) != len(prediction_dates):
            raise WalkForwardIntegrityError(
                "prediction_dates overlap across walk-forward splits."
            )
        first = pd.Timestamp(self.first_prediction_date)
        last = pd.Timestamp(self.last_prediction_date)
        if first != prediction_dates[0] or last != prediction_dates[-1]:
            raise WalkForwardIntegrityError(
                "first_prediction_date and last_prediction_date must match "
                "the actual prediction blocks."
            )
        object.__setattr__(self, "first_prediction_date", first)
        object.__setattr__(self, "last_prediction_date", last)

        first_position = all_dates.index(first) if first in all_dates else -1
        if first_position < 0 or last not in all_dates:
            raise WalkForwardIntegrityError(
                "Prediction boundaries must exist in all_score_dates."
            )
        last_position = all_dates.index(last)
        if last_position != len(all_dates) - 1:
            raise WalkForwardIntegrityError(
                "last_prediction_date must equal the final global score date; "
                f"last_prediction_date={last.isoformat()}."
            )
        expected_prediction_dates = all_dates[first_position : last_position + 1]
        if prediction_dates != expected_prediction_dates:
            raise WalkForwardIntegrityError(
                "Prediction blocks must cover every global score date exactly "
                "once without gaps."
            )
        if skipped != all_dates[:first_position]:
            raise WalkForwardIntegrityError(
                "skipped_initial_prediction_dates must exactly equal the score-date "
                "prefix before first_prediction_date."
            )
        for split in splits[:-1]:
            if len(split.prediction_dates) != self.config.retrain_frequency:
                raise WalkForwardIntegrityError(
                    "Every non-final prediction block must contain exactly "
                    f"retrain_frequency={self.config.retrain_frequency} dates."
                )
        if not 1 <= len(splits[-1].prediction_dates) <= self.config.retrain_frequency:
            raise WalkForwardIntegrityError(
                "The final prediction block must contain between 1 and "
                f"retrain_frequency={self.config.retrain_frequency} dates."
            )

    @property
    def n_splits(self) -> int:
        """Return the number of retraining blocks."""
        return len(self.splits)

    @property
    def n_score_dates(self) -> int:
        """Return the number of unique score dates in the dataset."""
        return len(self.all_score_dates)

    @property
    def n_prediction_dates(self) -> int:
        """Return the total number of predicted score dates."""
        return sum(len(split.prediction_dates) for split in self.splits)

    @property
    def n_skipped_initial_dates(self) -> int:
        """Return the number of score dates skipped before the first split."""
        return len(self.skipped_initial_prediction_dates)

    def summary(self) -> dict[str, Any]:
        """Return a complete JSON-safe compact plan summary."""
        return {
            "config": self.config.as_dict(),
            "n_splits": self.n_splits,
            "n_score_dates": self.n_score_dates,
            "n_skipped_initial_dates": self.n_skipped_initial_dates,
            "first_prediction_date": _iso(self.first_prediction_date),
            "last_prediction_date": _iso(self.last_prediction_date),
            "n_prediction_dates": self.n_prediction_dates,
            "n_prediction_rows": sum(
                split.n_prediction_rows for split in self.splits
            ),
            "splits": [split.summary() for split in self.splits],
        }


class _CandidateHistoryError(Exception):
    """Internal signal that a prediction start lacks a complete history."""


class WalkForwardSplitter:
    """Build strict date-level walk-forward plans from public ML metadata."""

    def __init__(self, config: WalkForwardConfig) -> None:
        if not isinstance(config, WalkForwardConfig):
            raise WalkForwardConfigError(
                "config must be a WalkForwardConfig."
            )
        self.config = config

    def build(self, dataset: MLDataset) -> WalkForwardPlan:
        """Build a gap-free walk-forward plan without reading features or labels."""
        if not isinstance(dataset, MLDataset):
            raise WalkForwardDataError(
                "dataset must be an MLDataset; "
                f"received {type(dataset).__name__}."
            )
        metadata = dataset.metadata
        metadata = self._validate_metadata(metadata, dataset.n_samples)
        all_score_dates = tuple(
            pd.Timestamp(value)
            for value in metadata["trade_date"].drop_duplicates().tolist()
        )
        date_max_exit = metadata.groupby("trade_date", sort=True)[
            "exit_trade_date"
        ].max()

        first_position: int | None = None
        first_window: dict[str, tuple[pd.Timestamp, ...]] | None = None
        skipped: list[pd.Timestamp] = []
        max_mature_history = 0
        last_reason = "no prediction date was evaluated"
        for position, prediction_start in enumerate(all_score_dates):
            mature_count = sum(
                date < prediction_start and date_max_exit.loc[date] < prediction_start
                for date in all_score_dates
            )
            max_mature_history = max(max_mature_history, int(mature_count))
            try:
                window = self._window_for_prediction(
                    prediction_start, all_score_dates, date_max_exit
                )
            except _CandidateHistoryError as exc:
                skipped.append(prediction_start)
                last_reason = str(exc)
                continue
            first_position = position
            first_window = window
            break

        if first_position is None or first_window is None:
            raise WalkForwardInsufficientHistoryError(
                "No valid walk-forward split can be generated: "
                f"n_score_dates={len(all_score_dates)}, "
                f"train_window_periods={self.config.train_window_periods}, "
                f"validation_periods={self.config.validation_periods}, "
                f"embargo_periods={self.config.embargo_periods}, "
                f"window_type={self.config.window_type!r}, "
                f"retrain_frequency={self.config.retrain_frequency}, "
                f"max_mature_history_periods={max_mature_history}, "
                f"reason={last_reason}."
            )

        splits: list[WalkForwardSplit] = []
        position = first_position
        retrain_id = 1
        while position < len(all_score_dates):
            prediction_dates = all_score_dates[
                position : position + self.config.retrain_frequency
            ]
            prediction_start = prediction_dates[0]
            if position == first_position:
                window = first_window
            else:
                try:
                    window = self._window_for_prediction(
                        prediction_start, all_score_dates, date_max_exit
                    )
                except _CandidateHistoryError as exc:
                    raise WalkForwardIntegrityError(
                        "Walk-forward history became invalid after the first split: "
                        f"retrain_id={retrain_id}, "
                        f"prediction_start_date={prediction_start.isoformat()}, "
                        f"reason={exc}."
                    ) from exc
            splits.append(
                self._materialize_split(
                    retrain_id,
                    metadata,
                    date_max_exit,
                    window,
                    prediction_dates,
                )
            )
            position += len(prediction_dates)
            retrain_id += 1

        return WalkForwardPlan(
            config=self.config,
            splits=tuple(splits),
            all_score_dates=all_score_dates,
            skipped_initial_prediction_dates=tuple(skipped),
            first_prediction_date=splits[0].prediction_start_date,
            last_prediction_date=splits[-1].prediction_end_date,
        )

    @staticmethod
    def _validate_metadata(
        metadata: pd.DataFrame, expected_rows: int
    ) -> pd.DataFrame:
        if not isinstance(metadata, pd.DataFrame):
            raise WalkForwardDataError("dataset.metadata must be a pandas DataFrame.")
        if list(metadata.columns) != list(METADATA_COLUMNS):
            raise WalkForwardDataError(
                "dataset.metadata columns must exactly be "
                f"{list(METADATA_COLUMNS)!r}; received {list(metadata.columns)!r}."
            )
        if metadata.empty:
            raise WalkForwardDataError("dataset.metadata contains 0 rows.")
        if len(metadata) != expected_rows:
            raise WalkForwardDataError(
                "dataset.metadata row count must match dataset.n_samples: "
                f"metadata_rows={len(metadata)}, n_samples={expected_rows}."
            )
        expected_index = pd.RangeIndex(len(metadata))
        if not metadata.index.equals(expected_index):
            raise WalkForwardDataError(
                "dataset.metadata index must be a zero-based continuous RangeIndex: "
                f"received={metadata.index!r}."
            )
        result = metadata.copy(deep=True)
        for column in ("trade_date", "entry_trade_date", "exit_trade_date"):
            original = result[column]
            converted = pd.to_datetime(original, errors="coerce", format="mixed")
            invalid = converted.isna()
            if invalid.any():
                raise WalkForwardDataError(
                    f"dataset.metadata.{column} contains missing or invalid dates: "
                    f"invalid_count={int(invalid.sum())}."
                )
            result[column] = converted.astype("datetime64[ns]")

        codes = result["ts_code"].astype("string").str.strip()
        invalid_codes = codes.isna() | codes.eq("")
        if invalid_codes.any():
            raise WalkForwardDataError(
                "dataset.metadata.ts_code contains missing or empty values: "
                f"invalid_count={int(invalid_codes.sum())}."
            )
        result["ts_code"] = codes
        duplicates = result.duplicated(["trade_date", "ts_code"], keep=False)
        if duplicates.any():
            duplicate_key_count = len(
                result.loc[duplicates, ["trade_date", "ts_code"]].drop_duplicates()
            )
            raise WalkForwardDataError(
                "dataset.metadata contains duplicate trade_date + ts_code keys: "
                f"duplicate_key_count={duplicate_key_count}."
            )

        expected_order = result.sort_values(
            ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
        )
        if not result.equals(expected_order):
            raise WalkForwardDataError(
                "dataset.metadata must already be stably sorted by "
                "trade_date, ts_code."
            )
        entry_before_score = result["entry_trade_date"] < result["trade_date"]
        if entry_before_score.any():
            raise WalkForwardDataError(
                "dataset.metadata requires trade_date <= entry_trade_date: "
                f"invalid_count={int(entry_before_score.sum())}."
            )
        exit_not_after_entry = result["exit_trade_date"] <= result["entry_trade_date"]
        if exit_not_after_entry.any():
            raise WalkForwardDataError(
                "dataset.metadata requires entry_trade_date < exit_trade_date: "
                f"invalid_count={int(exit_not_after_entry.sum())}."
            )
        return result

    def _window_for_prediction(
        self,
        prediction_start: pd.Timestamp,
        all_score_dates: tuple[pd.Timestamp, ...],
        date_max_exit: pd.Series,
    ) -> dict[str, tuple[pd.Timestamp, ...]]:
        history = tuple(date for date in all_score_dates if date < prediction_start)
        mature = tuple(
            date for date in history if date_max_exit.loc[date] < prediction_start
        )
        unavailable = tuple(
            date for date in history if date_max_exit.loc[date] >= prediction_start
        )
        if self.config.embargo_periods:
            embargo = mature[-self.config.embargo_periods :]
            after_embargo = mature[: -self.config.embargo_periods]
        else:
            embargo = ()
            after_embargo = mature
        if len(after_embargo) < self.config.validation_periods:
            raise _CandidateHistoryError(
                "mature history after embargo is shorter than validation window: "
                f"prediction_start_date={prediction_start.isoformat()}, "
                f"mature_periods={len(mature)}, embargoed_periods={len(embargo)}, "
                f"validation_periods={self.config.validation_periods}"
            )
        validation = after_embargo[-self.config.validation_periods :]
        validation_start = validation[0]
        train_candidates = tuple(
            date
            for date in history
            if date < validation_start and date_max_exit.loc[date] < validation_start
        )
        purged = tuple(
            date
            for date in history
            if date < validation_start and date_max_exit.loc[date] >= validation_start
        )
        if len(train_candidates) < self.config.train_window_periods:
            raise _CandidateHistoryError(
                "eligible training history is shorter than train_window_periods: "
                f"prediction_start_date={prediction_start.isoformat()}, "
                f"validation_start_date={validation_start.isoformat()}, "
                f"eligible_train_periods={len(train_candidates)}, "
                f"train_window_periods={self.config.train_window_periods}"
            )
        train = (
            train_candidates[-self.config.train_window_periods :]
            if self.config.window_type == "rolling"
            else train_candidates
        )
        return {
            "train": train,
            "validation": validation,
            "embargo": embargo,
            "purged": purged,
            "unavailable": unavailable,
        }

    @staticmethod
    def _materialize_split(
        retrain_id: int,
        metadata: pd.DataFrame,
        date_max_exit: pd.Series,
        window: dict[str, tuple[pd.Timestamp, ...]],
        prediction_dates: tuple[pd.Timestamp, ...],
    ) -> WalkForwardSplit:
        train_dates = window["train"]
        validation_dates = window["validation"]

        def indices_for(dates: tuple[pd.Timestamp, ...]) -> tuple[int, ...]:
            mask = metadata["trade_date"].isin(dates)
            return tuple(int(index) for index in metadata.index[mask])

        train_indices = indices_for(train_dates)
        validation_indices = indices_for(validation_dates)
        prediction_indices = indices_for(prediction_dates)
        max_train_exit = pd.Timestamp(
            max(date_max_exit.loc[date] for date in train_dates)
        )
        max_validation_exit = pd.Timestamp(
            max(date_max_exit.loc[date] for date in validation_dates)
        )
        if not max_train_exit < validation_dates[0]:
            raise WalkForwardIntegrityError(
                "Materialized training labels cross the validation cutoff: "
                f"retrain_id={retrain_id}, "
                f"max_train_exit_date={max_train_exit.isoformat()}, "
                f"validation_start_date={validation_dates[0].isoformat()}."
            )
        if not max_validation_exit < prediction_dates[0]:
            raise WalkForwardIntegrityError(
                "Materialized validation labels cross the prediction cutoff: "
                f"retrain_id={retrain_id}, "
                f"max_validation_exit_date={max_validation_exit.isoformat()}, "
                f"prediction_start_date={prediction_dates[0].isoformat()}."
            )
        return WalkForwardSplit(
            retrain_id=retrain_id,
            train_indices=train_indices,
            validation_indices=validation_indices,
            prediction_indices=prediction_indices,
            train_dates=train_dates,
            validation_dates=validation_dates,
            prediction_dates=prediction_dates,
            train_start_date=train_dates[0],
            train_end_date=train_dates[-1],
            validation_start_date=validation_dates[0],
            validation_end_date=validation_dates[-1],
            prediction_start_date=prediction_dates[0],
            prediction_end_date=prediction_dates[-1],
            n_train_rows=len(train_indices),
            n_validation_rows=len(validation_indices),
            n_prediction_rows=len(prediction_indices),
            max_train_exit_date=max_train_exit,
            max_validation_exit_date=max_validation_exit,
            embargo_dates=window["embargo"],
            train_validation_purged_dates=window["purged"],
            label_unavailable_dates=window["unavailable"],
        )
