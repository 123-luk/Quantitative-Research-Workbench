"""Pure in-memory contracts for auditable modeling panels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
import math
from numbers import Integral, Real
from typing import Any

import pandas as pd


MODELING_PANEL_SCHEMA_VERSION = "1.0"
MODELING_PANEL_KEY_COLUMNS = ("trade_date", "ts_code")
MODELING_PANEL_AUDIT_COLUMNS = (
    "entry_trade_date",
    "exit_trade_date",
    "entry_price",
    "exit_price",
)

_CONFIG_FIELDS = (
    "label_column",
    "include_features",
    "exclude_features",
    "unmatched_policy",
    "require_entry_after_signal",
    "allow_missing_labels",
)
_UNMATCHED_POLICIES = {"audit_and_drop", "error"}
_COUNT_FIELDS = (
    "factor_input_rows",
    "return_input_rows",
    "matched_rows",
    "output_rows",
    "date_count",
    "security_count",
    "feature_count",
    "label_missing_count",
    "label_non_finite_count",
    "duplicate_factor_key_count",
    "duplicate_return_key_count",
    "entry_before_signal_count",
    "entry_equal_signal_count",
    "exit_not_after_entry_count",
    "label_formula_mismatch_count",
)
_DISTRIBUTION_FIELDS = (
    (
        "per_date_security_count_min",
        "per_date_security_count_median",
        "per_date_security_count_max",
    ),
    (
        "per_security_observation_count_min",
        "per_security_observation_count_median",
        "per_security_observation_count_max",
    ),
)


class ModelingPanelError(ValueError):
    """Base error for Modeling Panel contract violations."""


class ModelingPanelConfigError(ModelingPanelError):
    """Raised when Modeling Panel configuration is invalid."""


class ModelingPanelDataError(ModelingPanelError):
    """Raised when input or output data violates the data contract."""


class ModelingPanelAlignmentError(ModelingPanelError):
    """Raised when factor and return observations are not safely aligned."""


class ModelingPanelLeakageError(ModelingPanelError):
    """Raised when a feature crosses a known leakage boundary."""


class ModelingPanelIntegrityError(ModelingPanelError):
    """Raised when contract objects are mutually inconsistent."""


def _normalized_name(field_name: str, value: object, error: type[ModelingPanelError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{field_name} must be a non-empty string.")
    return value.strip()


def _nonnegative_int(
    field_name: str, value: object, error: type[ModelingPanelError]
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise error(f"{field_name} must be a non-negative integer; received {value!r}.")
    return int(value)


def _positive_int(
    field_name: str, value: object, error: type[ModelingPanelError]
) -> int:
    normalized = _nonnegative_int(field_name, value, error)
    if normalized == 0:
        raise error(f"{field_name} must be a positive integer.")
    return normalized


def _normalized_date(
    field_name: str,
    value: object,
    *,
    allow_none: bool,
    allow_string: bool = False,
    error: type[ModelingPanelError] = ModelingPanelDataError,
) -> pd.Timestamp | None:
    if value is None or (isinstance(value, (pd.Timestamp, datetime, date)) and pd.isna(value)):
        if allow_none:
            return None
        raise error(f"{field_name} must be a valid date and cannot be NaT.")
    if isinstance(value, str) and not allow_string:
        raise error(f"{field_name} must be a Timestamp, datetime, or date.")
    if not isinstance(value, (pd.Timestamp, datetime, date, str)):
        raise error(f"{field_name} must be a valid date.")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise error(f"{field_name} must be a valid date.") from exc
    if pd.isna(timestamp):
        if allow_none:
            return None
        raise error(f"{field_name} must be a valid date and cannot be NaT.")
    if timestamp.tz is not None:
        raise error(f"{field_name} must be timezone-naive.")
    return timestamp.normalize()


def _date_string(value: pd.Timestamp | None) -> str | None:
    return value.strftime("%Y-%m-%d") if value is not None else None


def _strict_name_tuple(
    field_name: str,
    value: object,
    *,
    allow_empty: bool,
    error: type[ModelingPanelError],
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise error(f"{field_name} must be a tuple.")
    normalized = tuple(_normalized_name(field_name, item, error) for item in value)
    if not allow_empty and not normalized:
        raise error(f"{field_name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise error(f"{field_name} must not contain duplicate names.")
    return normalized


def _config_feature_tuple(
    field_name: str, value: object, *, allow_none: bool
) -> tuple[str, ...] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, (list, tuple)):
        expected = "None, a list, or a tuple" if allow_none else "a list or tuple"
        raise ModelingPanelConfigError(f"{field_name} must be {expected}.")
    normalized = tuple(
        _normalized_name(field_name, item, ModelingPanelConfigError) for item in value
    )
    if allow_none and not normalized:
        raise ModelingPanelConfigError(
            "include_features cannot be an explicit empty list or tuple; use None."
        )
    if len(set(normalized)) != len(normalized):
        raise ModelingPanelConfigError(
            f"{field_name} must not contain duplicate names."
        )
    return normalized


@dataclass(frozen=True)
class ModelingPanelConfig:
    """Immutable user-facing choices for a future Modeling Panel builder."""

    label_column: str = "forward_return"
    include_features: tuple[str, ...] | None = None
    exclude_features: tuple[str, ...] = ()
    unmatched_policy: str = "audit_and_drop"
    require_entry_after_signal: bool = True
    allow_missing_labels: bool = True

    def __post_init__(self) -> None:
        label = _normalized_name(
            "label_column", self.label_column, ModelingPanelConfigError
        )
        permanently_reserved = {
            *MODELING_PANEL_KEY_COLUMNS,
            *MODELING_PANEL_AUDIT_COLUMNS,
        }
        if label in permanently_reserved:
            raise ModelingPanelConfigError(
                f"label_column {label!r} conflicts with a reserved key/audit column."
            )

        include = _config_feature_tuple(
            "include_features", self.include_features, allow_none=True
        )
        exclude = _config_feature_tuple(
            "exclude_features", self.exclude_features, allow_none=False
        )
        if exclude is None:
            raise ModelingPanelConfigError("exclude_features cannot be None.")
        if include is not None and exclude:
            raise ModelingPanelConfigError(
                "include_features and exclude_features cannot both be configured."
            )
        reserved = permanently_reserved | {label}
        for field_name, values in (
            ("include_features", include or ()),
            ("exclude_features", exclude),
        ):
            conflicts = [name for name in values if name in reserved]
            if conflicts:
                raise ModelingPanelConfigError(
                    f"{field_name} contains reserved or label columns: {conflicts!r}."
                )

        if not isinstance(self.unmatched_policy, str):
            raise ModelingPanelConfigError("unmatched_policy must be a string.")
        unmatched_policy = self.unmatched_policy.strip().lower()
        if unmatched_policy not in _UNMATCHED_POLICIES:
            raise ModelingPanelConfigError(
                "unmatched_policy must be 'audit_and_drop' or 'error'."
            )
        for field_name in (
            "require_entry_after_signal",
            "allow_missing_labels",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ModelingPanelConfigError(f"{field_name} must be a bool.")

        object.__setattr__(self, "label_column", label)
        object.__setattr__(self, "include_features", include)
        object.__setattr__(self, "exclude_features", exclude)
        object.__setattr__(self, "unmatched_policy", unmatched_policy)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object] | ModelingPanelConfig | None
    ) -> ModelingPanelConfig:
        """Construct from a strict mapping, or return an existing config."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ModelingPanelConfigError(
                "ModelingPanelConfig input must be a mapping, config, or None."
            )
        unknown = sorted(set(value) - set(_CONFIG_FIELDS))
        if unknown:
            raise ModelingPanelConfigError(
                f"Unknown ModelingPanelConfig fields: {unknown!r}."
            )
        return cls(**dict(value))  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, Any]:
        """Return a detached, strictly JSON-safe configuration snapshot."""
        return {
            "label_column": self.label_column,
            "include_features": (
                list(self.include_features) if self.include_features is not None else None
            ),
            "exclude_features": list(self.exclude_features),
            "unmatched_policy": self.unmatched_policy,
            "require_entry_after_signal": self.require_entry_after_signal,
            "allow_missing_labels": self.allow_missing_labels,
        }


@dataclass(frozen=True)
class ModelingPanelUnmatchedAudit:
    """Compact, bounded audit metadata for one side of an unmatched join."""

    row_count: int
    date_count: int
    first_trade_date: pd.Timestamp | None
    last_trade_date: pd.Timestamp | None
    sampled_keys: tuple[tuple[pd.Timestamp, str], ...] = ()

    def __post_init__(self) -> None:
        row_count = _nonnegative_int(
            "row_count", self.row_count, ModelingPanelDataError
        )
        date_count = _nonnegative_int(
            "date_count", self.date_count, ModelingPanelDataError
        )
        if date_count > row_count:
            raise ModelingPanelDataError("date_count cannot exceed row_count.")
        first = _normalized_date(
            "first_trade_date",
            self.first_trade_date,
            allow_none=True,
        )
        last = _normalized_date(
            "last_trade_date",
            self.last_trade_date,
            allow_none=True,
        )
        if not isinstance(self.sampled_keys, tuple):
            raise ModelingPanelDataError("sampled_keys must be a tuple.")
        if len(self.sampled_keys) > 20:
            raise ModelingPanelDataError("sampled_keys cannot contain more than 20 keys.")

        samples: list[tuple[pd.Timestamp, str]] = []
        for index, item in enumerate(self.sampled_keys):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ModelingPanelDataError(
                    "sampled_keys must contain (trade_date, ts_code) tuples."
                )
            sample_date = _normalized_date(
                f"sampled_keys[{index}].trade_date",
                item[0],
                allow_none=False,
            )
            if sample_date is None:
                raise ModelingPanelDataError(
                    f"sampled_keys[{index}].trade_date cannot be None."
                )
            code = _normalized_name(
                f"sampled_keys[{index}].ts_code",
                item[1],
                ModelingPanelDataError,
            )
            samples.append((sample_date, code))

        if len(set(samples)) != len(samples):
            raise ModelingPanelDataError("sampled_keys must not contain duplicates.")
        if tuple(samples) != tuple(sorted(samples, key=lambda item: (item[0], item[1]))):
            raise ModelingPanelDataError(
                "sampled_keys must already be sorted by trade_date and ts_code."
            )
        if len(samples) > row_count:
            raise ModelingPanelDataError("sampled_keys count cannot exceed row_count.")

        if row_count == 0:
            if date_count != 0 or first is not None or last is not None or samples:
                raise ModelingPanelDataError(
                    "A zero-row unmatched audit requires zero dates, no date range, "
                    "and no sampled_keys."
                )
        else:
            if date_count == 0:
                raise ModelingPanelDataError(
                    "A non-empty unmatched audit requires date_count >= 1."
                )
            if first is None or last is None:
                raise ModelingPanelDataError(
                    "A non-empty unmatched audit requires first_trade_date and "
                    "last_trade_date."
                )
            if first > last:
                raise ModelingPanelDataError(
                    "first_trade_date cannot be later than last_trade_date."
                )
            if any(sample_date < first or sample_date > last for sample_date, _ in samples):
                raise ModelingPanelDataError(
                    "sampled_keys trade dates must fall within the audited date range."
                )

        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "date_count", date_count)
        object.__setattr__(self, "first_trade_date", first)
        object.__setattr__(self, "last_trade_date", last)
        object.__setattr__(self, "sampled_keys", tuple(samples))

    def as_dict(self) -> dict[str, Any]:
        """Return bounded, detached, strictly JSON-safe unmatched metadata."""
        return {
            "row_count": self.row_count,
            "date_count": self.date_count,
            "first_trade_date": _date_string(self.first_trade_date),
            "last_trade_date": _date_string(self.last_trade_date),
            "sampled_keys": [
                {"trade_date": _date_string(trade_date), "ts_code": ts_code}
                for trade_date, ts_code in self.sampled_keys
            ],
        }


FeatureCountItems = tuple[tuple[str, int], ...]
FeatureRateItems = tuple[tuple[str, float], ...]


def _feature_count_items(
    field_name: str,
    value: object,
    feature_names: tuple[str, ...],
    output_rows: int,
) -> FeatureCountItems:
    if not isinstance(value, tuple):
        raise ModelingPanelDataError(f"{field_name} must be a tuple of pairs.")
    normalized: list[tuple[str, int]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ModelingPanelDataError(f"{field_name} must contain (name, count) pairs.")
        name = _normalized_name(field_name, item[0], ModelingPanelDataError)
        count = _nonnegative_int(
            f"{field_name}[{name!r}]", item[1], ModelingPanelDataError
        )
        if count > output_rows:
            raise ModelingPanelDataError(
                f"{field_name}[{name!r}] cannot exceed output_rows."
            )
        normalized.append((name, count))
    if tuple(name for name, _ in normalized) != feature_names:
        raise ModelingPanelIntegrityError(
            f"{field_name} names and order must exactly match feature_names."
        )
    return tuple(normalized)


def _feature_rate_items(
    field_name: str,
    value: object,
    feature_names: tuple[str, ...],
) -> FeatureRateItems:
    if not isinstance(value, tuple):
        raise ModelingPanelDataError(f"{field_name} must be a tuple of pairs.")
    normalized: list[tuple[str, float]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ModelingPanelDataError(f"{field_name} must contain (name, rate) pairs.")
        name = _normalized_name(field_name, item[0], ModelingPanelDataError)
        rate = item[1]
        if (
            isinstance(rate, bool)
            or not isinstance(rate, Real)
            or not math.isfinite(float(rate))
            or not 0.0 <= float(rate) <= 1.0
        ):
            raise ModelingPanelDataError(
                f"{field_name}[{name!r}] must be a finite number between 0 and 1."
            )
        normalized.append((name, float(rate)))
    if tuple(name for name, _ in normalized) != feature_names:
        raise ModelingPanelIntegrityError(
            f"{field_name} names and order must exactly match feature_names."
        )
    return tuple(normalized)


def _ordered_feature_subset(
    field_name: str, value: object, feature_names: tuple[str, ...]
) -> tuple[str, ...]:
    names = _strict_name_tuple(
        field_name,
        value,
        allow_empty=True,
        error=ModelingPanelDataError,
    )
    unknown = [name for name in names if name not in feature_names]
    if unknown:
        raise ModelingPanelIntegrityError(
            f"{field_name} contains names absent from feature_names: {unknown!r}."
        )
    expected = tuple(name for name in feature_names if name in set(names))
    if names != expected:
        raise ModelingPanelIntegrityError(
            f"{field_name} must preserve feature_names order."
        )
    return names


def _date_pair(
    first_name: str,
    first_value: object,
    last_name: str,
    last_value: object,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    first = _normalized_date(first_name, first_value, allow_none=True)
    last = _normalized_date(last_name, last_value, allow_none=True)
    if (first is None) != (last is None):
        raise ModelingPanelDataError(
            f"{first_name} and {last_name} must both be dates or both be None."
        )
    if first is not None and last is not None and first > last:
        raise ModelingPanelDataError(f"{first_name} cannot be later than {last_name}.")
    return first, last


@dataclass(frozen=True)
class ModelingPanelAudit:
    """Immutable, JSON-safe provenance and quality metadata for one panel."""

    schema_version: str
    config: ModelingPanelConfig
    label_column: str
    factor_input_rows: int
    return_input_rows: int
    matched_rows: int
    output_rows: int
    factor_only: ModelingPanelUnmatchedAudit
    return_only: ModelingPanelUnmatchedAudit
    date_count: int
    security_count: int
    first_trade_date: pd.Timestamp
    last_trade_date: pd.Timestamp
    first_entry_trade_date: pd.Timestamp | None
    last_entry_trade_date: pd.Timestamp | None
    first_exit_trade_date: pd.Timestamp | None
    last_exit_trade_date: pd.Timestamp | None
    feature_count: int
    feature_names: tuple[str, ...]
    feature_missing_counts: FeatureCountItems
    feature_missing_rates: FeatureRateItems
    feature_non_finite_counts: FeatureCountItems
    all_missing_features: tuple[str, ...]
    constant_features: tuple[str, ...]
    suspicious_feature_names: tuple[str, ...]
    label_missing_count: int
    label_non_finite_count: int
    duplicate_factor_key_count: int
    duplicate_return_key_count: int
    entry_before_signal_count: int
    entry_equal_signal_count: int
    exit_not_after_entry_count: int
    label_formula_mismatch_count: int
    per_date_security_count_min: int
    per_date_security_count_median: float
    per_date_security_count_max: int
    per_security_observation_count_min: int
    per_security_observation_count_median: float
    per_security_observation_count_max: int
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != MODELING_PANEL_SCHEMA_VERSION:
            raise ModelingPanelIntegrityError(
                "schema_version must equal "
                f"{MODELING_PANEL_SCHEMA_VERSION!r}; received {self.schema_version!r}."
            )
        if not isinstance(self.config, ModelingPanelConfig):
            raise ModelingPanelIntegrityError(
                "config must be a ModelingPanelConfig."
            )
        label = _normalized_name(
            "label_column", self.label_column, ModelingPanelIntegrityError
        )
        if label != self.config.label_column:
            raise ModelingPanelIntegrityError(
                "label_column must match config.label_column."
            )
        if not isinstance(self.factor_only, ModelingPanelUnmatchedAudit):
            raise ModelingPanelIntegrityError(
                "factor_only must be a ModelingPanelUnmatchedAudit."
            )
        if not isinstance(self.return_only, ModelingPanelUnmatchedAudit):
            raise ModelingPanelIntegrityError(
                "return_only must be a ModelingPanelUnmatchedAudit."
            )

        counts: dict[str, int] = {}
        for field_name in _COUNT_FIELDS:
            counts[field_name] = _nonnegative_int(
                field_name, getattr(self, field_name), ModelingPanelDataError
            )
            object.__setattr__(self, field_name, counts[field_name])
        for field_name in ("date_count", "security_count", "feature_count"):
            if counts[field_name] == 0:
                raise ModelingPanelDataError(f"{field_name} must be positive.")
        if counts["output_rows"] == 0:
            raise ModelingPanelDataError("output_rows must be positive.")
        if counts["matched_rows"] != counts["output_rows"]:
            raise ModelingPanelAlignmentError(
                "matched_rows must equal output_rows."
            )
        if counts["output_rows"] > counts["factor_input_rows"]:
            raise ModelingPanelAlignmentError(
                "output_rows cannot exceed factor_input_rows."
            )
        if counts["output_rows"] > counts["return_input_rows"]:
            raise ModelingPanelAlignmentError(
                "output_rows cannot exceed return_input_rows."
            )
        if self.factor_only.row_count != (
            counts["factor_input_rows"] - counts["matched_rows"]
        ):
            raise ModelingPanelAlignmentError(
                "factor_only.row_count must equal factor_input_rows - matched_rows."
            )
        if self.return_only.row_count != (
            counts["return_input_rows"] - counts["matched_rows"]
        ):
            raise ModelingPanelAlignmentError(
                "return_only.row_count must equal return_input_rows - matched_rows."
            )

        first_trade = _normalized_date(
            "first_trade_date", self.first_trade_date, allow_none=False
        )
        last_trade = _normalized_date(
            "last_trade_date", self.last_trade_date, allow_none=False
        )
        if first_trade is None or last_trade is None:
            raise ModelingPanelDataError(
                "first_trade_date and last_trade_date cannot be None."
            )
        if first_trade > last_trade:
            raise ModelingPanelDataError(
                "first_trade_date cannot be later than last_trade_date."
            )
        first_entry, last_entry = _date_pair(
            "first_entry_trade_date",
            self.first_entry_trade_date,
            "last_entry_trade_date",
            self.last_entry_trade_date,
        )
        first_exit, last_exit = _date_pair(
            "first_exit_trade_date",
            self.first_exit_trade_date,
            "last_exit_trade_date",
            self.last_exit_trade_date,
        )
        for field_name, value in (
            ("first_trade_date", first_trade),
            ("last_trade_date", last_trade),
            ("first_entry_trade_date", first_entry),
            ("last_entry_trade_date", last_entry),
            ("first_exit_trade_date", first_exit),
            ("last_exit_trade_date", last_exit),
        ):
            object.__setattr__(self, field_name, value)

        feature_names = _strict_name_tuple(
            "feature_names",
            self.feature_names,
            allow_empty=False,
            error=ModelingPanelDataError,
        )
        if len(feature_names) != counts["feature_count"]:
            raise ModelingPanelIntegrityError(
                "feature_count must equal len(feature_names)."
            )
        reserved = {
            *MODELING_PANEL_KEY_COLUMNS,
            *MODELING_PANEL_AUDIT_COLUMNS,
            label,
        }
        conflicts = [name for name in feature_names if name in reserved]
        if conflicts:
            raise ModelingPanelLeakageError(
                f"feature_names contains reserved or label columns: {conflicts!r}."
            )
        missing_counts = _feature_count_items(
            "feature_missing_counts",
            self.feature_missing_counts,
            feature_names,
            counts["output_rows"],
        )
        missing_rates = _feature_rate_items(
            "feature_missing_rates", self.feature_missing_rates, feature_names
        )
        non_finite_counts = _feature_count_items(
            "feature_non_finite_counts",
            self.feature_non_finite_counts,
            feature_names,
            counts["output_rows"],
        )
        for (name, missing_count), (_, missing_rate) in zip(
            missing_counts, missing_rates, strict=True
        ):
            expected = missing_count / counts["output_rows"]
            if not math.isclose(
                missing_rate, expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ModelingPanelIntegrityError(
                    f"feature_missing_rates[{name!r}] must equal "
                    "feature_missing_counts / output_rows."
                )
        all_missing = _ordered_feature_subset(
            "all_missing_features", self.all_missing_features, feature_names
        )
        constant = _ordered_feature_subset(
            "constant_features", self.constant_features, feature_names
        )
        suspicious = _ordered_feature_subset(
            "suspicious_feature_names",
            self.suspicious_feature_names,
            feature_names,
        )
        all_missing_set = set(all_missing)
        for name, missing_count in missing_counts:
            if (missing_count == counts["output_rows"]) != (name in all_missing_set):
                raise ModelingPanelIntegrityError(
                    "all_missing_features must exactly identify features whose "
                    "missing count equals output_rows."
                )
        for field_name, value in (
            ("feature_names", feature_names),
            ("feature_missing_counts", missing_counts),
            ("feature_missing_rates", missing_rates),
            ("feature_non_finite_counts", non_finite_counts),
            ("all_missing_features", all_missing),
            ("constant_features", constant),
            ("suspicious_feature_names", suspicious),
        ):
            object.__setattr__(self, field_name, value)

        bounded_counts = (
            "label_missing_count",
            "label_non_finite_count",
            "entry_before_signal_count",
            "entry_equal_signal_count",
            "exit_not_after_entry_count",
            "label_formula_mismatch_count",
        )
        for field_name in bounded_counts:
            if counts[field_name] > counts["output_rows"]:
                raise ModelingPanelDataError(
                    f"{field_name} cannot exceed output_rows."
                )
        if counts["duplicate_factor_key_count"] > counts["factor_input_rows"]:
            raise ModelingPanelDataError(
                "duplicate_factor_key_count cannot exceed factor_input_rows."
            )
        if counts["duplicate_return_key_count"] > counts["return_input_rows"]:
            raise ModelingPanelDataError(
                "duplicate_return_key_count cannot exceed return_input_rows."
            )

        for minimum_name, median_name, maximum_name in _DISTRIBUTION_FIELDS:
            minimum = _positive_int(
                minimum_name, getattr(self, minimum_name), ModelingPanelDataError
            )
            maximum = _positive_int(
                maximum_name, getattr(self, maximum_name), ModelingPanelDataError
            )
            median_value = getattr(self, median_name)
            if (
                isinstance(median_value, bool)
                or not isinstance(median_value, Real)
                or not math.isfinite(float(median_value))
                or float(median_value) <= 0
            ):
                raise ModelingPanelDataError(
                    f"{median_name} must be a positive finite number."
                )
            median = float(median_value)
            if not minimum <= median <= maximum:
                raise ModelingPanelDataError(
                    f"{minimum_name} <= {median_name} <= {maximum_name} is required."
                )
            object.__setattr__(self, minimum_name, minimum)
            object.__setattr__(self, median_name, median)
            object.__setattr__(self, maximum_name, maximum)

        warnings = _strict_name_tuple(
            "warnings",
            self.warnings,
            allow_empty=True,
            error=ModelingPanelDataError,
        )
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "label_column", label)

    def as_dict(self) -> dict[str, Any]:
        """Return all audit metadata as a detached, strictly JSON-safe mapping."""
        return {
            "schema_version": self.schema_version,
            "config": self.config.as_dict(),
            "label_column": self.label_column,
            "factor_input_rows": self.factor_input_rows,
            "return_input_rows": self.return_input_rows,
            "matched_rows": self.matched_rows,
            "output_rows": self.output_rows,
            "factor_only": self.factor_only.as_dict(),
            "return_only": self.return_only.as_dict(),
            "date_count": self.date_count,
            "security_count": self.security_count,
            "first_trade_date": _date_string(self.first_trade_date),
            "last_trade_date": _date_string(self.last_trade_date),
            "first_entry_trade_date": _date_string(self.first_entry_trade_date),
            "last_entry_trade_date": _date_string(self.last_entry_trade_date),
            "first_exit_trade_date": _date_string(self.first_exit_trade_date),
            "last_exit_trade_date": _date_string(self.last_exit_trade_date),
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
            "feature_missing_counts": dict(self.feature_missing_counts),
            "feature_missing_rates": dict(self.feature_missing_rates),
            "feature_non_finite_counts": dict(self.feature_non_finite_counts),
            "all_missing_features": list(self.all_missing_features),
            "constant_features": list(self.constant_features),
            "suspicious_feature_names": list(self.suspicious_feature_names),
            "label_missing_count": self.label_missing_count,
            "label_non_finite_count": self.label_non_finite_count,
            "duplicate_factor_key_count": self.duplicate_factor_key_count,
            "duplicate_return_key_count": self.duplicate_return_key_count,
            "entry_before_signal_count": self.entry_before_signal_count,
            "entry_equal_signal_count": self.entry_equal_signal_count,
            "exit_not_after_entry_count": self.exit_not_after_entry_count,
            "label_formula_mismatch_count": self.label_formula_mismatch_count,
            "per_date_security_count_min": self.per_date_security_count_min,
            "per_date_security_count_median": self.per_date_security_count_median,
            "per_date_security_count_max": self.per_date_security_count_max,
            "per_security_observation_count_min": (
                self.per_security_observation_count_min
            ),
            "per_security_observation_count_median": (
                self.per_security_observation_count_median
            ),
            "per_security_observation_count_max": (
                self.per_security_observation_count_max
            ),
            "warnings": list(self.warnings),
        }


class ModelingPanelResult:
    """Defensively encapsulate a validated Modeling Panel and its metadata."""

    def __init__(
        self,
        panel: pd.DataFrame,
        *,
        feature_names: tuple[str, ...],
        label_column: str,
        audit: ModelingPanelAudit,
        config: ModelingPanelConfig,
        schema_version: str = MODELING_PANEL_SCHEMA_VERSION,
    ) -> None:
        if not isinstance(panel, pd.DataFrame):
            raise ModelingPanelDataError("panel must be a pandas DataFrame.")
        if panel.empty:
            raise ModelingPanelDataError("panel must not be empty.")
        if not panel.columns.is_unique:
            raise ModelingPanelDataError("panel columns must be unique.")
        names = _strict_name_tuple(
            "feature_names",
            feature_names,
            allow_empty=False,
            error=ModelingPanelDataError,
        )
        label = _normalized_name(
            "label_column", label_column, ModelingPanelDataError
        )
        reserved = {
            *MODELING_PANEL_KEY_COLUMNS,
            *MODELING_PANEL_AUDIT_COLUMNS,
            label,
        }
        conflicts = [name for name in names if name in reserved]
        if conflicts:
            raise ModelingPanelLeakageError(
                f"feature_names contains reserved or label columns: {conflicts!r}."
            )
        expected_columns = (
            *MODELING_PANEL_KEY_COLUMNS,
            *names,
            *MODELING_PANEL_AUDIT_COLUMNS,
            label,
        )
        if tuple(panel.columns) != expected_columns:
            raise ModelingPanelIntegrityError(
                "panel columns must exactly match the required ordered schema: "
                f"{expected_columns!r}."
            )
        if not isinstance(config, ModelingPanelConfig):
            raise ModelingPanelIntegrityError(
                "config must be a ModelingPanelConfig."
            )
        if not isinstance(audit, ModelingPanelAudit):
            raise ModelingPanelIntegrityError(
                "audit must be a ModelingPanelAudit."
            )
        if schema_version != MODELING_PANEL_SCHEMA_VERSION:
            raise ModelingPanelIntegrityError(
                "schema_version must equal "
                f"{MODELING_PANEL_SCHEMA_VERSION!r}."
            )
        if config.label_column != label:
            raise ModelingPanelIntegrityError(
                "label_column must match config.label_column."
            )
        if audit.config != config:
            raise ModelingPanelIntegrityError("audit.config must match config.")
        if audit.schema_version != schema_version:
            raise ModelingPanelIntegrityError(
                "audit.schema_version must match schema_version."
            )
        if audit.label_column != label:
            raise ModelingPanelIntegrityError(
                "audit.label_column must match label_column."
            )
        if audit.output_rows != len(panel):
            raise ModelingPanelIntegrityError(
                "audit.output_rows must match panel row count."
            )
        if audit.feature_count != len(names) or audit.feature_names != names:
            raise ModelingPanelIntegrityError(
                "audit feature_count and feature_names must match feature_names."
            )

        if audit.duplicate_factor_key_count or audit.duplicate_return_key_count:
            raise ModelingPanelIntegrityError(
                "A successful result requires zero duplicate key counts."
            )
        if audit.entry_before_signal_count:
            raise ModelingPanelIntegrityError(
                "A successful result requires zero entry_before_signal_count."
            )
        if config.require_entry_after_signal and audit.entry_equal_signal_count:
            raise ModelingPanelIntegrityError(
                "Strict entry ordering requires zero entry_equal_signal_count."
            )
        if audit.exit_not_after_entry_count:
            raise ModelingPanelIntegrityError(
                "A successful result requires zero exit_not_after_entry_count."
            )
        if audit.label_formula_mismatch_count:
            raise ModelingPanelIntegrityError(
                "A successful result requires zero label_formula_mismatch_count."
            )
        if config.unmatched_policy == "error" and (
            audit.factor_only.row_count or audit.return_only.row_count
        ):
            raise ModelingPanelAlignmentError(
                "unmatched_policy='error' requires zero unmatched rows."
            )
        if not config.allow_missing_labels and audit.label_missing_count:
            raise ModelingPanelIntegrityError(
                "allow_missing_labels=False requires zero label_missing_count."
            )
        if audit.all_missing_features:
            raise ModelingPanelIntegrityError(
                "A successful result cannot contain all-missing features."
            )
        if audit.label_non_finite_count:
            raise ModelingPanelIntegrityError(
                "A successful result requires zero label_non_finite_count."
            )
        if any(count for _, count in audit.feature_non_finite_counts):
            raise ModelingPanelIntegrityError(
                "A successful result requires zero feature non-finite counts."
            )

        self._panel = panel.copy(deep=True)
        self._feature_names = names
        self._label_column = label
        self._audit = audit
        self._config = config
        self._schema_version = schema_version

    @property
    def panel(self) -> pd.DataFrame:
        """Return a deep defensive copy of the Modeling Panel."""
        return self._panel.copy(deep=True)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the immutable ordered feature names."""
        return self._feature_names

    @property
    def label_column(self) -> str:
        """Return the configured label column."""
        return self._label_column

    @property
    def audit(self) -> ModelingPanelAudit:
        """Return immutable audit metadata."""
        return self._audit

    @property
    def config(self) -> ModelingPanelConfig:
        """Return immutable builder configuration."""
        return self._config

    @property
    def schema_version(self) -> str:
        """Return the immutable Modeling Panel schema version."""
        return self._schema_version
