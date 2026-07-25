"""Public, model-free contracts for auditable machine-learning datasets."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import pandas as pd


METADATA_COLUMNS = (
    "trade_date",
    "ts_code",
    "entry_trade_date",
    "exit_trade_date",
)
_ALWAYS_FORBIDDEN_FEATURE_COLUMNS = {
    *METADATA_COLUMNS,
    "entry_price",
    "exit_price",
}


class MLDatasetError(ValueError):
    """Base error for invalid ML dataset inputs and contracts."""


class MLDatasetSchemaError(MLDatasetError):
    """Raised when required columns or object schemas are invalid."""


class MLDatasetDuplicateKeyError(MLDatasetError):
    """Raised when ``trade_date + ts_code`` is not unique."""


class MLDatasetAlignmentError(MLDatasetError):
    """Raised when feature and label key sets are not identical."""


class MLDatasetValueError(MLDatasetError):
    """Raised when a present field contains an invalid value."""


@dataclass(frozen=True)
class MLDatasetConfig:
    """Configure the single regression-label column for V3-A."""

    label_col: str = "forward_return"

    def __post_init__(self) -> None:
        if not isinstance(self.label_col, str) or not self.label_col.strip():
            raise MLDatasetSchemaError(
                "label_col must be a non-empty string; received an empty value."
            )
        label_col = self.label_col.strip()
        if label_col in METADATA_COLUMNS:
            raise MLDatasetSchemaError(
                f"label_col {label_col!r} conflicts with reserved metadata column "
                f"{label_col!r}."
            )
        object.__setattr__(self, "label_col", label_col)


FeatureCountItems = tuple[tuple[str, int], ...]
FeatureRateItems = tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class MLDatasetAudit:
    """Immutable counts and coverage statistics for one built dataset."""

    input_feature_rows: int
    input_label_rows: int
    aligned_rows: int
    output_rows: int
    dropped_label_rows: int
    missing_label_rows: int
    nonfinite_label_rows: int
    label_coverage: float
    feature_count: int
    feature_missing_counts: FeatureCountItems
    feature_missing_rates: FeatureRateItems
    feature_nonfinite_counts: FeatureCountItems
    min_trade_date: pd.Timestamp | None
    max_trade_date: pd.Timestamp | None

    def __post_init__(self) -> None:
        count_fields = (
            "input_feature_rows",
            "input_label_rows",
            "aligned_rows",
            "output_rows",
            "dropped_label_rows",
            "missing_label_rows",
            "nonfinite_label_rows",
            "feature_count",
        )
        for field_name in count_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise MLDatasetValueError(
                    f"{field_name} must be a non-negative integer; received {value!r}."
                )
            object.__setattr__(self, field_name, int(value))

        if (
            isinstance(self.label_coverage, bool)
            or not isinstance(self.label_coverage, Real)
            or not 0.0 <= float(self.label_coverage) <= 1.0
        ):
            raise MLDatasetValueError(
                "label_coverage must be between 0 and 1; "
                f"received {self.label_coverage!r}."
            )
        object.__setattr__(self, "label_coverage", float(self.label_coverage))

        counts = self._normalize_count_items(
            "feature_missing_counts", self.feature_missing_counts
        )
        rates = self._normalize_rate_items(
            "feature_missing_rates", self.feature_missing_rates
        )
        nonfinite = self._normalize_count_items(
            "feature_nonfinite_counts", self.feature_nonfinite_counts
        )
        names = tuple(name for name, _ in counts)
        if tuple(name for name, _ in rates) != names:
            raise MLDatasetSchemaError(
                "feature_missing_rates names and order must match "
                "feature_missing_counts."
            )
        if tuple(name for name, _ in nonfinite) != names:
            raise MLDatasetSchemaError(
                "feature_nonfinite_counts names and order must match "
                "feature_missing_counts."
            )
        if len(names) != self.feature_count:
            raise MLDatasetSchemaError(
                "feature_count does not match feature statistics: "
                f"feature_count={self.feature_count}, statistics={len(names)}."
            )
        object.__setattr__(self, "feature_missing_counts", counts)
        object.__setattr__(self, "feature_missing_rates", rates)
        object.__setattr__(self, "feature_nonfinite_counts", nonfinite)

        if self.dropped_label_rows != (
            self.missing_label_rows + self.nonfinite_label_rows
        ):
            raise MLDatasetValueError(
                "dropped_label_rows must equal missing_label_rows + "
                "nonfinite_label_rows."
            )
        if self.output_rows + self.dropped_label_rows != self.aligned_rows:
            raise MLDatasetValueError(
                "output_rows + dropped_label_rows must equal aligned_rows."
            )
        expected_coverage = (
            self.output_rows / self.aligned_rows if self.aligned_rows else 0.0
        )
        if abs(self.label_coverage - expected_coverage) > 1e-12:
            raise MLDatasetValueError(
                "label_coverage does not match output_rows / aligned_rows: "
                f"{self.label_coverage} != {expected_coverage}."
            )

        min_date = self._normalize_optional_timestamp(
            "min_trade_date", self.min_trade_date
        )
        max_date = self._normalize_optional_timestamp(
            "max_trade_date", self.max_trade_date
        )
        if min_date is not None and max_date is not None and min_date > max_date:
            raise MLDatasetValueError(
                "min_trade_date cannot be later than max_trade_date."
            )
        object.__setattr__(self, "min_trade_date", min_date)
        object.__setattr__(self, "max_trade_date", max_date)

    @staticmethod
    def _normalize_count_items(name: str, value: object) -> FeatureCountItems:
        try:
            items = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise MLDatasetSchemaError(f"{name} must be tuple pairs.") from exc
        normalized: list[tuple[str, int]] = []
        for item in items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise MLDatasetSchemaError(f"{name} must contain (name, count) pairs.")
            feature_name, count = item
            if not isinstance(feature_name, str) or not feature_name:
                raise MLDatasetSchemaError(
                    f"{name} contains an empty feature name."
                )
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
                raise MLDatasetValueError(
                    f"{name}[{feature_name!r}] must be a non-negative integer."
                )
            normalized.append((feature_name, int(count)))
        names = [item[0] for item in normalized]
        if len(set(names)) != len(names):
            raise MLDatasetSchemaError(f"{name} contains duplicate feature names.")
        return tuple(normalized)

    @staticmethod
    def _normalize_rate_items(name: str, value: object) -> FeatureRateItems:
        try:
            items = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise MLDatasetSchemaError(f"{name} must be tuple pairs.") from exc
        normalized: list[tuple[str, float]] = []
        for item in items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise MLDatasetSchemaError(f"{name} must contain (name, rate) pairs.")
            feature_name, rate = item
            if not isinstance(feature_name, str) or not feature_name:
                raise MLDatasetSchemaError(
                    f"{name} contains an empty feature name."
                )
            if (
                isinstance(rate, bool)
                or not isinstance(rate, Real)
                or not 0.0 <= float(rate) <= 1.0
            ):
                raise MLDatasetValueError(
                    f"{name}[{feature_name!r}] must be between 0 and 1."
                )
            normalized.append((feature_name, float(rate)))
        names = [item[0] for item in normalized]
        if len(set(names)) != len(names):
            raise MLDatasetSchemaError(f"{name} contains duplicate feature names.")
        return tuple(normalized)

    @staticmethod
    def _normalize_optional_timestamp(
        field_name: str, value: object
    ) -> pd.Timestamp | None:
        if value is None:
            return None
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise MLDatasetValueError(
                f"{field_name} must be a valid Timestamp or None."
            ) from exc
        if pd.isna(timestamp):
            raise MLDatasetValueError(
                f"{field_name} must be a valid Timestamp or None."
            )
        return timestamp

    def as_dict(self) -> dict[str, Any]:
        """Return a detached, directly JSON-serializable audit dictionary."""
        return {
            "input_feature_rows": self.input_feature_rows,
            "input_label_rows": self.input_label_rows,
            "aligned_rows": self.aligned_rows,
            "output_rows": self.output_rows,
            "dropped_label_rows": self.dropped_label_rows,
            "missing_label_rows": self.missing_label_rows,
            "nonfinite_label_rows": self.nonfinite_label_rows,
            "label_coverage": self.label_coverage,
            "feature_count": self.feature_count,
            "feature_missing_counts": dict(self.feature_missing_counts),
            "feature_missing_rates": dict(self.feature_missing_rates),
            "feature_nonfinite_counts": dict(self.feature_nonfinite_counts),
            "min_trade_date": (
                self.min_trade_date.isoformat()
                if self.min_trade_date is not None
                else None
            ),
            "max_trade_date": (
                self.max_trade_date.isoformat()
                if self.max_trade_date is not None
                else None
            ),
        }


class MLDataset:
    """Defensively encapsulate aligned ML features, labels, and audit metadata."""

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        metadata: pd.DataFrame,
        feature_names: tuple[str, ...],
        label_name: str,
        audit: MLDatasetAudit,
    ) -> None:
        if not isinstance(features, pd.DataFrame):
            raise MLDatasetSchemaError("features must be a pandas DataFrame.")
        if not isinstance(labels, pd.Series):
            raise MLDatasetSchemaError("labels must be a pandas Series.")
        if not isinstance(metadata, pd.DataFrame):
            raise MLDatasetSchemaError("metadata must be a pandas DataFrame.")
        names = self._validate_feature_names(feature_names)
        if list(features.columns) != list(names):
            raise MLDatasetSchemaError(
                "features columns must exactly match feature_names in order: "
                f"columns={list(features.columns)!r}, feature_names={list(names)!r}."
            )
        if not isinstance(label_name, str) or not label_name.strip():
            raise MLDatasetSchemaError("label_name must be a non-empty string.")
        label_name = label_name.strip()
        if labels.name != label_name:
            raise MLDatasetSchemaError(
                f"labels.name must equal label_name {label_name!r}; "
                f"received {labels.name!r}."
            )
        if list(metadata.columns) != list(METADATA_COLUMNS):
            raise MLDatasetSchemaError(
                "metadata columns must exactly be "
                f"{list(METADATA_COLUMNS)!r}; received {list(metadata.columns)!r}."
            )
        if label_name in features.columns:
            raise MLDatasetSchemaError(
                f"features cannot contain label column {label_name!r}."
            )
        forbidden = sorted(set(features.columns) & _ALWAYS_FORBIDDEN_FEATURE_COLUMNS)
        if forbidden:
            raise MLDatasetSchemaError(
                f"features contain reserved key/date/price columns: {forbidden!r}."
            )
        lengths = (len(features), len(labels), len(metadata))
        if len(set(lengths)) != 1:
            raise MLDatasetAlignmentError(
                "features, labels, and metadata row counts must match: "
                f"features={lengths[0]}, labels={lengths[1]}, metadata={lengths[2]}."
            )
        expected_index = pd.RangeIndex(len(features))
        for object_name, index in (
            ("features", features.index),
            ("labels", labels.index),
            ("metadata", metadata.index),
        ):
            if not index.equals(expected_index):
                raise MLDatasetAlignmentError(
                    f"{object_name} index must be RangeIndex(0, {len(features)}); "
                    f"received {index!r}."
                )
        if not features.index.equals(labels.index) or not features.index.equals(
            metadata.index
        ):
            raise MLDatasetAlignmentError(
                "features, labels, and metadata indexes must be identical."
            )
        if not isinstance(audit, MLDatasetAudit):
            raise MLDatasetSchemaError("audit must be an MLDatasetAudit.")
        if audit.output_rows != len(features):
            raise MLDatasetAlignmentError(
                "audit.output_rows must match dataset rows: "
                f"audit={audit.output_rows}, dataset={len(features)}."
            )
        if audit.feature_count != len(names):
            raise MLDatasetAlignmentError(
                "audit.feature_count must match feature_names: "
                f"audit={audit.feature_count}, features={len(names)}."
            )
        audit_names = tuple(name for name, _ in audit.feature_missing_counts)
        if audit_names != names:
            raise MLDatasetAlignmentError(
                "audit feature statistic order must match feature_names."
            )

        self._features = features.copy(deep=True)
        self._labels = labels.copy(deep=True)
        self._metadata = metadata.copy(deep=True)
        self.feature_names = names
        self.label_name = label_name
        self.audit = audit

    @staticmethod
    def _validate_feature_names(value: object) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise MLDatasetSchemaError("feature_names must be a non-empty tuple.")
        try:
            names = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise MLDatasetSchemaError(
                "feature_names must be a non-empty tuple."
            ) from exc
        if not names:
            raise MLDatasetSchemaError("feature_names must not be empty.")
        if any(not isinstance(name, str) or not name for name in names):
            raise MLDatasetSchemaError(
                "feature_names must contain non-empty strings."
            )
        if len(set(names)) != len(names):
            raise MLDatasetSchemaError("feature_names must be unique.")
        return names

    @property
    def features(self) -> pd.DataFrame:
        """Return a deep defensive copy of the feature matrix."""
        return self._features.copy(deep=True)

    @property
    def labels(self) -> pd.Series:
        """Return a deep defensive copy of the regression labels."""
        return self._labels.copy(deep=True)

    @property
    def metadata(self) -> pd.DataFrame:
        """Return a deep defensive copy of sample audit metadata."""
        return self._metadata.copy(deep=True)

    @property
    def n_samples(self) -> int:
        """Return the number of retained labeled samples."""
        return len(self._labels)

    @property
    def n_features(self) -> int:
        """Return the number of ordered feature columns."""
        return len(self.feature_names)

    @property
    def date_range(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Return the retained minimum and maximum score dates."""
        return self.audit.min_trade_date, self.audit.max_trade_date

    def summary(self) -> dict[str, Any]:
        """Return compact JSON-safe metadata without exposing sample values."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names),
            "label_name": self.label_name,
            "date_range": {
                "min": (
                    self.audit.min_trade_date.isoformat()
                    if self.audit.min_trade_date is not None
                    else None
                ),
                "max": (
                    self.audit.max_trade_date.isoformat()
                    if self.audit.max_trade_date is not None
                    else None
                ),
            },
            "audit": self.audit.as_dict(),
        }
