"""Strict construction of model-free ML datasets from published V2 tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.ml.contracts import (
    METADATA_COLUMNS,
    MLDataset,
    MLDatasetAlignmentError,
    MLDatasetAudit,
    MLDatasetConfig,
    MLDatasetDuplicateKeyError,
    MLDatasetSchemaError,
    MLDatasetValueError,
)


_KEY_COLUMNS = ("trade_date", "ts_code")
_RESERVED_FEATURE_COLUMNS = {
    *_KEY_COLUMNS,
    "entry_trade_date",
    "exit_trade_date",
    "entry_price",
    "exit_price",
}


class MLDatasetBuilder:
    """Build one strictly aligned, audited single-label regression dataset."""

    def __init__(self, config: MLDatasetConfig | None = None) -> None:
        if config is not None and not isinstance(config, MLDatasetConfig):
            raise TypeError("config must be an MLDatasetConfig or None.")
        self.config = config or MLDatasetConfig()

    def build(
        self,
        factor_panel: pd.DataFrame,
        forward_returns: pd.DataFrame,
        feature_names: Sequence[str],
    ) -> MLDataset:
        """Validate, align, filter, and audit V2 feature and label tables."""
        if not isinstance(factor_panel, pd.DataFrame):
            raise MLDatasetSchemaError(
                "factor_panel must be a pandas DataFrame."
            )
        if not isinstance(forward_returns, pd.DataFrame):
            raise MLDatasetSchemaError(
                "forward_returns must be a pandas DataFrame."
            )
        if factor_panel.empty:
            raise MLDatasetValueError(
                "factor_panel contains 0 rows; at least one feature row is required."
            )
        if forward_returns.empty:
            raise MLDatasetValueError(
                "forward_returns contains 0 rows; at least one label row is required."
            )

        names = self._normalize_feature_names(feature_names)
        features = factor_panel.copy(deep=True)
        labels = forward_returns.copy(deep=True)
        input_feature_rows = len(features)
        input_label_rows = len(labels)

        self._validate_required_columns(
            features,
            [*_KEY_COLUMNS, *names],
            "factor_panel",
        )
        self._validate_required_columns(
            labels,
            [
                *_KEY_COLUMNS,
                "entry_trade_date",
                "exit_trade_date",
                self.config.label_col,
            ],
            "forward_returns",
        )

        features = self._normalize_keys(features, "factor_panel")
        labels = self._normalize_keys(labels, "forward_returns")
        self._validate_unique_keys(features, "factor_panel")
        self._validate_unique_keys(labels, "forward_returns")
        self._validate_key_alignment(features, labels)
        aligned_rows = len(features)
        if aligned_rows == 0:
            raise MLDatasetAlignmentError(
                "Strict key alignment produced 0 rows; no dataset can be built."
            )

        feature_nonfinite_counts: list[tuple[str, int]] = []
        for name in names:
            numeric = self._numeric_series(features[name], name, "feature")
            infinite = np.isinf(numeric.to_numpy(dtype=float))
            feature_nonfinite_counts.append((name, int(infinite.sum())))
            features[name] = numeric.mask(infinite, np.nan).astype("float64")

        label_numeric = self._numeric_series(
            labels[self.config.label_col],
            self.config.label_col,
            "label",
        )
        raw_label_missing = labels[self.config.label_col].isna()
        label_infinite = pd.Series(
            np.isinf(label_numeric.to_numpy(dtype=float)),
            index=labels.index,
        )
        missing_label_rows = int(raw_label_missing.sum())
        nonfinite_label_rows = int(label_infinite.sum())
        dropped_mask = raw_label_missing | label_infinite
        dropped_label_rows = int(dropped_mask.sum())
        labels[self.config.label_col] = label_numeric.astype("float64")
        labels["_drop_label"] = dropped_mask.astype(bool)

        labels["entry_trade_date"] = self._normalize_audit_date(
            labels["entry_trade_date"], "entry_trade_date"
        )
        labels["exit_trade_date"] = self._normalize_audit_date(
            labels["exit_trade_date"], "exit_trade_date"
        )

        feature_columns = [*_KEY_COLUMNS, *names]
        label_columns = [
            *_KEY_COLUMNS,
            "entry_trade_date",
            "exit_trade_date",
            self.config.label_col,
            "_drop_label",
        ]
        merged = features.loc[:, feature_columns].merge(
            labels.loc[:, label_columns],
            on=list(_KEY_COLUMNS),
            how="inner",
            sort=False,
            validate="one_to_one",
        )
        valid = merged.loc[~merged["_drop_label"]].copy(deep=True)
        if valid.empty:
            raise MLDatasetValueError(
                "All aligned labels were dropped: "
                f"aligned_rows={aligned_rows}, missing_label_rows={missing_label_rows}, "
                f"nonfinite_label_rows={nonfinite_label_rows}."
            )

        self._validate_retained_dates(valid)
        valid = valid.sort_values(
            list(_KEY_COLUMNS), kind="mergesort", ignore_index=True
        )

        output_features = valid.loc[:, list(names)].astype("float64")
        feature_missing_counts: list[tuple[str, int]] = []
        feature_missing_rates: list[tuple[str, float]] = []
        for name in names:
            missing_count = int(output_features[name].isna().sum())
            if missing_count == len(output_features):
                raise MLDatasetValueError(
                    f"Feature {name!r} is entirely missing after label filtering: "
                    f"missing_rows={missing_count}, output_rows={len(output_features)}."
                )
            feature_missing_counts.append((name, missing_count))
            feature_missing_rates.append(
                (name, float(missing_count / len(output_features)))
            )

        output_labels = valid.loc[:, self.config.label_col].astype("float64")
        output_labels.name = self.config.label_col
        output_metadata = valid.loc[:, list(METADATA_COLUMNS)].copy(deep=True)
        output_metadata["ts_code"] = output_metadata["ts_code"].astype("string")
        for name in ("trade_date", "entry_trade_date", "exit_trade_date"):
            output_metadata[name] = output_metadata[name].astype("datetime64[ns]")

        output_features = output_features.reset_index(drop=True)
        output_labels = output_labels.reset_index(drop=True)
        output_metadata = output_metadata.reset_index(drop=True)
        output_rows = len(output_features)
        audit = MLDatasetAudit(
            input_feature_rows=input_feature_rows,
            input_label_rows=input_label_rows,
            aligned_rows=aligned_rows,
            output_rows=output_rows,
            dropped_label_rows=dropped_label_rows,
            missing_label_rows=missing_label_rows,
            nonfinite_label_rows=nonfinite_label_rows,
            label_coverage=float(output_rows / aligned_rows),
            feature_count=len(names),
            feature_missing_counts=tuple(feature_missing_counts),
            feature_missing_rates=tuple(feature_missing_rates),
            feature_nonfinite_counts=tuple(feature_nonfinite_counts),
            min_trade_date=pd.Timestamp(output_metadata["trade_date"].min()),
            max_trade_date=pd.Timestamp(output_metadata["trade_date"].max()),
        )
        return MLDataset(
            output_features,
            output_labels,
            output_metadata,
            names,
            self.config.label_col,
            audit,
        )

    def _normalize_feature_names(
        self, feature_names: Sequence[str]
    ) -> tuple[str, ...]:
        if isinstance(feature_names, (str, bytes)):
            raise MLDatasetSchemaError(
                "feature_names must be a non-empty sequence, not a string."
            )
        try:
            raw_names = tuple(feature_names)
        except TypeError as exc:
            raise MLDatasetSchemaError(
                "feature_names must be a non-empty sequence."
            ) from exc
        if not raw_names:
            raise MLDatasetSchemaError("feature_names must not be empty.")
        if any(not isinstance(name, str) or not name.strip() for name in raw_names):
            raise MLDatasetSchemaError(
                "feature_names must contain only non-empty strings."
            )
        names = tuple(name.strip() for name in raw_names)
        if len(set(names)) != len(names):
            raise MLDatasetSchemaError(
                f"feature_names contains duplicates: {self._duplicates(names)!r}."
            )
        forbidden_columns = _RESERVED_FEATURE_COLUMNS | {self.config.label_col}
        forbidden = [name for name in names if name in forbidden_columns]
        if forbidden:
            raise MLDatasetSchemaError(
                f"feature_names contains reserved columns: {forbidden!r}."
            )
        return names

    @staticmethod
    def _validate_required_columns(
        frame: pd.DataFrame, required: Sequence[str], frame_name: str
    ) -> None:
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise MLDatasetSchemaError(
                f"{frame_name} is missing required columns: {missing!r}."
            )

    @staticmethod
    def _normalize_keys(frame: pd.DataFrame, frame_name: str) -> pd.DataFrame:
        raw_dates = frame["trade_date"]
        missing_dates = raw_dates.isna()
        converted_dates = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
        invalid_dates = raw_dates.notna() & converted_dates.isna()
        if missing_dates.any() or invalid_dates.any():
            raise MLDatasetValueError(
                f"{frame_name}.trade_date contains invalid values: "
                f"missing={int(missing_dates.sum())}, "
                f"unparseable={int(invalid_dates.sum())}."
            )
        frame["trade_date"] = converted_dates

        codes = frame["ts_code"].astype("string").str.strip()
        invalid_codes = codes.isna() | codes.eq("")
        if invalid_codes.any():
            raise MLDatasetValueError(
                f"{frame_name}.ts_code contains missing or empty values: "
                f"invalid_count={int(invalid_codes.sum())}."
            )
        frame["ts_code"] = codes
        return frame

    @staticmethod
    def _validate_unique_keys(frame: pd.DataFrame, frame_name: str) -> None:
        duplicate_rows = frame.duplicated(list(_KEY_COLUMNS), keep=False)
        if not duplicate_rows.any():
            return
        duplicate_keys = (
            frame.loc[duplicate_rows, list(_KEY_COLUMNS)]
            .drop_duplicates()
            .sort_values(list(_KEY_COLUMNS), kind="mergesort")
        )
        raise MLDatasetDuplicateKeyError(
            f"{frame_name} contains duplicate trade_date + ts_code keys: "
            f"duplicate_key_count={len(duplicate_keys)}, "
            f"examples={MLDatasetBuilder._key_examples(duplicate_keys)}."
        )

    @staticmethod
    def _validate_key_alignment(
        features: pd.DataFrame, labels: pd.DataFrame
    ) -> None:
        feature_keys = pd.MultiIndex.from_frame(features.loc[:, list(_KEY_COLUMNS)])
        label_keys = pd.MultiIndex.from_frame(labels.loc[:, list(_KEY_COLUMNS)])
        feature_only = feature_keys.difference(label_keys, sort=True)
        label_only = label_keys.difference(feature_keys, sort=True)
        if not feature_only.empty or not label_only.empty:
            raise MLDatasetAlignmentError(
                "factor_panel and forward_returns key sets must match exactly: "
                f"feature_only_key_count={len(feature_only)}, "
                f"label_only_key_count={len(label_only)}, "
                f"feature_only_examples={MLDatasetBuilder._index_examples(feature_only)}, "
                f"label_only_examples={MLDatasetBuilder._index_examples(label_only)}."
            )

    @staticmethod
    def _numeric_series(
        values: pd.Series, column_name: str, value_role: str
    ) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        invalid = values.notna() & numeric.isna()
        if invalid.any():
            examples = values.loc[invalid].astype("string").drop_duplicates().head(5)
            raise MLDatasetValueError(
                f"{value_role} column {column_name!r} contains non-numeric "
                f"non-empty values: invalid_count={int(invalid.sum())}, "
                f"examples={examples.tolist()!r}."
            )
        return numeric.astype("float64")

    @staticmethod
    def _normalize_audit_date(values: pd.Series, column_name: str) -> pd.Series:
        converted = pd.to_datetime(values, errors="coerce", format="mixed")
        invalid = values.notna() & converted.isna()
        if invalid.any():
            raise MLDatasetValueError(
                f"{column_name} contains unparseable non-empty dates: "
                f"invalid_count={int(invalid.sum())}."
            )
        return converted

    @staticmethod
    def _validate_retained_dates(frame: pd.DataFrame) -> None:
        for name in ("entry_trade_date", "exit_trade_date"):
            missing = frame[name].isna()
            if missing.any():
                raise MLDatasetValueError(
                    f"{name} is missing for retained valid-label samples: "
                    f"missing_count={int(missing.sum())}."
                )
        entry_before_score = frame["entry_trade_date"] < frame["trade_date"]
        if entry_before_score.any():
            raise MLDatasetValueError(
                "Date order requires trade_date <= entry_trade_date: "
                f"invalid_count={int(entry_before_score.sum())}."
            )
        exit_not_after_entry = frame["exit_trade_date"] <= frame["entry_trade_date"]
        if exit_not_after_entry.any():
            raise MLDatasetValueError(
                "Date order requires entry_trade_date < exit_trade_date: "
                f"invalid_count={int(exit_not_after_entry.sum())}."
            )

    @staticmethod
    def _key_examples(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                "trade_date": pd.Timestamp(row.trade_date).isoformat(),
                "ts_code": str(row.ts_code),
            }
            for row in frame.head(5).itertuples(index=False)
        ]

    @staticmethod
    def _index_examples(index: pd.MultiIndex) -> list[dict[str, Any]]:
        return [
            {
                "trade_date": pd.Timestamp(trade_date).isoformat(),
                "ts_code": str(ts_code),
            }
            for trade_date, ts_code in list(index[:5])
        ]

    @staticmethod
    def _duplicates(values: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for value in values:
            if value in seen and value not in duplicates:
                duplicates.append(value)
            seen.add(value)
        return duplicates
