"""Pure in-memory construction of auditable Modeling Panels."""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pandas.errors import MergeError

from src.modeling_panel.contracts import (
    MODELING_PANEL_AUDIT_COLUMNS,
    MODELING_PANEL_KEY_COLUMNS,
    MODELING_PANEL_SCHEMA_VERSION,
    ModelingPanelAlignmentError,
    ModelingPanelAudit,
    ModelingPanelConfig,
    ModelingPanelConfigError,
    ModelingPanelDataError,
    ModelingPanelIntegrityError,
    ModelingPanelLeakageError,
    ModelingPanelResult,
    ModelingPanelUnmatchedAudit,
)


_SAMPLE_LIMIT = 20
_HIGH_MISSING_RATE = 0.50
_FORMULA_RTOL = 1e-10
_FORMULA_ATOL = 1e-12
_SUSPICIOUS_PREFIXES = ("future_", "next_", "lead_", "target_", "label_")


class ModelingPanelBuilder:
    """Validate and align factor and forward-return tables entirely in memory."""

    def __init__(self, config: ModelingPanelConfig | None = None) -> None:
        if config is not None and not isinstance(config, ModelingPanelConfig):
            raise ModelingPanelConfigError(
                "config must be a ModelingPanelConfig or None."
            )
        self._config = ModelingPanelConfig() if config is None else config

    @property
    def config(self) -> ModelingPanelConfig:
        """Return the immutable configuration used by every build."""
        return self._config

    def build(
        self,
        factor_panel: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> ModelingPanelResult:
        """Build one strictly validated, deterministically ordered panel."""
        factors = self._copy_input(factor_panel, "factor_panel")
        returns = self._copy_input(forward_returns, "forward_returns")
        factor_input_rows = len(factors)
        return_input_rows = len(returns)

        self._validate_columns(factors, returns)
        factors["trade_date"] = self._normalize_date_column(
            factors["trade_date"],
            "factor_panel.trade_date",
            allow_missing=False,
        )
        returns["trade_date"] = self._normalize_date_column(
            returns["trade_date"],
            "forward_returns.trade_date",
            allow_missing=False,
        )
        for column in ("entry_trade_date", "exit_trade_date"):
            returns[column] = self._normalize_date_column(
                returns[column],
                f"forward_returns.{column}",
                allow_missing=True,
            )
        factors["ts_code"] = self._normalize_codes(
            factors["ts_code"], "factor_panel.ts_code"
        )
        returns["ts_code"] = self._normalize_codes(
            returns["ts_code"], "forward_returns.ts_code"
        )

        self._validate_unique_keys(factors, "factor_panel")
        self._validate_unique_keys(returns, "forward_returns")
        feature_names, ignored_factor_columns = self._resolve_features(factors)
        extra_return_columns = self._extra_return_columns(returns)
        self._validate_non_key_collisions(
            factors, returns, feature_names, extra_return_columns
        )
        self._validate_numeric_inputs(factors, returns, feature_names)

        factor_keys = self._key_tuples(factors)
        return_keys = self._key_tuples(returns)
        factor_key_set = set(factor_keys)
        return_key_set = set(return_keys)
        factor_only_keys = [
            key for key in factor_keys if key not in return_key_set
        ]
        return_only_keys = [
            key for key in return_keys if key not in factor_key_set
        ]
        matched_keys = factor_key_set & return_key_set
        factor_only = self._unmatched_audit(factor_only_keys)
        return_only = self._unmatched_audit(return_only_keys)
        if not matched_keys:
            raise ModelingPanelAlignmentError(
                "factor_panel and forward_returns have no matched keys; "
                f"factor_only={factor_only.row_count}, "
                f"return_only={return_only.row_count}."
            )
        if self.config.unmatched_policy == "error" and (
            factor_only.row_count or return_only.row_count
        ):
            raise ModelingPanelAlignmentError(
                "unmatched_policy='error' rejected unmatched keys: "
                f"factor_only={factor_only.row_count}, "
                f"return_only={return_only.row_count}, "
                f"factor_sample={factor_only.as_dict()['sampled_keys']!r}, "
                f"return_sample={return_only.as_dict()['sampled_keys']!r}."
            )

        merged = self._merge(factors, returns, feature_names)
        matched_rows = len(matched_keys)
        if len(merged) != matched_rows:
            raise ModelingPanelIntegrityError(
                "one-to-one merge row count does not equal the key intersection: "
                f"merged_rows={len(merged)}, matched_keys={matched_rows}."
            )
        if len(merged) > min(factor_input_rows, return_input_rows):
            raise ModelingPanelIntegrityError(
                "one-to-one merge unexpectedly expanded the input row count."
            )

        merged = merged.sort_values(
            list(MODELING_PANEL_KEY_COLUMNS),
            kind="mergesort",
            ignore_index=True,
        )
        self._validate_structural_completeness(merged)
        (
            entry_before_signal_count,
            entry_equal_signal_count,
            exit_not_after_entry_count,
        ) = self._validate_time_order(merged)
        label_formula_mismatch_count = self._validate_label_formula(merged)

        feature_missing_counts = tuple(
            (name, int(merged[name].isna().sum())) for name in feature_names
        )
        feature_missing_rates = tuple(
            (name, float(count / len(merged)))
            for name, count in feature_missing_counts
        )
        self._validate_trainable_feature_coverage(
            merged, feature_names, feature_missing_counts
        )
        constant_features = tuple(
            name
            for name in feature_names
            if int(merged[name].nunique(dropna=True)) <= 1
        )
        suspicious_features = tuple(
            name
            for name in feature_names
            if name.lower().startswith(_SUSPICIOUS_PREFIXES)
        )
        high_missing_features = tuple(
            name
            for name, rate in feature_missing_rates
            if rate >= _HIGH_MISSING_RATE
        )
        label_missing_count = int(merged[self.config.label_column].isna().sum())

        per_date_counts = merged.groupby("trade_date", sort=False).size()
        per_security_counts = merged.groupby("ts_code", sort=False).size()
        warnings = self._warnings(
            factor_only=factor_only,
            return_only=return_only,
            ignored_factor_columns=ignored_factor_columns,
            extra_return_columns=extra_return_columns,
            suspicious_features=suspicious_features,
            constant_features=constant_features,
            high_missing_features=high_missing_features,
            label_missing_count=label_missing_count,
            minimum_cross_section=int(per_date_counts.min()),
            minimum_security_history=int(per_security_counts.min()),
        )

        first_entry, last_entry = self._optional_range(
            merged["entry_trade_date"]
        )
        first_exit, last_exit = self._optional_range(
            merged["exit_trade_date"]
        )
        audit = ModelingPanelAudit(
            schema_version=MODELING_PANEL_SCHEMA_VERSION,
            config=self.config,
            label_column=self.config.label_column,
            factor_input_rows=factor_input_rows,
            return_input_rows=return_input_rows,
            matched_rows=matched_rows,
            output_rows=len(merged),
            factor_only=factor_only,
            return_only=return_only,
            date_count=int(merged["trade_date"].nunique()),
            security_count=int(merged["ts_code"].nunique()),
            first_trade_date=pd.Timestamp(merged["trade_date"].min()),
            last_trade_date=pd.Timestamp(merged["trade_date"].max()),
            first_entry_trade_date=first_entry,
            last_entry_trade_date=last_entry,
            first_exit_trade_date=first_exit,
            last_exit_trade_date=last_exit,
            feature_count=len(feature_names),
            feature_names=feature_names,
            feature_missing_counts=feature_missing_counts,
            feature_missing_rates=feature_missing_rates,
            feature_non_finite_counts=tuple((name, 0) for name in feature_names),
            all_missing_features=(),
            constant_features=constant_features,
            suspicious_feature_names=suspicious_features,
            label_missing_count=label_missing_count,
            label_non_finite_count=0,
            duplicate_factor_key_count=0,
            duplicate_return_key_count=0,
            entry_before_signal_count=entry_before_signal_count,
            entry_equal_signal_count=entry_equal_signal_count,
            exit_not_after_entry_count=exit_not_after_entry_count,
            label_formula_mismatch_count=label_formula_mismatch_count,
            per_date_security_count_min=int(per_date_counts.min()),
            per_date_security_count_median=float(per_date_counts.median()),
            per_date_security_count_max=int(per_date_counts.max()),
            per_security_observation_count_min=int(per_security_counts.min()),
            per_security_observation_count_median=float(
                per_security_counts.median()
            ),
            per_security_observation_count_max=int(per_security_counts.max()),
            warnings=warnings,
        )
        output_columns = (
            *MODELING_PANEL_KEY_COLUMNS,
            *feature_names,
            *MODELING_PANEL_AUDIT_COLUMNS,
            self.config.label_column,
        )
        output = merged.loc[:, list(output_columns)].copy(deep=True)
        output.index = pd.RangeIndex(len(output))
        return ModelingPanelResult(
            output,
            feature_names=feature_names,
            label_column=self.config.label_column,
            audit=audit,
            config=self.config,
        )

    @staticmethod
    def _copy_input(value: object, name: str) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise ModelingPanelDataError(f"{name} must be a pandas DataFrame.")
        if value.empty:
            raise ModelingPanelDataError(f"{name} must not be empty.")
        if isinstance(value.columns, pd.MultiIndex):
            raise ModelingPanelDataError(f"{name} cannot use MultiIndex columns.")
        if not value.columns.is_unique:
            raise ModelingPanelDataError(f"{name} columns must be unique.")
        if any(not isinstance(column, str) for column in value.columns):
            raise ModelingPanelDataError(
                f"{name} columns must contain only string names."
            )
        return value.copy(deep=True)

    def _validate_columns(
        self, factors: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        missing_factor = [
            name for name in MODELING_PANEL_KEY_COLUMNS if name not in factors
        ]
        if missing_factor:
            raise ModelingPanelDataError(
                f"factor_panel is missing required columns: {missing_factor!r}."
            )
        required_return = (
            *MODELING_PANEL_KEY_COLUMNS,
            *MODELING_PANEL_AUDIT_COLUMNS,
            self.config.label_column,
        )
        missing_return = [name for name in required_return if name not in returns]
        if missing_return:
            raise ModelingPanelDataError(
                "forward_returns is missing required columns: "
                f"{missing_return!r}."
            )
        forbidden = {
            *MODELING_PANEL_AUDIT_COLUMNS,
            self.config.label_column,
            "forward_return",
        }
        leaked = [name for name in factors.columns if name in forbidden]
        if leaked:
            raise ModelingPanelLeakageError(
                "factor_panel contains reserved label/audit columns: "
                f"{leaked!r}."
            )

    @staticmethod
    def _normalize_date_column(
        series: pd.Series,
        name: str,
        *,
        allow_missing: bool,
    ) -> pd.Series:
        normalized: list[pd.Timestamp | pd.NaT] = []
        for value in series.tolist():
            missing_scalar = (
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
            if missing_scalar:
                if allow_missing:
                    normalized.append(pd.NaT)
                    continue
                raise ModelingPanelDataError(f"{name} cannot contain missing dates.")
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    raise ModelingPanelDataError(
                        f"{name} contains an invalid ISO date."
                    )
                try:
                    parsed_value: object = datetime.fromisoformat(stripped)
                except ValueError:
                    try:
                        parsed_value = date.fromisoformat(stripped)
                    except ValueError as exc:
                        raise ModelingPanelDataError(
                            f"{name} contains an invalid ISO date."
                        ) from exc
            elif isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
                parsed_value = value
            else:
                raise ModelingPanelDataError(
                    f"{name} must contain dates or ISO date strings."
                )
            try:
                timestamp = pd.Timestamp(parsed_value)
            except (TypeError, ValueError) as exc:
                raise ModelingPanelDataError(
                    f"{name} contains an invalid date."
                ) from exc
            if pd.isna(timestamp):
                if allow_missing:
                    normalized.append(pd.NaT)
                    continue
                raise ModelingPanelDataError(f"{name} cannot contain NaT.")
            if timestamp.tz is not None:
                raise ModelingPanelDataError(f"{name} must be timezone-naive.")
            if timestamp != timestamp.normalize():
                raise ModelingPanelDataError(
                    f"{name} must contain day-granularity dates with no time part."
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
            raise ModelingPanelDataError(
                f"{name} contains a date outside datetime64[ns] range."
            ) from exc

    @staticmethod
    def _normalize_codes(series: pd.Series, name: str) -> pd.Series:
        normalized: list[str] = []
        for value in series.tolist():
            if not isinstance(value, (str, np.str_)):
                raise ModelingPanelDataError(
                    f"{name} must contain only non-empty string values."
                )
            code = str(value).strip()
            if not code:
                raise ModelingPanelDataError(
                    f"{name} must contain only non-empty string values."
                )
            normalized.append(code)
        return pd.Series(
            normalized, index=series.index, name=series.name, dtype="string"
        )

    @classmethod
    def _validate_unique_keys(cls, frame: pd.DataFrame, name: str) -> None:
        duplicate_mask = frame.duplicated(
            list(MODELING_PANEL_KEY_COLUMNS), keep=False
        )
        duplicate_rows = int(duplicate_mask.sum())
        if not duplicate_rows:
            return
        duplicate_frame = frame.loc[
            duplicate_mask, list(MODELING_PANEL_KEY_COLUMNS)
        ]
        group_sizes = duplicate_frame.groupby(
            list(MODELING_PANEL_KEY_COLUMNS), sort=False
        ).size()
        duplicate_keys = int((group_sizes > 1).sum())
        samples = cls._sample_keys(cls._key_tuples(duplicate_frame))
        raise ModelingPanelAlignmentError(
            f"{name} contains duplicate keys: duplicate_rows={duplicate_rows}, "
            f"duplicate_unique_keys={duplicate_keys}, sampled_keys={samples!r}."
        )

    def _resolve_features(
        self, factors: pd.DataFrame
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        candidates = tuple(
            name for name in factors.columns if name not in MODELING_PANEL_KEY_COLUMNS
        )
        if self.config.include_features is not None:
            missing = [
                name
                for name in self.config.include_features
                if name not in candidates
            ]
            if missing:
                raise ModelingPanelDataError(
                    f"include_features are absent from factor_panel: {missing!r}."
                )
            feature_names = self.config.include_features
        else:
            missing_excluded = [
                name for name in self.config.exclude_features if name not in candidates
            ]
            if missing_excluded:
                raise ModelingPanelDataError(
                    "exclude_features are absent from factor_panel candidates: "
                    f"{missing_excluded!r}."
                )
            excluded = set(self.config.exclude_features)
            feature_names = tuple(name for name in candidates if name not in excluded)
        if not feature_names:
            raise ModelingPanelDataError(
                "Feature resolution produced no Modeling Panel features."
            )
        ignored = tuple(name for name in candidates if name not in set(feature_names))
        return feature_names, ignored

    def _extra_return_columns(self, returns: pd.DataFrame) -> tuple[str, ...]:
        required = {
            *MODELING_PANEL_KEY_COLUMNS,
            *MODELING_PANEL_AUDIT_COLUMNS,
            self.config.label_column,
        }
        return tuple(name for name in returns.columns if name not in required)

    @staticmethod
    def _validate_non_key_collisions(
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        feature_names: tuple[str, ...],
        extra_return_columns: tuple[str, ...],
    ) -> None:
        factor_non_keys = set(factors.columns) - set(MODELING_PANEL_KEY_COLUMNS)
        collisions = [
            name for name in extra_return_columns if name in factor_non_keys
        ]
        if collisions:
            selected = [name for name in collisions if name in feature_names]
            detail = selected if selected else collisions
            raise ModelingPanelDataError(
                "factor_panel and forward_returns contain conflicting non-key "
                f"columns: {detail!r}."
            )

    def _validate_numeric_inputs(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        feature_names: tuple[str, ...],
    ) -> None:
        for name in feature_names:
            self._validate_numeric_series(
                factors[name], f"factor_panel.{name}", allow_missing=True
            )
        self._validate_numeric_series(
            returns[self.config.label_column],
            f"forward_returns.{self.config.label_column}",
            allow_missing=self.config.allow_missing_labels,
        )
        for name in ("entry_price", "exit_price"):
            self._validate_numeric_series(
                returns[name], f"forward_returns.{name}", allow_missing=True
            )
            nonmissing = returns[name].notna()
            if bool((returns.loc[nonmissing, name] <= 0).any()):
                raise ModelingPanelDataError(
                    f"forward_returns.{name} must contain strictly positive "
                    "non-missing prices."
                )

    @staticmethod
    def _validate_numeric_series(
        series: pd.Series, name: str, *, allow_missing: bool
    ) -> None:
        if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
            raise ModelingPanelDataError(
                f"{name} must use a non-boolean numeric dtype."
            )
        values = series.to_numpy(dtype=float, na_value=np.nan)
        infinite_count = int(np.isinf(values).sum())
        if infinite_count:
            raise ModelingPanelDataError(
                f"{name} contains non-finite values: infinite_count={infinite_count}."
            )
        missing_count = int(series.isna().sum())
        if missing_count and not allow_missing:
            raise ModelingPanelDataError(
                f"{name} contains missing values: missing_count={missing_count}."
            )

    @staticmethod
    def _key_tuples(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
        return [
            (pd.Timestamp(trade_date), str(ts_code))
            for trade_date, ts_code in frame.loc[
                :, list(MODELING_PANEL_KEY_COLUMNS)
            ].itertuples(index=False, name=None)
        ]

    @staticmethod
    def _sample_keys(
        keys: Iterable[tuple[pd.Timestamp, str]],
    ) -> tuple[tuple[pd.Timestamp, str], ...]:
        ordered = sorted(keys, key=lambda item: (item[0], item[1]))
        samples: list[tuple[pd.Timestamp, str]] = []
        seen: set[tuple[pd.Timestamp, str]] = set()
        for key in ordered:
            if key in seen:
                continue
            seen.add(key)
            samples.append(key)
            if len(samples) == _SAMPLE_LIMIT:
                break
        return tuple(samples)

    @classmethod
    def _unmatched_audit(
        cls, keys: list[tuple[pd.Timestamp, str]]
    ) -> ModelingPanelUnmatchedAudit:
        if not keys:
            return ModelingPanelUnmatchedAudit(0, 0, None, None)
        dates = [key[0] for key in keys]
        return ModelingPanelUnmatchedAudit(
            row_count=len(keys),
            date_count=len(set(dates)),
            first_trade_date=min(dates),
            last_trade_date=max(dates),
            sampled_keys=cls._sample_keys(keys),
        )

    def _merge(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        feature_names: tuple[str, ...],
    ) -> pd.DataFrame:
        factor_columns = [*MODELING_PANEL_KEY_COLUMNS, *feature_names]
        return_columns = [
            *MODELING_PANEL_KEY_COLUMNS,
            *MODELING_PANEL_AUDIT_COLUMNS,
            self.config.label_column,
        ]
        try:
            merged = factors.loc[:, factor_columns].merge(
                returns.loc[:, return_columns],
                on=list(MODELING_PANEL_KEY_COLUMNS),
                how="inner",
                validate="one_to_one",
                sort=False,
                indicator=True,
            )
        except MergeError as exc:
            raise ModelingPanelAlignmentError(
                "pandas rejected the requested one-to-one Modeling Panel merge."
            ) from exc
        if not bool(merged["_merge"].eq("both").all()):
            raise ModelingPanelIntegrityError(
                "inner merge produced a non-'both' indicator state."
            )
        columns_without_indicator = [
            column for column in merged.columns if column != "_merge"
        ]
        return merged.loc[:, columns_without_indicator].copy(deep=True)

    def _validate_structural_completeness(self, frame: pd.DataFrame) -> None:
        rules = (
            (
                frame["exit_trade_date"].notna()
                & frame["entry_trade_date"].isna(),
                "exit_trade_date requires entry_trade_date",
            ),
            (
                frame["entry_price"].notna()
                & frame["entry_trade_date"].isna(),
                "entry_price requires entry_trade_date",
            ),
            (
                frame["exit_price"].notna()
                & frame["exit_trade_date"].isna(),
                "exit_price requires exit_trade_date",
            ),
        )
        for mask, message in rules:
            count = int(mask.sum())
            if count:
                raise ModelingPanelDataError(
                    f"{message}: violation_count={count}."
                )
        label_present = frame[self.config.label_column].notna()
        for column in (
            "entry_trade_date",
            "exit_trade_date",
            "entry_price",
            "exit_price",
        ):
            missing = label_present & frame[column].isna()
            count = int(missing.sum())
            if count:
                raise ModelingPanelDataError(
                    f"non-missing {self.config.label_column} requires {column}: "
                    f"violation_count={count}."
                )

    def _validate_time_order(
        self, frame: pd.DataFrame
    ) -> tuple[int, int, int]:
        entry_present = frame["entry_trade_date"].notna()
        exit_present = frame["exit_trade_date"].notna()
        entry_before = entry_present & (
            frame["entry_trade_date"] < frame["trade_date"]
        )
        entry_equal = entry_present & (
            frame["entry_trade_date"] == frame["trade_date"]
        )
        exit_not_after = entry_present & exit_present & (
            frame["exit_trade_date"] <= frame["entry_trade_date"]
        )
        entry_before_count = int(entry_before.sum())
        entry_equal_count = int(entry_equal.sum())
        exit_not_after_count = int(exit_not_after.sum())
        if entry_before_count:
            raise ModelingPanelLeakageError(
                "entry_trade_date precedes trade_date: "
                f"violation_count={entry_before_count}, "
                f"sampled_keys={self._violation_samples(frame, entry_before)!r}."
            )
        if self.config.require_entry_after_signal and entry_equal_count:
            raise ModelingPanelLeakageError(
                "entry_trade_date equals trade_date under strict entry ordering: "
                f"violation_count={entry_equal_count}, "
                f"sampled_keys={self._violation_samples(frame, entry_equal)!r}."
            )
        if exit_not_after_count:
            raise ModelingPanelLeakageError(
                "exit_trade_date must be later than entry_trade_date: "
                f"violation_count={exit_not_after_count}, "
                f"sampled_keys={self._violation_samples(frame, exit_not_after)!r}."
            )
        return entry_before_count, entry_equal_count, exit_not_after_count

    def _validate_label_formula(self, frame: pd.DataFrame) -> int:
        complete = (
            frame[self.config.label_column].notna()
            & frame["entry_price"].notna()
            & frame["exit_price"].notna()
        )
        mismatch = pd.Series(False, index=frame.index)
        if bool(complete.any()):
            expected = (
                frame.loc[complete, "exit_price"]
                / frame.loc[complete, "entry_price"]
                - 1.0
            )
            actual = frame.loc[complete, self.config.label_column]
            close = np.isclose(
                actual.to_numpy(dtype=float),
                expected.to_numpy(dtype=float),
                rtol=_FORMULA_RTOL,
                atol=_FORMULA_ATOL,
                equal_nan=False,
            )
            mismatch.loc[complete] = ~close
        mismatch_count = int(mismatch.sum())
        if mismatch_count:
            raise ModelingPanelLeakageError(
                "label values do not match exit_price / entry_price - 1: "
                f"label_formula_mismatch_count={mismatch_count}, "
                f"sampled_keys={self._violation_samples(frame, mismatch)!r}."
            )
        return mismatch_count

    def _validate_trainable_feature_coverage(
        self,
        frame: pd.DataFrame,
        feature_names: tuple[str, ...],
        missing_counts: tuple[tuple[str, int], ...],
    ) -> None:
        all_missing = [
            name for name, count in missing_counts if count == len(frame)
        ]
        if all_missing:
            raise ModelingPanelDataError(
                "Features are entirely missing in the output panel: "
                f"{all_missing!r}."
            )
        labeled = frame[self.config.label_column].notna()
        if not bool(labeled.any()):
            return
        untrainable = [
            name for name in feature_names if bool(frame.loc[labeled, name].isna().all())
        ]
        if untrainable:
            raise ModelingPanelDataError(
                "Features are entirely missing on non-missing-label rows: "
                f"{untrainable!r}."
            )

    @classmethod
    def _violation_samples(
        cls, frame: pd.DataFrame, mask: pd.Series
    ) -> tuple[tuple[pd.Timestamp, str], ...]:
        return cls._sample_keys(cls._key_tuples(frame.loc[mask]))

    @staticmethod
    def _optional_range(
        series: pd.Series,
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        present = series[series.notna()]
        if present.empty:
            return None, None
        return pd.Timestamp(present.min()), pd.Timestamp(present.max())

    @staticmethod
    def _warnings(
        *,
        factor_only: ModelingPanelUnmatchedAudit,
        return_only: ModelingPanelUnmatchedAudit,
        ignored_factor_columns: tuple[str, ...],
        extra_return_columns: tuple[str, ...],
        suspicious_features: tuple[str, ...],
        constant_features: tuple[str, ...],
        high_missing_features: tuple[str, ...],
        label_missing_count: int,
        minimum_cross_section: int,
        minimum_security_history: int,
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        if factor_only.row_count or return_only.row_count:
            warnings.append(
                "Unmatched keys were dropped: "
                f"factor_only={factor_only.row_count}, "
                f"return_only={return_only.row_count}."
            )
        if ignored_factor_columns:
            warnings.append(
                f"Ignored factor columns: {list(ignored_factor_columns)!r}."
            )
        if extra_return_columns:
            warnings.append(
                "Ignored extra forward-return columns: "
                f"{list(extra_return_columns)!r}."
            )
        if suspicious_features:
            warnings.append(
                "Suspicious feature names require provenance review: "
                f"{list(suspicious_features)!r}."
            )
        if constant_features:
            warnings.append(
                f"Constant features were retained: {list(constant_features)!r}."
            )
        if high_missing_features:
            warnings.append(
                "Features with missing rate >= 0.50 were retained: "
                f"{list(high_missing_features)!r}."
            )
        if label_missing_count:
            warnings.append(
                f"Missing labels were retained: count={label_missing_count}."
            )
        if minimum_cross_section < 2:
            warnings.append(
                "At least one trade_date has fewer than 2 securities: "
                f"minimum={minimum_cross_section}."
            )
        if minimum_security_history < 2:
            warnings.append(
                "At least one security has fewer than 2 observations: "
                f"minimum={minimum_security_history}."
            )
        return tuple(warnings)
