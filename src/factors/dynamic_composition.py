"""Rolling IC and RankIC weighting for already processed factor values.

This module consumes evaluation results created by V2-E1 and factor values
that have completed V2-D1 direction unification and cross-sectional
standardization. Optional V2-D2 neutralization must also happen before
composition. For a score date ``t``, weights use only evaluation rows whose
``trade_date`` is strictly earlier than ``t``; current and future evaluation
information is never eligible.

F2 does not repeat direction adjustment, filling, winsorization,
standardization, or neutralization. The ``absolute`` policy measures historical
correlation strength and does not flip factor values. A higher composite score
means a more preferred combined direction, not an expected return. This module
does not construct forward returns, select stocks, rebalance, model costs, or
run a backtest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from src.factors.registry import FactorRegistry


WEIGHT_HISTORY_COLUMNS = [
    "trade_date",
    "factor_name",
    "metric",
    "history_periods",
    "history_start_date",
    "history_end_date",
    "historical_mean",
    "raw_weight",
    "normalized_weight",
    "used_fallback",
]


@dataclass(frozen=True)
class RollingICWeightConfig:
    """Configure rolling evaluation history, weight conversion, and scoring."""

    metric: str = "rank_ic"
    lookback_periods: int = 12
    min_periods: int = 6
    negative_policy: str = "zero"
    fallback_method: str = "equal"
    missing_policy: str = "renormalize"
    min_valid_factors: int = 1
    score_col: str = "composite_score"

    def __post_init__(self) -> None:
        """Validate methods, positive integer limits, and output naming."""
        if self.metric not in {"ic", "rank_ic"}:
            raise ValueError("metric must be either 'ic' or 'rank_ic'.")
        lookback = self._positive_integer("lookback_periods", self.lookback_periods)
        minimum = self._positive_integer("min_periods", self.min_periods)
        if minimum > lookback:
            raise ValueError("min_periods cannot exceed lookback_periods.")
        if self.negative_policy not in {"zero", "absolute"}:
            raise ValueError("negative_policy must be either 'zero' or 'absolute'.")
        if self.fallback_method not in {"equal", "none"}:
            raise ValueError("fallback_method must be either 'equal' or 'none'.")
        if self.missing_policy not in {"require_all", "renormalize"}:
            raise ValueError(
                "missing_policy must be either 'require_all' or 'renormalize'."
            )
        min_valid = self._positive_integer(
            "min_valid_factors", self.min_valid_factors
        )
        if not isinstance(self.score_col, str) or not self.score_col.strip():
            raise ValueError("score_col must be a non-empty string.")

        object.__setattr__(self, "lookback_periods", lookback)
        object.__setattr__(self, "min_periods", minimum)
        object.__setattr__(self, "min_valid_factors", min_valid)
        object.__setattr__(self, "score_col", self.score_col.strip())

    @staticmethod
    def _positive_integer(field_name: str, value: object) -> int:
        """Return a positive integer while rejecting booleans."""
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{field_name} must be an integer >= 1.")
        normalized = int(value)
        if normalized < 1:
            raise ValueError(f"{field_name} must be an integer >= 1.")
        return normalized

    def to_dict(self) -> Dict[str, object]:
        """Return a serialization-friendly configuration dictionary."""
        return asdict(self)


class RollingICFactorComposer:
    """Build strictly historical dynamic weights and row-level factor scores."""

    def __init__(
        self,
        registry: FactorRegistry,
        config: RollingICWeightConfig | None = None,
    ) -> None:
        if not isinstance(registry, FactorRegistry):
            raise TypeError("registry must be a FactorRegistry.")
        if config is not None and not isinstance(config, RollingICWeightConfig):
            raise TypeError("config must be a RollingICWeightConfig or None.")
        self.registry = registry
        self.config = config if config is not None else RollingICWeightConfig()

    def describe_config(self) -> Dict[str, object]:
        """Return a serialization-friendly summary of the active configuration."""
        return self.config.to_dict()

    def build_weight_history(
        self,
        score_dates: Sequence,
        ic_results: pd.DataFrame,
        factor_names: Sequence[str],
    ) -> pd.DataFrame:
        """Return one auditable dynamic-weight row per date and factor."""
        names = self._validate_factor_names(factor_names)
        dates = self._normalize_score_dates(score_dates)
        evaluations = self._normalize_ic_results(ic_results)
        if not dates:
            return self._empty_weight_history()

        rows = []
        for score_date in dates:
            date_rows = []
            for name in sorted(names):
                history = evaluations.loc[
                    (evaluations["factor_name"] == name)
                    & (evaluations["trade_date"] < score_date)
                    & evaluations[self.config.metric].notna(),
                    ["trade_date", self.config.metric],
                ].sort_values("trade_date", kind="mergesort")
                history = history.tail(self.config.lookback_periods)
                periods = int(len(history))
                historical_mean = (
                    float(history[self.config.metric].mean())
                    if periods > 0
                    else np.nan
                )
                if periods >= self.config.min_periods:
                    if self.config.negative_policy == "zero":
                        raw_weight = max(historical_mean, 0.0)
                    else:
                        raw_weight = abs(historical_mean)
                else:
                    raw_weight = np.nan
                date_rows.append(
                    {
                        "trade_date": score_date,
                        "factor_name": name,
                        "metric": self.config.metric,
                        "history_periods": periods,
                        "history_start_date": (
                            history["trade_date"].iloc[0] if periods else pd.NaT
                        ),
                        "history_end_date": (
                            history["trade_date"].iloc[-1] if periods else pd.NaT
                        ),
                        "historical_mean": historical_mean,
                        "raw_weight": raw_weight,
                        "normalized_weight": np.nan,
                        "used_fallback": False,
                    }
                )

            positive_total = float(
                sum(
                    row["raw_weight"]
                    for row in date_rows
                    if np.isfinite(row["raw_weight"]) and row["raw_weight"] > 0.0
                )
            )
            if positive_total > 0.0:
                for row in date_rows:
                    raw_weight = row["raw_weight"]
                    row["normalized_weight"] = (
                        float(raw_weight / positive_total)
                        if np.isfinite(raw_weight) and raw_weight >= 0.0
                        else 0.0
                    )
            elif self.config.fallback_method == "equal":
                equal_weight = 1.0 / len(date_rows)
                for row in date_rows:
                    row["normalized_weight"] = equal_weight
                    row["used_fallback"] = True
            rows.extend(date_rows)

        result = pd.DataFrame(rows, columns=WEIGHT_HISTORY_COLUMNS)
        result["history_periods"] = result["history_periods"].astype("int64")
        result["used_fallback"] = result["used_fallback"].astype(bool)
        result["normalized_weight"] = result["normalized_weight"].clip(0.0, 1.0)
        return result.replace([np.inf, -np.inf], np.nan).sort_values(
            ["trade_date", "factor_name"], kind="mergesort", ignore_index=True
        )

    def compose(
        self,
        factor_panel: pd.DataFrame,
        ic_results: pd.DataFrame,
        factor_names: Sequence[str],
    ) -> pd.DataFrame:
        """Apply date-specific historical weights to each factor-panel row."""
        names = self._validate_factor_names(factor_names)
        panel = self._normalize_factor_panel(factor_panel, names)
        # Validate IC input even when the factor panel is empty.
        evaluations = self._normalize_ic_results(ic_results)
        output_columns = [
            "trade_date",
            "ts_code",
            self.config.score_col,
            "valid_factor_count",
            "weight_coverage",
        ]
        if panel.empty:
            result = panel.loc[:, ["trade_date", "ts_code"]].copy()
            result[self.config.score_col] = pd.Series(dtype=float)
            result["valid_factor_count"] = pd.Series(dtype="int64")
            result["weight_coverage"] = pd.Series(dtype=float)
            return result.loc[:, output_columns]

        weights = self.build_weight_history(
            panel["trade_date"].drop_duplicates().tolist(), evaluations, names
        )
        weight_lookup = {
            trade_date: group.set_index("factor_name")["normalized_weight"].to_dict()
            for trade_date, group in weights.groupby("trade_date", sort=False)
        }

        score_values = np.full(len(panel), np.nan, dtype=float)
        valid_counts = np.zeros(len(panel), dtype=np.int64)
        coverage_values = np.full(len(panel), np.nan, dtype=float)
        ordered_names = sorted(names)

        for trade_date, row_indexes in panel.groupby("trade_date", sort=True).groups.items():
            row_positions = panel.index.get_indexer(row_indexes)
            date_weights = weight_lookup[trade_date]
            base_weights = np.array(
                [date_weights[name] for name in ordered_names], dtype=float
            )
            has_weight_system = np.isfinite(base_weights).any()
            positive_weights = np.isfinite(base_weights) & (base_weights > 0.0)
            values = panel.loc[row_indexes, ordered_names].to_numpy(dtype=float)
            valid = np.isfinite(values) & positive_weights
            valid_count = valid.sum(axis=1).astype(np.int64)
            valid_counts[row_positions] = valid_count

            if not has_weight_system:
                continue
            coverage = (valid * np.where(positive_weights, base_weights, 0.0)).sum(
                axis=1
            )
            coverage = np.clip(coverage, 0.0, 1.0)
            coverage_values[row_positions] = coverage
            weighted_sum = (
                np.where(valid, values, 0.0)
                * np.where(positive_weights, base_weights, 0.0)
            ).sum(axis=1)

            if self.config.missing_policy == "require_all":
                required_count = int(positive_weights.sum())
                eligible = (required_count > 0) & (valid_count == required_count)
                scores = np.where(eligible, weighted_sum, np.nan)
            else:
                eligible = (
                    (valid_count >= self.config.min_valid_factors)
                    & (coverage > 0.0)
                )
                scores = np.full(len(row_indexes), np.nan, dtype=float)
                scores[eligible] = weighted_sum[eligible] / coverage[eligible]
            score_values[row_positions] = scores

        score_values[~np.isfinite(score_values)] = np.nan
        result = panel.loc[:, ["trade_date", "ts_code"]].copy()
        result[self.config.score_col] = score_values
        result["valid_factor_count"] = valid_counts
        result["weight_coverage"] = coverage_values
        return result.loc[:, output_columns].sort_values(
            ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
        )

    def _validate_factor_names(self, factor_names: Sequence[str]) -> list[str]:
        """Validate names and confirm that every requested factor is registered."""
        if isinstance(factor_names, (str, bytes)):
            raise TypeError("factor_names must be a sequence of factor names.")
        try:
            names = list(factor_names)
        except TypeError as exc:
            raise TypeError("factor_names must be a sequence of factor names.") from exc
        if not names:
            raise ValueError("factor_names must contain at least one factor name.")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("factor_names cannot contain empty values.")
        names = [name.strip() for name in names]
        if len(set(names)) != len(names):
            raise ValueError("factor_names cannot contain duplicate names.")
        for name in names:
            try:
                self.registry.get(name)
            except KeyError as exc:
                raise ValueError(f"Factor '{name}' is not registered.") from exc
        return names

    @staticmethod
    def _normalize_score_dates(score_dates: Sequence) -> list[pd.Timestamp]:
        """Return unique, sorted, valid score dates."""
        if isinstance(score_dates, (str, bytes)):
            raise TypeError("score_dates must be a sequence of dates.")
        try:
            values = list(score_dates)
        except TypeError as exc:
            raise TypeError("score_dates must be a sequence of dates.") from exc
        if not values:
            return []
        normalized = pd.to_datetime(pd.Series(values), errors="coerce")
        if normalized.isna().any():
            raise ValueError("score_dates must contain valid, non-empty dates.")
        return sorted(pd.Timestamp(value) for value in normalized.unique())

    def _normalize_ic_results(self, ic_results: pd.DataFrame) -> pd.DataFrame:
        """Copy and validate the selected E1 evaluation metric."""
        if not isinstance(ic_results, pd.DataFrame):
            raise TypeError("ic_results must be a pandas DataFrame.")
        required = ["trade_date", "factor_name", self.config.metric]
        missing = [column for column in required if column not in ic_results.columns]
        if missing:
            raise ValueError(
                "ic_results is missing required columns: " + ", ".join(missing) + "."
            )
        normalized = ic_results.loc[:, required].copy(deep=True)
        dates = pd.to_datetime(normalized["trade_date"], errors="coerce")
        if dates.isna().any():
            raise ValueError(
                "ic_results trade_date must contain valid, non-empty dates."
            )
        normalized["trade_date"] = dates
        factor_names = normalized["factor_name"].astype("string").str.strip()
        if factor_names.isna().any() or factor_names.eq("").any():
            raise ValueError("ic_results factor_name cannot contain empty values.")
        normalized["factor_name"] = factor_names
        if normalized.duplicated(["trade_date", "factor_name"]).any():
            raise ValueError(
                "ic_results trade_date and factor_name combinations must be unique."
            )
        normalized[self.config.metric] = pd.to_numeric(
            normalized[self.config.metric], errors="coerce"
        ).astype(float).replace([np.inf, -np.inf], np.nan)
        return normalized.sort_values(
            ["trade_date", "factor_name"], kind="mergesort", ignore_index=True
        )

    @staticmethod
    def _normalize_factor_panel(
        factor_panel: pd.DataFrame, factor_names: Sequence[str]
    ) -> pd.DataFrame:
        """Copy and validate factor values and their date-stock keys."""
        if not isinstance(factor_panel, pd.DataFrame):
            raise TypeError("factor_panel must be a pandas DataFrame.")
        required = ["trade_date", "ts_code"] + list(factor_names)
        missing = [column for column in required if column not in factor_panel.columns]
        if missing:
            raise ValueError(
                "factor_panel is missing required columns: "
                + ", ".join(missing)
                + "."
            )
        normalized = factor_panel.loc[:, required].copy(deep=True)
        dates = pd.to_datetime(normalized["trade_date"], errors="coerce")
        if dates.isna().any():
            raise ValueError(
                "factor_panel trade_date must contain valid, non-empty dates."
            )
        normalized["trade_date"] = dates
        codes = normalized["ts_code"].astype("string").str.strip()
        if codes.isna().any() or codes.eq("").any():
            raise ValueError("factor_panel ts_code cannot contain empty values.")
        normalized["ts_code"] = codes
        if normalized.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(
                "factor_panel trade_date and ts_code combinations must be unique."
            )
        for name in factor_names:
            normalized[name] = pd.to_numeric(
                normalized[name], errors="coerce"
            ).astype(float).replace([np.inf, -np.inf], np.nan)
        return normalized.sort_values(
            ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
        )

    @staticmethod
    def _empty_weight_history() -> pd.DataFrame:
        """Return an empty weight history with stable output dtypes."""
        result = pd.DataFrame(columns=WEIGHT_HISTORY_COLUMNS)
        result["trade_date"] = pd.Series(dtype="datetime64[ns]")
        result["history_periods"] = pd.Series(dtype="int64")
        result["history_start_date"] = pd.Series(dtype="datetime64[ns]")
        result["history_end_date"] = pd.Series(dtype="datetime64[ns]")
        result["historical_mean"] = pd.Series(dtype=float)
        result["raw_weight"] = pd.Series(dtype=float)
        result["normalized_weight"] = pd.Series(dtype=float)
        result["used_fallback"] = pd.Series(dtype=bool)
        return result.loc[:, WEIGHT_HISTORY_COLUMNS]
