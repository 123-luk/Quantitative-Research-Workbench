"""Same-date quantile and long-short evaluation using prepared forward returns.

This module evaluates cross-sectional discrimination; it is not a realizable
portfolio backtest. It does not construct forward returns, rebalance holdings,
compound capital, or model costs, slippage, suspensions, or price limits.
Long-short return is only the high-quantile mean forward return minus the
low-quantile mean forward return. Factor direction is assumed to be unified by
V2-D1 and is never adjusted again here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Dict, Sequence

import numpy as np
import pandas as pd


QUANTILE_RESULT_COLUMNS = [
    "trade_date",
    "factor_name",
    "quantile",
    "universe_count",
    "n_obs",
    "coverage",
    "group_count",
    "group_return",
]

LONG_SHORT_RESULT_COLUMNS = [
    "trade_date",
    "factor_name",
    "low_group_return",
    "high_group_return",
    "long_short_return",
    "nonempty_quantiles",
    "monotonicity",
    "coverage",
    "n_obs",
]

QUANTILE_SUMMARY_COLUMNS = [
    "factor_name",
    "quantile",
    "total_periods",
    "valid_periods",
    "mean_group_return",
    "std_group_return",
    "positive_return_ratio",
    "mean_group_count",
]

LONG_SHORT_SUMMARY_COLUMNS = [
    "factor_name",
    "total_periods",
    "valid_long_short_periods",
    "mean_long_short_return",
    "std_long_short_return",
    "long_short_ir",
    "positive_long_short_ratio",
    "valid_monotonicity_periods",
    "mean_monotonicity",
    "positive_monotonicity_ratio",
    "mean_coverage",
    "mean_n_obs",
]


@dataclass(frozen=True)
class QuantileEvaluationConfig:
    """Configure rank-based same-date quantile return evaluation."""

    return_col: str = "forward_return"
    quantiles: int = 5
    min_cross_section_size: int = 20
    min_group_size: int = 1
    compute_monotonicity: bool = True

    def __post_init__(self) -> None:
        """Validate column naming, integer limits, and monotonicity switch."""
        if not isinstance(self.return_col, str) or not self.return_col.strip():
            raise ValueError("return_col must be a non-empty string.")
        quantiles = self._valid_integer("quantiles", self.quantiles, minimum=2)
        min_cross_section_size = self._valid_integer(
            "min_cross_section_size", self.min_cross_section_size, minimum=2
        )
        min_group_size = self._valid_integer(
            "min_group_size", self.min_group_size, minimum=1
        )
        if min_cross_section_size < quantiles:
            raise ValueError("min_cross_section_size must be >= quantiles.")
        if not isinstance(self.compute_monotonicity, bool):
            raise ValueError("compute_monotonicity must be a bool.")
        object.__setattr__(self, "return_col", self.return_col.strip())
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(
            self, "min_cross_section_size", min_cross_section_size
        )
        object.__setattr__(self, "min_group_size", min_group_size)

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


class FactorQuantileEvaluator:
    """Evaluate rank quantiles, long-short spreads, and monotonicity by date."""

    _RETURN_INTERNAL = "__quantile_forward_return__"

    def __init__(self, config: QuantileEvaluationConfig | None = None) -> None:
        if config is not None and not isinstance(config, QuantileEvaluationConfig):
            raise TypeError("config must be a QuantileEvaluationConfig or None.")
        self.config = config if config is not None else QuantileEvaluationConfig()

    def describe_config(self) -> Dict[str, object]:
        """Return a serialization-friendly summary of the active configuration."""
        return self.config.to_dict()

    def evaluate_quantiles(
        self,
        factor_panel: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_names: Sequence[str],
    ) -> pd.DataFrame:
        """Return complete same-date quantile rows for every requested factor."""
        names = self._validate_factor_names(factor_names)
        factors = self._normalize_factor_panel(factor_panel, names)
        returns = self._normalize_forward_returns(forward_returns).rename(
            columns={self.config.return_col: self._RETURN_INTERNAL}
        )
        merged = factors.merge(
            returns,
            on=["trade_date", "ts_code"],
            how="left",
            sort=False,
            validate="one_to_one",
        )

        rows = []
        for trade_date, date_frame in merged.groupby("trade_date", sort=True):
            universe_count = int(len(date_frame))
            forward_return = date_frame[self._RETURN_INTERNAL]
            for name in names:
                factor_values = date_frame[name]
                valid = (
                    factor_values.notna()
                    & np.isfinite(factor_values)
                    & forward_return.notna()
                    & np.isfinite(forward_return)
                )
                n_obs = int(valid.sum())
                coverage = (
                    float(n_obs / universe_count) if universe_count > 0 else np.nan
                )
                assignments = pd.Series(pd.NA, index=date_frame.index, dtype="Int64")
                if (
                    n_obs >= self.config.min_cross_section_size
                    and factor_values.loc[valid].nunique() >= 2
                ):
                    ranks = factor_values.loc[valid].rank(method="average")
                    positions = (ranks - 1.0) / (n_obs - 1.0)
                    groups = np.floor(positions * self.config.quantiles) + 1.0
                    assignments.loc[valid] = groups.clip(
                        lower=1, upper=self.config.quantiles
                    ).astype(int)

                for quantile in range(1, self.config.quantiles + 1):
                    group_mask = assignments.eq(quantile).fillna(False)
                    group_count = int(group_mask.sum())
                    group_return = np.nan
                    if group_count >= self.config.min_group_size:
                        group_return = float(forward_return.loc[group_mask].mean())
                        if not np.isfinite(group_return):
                            group_return = np.nan
                    rows.append(
                        {
                            "trade_date": trade_date,
                            "factor_name": name,
                            "quantile": quantile,
                            "universe_count": universe_count,
                            "n_obs": n_obs,
                            "coverage": coverage,
                            "group_count": group_count,
                            "group_return": group_return,
                        }
                    )

        result = pd.DataFrame(rows, columns=QUANTILE_RESULT_COLUMNS)
        if result.empty:
            return result
        result = result.replace([np.inf, -np.inf], np.nan)
        result["coverage"] = result["coverage"].clip(lower=0.0, upper=1.0)
        return result.sort_values(
            ["trade_date", "factor_name", "quantile"],
            kind="mergesort",
            ignore_index=True,
        )

    def evaluate_long_short(self, quantile_results: pd.DataFrame) -> pd.DataFrame:
        """Return high-minus-low spreads and optional group-return monotonicity."""
        normalized = self._normalize_quantile_results(quantile_results)
        rows = []
        for (trade_date, factor_name), group in normalized.groupby(
            ["trade_date", "factor_name"], sort=True
        ):
            returns = group.set_index("quantile")["group_return"]
            low = self._finite_value(returns.get(1, np.nan))
            high = self._finite_value(
                returns.get(self.config.quantiles, np.nan)
            )
            long_short = (
                float(high - low)
                if np.isfinite(low) and np.isfinite(high)
                else np.nan
            )
            finite_groups = group[np.isfinite(group["group_return"])].copy()
            monotonicity = np.nan
            if self.config.compute_monotonicity and len(finite_groups) >= 2:
                quantile_rank = finite_groups["quantile"].rank(method="average")
                return_rank = finite_groups["group_return"].rank(method="average")
                monotonicity = self._finite_correlation(quantile_rank, return_rank)
            rows.append(
                {
                    "trade_date": trade_date,
                    "factor_name": factor_name,
                    "low_group_return": low,
                    "high_group_return": high,
                    "long_short_return": long_short,
                    "nonempty_quantiles": int(len(finite_groups)),
                    "monotonicity": monotonicity,
                    "coverage": self._finite_value(group["coverage"].iloc[0]),
                    "n_obs": self._finite_value(group["n_obs"].iloc[0]),
                }
            )
        result = pd.DataFrame(rows, columns=LONG_SHORT_RESULT_COLUMNS)
        return result.replace([np.inf, -np.inf], np.nan).sort_values(
            ["trade_date", "factor_name"], kind="mergesort", ignore_index=True
        )

    def summarize_quantiles(self, quantile_results: pd.DataFrame) -> pd.DataFrame:
        """Summarize non-annualized group returns by factor and quantile."""
        normalized = self._normalize_quantile_results(quantile_results)
        rows = []
        for (factor_name, quantile), group in normalized.groupby(
            ["factor_name", "quantile"], sort=True
        ):
            stats = self._return_summary(group["group_return"])
            rows.append(
                {
                    "factor_name": factor_name,
                    "quantile": int(quantile),
                    "total_periods": int(group["trade_date"].nunique()),
                    "valid_periods": stats["valid_periods"],
                    "mean_group_return": stats["mean"],
                    "std_group_return": stats["std"],
                    "positive_return_ratio": stats["positive_ratio"],
                    "mean_group_count": float(group["group_count"].mean()),
                }
            )
        result = pd.DataFrame(rows, columns=QUANTILE_SUMMARY_COLUMNS)
        return result.replace([np.inf, -np.inf], np.nan).sort_values(
            ["factor_name", "quantile"], kind="mergesort", ignore_index=True
        )

    def summarize_long_short(
        self, long_short_results: pd.DataFrame
    ) -> pd.DataFrame:
        """Summarize non-annualized spread and monotonicity statistics."""
        normalized = self._normalize_long_short_results(long_short_results)
        rows = []
        for factor_name, group in normalized.groupby("factor_name", sort=True):
            spread = self._return_summary(group["long_short_return"])
            monotonicity = self._return_summary(group["monotonicity"])
            rows.append(
                {
                    "factor_name": factor_name,
                    "total_periods": int(group["trade_date"].nunique()),
                    "valid_long_short_periods": spread["valid_periods"],
                    "mean_long_short_return": spread["mean"],
                    "std_long_short_return": spread["std"],
                    "long_short_ir": spread["ratio"],
                    "positive_long_short_ratio": spread["positive_ratio"],
                    "valid_monotonicity_periods": monotonicity["valid_periods"],
                    "mean_monotonicity": monotonicity["mean"],
                    "positive_monotonicity_ratio": monotonicity["positive_ratio"],
                    "mean_coverage": float(group["coverage"].mean()),
                    "mean_n_obs": float(group["n_obs"].mean()),
                }
            )
        result = pd.DataFrame(rows, columns=LONG_SHORT_SUMMARY_COLUMNS)
        return result.replace([np.inf, -np.inf], np.nan).sort_values(
            "factor_name", kind="mergesort", ignore_index=True
        )

    @staticmethod
    def _return_summary(values: pd.Series) -> Dict[str, object]:
        """Return finite-period mean, sample std, ratio, and positive share."""
        finite = values[np.isfinite(values)].astype(float)
        valid_periods = int(len(finite))
        if valid_periods == 0:
            return {
                "valid_periods": 0,
                "mean": np.nan,
                "std": np.nan,
                "ratio": np.nan,
                "positive_ratio": np.nan,
            }
        mean = float(finite.mean())
        positive_ratio = float((finite > 0.0).mean())
        if valid_periods < 2:
            std = np.nan
            ratio = np.nan
        else:
            std = float(finite.std(ddof=1))
            ratio = (
                float(mean / std)
                if np.isfinite(std) and not np.isclose(std, 0.0, atol=1e-12, rtol=0.0)
                else np.nan
            )
        return {
            "valid_periods": valid_periods,
            "mean": mean,
            "std": std,
            "ratio": ratio,
            "positive_ratio": positive_ratio,
        }

    def _normalize_quantile_results(self, results: pd.DataFrame) -> pd.DataFrame:
        """Validate and numericize quantile result rows without mutating input."""
        return self._normalize_result_frame(
            results, QUANTILE_RESULT_COLUMNS, "quantile_results"
        )

    def _normalize_long_short_results(self, results: pd.DataFrame) -> pd.DataFrame:
        """Validate spread rows while allowing absent optional coverage fields."""
        core = LONG_SHORT_RESULT_COLUMNS[:-2]
        if not isinstance(results, pd.DataFrame):
            raise TypeError("long_short_results must be a pandas DataFrame.")
        missing = [column for column in core if column not in results.columns]
        if missing:
            raise ValueError(
                f"long_short_results is missing required columns: {', '.join(missing)}."
            )
        normalized = results.copy(deep=True)
        for optional in ("coverage", "n_obs"):
            if optional not in normalized.columns:
                normalized[optional] = np.nan
        return self._normalize_result_frame(
            normalized, LONG_SHORT_RESULT_COLUMNS, "long_short_results"
        )

    @staticmethod
    def _normalize_result_frame(
        results: pd.DataFrame, columns: Sequence[str], frame_name: str
    ) -> pd.DataFrame:
        """Validate a result schema and coerce its metric columns to finite numeric."""
        if not isinstance(results, pd.DataFrame):
            raise TypeError(f"{frame_name} must be a pandas DataFrame.")
        missing = [column for column in columns if column not in results.columns]
        if missing:
            raise ValueError(
                f"{frame_name} is missing required columns: {', '.join(missing)}."
            )
        normalized = results.loc[:, list(columns)].copy(deep=True)
        numeric_columns = [
            column
            for column in columns
            if column not in {"trade_date", "factor_name"}
        ]
        for column in numeric_columns:
            normalized[column] = pd.to_numeric(
                normalized[column], errors="coerce"
            ).astype(float).replace([np.inf, -np.inf], np.nan)
        return normalized

    @staticmethod
    def _finite_correlation(first: pd.Series, second: pd.Series) -> float:
        """Return finite Pearson correlation or NaN for constant vectors."""
        first_values = first.to_numpy(dtype=float)
        second_values = second.to_numpy(dtype=float)
        first_std = float(first_values.std(ddof=0))
        second_std = float(second_values.std(ddof=0))
        if first_std == 0.0 or second_std == 0.0:
            return np.nan
        correlation = float(np.corrcoef(first_values, second_values)[0, 1])
        return float(np.clip(correlation, -1.0, 1.0)) if np.isfinite(correlation) else np.nan

    @staticmethod
    def _finite_value(value: object) -> float:
        """Return a finite float or NaN."""
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            return np.nan
        return normalized if np.isfinite(normalized) else np.nan

    @staticmethod
    def _validate_factor_names(factor_names: Sequence[str]) -> list[str]:
        """Validate requested factor names while retaining caller order."""
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
        return names

    def _normalize_factor_panel(
        self, factor_panel: pd.DataFrame, factor_names: Sequence[str]
    ) -> pd.DataFrame:
        """Copy, validate, and numericize selected factor columns."""
        required = ["trade_date", "ts_code"] + list(factor_names)
        normalized = self._normalize_panel(factor_panel, required, "factor_panel")
        for name in factor_names:
            normalized[name] = pd.to_numeric(
                normalized[name], errors="coerce"
            ).astype(float).replace([np.inf, -np.inf], np.nan)
        return normalized

    def _normalize_forward_returns(
        self, forward_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Copy, validate, and numericize caller-provided forward returns."""
        required = ["trade_date", "ts_code", self.config.return_col]
        normalized = self._normalize_panel(
            forward_returns, required, "forward_returns"
        )
        normalized[self.config.return_col] = pd.to_numeric(
            normalized[self.config.return_col], errors="coerce"
        ).astype(float).replace([np.inf, -np.inf], np.nan)
        return normalized

    @staticmethod
    def _normalize_panel(
        panel: pd.DataFrame, required: Sequence[str], panel_name: str
    ) -> pd.DataFrame:
        """Validate keys after normalizing dates and stripped stock codes."""
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(f"{panel_name} must be a pandas DataFrame.")
        missing = [column for column in required if column not in panel.columns]
        if missing:
            raise ValueError(
                f"{panel_name} is missing required columns: {', '.join(missing)}."
            )
        normalized = panel.loc[:, list(dict.fromkeys(required))].copy(deep=True)
        trade_dates = pd.to_datetime(normalized["trade_date"], errors="coerce")
        if trade_dates.isna().any():
            raise ValueError(
                f"{panel_name} trade_date must contain valid, non-empty dates."
            )
        normalized["trade_date"] = trade_dates
        codes = normalized["ts_code"].astype("string").str.strip()
        if codes.isna().any() or codes.eq("").any():
            raise ValueError(f"{panel_name} ts_code cannot contain empty values.")
        normalized["ts_code"] = codes
        if normalized.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(
                f"{panel_name} trade_date and ts_code combinations must be unique."
            )
        return normalized
