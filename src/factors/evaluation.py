"""Same-date IC and RankIC evaluation for already processed factor panels.

Forward returns must be prepared by the caller with an explicit holding-period
meaning. This module never constructs labels, shifts prices, fills observations,
or repeats factor-direction adjustment. Inputs are assumed to have completed
V2-D1 direction handling and may optionally have completed V2-D2 neutralization.
Positive IC therefore means higher factor values align with higher future returns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Dict, Sequence

import numpy as np
import pandas as pd


IC_RESULT_COLUMNS = [
    "trade_date",
    "factor_name",
    "universe_count",
    "n_obs",
    "coverage",
    "ic",
    "rank_ic",
]

IC_SUMMARY_COLUMNS = [
    "factor_name",
    "total_periods",
    "valid_ic_periods",
    "mean_ic",
    "std_ic",
    "icir",
    "positive_ic_ratio",
    "valid_rank_ic_periods",
    "mean_rank_ic",
    "std_rank_ic",
    "rank_icir",
    "positive_rank_ic_ratio",
    "mean_coverage",
    "mean_n_obs",
]


@dataclass(frozen=True)
class FactorEvaluationConfig:
    """Configure same-date Pearson IC and Spearman RankIC evaluation."""

    return_col: str = "forward_return"
    min_cross_section_size: int = 20
    compute_ic: bool = True
    compute_rank_ic: bool = True

    def __post_init__(self) -> None:
        """Validate the return field, sample threshold, and metric switches."""
        if not isinstance(self.return_col, str) or not self.return_col.strip():
            raise ValueError("return_col must be a non-empty string.")
        if isinstance(self.min_cross_section_size, bool) or not isinstance(
            self.min_cross_section_size, Integral
        ):
            raise ValueError("min_cross_section_size must be an integer >= 2.")
        if self.min_cross_section_size < 2:
            raise ValueError("min_cross_section_size must be an integer >= 2.")
        if not isinstance(self.compute_ic, bool):
            raise ValueError("compute_ic must be a bool.")
        if not isinstance(self.compute_rank_ic, bool):
            raise ValueError("compute_rank_ic must be a bool.")
        if not self.compute_ic and not self.compute_rank_ic:
            raise ValueError("At least one of compute_ic or compute_rank_ic must be True.")

        object.__setattr__(self, "return_col", self.return_col.strip())
        object.__setattr__(
            self, "min_cross_section_size", int(self.min_cross_section_size)
        )

    def to_dict(self) -> Dict[str, object]:
        """Return a serialization-friendly configuration dictionary."""
        return asdict(self)


class FactorEvaluator:
    """Evaluate factor IC and RankIC independently within each trade date."""

    _RETURN_INTERNAL = "__evaluation_forward_return__"

    def __init__(self, config: FactorEvaluationConfig | None = None) -> None:
        if config is not None and not isinstance(config, FactorEvaluationConfig):
            raise TypeError("config must be a FactorEvaluationConfig or None.")
        self.config = config if config is not None else FactorEvaluationConfig()

    def describe_config(self) -> Dict[str, object]:
        """Return a serialization-friendly summary of the active configuration."""
        return self.config.to_dict()

    def evaluate_ic(
        self,
        factor_panel: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_names: Sequence[str],
    ) -> pd.DataFrame:
        """Return same-date IC observations for every requested factor."""
        names = self._validate_factor_names(factor_names)
        factors = self._normalize_factor_panel(factor_panel, names)
        returns = self._normalize_forward_returns(forward_returns)
        returns = returns.rename(columns={self.config.return_col: self._RETURN_INTERNAL})
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
                paired_factor = factor_values.loc[valid]
                paired_return = forward_return.loc[valid]
                n_obs = int(valid.sum())
                coverage = (
                    float(n_obs / universe_count) if universe_count > 0 else np.nan
                )
                ic = np.nan
                rank_ic = np.nan
                if n_obs >= self.config.min_cross_section_size:
                    if self.config.compute_ic:
                        ic = self._finite_correlation(paired_factor, paired_return)
                    if self.config.compute_rank_ic:
                        factor_rank = paired_factor.rank(method="average")
                        return_rank = paired_return.rank(method="average")
                        rank_ic = self._finite_correlation(factor_rank, return_rank)
                rows.append(
                    {
                        "trade_date": trade_date,
                        "factor_name": name,
                        "universe_count": universe_count,
                        "n_obs": n_obs,
                        "coverage": coverage,
                        "ic": ic,
                        "rank_ic": rank_ic,
                    }
                )

        result = pd.DataFrame(rows, columns=IC_RESULT_COLUMNS)
        if result.empty:
            return result
        result = result.replace([np.inf, -np.inf], np.nan)
        result["coverage"] = result["coverage"].clip(lower=0.0, upper=1.0)
        return result.sort_values(
            ["trade_date", "factor_name"], kind="mergesort", ignore_index=True
        )

    def summarize_ic(self, ic_results: pd.DataFrame) -> pd.DataFrame:
        """Summarize finite IC and RankIC observations independently by factor."""
        if not isinstance(ic_results, pd.DataFrame):
            raise TypeError("ic_results must be a pandas DataFrame.")
        missing = [
            column for column in IC_RESULT_COLUMNS if column not in ic_results.columns
        ]
        if missing:
            raise ValueError(
                f"ic_results is missing required columns: {', '.join(missing)}."
            )
        normalized = ic_results.loc[:, IC_RESULT_COLUMNS].copy(deep=True)
        for column in ("ic", "rank_ic", "coverage", "n_obs"):
            normalized[column] = pd.to_numeric(
                normalized[column], errors="coerce"
            ).astype(float).replace([np.inf, -np.inf], np.nan)

        rows = []
        for factor_name, group in normalized.groupby("factor_name", sort=True):
            ic_stats = self._metric_summary(group["ic"])
            rank_stats = self._metric_summary(group["rank_ic"])
            rows.append(
                {
                    "factor_name": factor_name,
                    "total_periods": int(group["trade_date"].nunique()),
                    "valid_ic_periods": ic_stats["valid_periods"],
                    "mean_ic": ic_stats["mean"],
                    "std_ic": ic_stats["std"],
                    "icir": ic_stats["ratio"],
                    "positive_ic_ratio": ic_stats["positive_ratio"],
                    "valid_rank_ic_periods": rank_stats["valid_periods"],
                    "mean_rank_ic": rank_stats["mean"],
                    "std_rank_ic": rank_stats["std"],
                    "rank_icir": rank_stats["ratio"],
                    "positive_rank_ic_ratio": rank_stats["positive_ratio"],
                    "mean_coverage": float(group["coverage"].mean()),
                    "mean_n_obs": float(group["n_obs"].mean()),
                }
            )
        result = pd.DataFrame(rows, columns=IC_SUMMARY_COLUMNS)
        return result.replace([np.inf, -np.inf], np.nan).sort_values(
            "factor_name", kind="mergesort", ignore_index=True
        )

    @staticmethod
    def _finite_correlation(first: pd.Series, second: pd.Series) -> float:
        """Return finite Pearson correlation, or NaN for constant inputs."""
        first_values = first.to_numpy(dtype=float)
        second_values = second.to_numpy(dtype=float)
        if len(first_values) < 2:
            return np.nan
        first_std = float(first_values.std(ddof=0))
        second_std = float(second_values.std(ddof=0))
        if (
            not np.isfinite(first_std)
            or not np.isfinite(second_std)
            or first_std == 0.0
            or second_std == 0.0
        ):
            return np.nan
        correlation = float(np.corrcoef(first_values, second_values)[0, 1])
        if not np.isfinite(correlation):
            return np.nan
        return float(np.clip(correlation, -1.0, 1.0))

    @staticmethod
    def _metric_summary(values: pd.Series) -> Dict[str, object]:
        """Return non-annualized summary statistics for one IC metric."""
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
                if np.isfinite(std) and std != 0.0
                else np.nan
            )
        return {
            "valid_periods": valid_periods,
            "mean": mean,
            "std": std,
            "ratio": ratio,
            "positive_ratio": positive_ratio,
        }

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
        """Copy, validate, and numericize the caller-provided return column."""
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
