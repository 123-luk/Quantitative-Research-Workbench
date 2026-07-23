"""Cross-sectional factor preprocessing performed independently by trade date."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from src.factors.registry import FactorRegistry


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configure missing-value, winsorization, and standardization methods."""

    missing_method: str = "median"
    winsor_method: str = "mad"
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    mad_limit: float = 3.0
    standardize_method: str = "zscore"
    min_cross_section_size: int = 3

    def __post_init__(self) -> None:
        """Validate methods and numeric boundaries, then normalize numeric types."""
        if self.missing_method not in {"none", "median"}:
            raise ValueError("missing_method must be either 'none' or 'median'.")
        if self.winsor_method not in {"none", "quantile", "mad"}:
            raise ValueError(
                "winsor_method must be 'none', 'quantile', or 'mad'."
            )
        if self.standardize_method not in {"none", "zscore", "rank"}:
            raise ValueError(
                "standardize_method must be 'none', 'zscore', or 'rank'."
            )

        lower = self._finite_real("lower_quantile", self.lower_quantile)
        upper = self._finite_real("upper_quantile", self.upper_quantile)
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(
                "Quantile boundaries must satisfy "
                "0 <= lower_quantile < upper_quantile <= 1."
            )

        mad_limit = self._finite_real("mad_limit", self.mad_limit)
        if mad_limit <= 0.0:
            raise ValueError("mad_limit must be greater than 0.")
        if isinstance(self.min_cross_section_size, bool) or not isinstance(
            self.min_cross_section_size, Integral
        ):
            raise ValueError("min_cross_section_size must be an integer >= 1.")
        if self.min_cross_section_size < 1:
            raise ValueError("min_cross_section_size must be an integer >= 1.")

        object.__setattr__(self, "lower_quantile", lower)
        object.__setattr__(self, "upper_quantile", upper)
        object.__setattr__(self, "mad_limit", mad_limit)
        object.__setattr__(
            self, "min_cross_section_size", int(self.min_cross_section_size)
        )

    @staticmethod
    def _finite_real(name: str, value: object) -> float:
        """Return a finite float while rejecting booleans and non-real values."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number.")
        normalized = float(value)
        if not np.isfinite(normalized):
            raise ValueError(f"{name} must be a finite real number.")
        return normalized

    def to_dict(self) -> Dict[str, object]:
        """Return the configuration as a serialization-friendly dictionary."""
        return asdict(self)


class FactorPreprocessor:
    """Apply factor preprocessing to each trade-date cross-section in isolation."""

    def __init__(
        self,
        registry: FactorRegistry,
        config: PreprocessingConfig | None = None,
    ) -> None:
        if not isinstance(registry, FactorRegistry):
            raise TypeError("registry must be a FactorRegistry.")
        if config is not None and not isinstance(config, PreprocessingConfig):
            raise TypeError("config must be a PreprocessingConfig or None.")
        self.registry = registry
        self.config = config if config is not None else PreprocessingConfig()

    def describe_config(self) -> Dict[str, object]:
        """Return a serialization-friendly summary of the active configuration."""
        return self.config.to_dict()

    def transform(
        self,
        factor_panel: pd.DataFrame,
        factor_names: Sequence[str],
    ) -> pd.DataFrame:
        """Return selected factors preprocessed independently for each trade date."""
        names, factors = self._resolve_factors(factor_names)
        normalized = self._normalize_input(factor_panel, names)
        output_columns = ["trade_date", "ts_code"] + names
        if normalized.empty:
            return normalized.loc[:, output_columns]

        frames = []
        for _, cross_section in normalized.groupby("trade_date", sort=True):
            processed = cross_section.loc[:, ["trade_date", "ts_code"]].copy()
            for name, factor in zip(names, factors):
                values = pd.to_numeric(cross_section[name], errors="coerce").astype(
                    float
                )
                values = values.replace([np.inf, -np.inf], np.nan)
                values = self._handle_missing(values)
                values = self._winsorize(values)
                values = values * factor.metadata.direction
                values = self._standardize(values)
                processed[name] = values.replace([np.inf, -np.inf], np.nan)
            frames.append(processed)

        result = pd.concat(frames, ignore_index=True)
        return result.loc[:, output_columns].sort_values(
            ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
        )

    def _resolve_factors(self, factor_names: Sequence[str]) -> tuple[list[str], list[object]]:
        """Validate requested factor names and resolve registered factor objects."""
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
        if len(set(names)) != len(names):
            raise ValueError("factor_names cannot contain duplicate names.")

        factors = []
        for name in names:
            try:
                factors.append(self.registry.get(name))
            except KeyError as exc:
                raise KeyError(f"Requested factor '{name}' is not registered.") from exc
        return names, factors

    @staticmethod
    def _normalize_input(
        factor_panel: pd.DataFrame,
        factor_names: Sequence[str],
    ) -> pd.DataFrame:
        """Copy and validate panel keys and requested factor columns."""
        if not isinstance(factor_panel, pd.DataFrame):
            raise TypeError("factor_panel must be a pandas DataFrame.")
        required = ["trade_date", "ts_code"] + list(factor_names)
        missing = [column for column in required if column not in factor_panel.columns]
        if missing:
            raise ValueError(f"factor_panel is missing required columns: {', '.join(missing)}.")

        normalized = factor_panel.loc[:, required].copy(deep=True)
        trade_dates = pd.to_datetime(normalized["trade_date"], errors="coerce")
        if trade_dates.isna().any():
            raise ValueError("trade_date must contain valid, non-empty dates.")
        normalized["trade_date"] = trade_dates

        codes = normalized["ts_code"].astype("string")
        if codes.isna().any() or codes.str.strip().eq("").any():
            raise ValueError("ts_code cannot contain empty values.")
        normalized["ts_code"] = codes
        if normalized.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError("trade_date and ts_code combinations must be unique.")
        return normalized

    def _handle_missing(self, values: pd.Series) -> pd.Series:
        """Apply the configured same-date missing-value rule."""
        result = values.copy()
        if self.config.missing_method == "median":
            finite = result[np.isfinite(result)]
            if not finite.empty:
                result = result.fillna(float(finite.median()))
        return result

    def _winsorize(self, values: pd.Series) -> pd.Series:
        """Apply the configured same-date winsorization rule."""
        result = values.copy()
        finite = result[np.isfinite(result)]
        if finite.empty or self.config.winsor_method == "none":
            return result
        if self.config.winsor_method == "quantile":
            lower = float(finite.quantile(self.config.lower_quantile))
            upper = float(finite.quantile(self.config.upper_quantile))
        else:
            median = float(finite.median())
            mad = float((finite - median).abs().median())
            if mad == 0.0 or not np.isfinite(mad):
                return result
            robust_scale = 1.4826 * mad
            lower = median - self.config.mad_limit * robust_scale
            upper = median + self.config.mad_limit * robust_scale
        result.loc[finite.index] = finite.clip(lower=lower, upper=upper)
        return result

    def _standardize(self, values: pd.Series) -> pd.Series:
        """Apply the configured same-date standardization rule."""
        result = values.copy()
        if self.config.standardize_method == "none":
            return result
        finite = result[np.isfinite(result)]
        if len(finite) < self.config.min_cross_section_size:
            return pd.Series(np.nan, index=result.index, dtype=float)

        if self.config.standardize_method == "zscore":
            mean = float(finite.mean())
            std = float(finite.std(ddof=0))
            if std == 0.0 or not np.isfinite(mean) or not np.isfinite(std):
                return pd.Series(np.nan, index=result.index, dtype=float)
            result.loc[finite.index] = (finite - mean) / std
            return result

        if len(finite) <= 1:
            return pd.Series(np.nan, index=result.index, dtype=float)
        if finite.nunique(dropna=True) == 1:
            result.loc[finite.index] = 0.0
            return result
        ranks = finite.rank(method="average")
        result.loc[finite.index] = 2.0 * (ranks - 1.0) / (len(finite) - 1.0) - 1.0
        return result
