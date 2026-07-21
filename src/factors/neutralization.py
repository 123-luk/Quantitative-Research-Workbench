"""Same-date industry and size neutralization for preprocessed factor panels.

The recommended order is raw factor calculation, V2-D1 preprocessing, this
V2-D2 neutralization step, optional residual z-scoring, and factor evaluation.
Small industries are excluded rather than pooled, and no exposure is filled
across dates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NeutralizationConfig:
    """Configure same-date industry and size OLS neutralization."""

    neutralize_industry: bool = True
    neutralize_size: bool = True
    industry_col: str = "industry"
    size_col: str = "log_total_mv"
    min_cross_section_size: int = 10
    min_industry_size: int = 2
    standardize_residuals: bool = True
    size_exempt_factors: Tuple[str, ...] = ("log_total_mv", "log_circ_mv")

    def __post_init__(self) -> None:
        """Validate flags, column names, sample limits, and exemptions."""
        for name in (
            "neutralize_industry",
            "neutralize_size",
            "standardize_residuals",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool.")
        if not self.neutralize_industry and not self.neutralize_size:
            raise ValueError(
                "At least one of neutralize_industry or neutralize_size must be True."
            )

        industry_col = self._non_empty_name("industry_col", self.industry_col)
        size_col = self._non_empty_name("size_col", self.size_col)
        min_cross_section_size = self._valid_integer(
            "min_cross_section_size", self.min_cross_section_size, minimum=2
        )
        min_industry_size = self._valid_integer(
            "min_industry_size", self.min_industry_size, minimum=1
        )

        if isinstance(self.size_exempt_factors, (str, bytes)):
            raise ValueError("size_exempt_factors must be an iterable of names.")
        try:
            exemptions = tuple(self.size_exempt_factors)
        except TypeError as exc:
            raise ValueError("size_exempt_factors must be an iterable of names.") from exc
        normalized_exemptions = tuple(
            self._non_empty_name("size_exempt_factors entry", name)
            for name in exemptions
        )
        if len(set(normalized_exemptions)) != len(normalized_exemptions):
            raise ValueError("size_exempt_factors cannot contain duplicate names.")

        object.__setattr__(self, "industry_col", industry_col)
        object.__setattr__(self, "size_col", size_col)
        object.__setattr__(
            self, "min_cross_section_size", min_cross_section_size
        )
        object.__setattr__(self, "min_industry_size", min_industry_size)
        object.__setattr__(self, "size_exempt_factors", normalized_exemptions)

    @staticmethod
    def _non_empty_name(field_name: str, value: object) -> str:
        """Return a stripped non-empty string."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _valid_integer(field_name: str, value: object, minimum: int) -> int:
        """Return an integer at or above the required minimum, excluding bool."""
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{field_name} must be an integer >= {minimum}.")
        normalized = int(value)
        if normalized < minimum:
            raise ValueError(f"{field_name} must be an integer >= {minimum}.")
        return normalized

    def to_dict(self) -> Dict[str, object]:
        """Return a serialization-friendly dictionary of all configuration."""
        return asdict(self)


class FactorNeutralizer:
    """Neutralize factor values through independent same-date OLS regressions."""

    _INDUSTRY_INTERNAL = "__neutralization_industry__"
    _SIZE_INTERNAL = "__neutralization_size__"

    def __init__(self, config: NeutralizationConfig | None = None) -> None:
        if config is not None and not isinstance(config, NeutralizationConfig):
            raise TypeError("config must be a NeutralizationConfig or None.")
        self.config = config if config is not None else NeutralizationConfig()

    def describe_config(self) -> Dict[str, object]:
        """Return a serialization-friendly summary of the active configuration."""
        return self.config.to_dict()

    def transform(
        self,
        factor_panel: pd.DataFrame,
        exposure_panel: pd.DataFrame,
        factor_names: Sequence[str],
    ) -> pd.DataFrame:
        """Return factors neutralized independently for every trade date."""
        names = self._validate_factor_names(factor_names)
        factors = self._normalize_factor_panel(factor_panel, names)
        exposures = self._normalize_exposure_panel(exposure_panel)
        output_columns = ["trade_date", "ts_code"] + names

        exposure_columns = ["trade_date", "ts_code"]
        rename_columns = {}
        if self.config.neutralize_industry:
            exposure_columns.append(self.config.industry_col)
            rename_columns[self.config.industry_col] = self._INDUSTRY_INTERNAL
        if self.config.neutralize_size:
            exposure_columns.append(self.config.size_col)
            rename_columns[self.config.size_col] = self._SIZE_INTERNAL
        exposures = exposures.loc[:, exposure_columns].rename(columns=rename_columns)
        merged = factors.merge(
            exposures,
            on=["trade_date", "ts_code"],
            how="left",
            sort=False,
            validate="one_to_one",
        )

        result = merged.loc[:, ["trade_date", "ts_code"]].copy()
        for name in names:
            result[name] = np.nan

        for _, date_frame in merged.groupby("trade_date", sort=True):
            for name in names:
                result.loc[date_frame.index, name] = self._neutralize_cross_section(
                    date_frame, name
                )

        result = result.loc[:, output_columns].replace([np.inf, -np.inf], np.nan)
        return result.sort_values(
            ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
        )

    @staticmethod
    def _validate_factor_names(factor_names: Sequence[str]) -> list[str]:
        """Validate and normalize the requested factor-name sequence."""
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
        """Copy and validate factor keys and selected numeric factor columns."""
        required = ["trade_date", "ts_code"] + list(factor_names)
        normalized = self._normalize_panel(factor_panel, required, "factor_panel")
        for name in factor_names:
            normalized[name] = pd.to_numeric(
                normalized[name], errors="coerce"
            ).astype(float).replace([np.inf, -np.inf], np.nan)
        return normalized

    def _normalize_exposure_panel(self, exposure_panel: pd.DataFrame) -> pd.DataFrame:
        """Copy and validate exposure keys and configured exposure columns."""
        required = ["trade_date", "ts_code"]
        if self.config.neutralize_industry:
            required.append(self.config.industry_col)
        if self.config.neutralize_size:
            required.append(self.config.size_col)
        normalized = self._normalize_panel(exposure_panel, required, "exposure_panel")
        if self.config.neutralize_industry:
            industry = normalized[self.config.industry_col].astype("string").str.strip()
            normalized[self.config.industry_col] = industry.mask(industry.eq(""))
        if self.config.neutralize_size:
            normalized[self.config.size_col] = pd.to_numeric(
                normalized[self.config.size_col], errors="coerce"
            ).astype(float).replace([np.inf, -np.inf], np.nan)
        return normalized

    @staticmethod
    def _normalize_panel(
        panel: pd.DataFrame, required: Sequence[str], panel_name: str
    ) -> pd.DataFrame:
        """Validate a panel without mutating the caller's DataFrame."""
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
        codes = normalized["ts_code"].astype("string")
        if codes.isna().any() or codes.str.strip().eq("").any():
            raise ValueError(f"{panel_name} ts_code cannot contain empty values.")
        normalized["ts_code"] = codes
        if normalized.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(
                f"{panel_name} trade_date and ts_code combinations must be unique."
            )
        return normalized

    def _neutralize_cross_section(
        self, date_frame: pd.DataFrame, factor_name: str
    ) -> pd.Series:
        """Return one factor's residuals aligned to the date-frame index."""
        output = pd.Series(np.nan, index=date_frame.index, dtype=float)
        y = date_frame[factor_name]
        valid = y.notna() & np.isfinite(y)
        use_size = (
            self.config.neutralize_size
            and factor_name not in self.config.size_exempt_factors
        )

        if self.config.neutralize_industry:
            industry = date_frame[self._INDUSTRY_INTERNAL]
            valid &= industry.notna()
        else:
            industry = pd.Series(pd.NA, index=date_frame.index, dtype="string")
        if use_size:
            size = date_frame[self._SIZE_INTERNAL]
            valid &= size.notna() & np.isfinite(size)
        else:
            size = pd.Series(np.nan, index=date_frame.index, dtype=float)

        if self.config.neutralize_industry:
            counts = industry.loc[valid].value_counts()
            eligible = counts[counts >= self.config.min_industry_size].index
            valid &= industry.isin(eligible)

        sample_index = date_frame.index[valid]
        if len(sample_index) < self.config.min_cross_section_size:
            return output

        columns = [np.ones(len(sample_index), dtype=float)]
        if self.config.neutralize_industry:
            sample_industry = industry.loc[sample_index]
            categories = sorted(sample_industry.astype(str).unique())
            for category in categories[1:]:
                columns.append((sample_industry == category).to_numpy(dtype=float))

        if use_size:
            sample_size = size.loc[sample_index].to_numpy(dtype=float)
            size_mean = float(sample_size.mean())
            size_std = float(sample_size.std(ddof=0))
            if (
                not np.isfinite(size_mean)
                or not np.isfinite(size_std)
                or np.isclose(size_std, 0.0, atol=1e-12, rtol=0.0)
            ):
                return output
            columns.append((sample_size - size_mean) / size_std)

        design = np.column_stack(columns)
        rank = int(np.linalg.matrix_rank(design))
        if len(sample_index) <= rank:
            return output

        sample_y = y.loc[sample_index].to_numpy(dtype=float)
        try:
            beta = np.linalg.lstsq(design, sample_y, rcond=None)[0]
            residuals = sample_y - design @ beta
        except np.linalg.LinAlgError:
            return output
        if not np.isfinite(residuals).all():
            return output

        if self.config.standardize_residuals:
            residual_mean = float(residuals.mean())
            residual_std = float(residuals.std(ddof=0))
            if (
                not np.isfinite(residual_mean)
                or not np.isfinite(residual_std)
                or np.isclose(residual_std, 0.0, atol=1e-12, rtol=0.0)
            ):
                return output
            residuals = (residuals - residual_mean) / residual_std

        output.loc[sample_index] = residuals
        return output
