"""Equal- and fixed-weight composition of already processed factor values.

Inputs are expected to have completed V2-D1 direction unification and
cross-sectional standardization, and may optionally have completed V2-D2
neutralization. Larger input values must already mean a more preferred
direction. This module only performs row-local weighted composition: it does
not repeat direction adjustment, filling, winsorization, standardization, or
neutralization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from src.factors.registry import FactorRegistry


@dataclass(frozen=True)
class FactorCompositionConfig:
    """Configure factor weights and row-level missing-value handling."""

    method: str = "equal"
    fixed_weights: Tuple[Tuple[str, float], ...] = ()
    normalize_weights: bool = True
    missing_policy: str = "renormalize"
    min_valid_factors: int = 1
    score_col: str = "composite_score"

    def __post_init__(self) -> None:
        """Validate and normalize the immutable configuration."""
        if self.method not in {"equal", "fixed"}:
            raise ValueError("method must be either 'equal' or 'fixed'.")
        if self.missing_policy not in {"require_all", "renormalize"}:
            raise ValueError(
                "missing_policy must be either 'require_all' or 'renormalize'."
            )
        if not isinstance(self.normalize_weights, bool):
            raise ValueError("normalize_weights must be a bool.")
        if isinstance(self.min_valid_factors, bool) or not isinstance(
            self.min_valid_factors, Integral
        ):
            raise ValueError("min_valid_factors must be an integer >= 1.")
        if self.min_valid_factors < 1:
            raise ValueError("min_valid_factors must be an integer >= 1.")
        if not isinstance(self.score_col, str) or not self.score_col.strip():
            raise ValueError("score_col must be a non-empty string.")

        if isinstance(self.fixed_weights, (str, bytes)):
            raise ValueError("fixed_weights must contain (factor_name, weight) pairs.")
        try:
            pairs = tuple(self.fixed_weights)
        except TypeError as exc:
            raise ValueError(
                "fixed_weights must contain (factor_name, weight) pairs."
            ) from exc

        normalized_pairs = []
        for pair in pairs:
            if isinstance(pair, (str, bytes)):
                raise ValueError(
                    "fixed_weights must contain (factor_name, weight) pairs."
                )
            try:
                name, weight = pair
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "fixed_weights must contain (factor_name, weight) pairs."
                ) from exc
            if not isinstance(name, str) or not name.strip():
                raise ValueError("fixed_weights factor names cannot be empty.")
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise ValueError("fixed_weights must contain finite real weights.")
            numeric_weight = float(weight)
            if not np.isfinite(numeric_weight):
                raise ValueError("fixed_weights must contain finite real weights.")
            if numeric_weight < 0.0:
                raise ValueError("fixed_weights cannot contain negative weights.")
            normalized_pairs.append((name.strip(), numeric_weight))

        names = [name for name, _ in normalized_pairs]
        if len(set(names)) != len(names):
            raise ValueError("fixed_weights cannot contain duplicate factor names.")
        if normalized_pairs and not any(weight > 0.0 for _, weight in normalized_pairs):
            raise ValueError("fixed_weights must contain at least one positive weight.")
        if self.method == "fixed" and not normalized_pairs:
            raise ValueError("fixed_weights cannot be empty when method='fixed'.")

        object.__setattr__(self, "fixed_weights", tuple(normalized_pairs))
        object.__setattr__(self, "min_valid_factors", int(self.min_valid_factors))
        object.__setattr__(self, "score_col", self.score_col.strip())

    def to_dict(self) -> Dict[str, object]:
        """Return a serialization-friendly dictionary of all configuration."""
        return asdict(self)


class FactorComposer:
    """Compose comparable factor values into an auditable row-level score.

    A higher composite score means a more preferred combined factor direction.
    The score is a research ranking signal, not a return forecast. With
    ``renormalize``, the effective weights can vary by row; ``weight_coverage``
    preserves the share of configured base weight that was actually observed.
    """

    def __init__(
        self,
        registry: FactorRegistry,
        config: FactorCompositionConfig | None = None,
    ) -> None:
        if not isinstance(registry, FactorRegistry):
            raise TypeError("registry must be a FactorRegistry.")
        if config is not None and not isinstance(config, FactorCompositionConfig):
            raise TypeError("config must be a FactorCompositionConfig or None.")
        self.registry = registry
        self.config = config if config is not None else FactorCompositionConfig()

    def describe_config(self) -> Dict[str, object]:
        """Return a serialization-friendly summary of the active configuration."""
        return self.config.to_dict()

    def resolve_weights(self, factor_names: Sequence[str]) -> Dict[str, float]:
        """Validate factor names and return stable base weights keyed by name."""
        names = self._validate_factor_names(factor_names)
        ordered_names = sorted(names)
        if self.config.method == "equal":
            weight = 1.0 / len(ordered_names)
            return {name: weight for name in ordered_names}

        configured = dict(self.config.fixed_weights)
        selected = set(ordered_names)
        configured_names = set(configured)
        if configured_names != selected:
            missing = sorted(selected - configured_names)
            extra = sorted(configured_names - selected)
            details = []
            if missing:
                details.append(f"missing weights for: {', '.join(missing)}")
            if extra:
                details.append(f"extra weights for: {', '.join(extra)}")
            raise ValueError(
                "fixed_weights factor set must exactly match factor_names"
                + (f" ({'; '.join(details)})" if details else "")
                + "."
            )

        weights = {name: float(configured[name]) for name in ordered_names}
        total = float(sum(weights.values()))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Resolved fixed weights must have a positive finite sum.")
        if self.config.normalize_weights:
            weights = {name: weight / total for name, weight in weights.items()}
        return weights

    def compose(
        self,
        factor_panel: pd.DataFrame,
        factor_names: Sequence[str],
    ) -> pd.DataFrame:
        """Return one independently computed composite score per input row."""
        weights = self.resolve_weights(factor_names)
        names = list(weights)
        normalized = self._normalize_panel(factor_panel, names)

        output_columns = [
            "trade_date",
            "ts_code",
            self.config.score_col,
            "valid_factor_count",
            "weight_coverage",
        ]
        if normalized.empty:
            empty = normalized.loc[:, ["trade_date", "ts_code"]].copy()
            empty[self.config.score_col] = pd.Series(dtype=float)
            empty["valid_factor_count"] = pd.Series(dtype="int64")
            empty["weight_coverage"] = pd.Series(dtype=float)
            return empty.loc[:, output_columns]

        values = normalized.loc[:, names].to_numpy(dtype=float)
        base_weights = np.array([weights[name] for name in names], dtype=float)
        valid = np.isfinite(values)
        valid_count = valid.sum(axis=1).astype(np.int64)
        valid_weight = (valid * base_weights).sum(axis=1)
        total_weight = float(base_weights.sum())
        coverage = np.clip(valid_weight / total_weight, 0.0, 1.0)

        safe_values = np.where(valid, values, 0.0)
        weighted_sum = (safe_values * base_weights).sum(axis=1)
        scores = np.full(len(normalized), np.nan, dtype=float)
        if self.config.missing_policy == "require_all":
            complete = valid_count == len(names)
            scores[complete] = weighted_sum[complete]
        else:
            eligible = (
                (valid_count >= self.config.min_valid_factors)
                & np.isfinite(valid_weight)
                & (valid_weight > 0.0)
            )
            scores[eligible] = weighted_sum[eligible] / valid_weight[eligible]

        scores[~np.isfinite(scores)] = np.nan
        result = normalized.loc[:, ["trade_date", "ts_code"]].copy()
        result[self.config.score_col] = scores
        result["valid_factor_count"] = valid_count
        result["weight_coverage"] = coverage
        return result.loc[:, output_columns].sort_values(
            ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
        )

    def _validate_factor_names(self, factor_names: Sequence[str]) -> list[str]:
        """Validate requested names and confirm every factor is registered."""
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
        normalized = [name.strip() for name in names]
        if len(set(normalized)) != len(normalized):
            raise ValueError("factor_names cannot contain duplicate names.")
        for name in normalized:
            try:
                self.registry.get(name)
            except KeyError as exc:
                raise ValueError(f"Factor '{name}' is not registered.") from exc
        return normalized

    @staticmethod
    def _normalize_panel(
        factor_panel: pd.DataFrame, factor_names: Sequence[str]
    ) -> pd.DataFrame:
        """Copy, validate, key-normalize, and numericize selected factors."""
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
        return normalized
