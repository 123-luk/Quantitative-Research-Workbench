"""Pure in-memory orchestration of the V2 factor-research components.

The runner composes existing public interfaces without copying their formulas.
Its fixed order is full-history factor calculation, exact score-key selection,
D1 preprocessing, optional D2 neutralization, G1 label construction, E1/E2
component evaluation, F1/F2 composition, and composite evaluation.

G2 does not download data or write result files. It does not construct real
historical index membership, adapt financial vendor fields, select Top N
stocks, manage holdings, rebalance, or run a backtest. Financial inputs must
already be point-in-time aligned and mapped to the standardized ``fin_*``
fields. Forward returns are evaluation labels only and never enter feature
processing. Rolling weights continue to use only IC or RankIC observations
strictly earlier than each score date. G3 owns result persistence and G4 owns
integration with V1 pipeline configuration and the CLI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar, Dict, Tuple

import pandas as pd

from src.factors.composition import FactorComposer, FactorCompositionConfig
from src.factors.dynamic_composition import (
    WEIGHT_HISTORY_COLUMNS,
    RollingICFactorComposer,
    RollingICWeightConfig,
)
from src.factors.evaluation import (
    IC_RESULT_COLUMNS,
    IC_SUMMARY_COLUMNS,
    FactorEvaluationConfig,
    FactorEvaluator,
)
from src.factors.factor_engine import FactorEngine
from src.factors.forward_returns import ForwardReturnBuilder, ForwardReturnConfig
from src.factors.neutralization import FactorNeutralizer, NeutralizationConfig
from src.factors.preprocessing import FactorPreprocessor, PreprocessingConfig
from src.factors.quantile_evaluation import (
    LONG_SHORT_RESULT_COLUMNS,
    LONG_SHORT_SUMMARY_COLUMNS,
    QUANTILE_RESULT_COLUMNS,
    QUANTILE_SUMMARY_COLUMNS,
    FactorQuantileEvaluator,
    QuantileEvaluationConfig,
)
from src.factors.registry import FactorRegistry


@dataclass(frozen=True)
class FactorResearchConfig:
    """Select factors, optional stages, composition, and evaluation outputs."""

    factor_names: Tuple[str, ...] = ()
    use_neutralization: bool = False
    composition_method: str = "equal"
    evaluate_components: bool = True
    evaluate_composite: bool = True

    def __post_init__(self) -> None:
        """Normalize factor names and validate stage dependencies."""
        if isinstance(self.factor_names, (str, bytes)):
            raise ValueError("factor_names must be a non-empty iterable of names.")
        try:
            factor_names = tuple(self.factor_names)
        except TypeError as exc:
            raise ValueError(
                "factor_names must be a non-empty iterable of names."
            ) from exc
        if not factor_names:
            raise ValueError("factor_names must contain at least one factor name.")
        if any(not isinstance(name, str) or not name.strip() for name in factor_names):
            raise ValueError("factor_names cannot contain empty values.")
        factor_names = tuple(name.strip() for name in factor_names)
        if len(set(factor_names)) != len(factor_names):
            raise ValueError("factor_names cannot contain duplicate names.")

        for field_name in (
            "use_neutralization",
            "evaluate_components",
            "evaluate_composite",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a bool.")
        allowed_methods = {
            "none",
            "equal",
            "fixed",
            "rolling_ic",
            "rolling_rank_ic",
        }
        if self.composition_method not in allowed_methods:
            raise ValueError(
                "composition_method must be 'none', 'equal', 'fixed', "
                "'rolling_ic', or 'rolling_rank_ic'."
            )
        if self.composition_method == "none" and self.evaluate_composite:
            raise ValueError(
                "evaluate_composite must be False when composition_method='none'."
            )
        if (
            self.composition_method in {"rolling_ic", "rolling_rank_ic"}
            and not self.evaluate_components
        ):
            raise ValueError(
                "Rolling composition requires evaluate_components=True."
            )
        object.__setattr__(self, "factor_names", factor_names)

    def to_dict(self) -> Dict[str, object]:
        """Return a serialization-friendly configuration dictionary."""
        return asdict(self)


@dataclass
class FactorResearchResult:
    """Hold all in-memory research tables and compact audit metadata."""

    requirements: Dict[str, object]
    raw_factor_panel: pd.DataFrame
    processed_factor_panel: pd.DataFrame
    final_factor_panel: pd.DataFrame
    forward_returns: pd.DataFrame
    factor_ic_results: pd.DataFrame
    factor_ic_summary: pd.DataFrame
    factor_quantile_results: pd.DataFrame
    factor_long_short_results: pd.DataFrame
    factor_quantile_summary: pd.DataFrame
    factor_long_short_summary: pd.DataFrame
    weight_history: pd.DataFrame
    composite_scores: pd.DataFrame
    composite_ic_results: pd.DataFrame
    composite_ic_summary: pd.DataFrame
    composite_quantile_results: pd.DataFrame
    composite_long_short_results: pd.DataFrame
    composite_quantile_summary: pd.DataFrame
    composite_long_short_summary: pd.DataFrame
    factor_names: Tuple[str, ...]
    used_neutralization: bool
    composition_method: str
    composite_score_col: str
    forward_return_col: str

    TABLE_FIELDS: ClassVar[Tuple[str, ...]] = (
        "raw_factor_panel",
        "processed_factor_panel",
        "final_factor_panel",
        "forward_returns",
        "factor_ic_results",
        "factor_ic_summary",
        "factor_quantile_results",
        "factor_long_short_results",
        "factor_quantile_summary",
        "factor_long_short_summary",
        "weight_history",
        "composite_scores",
        "composite_ic_results",
        "composite_ic_summary",
        "composite_quantile_results",
        "composite_long_short_results",
        "composite_quantile_summary",
        "composite_long_short_summary",
    )

    def table_shapes(self) -> Dict[str, Tuple[int, int]]:
        """Return row-column shapes without materializing table contents."""
        return {
            name: tuple(getattr(self, name).shape)
            for name in self.TABLE_FIELDS
        }

    def to_dict(self) -> Dict[str, object]:
        """Return compact metadata and shapes, never full DataFrame contents."""
        return {
            "requirements": self.requirements,
            "factor_names": list(self.factor_names),
            "used_neutralization": self.used_neutralization,
            "composition_method": self.composition_method,
            "composite_score_col": self.composite_score_col,
            "forward_return_col": self.forward_return_col,
            "table_shapes": self.table_shapes(),
        }


class FactorResearchRunner:
    """Orchestrate one deterministic, state-independent in-memory research run."""

    def __init__(
        self,
        registry: FactorRegistry,
        config: FactorResearchConfig,
        preprocessing_config: PreprocessingConfig | None = None,
        neutralization_config: NeutralizationConfig | None = None,
        evaluation_config: FactorEvaluationConfig | None = None,
        quantile_config: QuantileEvaluationConfig | None = None,
        composition_config: FactorCompositionConfig | None = None,
        rolling_config: RollingICWeightConfig | None = None,
        forward_return_config: ForwardReturnConfig | None = None,
    ) -> None:
        if not isinstance(registry, FactorRegistry):
            raise TypeError("registry must be a FactorRegistry.")
        if not isinstance(config, FactorResearchConfig):
            raise TypeError("config must be a FactorResearchConfig.")
        self.registry = registry
        self.config = config
        self.engine = FactorEngine(registry)
        # Resolving requirements validates every configured factor up front.
        self.engine.describe_requirements(list(config.factor_names))

        self.preprocessing_config = self._optional_config(
            preprocessing_config, PreprocessingConfig, "preprocessing_config"
        ) or PreprocessingConfig()
        self.neutralization_config = self._optional_config(
            neutralization_config, NeutralizationConfig, "neutralization_config"
        ) or NeutralizationConfig()
        self.evaluation_config = self._optional_config(
            evaluation_config, FactorEvaluationConfig, "evaluation_config"
        ) or FactorEvaluationConfig()
        self.quantile_config = self._optional_config(
            quantile_config, QuantileEvaluationConfig, "quantile_config"
        ) or QuantileEvaluationConfig()
        self.forward_return_config = self._optional_config(
            forward_return_config, ForwardReturnConfig, "forward_return_config"
        ) or ForwardReturnConfig()

        self.composition_config = self._resolve_composition_config(
            composition_config
        )
        self.rolling_config = self._resolve_rolling_config(rolling_config)
        return_columns = {
            self.evaluation_config.return_col,
            self.quantile_config.return_col,
            self.forward_return_config.return_col,
        }
        if len(return_columns) != 1:
            raise ValueError(
                "evaluation, quantile, and forward-return return_col values "
                "must match."
            )

    @staticmethod
    def _optional_config(value: object, expected_type: type, name: str) -> object:
        """Validate one optional component configuration."""
        if value is not None and not isinstance(value, expected_type):
            raise TypeError(f"{name} must be a {expected_type.__name__} or None.")
        return value

    def _resolve_composition_config(
        self, config: FactorCompositionConfig | None
    ) -> FactorCompositionConfig | None:
        """Validate or create the static-composition configuration."""
        if config is not None and not isinstance(config, FactorCompositionConfig):
            raise TypeError(
                "composition_config must be a FactorCompositionConfig or None."
            )
        method = self.config.composition_method
        if method == "equal":
            resolved = config or FactorCompositionConfig(method="equal")
            if resolved.method != "equal":
                raise ValueError(
                    "composition_method='equal' requires an equal composition_config."
                )
            return resolved
        if method == "fixed":
            if config is None or config.method != "fixed":
                raise ValueError(
                    "composition_method='fixed' requires a fixed composition_config."
                )
            return config
        return config

    def _resolve_rolling_config(
        self, config: RollingICWeightConfig | None
    ) -> RollingICWeightConfig | None:
        """Validate or create the dynamic-composition configuration."""
        if config is not None and not isinstance(config, RollingICWeightConfig):
            raise TypeError(
                "rolling_config must be a RollingICWeightConfig or None."
            )
        method = self.config.composition_method
        if method == "rolling_ic":
            resolved = config or RollingICWeightConfig(metric="ic")
            if resolved.metric != "ic":
                raise ValueError(
                    "composition_method='rolling_ic' requires metric='ic'."
                )
            return resolved
        if method == "rolling_rank_ic":
            resolved = config or RollingICWeightConfig(metric="rank_ic")
            if resolved.metric != "rank_ic":
                raise ValueError(
                    "composition_method='rolling_rank_ic' requires metric='rank_ic'."
                )
            return resolved
        return config

    def describe_config(self) -> Dict[str, object]:
        """Return every active component configuration as serializable metadata."""
        return {
            "research_config": self.config.to_dict(),
            "preprocessing_config": self.preprocessing_config.to_dict(),
            "neutralization_config": self.neutralization_config.to_dict(),
            "evaluation_config": self.evaluation_config.to_dict(),
            "quantile_config": self.quantile_config.to_dict(),
            "composition_config": (
                self.composition_config.to_dict()
                if self.composition_config is not None
                else None
            ),
            "rolling_config": (
                self.rolling_config.to_dict()
                if self.rolling_config is not None
                else None
            ),
            "forward_return_config": self.forward_return_config.to_dict(),
        }

    def run(
        self,
        factor_input: pd.DataFrame,
        score_panel: pd.DataFrame,
        price_panel: pd.DataFrame,
        exposure_panel: pd.DataFrame | None = None,
    ) -> FactorResearchResult:
        """Execute the fixed research order and return only in-memory results."""
        if not isinstance(factor_input, pd.DataFrame):
            raise TypeError("factor_input must be a pandas DataFrame.")
        if not isinstance(score_panel, pd.DataFrame):
            raise TypeError("score_panel must be a pandas DataFrame.")
        if not isinstance(price_panel, pd.DataFrame):
            raise TypeError("price_panel must be a pandas DataFrame.")
        if self.config.use_neutralization and exposure_panel is None:
            raise ValueError(
                "exposure_panel is required when use_neutralization=True."
            )

        names = list(self.config.factor_names)
        requirements = self.engine.describe_requirements(names)
        full_factor_panel = self.engine.compute_factor_panel(factor_input, names)
        score_keys = self._normalize_score_panel(score_panel)
        raw_factor_panel = self._select_score_keys(
            score_keys, full_factor_panel, names
        )

        processed_factor_panel = FactorPreprocessor(
            self.registry, self.preprocessing_config
        ).transform(raw_factor_panel, names)
        if self.config.use_neutralization:
            final_factor_panel = FactorNeutralizer(
                self.neutralization_config
            ).transform(
                processed_factor_panel,
                exposure_panel,  # type: ignore[arg-type]
                names,
            )
        else:
            final_factor_panel = processed_factor_panel.copy(deep=True)

        forward_returns = ForwardReturnBuilder(
            self.forward_return_config
        ).build(score_keys, price_panel)

        if self.config.evaluate_components:
            (
                factor_ic_results,
                factor_ic_summary,
                factor_quantile_results,
                factor_long_short_results,
                factor_quantile_summary,
                factor_long_short_summary,
            ) = self._evaluate(final_factor_panel, forward_returns, names)
        else:
            (
                factor_ic_results,
                factor_ic_summary,
                factor_quantile_results,
                factor_long_short_results,
                factor_quantile_summary,
                factor_long_short_summary,
            ) = self._empty_evaluation_tables()

        score_col = self._composite_score_col()
        composite_scores = self._empty_composite_scores(score_col)
        weight_history = self._empty_table(WEIGHT_HISTORY_COLUMNS)
        method = self.config.composition_method
        if method in {"equal", "fixed"}:
            composer = FactorComposer(
                self.registry,
                self.composition_config,  # type: ignore[arg-type]
            )
            composite_scores = composer.compose(final_factor_panel, names)
        elif method in {"rolling_ic", "rolling_rank_ic"}:
            composer = RollingICFactorComposer(
                self.registry,
                self.rolling_config,  # type: ignore[arg-type]
            )
            weight_history = composer.build_weight_history(
                final_factor_panel["trade_date"].drop_duplicates().tolist(),
                factor_ic_results,
                names,
            )
            composite_scores = composer.compose(
                final_factor_panel, factor_ic_results, names
            )

        if self.config.evaluate_composite:
            (
                composite_ic_results,
                composite_ic_summary,
                composite_quantile_results,
                composite_long_short_results,
                composite_quantile_summary,
                composite_long_short_summary,
            ) = self._evaluate(composite_scores, forward_returns, [score_col])
        else:
            (
                composite_ic_results,
                composite_ic_summary,
                composite_quantile_results,
                composite_long_short_results,
                composite_quantile_summary,
                composite_long_short_summary,
            ) = self._empty_evaluation_tables()

        return FactorResearchResult(
            requirements=requirements,
            raw_factor_panel=raw_factor_panel,
            processed_factor_panel=processed_factor_panel,
            final_factor_panel=final_factor_panel,
            forward_returns=forward_returns,
            factor_ic_results=factor_ic_results,
            factor_ic_summary=factor_ic_summary,
            factor_quantile_results=factor_quantile_results,
            factor_long_short_results=factor_long_short_results,
            factor_quantile_summary=factor_quantile_summary,
            factor_long_short_summary=factor_long_short_summary,
            weight_history=weight_history,
            composite_scores=composite_scores,
            composite_ic_results=composite_ic_results,
            composite_ic_summary=composite_ic_summary,
            composite_quantile_results=composite_quantile_results,
            composite_long_short_results=composite_long_short_results,
            composite_quantile_summary=composite_quantile_summary,
            composite_long_short_summary=composite_long_short_summary,
            factor_names=self.config.factor_names,
            used_neutralization=self.config.use_neutralization,
            composition_method=method,
            composite_score_col=score_col,
            forward_return_col=self.forward_return_config.return_col,
        )

    def _evaluate(
        self,
        factor_panel: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_names: list[str],
    ) -> Tuple[pd.DataFrame, ...]:
        """Call E1 and E2 public APIs for one factor panel."""
        evaluator = FactorEvaluator(self.evaluation_config)
        ic_results = evaluator.evaluate_ic(
            factor_panel, forward_returns, factor_names
        )
        ic_summary = evaluator.summarize_ic(ic_results)
        quantile_evaluator = FactorQuantileEvaluator(self.quantile_config)
        quantile_results = quantile_evaluator.evaluate_quantiles(
            factor_panel, forward_returns, factor_names
        )
        long_short_results = quantile_evaluator.evaluate_long_short(
            quantile_results
        )
        quantile_summary = quantile_evaluator.summarize_quantiles(
            quantile_results
        )
        long_short_summary = quantile_evaluator.summarize_long_short(
            long_short_results
        )
        return (
            ic_results,
            ic_summary,
            quantile_results,
            long_short_results,
            quantile_summary,
            long_short_summary,
        )

    @staticmethod
    def _normalize_score_panel(score_panel: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize exact score keys without copying extra fields."""
        required = ["trade_date", "ts_code"]
        missing = [column for column in required if column not in score_panel.columns]
        if missing:
            raise ValueError(
                "score_panel is missing required columns: "
                + ", ".join(missing)
                + "."
            )
        normalized = score_panel.loc[:, required].copy(deep=True)
        dates = pd.to_datetime(normalized["trade_date"], errors="coerce")
        if dates.isna().any():
            raise ValueError(
                "score_panel trade_date must contain valid, non-empty dates."
            )
        normalized["trade_date"] = dates
        codes = normalized["ts_code"].astype("string").str.strip()
        if codes.isna().any() or codes.eq("").any():
            raise ValueError("score_panel ts_code cannot contain empty values.")
        normalized["ts_code"] = codes
        if normalized.duplicated(required).any():
            raise ValueError(
                "score_panel trade_date and ts_code combinations must be unique."
            )
        return normalized.sort_values(
            required, kind="mergesort", ignore_index=True
        )

    @staticmethod
    def _select_score_keys(
        score_keys: pd.DataFrame,
        full_factor_panel: pd.DataFrame,
        factor_names: list[str],
    ) -> pd.DataFrame:
        """Select exact score keys after full-history factor calculation."""
        selected = score_keys.merge(
            full_factor_panel.loc[
                :, ["trade_date", "ts_code"] + factor_names
            ],
            on=["trade_date", "ts_code"],
            how="left",
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
        missing = selected["_merge"].eq("left_only")
        if missing.any():
            keys = selected.loc[missing, ["trade_date", "ts_code"]]
            details = ", ".join(
                f"({row.trade_date}, {row.ts_code})"
                for row in keys.itertuples(index=False)
            )
            raise ValueError(
                "Every score_panel key must exist in the computed factor panel; "
                f"missing: {details}."
            )
        columns = ["trade_date", "ts_code"] + factor_names
        return selected.loc[:, columns].sort_values(
            ["trade_date", "ts_code"], kind="mergesort", ignore_index=True
        )

    def _composite_score_col(self) -> str:
        """Return the active or conventional composite-score column name."""
        if self.config.composition_method in {"equal", "fixed"}:
            return self.composition_config.score_col  # type: ignore[union-attr]
        if self.config.composition_method in {"rolling_ic", "rolling_rank_ic"}:
            return self.rolling_config.score_col  # type: ignore[union-attr]
        return "composite_score"

    @staticmethod
    def _empty_table(columns: list[str]) -> pd.DataFrame:
        """Return an empty table with a stable ordered schema."""
        return pd.DataFrame(columns=columns)

    @classmethod
    def _empty_evaluation_tables(cls) -> Tuple[pd.DataFrame, ...]:
        """Return stable empty E1 and E2 result schemas."""
        return (
            cls._empty_table(IC_RESULT_COLUMNS),
            cls._empty_table(IC_SUMMARY_COLUMNS),
            cls._empty_table(QUANTILE_RESULT_COLUMNS),
            cls._empty_table(LONG_SHORT_RESULT_COLUMNS),
            cls._empty_table(QUANTILE_SUMMARY_COLUMNS),
            cls._empty_table(LONG_SHORT_SUMMARY_COLUMNS),
        )

    @staticmethod
    def _empty_composite_scores(score_col: str) -> pd.DataFrame:
        """Return the stable F1/F2 score schema when composition is disabled."""
        return pd.DataFrame(
            columns=[
                "trade_date",
                "ts_code",
                score_col,
                "valid_factor_count",
                "weight_coverage",
            ]
        )
