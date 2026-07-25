"""In-memory orchestration for the public V3 machine-learning stages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from numbers import Integral
from typing import Mapping

import pandas as pd

from src.ml.contracts import (
    METADATA_COLUMNS,
    MLDataset,
    MLDatasetAudit,
    MLDatasetConfig,
)
from src.ml.dataset import MLDatasetBuilder
from src.ml.evaluation import (
    ModelEvaluationAudit,
    ModelEvaluationConfig,
    ModelEvaluationResult,
    OOSModelEvaluator,
)
from src.ml.importance import (
    WalkForwardPermutationImportanceConfig,
    WalkForwardPermutationImportanceResult,
    WalkForwardPermutationImportanceRunner,
)
from src.ml.models import ModelRegistry, create_default_model_registry
from src.ml.splitting import (
    WalkForwardConfig,
    WalkForwardPlan,
    WalkForwardSplitter,
)
from src.ml.training import (
    WalkForwardTrainingConfig,
    WalkForwardTrainingResult,
    WalkForwardTrainer,
)


class MLExperimentError(Exception):
    """Base error for an in-memory ML experiment."""


class MLExperimentConfigError(MLExperimentError):
    """Raised when experiment configuration is invalid."""


class MLExperimentDataError(MLExperimentError):
    """Raised when the input or a stage output has the wrong data type."""


class MLExperimentIntegrityError(MLExperimentError):
    """Raised when successful stage outputs disagree."""


class MLExperimentStageError(MLExperimentError):
    """Wrap one failed public stage call with compact context."""

    def __init__(self, stage: str, cause: Exception) -> None:
        allowed = {
            "dataset_build",
            "walk_forward_split",
            "training",
            "evaluation",
            "permutation_importance",
            "integrity_validation",
            "result_build",
        }
        if stage not in allowed:
            raise MLExperimentConfigError(f"unknown experiment stage {stage!r}")
        self.stage = stage
        super().__init__(
            f"ML experiment stage {stage} failed; "
            f"cause={type(cause).__name__}"
        )


@dataclass(frozen=True)
class PermutationImportanceOptionsConfig:
    """Importance options without a second source of model configuration."""

    scoring: str = "rmse"
    n_repeats: int = 5
    random_state: int = 42
    permutation_scope: str = "within_trade_date"

    def __post_init__(self) -> None:
        if not isinstance(self.scoring, str):
            raise MLExperimentConfigError("scoring must be rmse or mae")
        scoring = self.scoring.strip().lower()
        if scoring not in {"rmse", "mae"}:
            raise MLExperimentConfigError("scoring must be rmse or mae")
        if (
            isinstance(self.n_repeats, bool)
            or not isinstance(self.n_repeats, Integral)
            or int(self.n_repeats) < 1
        ):
            raise MLExperimentConfigError(
                "n_repeats must be an integer >= 1"
            )
        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, Integral)
            or int(self.random_state) < 0
        ):
            raise MLExperimentConfigError(
                "random_state must be a non-negative integer"
            )
        if not isinstance(self.permutation_scope, str):
            raise MLExperimentConfigError(
                "permutation_scope must be within_trade_date"
            )
        scope = self.permutation_scope.strip().lower()
        if scope != "within_trade_date":
            raise MLExperimentConfigError(
                "permutation_scope must be within_trade_date"
            )
        object.__setattr__(self, "scoring", scoring)
        object.__setattr__(self, "n_repeats", int(self.n_repeats))
        object.__setattr__(self, "random_state", int(self.random_state))
        object.__setattr__(self, "permutation_scope", scope)

    @classmethod
    def from_dict(
        cls, values: Mapping[str, object] | None
    ) -> "PermutationImportanceOptionsConfig":
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise MLExperimentConfigError(
                "permutation_importance must be a Mapping or None"
            )
        allowed = {
            "scoring",
            "n_repeats",
            "random_state",
            "permutation_scope",
        }
        unknown = [key for key in values if key not in allowed]
        if unknown:
            raise MLExperimentConfigError(
                f"unknown permutation_importance field(s): {unknown!r}"
            )
        return cls(
            scoring=values.get("scoring", "rmse"),  # type: ignore[arg-type]
            n_repeats=values.get("n_repeats", 5),  # type: ignore[arg-type]
            random_state=values.get("random_state", 42),  # type: ignore[arg-type]
            permutation_scope=values.get(
                "permutation_scope", "within_trade_date"
            ),  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "scoring": self.scoring,
            "n_repeats": self.n_repeats,
            "random_state": self.random_state,
            "permutation_scope": self.permutation_scope,
        }


@dataclass(frozen=True)
class MLExperimentConfig:
    """Immutable complete configuration for one in-memory experiment."""

    dataset_config: MLDatasetConfig
    walk_forward_config: WalkForwardConfig
    training_config: WalkForwardTrainingConfig
    evaluation_config: ModelEvaluationConfig
    permutation_importance: PermutationImportanceOptionsConfig | None = None

    def __post_init__(self) -> None:
        expected = (
            ("dataset_config", self.dataset_config, MLDatasetConfig),
            (
                "walk_forward_config",
                self.walk_forward_config,
                WalkForwardConfig,
            ),
            (
                "training_config",
                self.training_config,
                WalkForwardTrainingConfig,
            ),
            (
                "evaluation_config",
                self.evaluation_config,
                ModelEvaluationConfig,
            ),
        )
        for name, value, expected_type in expected:
            if not isinstance(value, expected_type):
                raise MLExperimentConfigError(
                    f"{name} must be {expected_type.__name__}"
                )
        if self.permutation_importance is not None and not isinstance(
            self.permutation_importance,
            PermutationImportanceOptionsConfig,
        ):
            raise MLExperimentConfigError(
                "permutation_importance must be "
                "PermutationImportanceOptionsConfig or None"
            )
        json.dumps(self.as_dict(), allow_nan=False)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "MLExperimentConfig":
        if not isinstance(values, Mapping):
            raise MLExperimentConfigError(
                "experiment config must be a Mapping"
            )
        allowed = {
            "dataset",
            "walk_forward",
            "training",
            "evaluation",
            "permutation_importance",
        }
        unknown = [key for key in values if key not in allowed]
        if unknown:
            raise MLExperimentConfigError(
                f"unknown experiment config field(s): {unknown!r}"
            )
        for required in ("walk_forward", "training"):
            if required not in values:
                raise MLExperimentConfigError(
                    f"experiment config requires {required}"
                )
        try:
            dataset_config = cls._dataset_config(values.get("dataset"))
            walk_forward_config = cls._walk_forward_config(
                values["walk_forward"]
            )
            training_config = cls._training_config(values["training"])
            evaluation_config = cls._evaluation_config(
                values.get("evaluation")
            )
            importance = cls._importance_config(
                values.get("permutation_importance")
            )
        except MLExperimentConfigError:
            raise
        except Exception as exc:
            raise MLExperimentConfigError(
                f"invalid nested experiment config; cause={type(exc).__name__}"
            ) from exc
        return cls(
            dataset_config,
            walk_forward_config,
            training_config,
            evaluation_config,
            importance,
        )

    @staticmethod
    def _dataset_config(value: object) -> MLDatasetConfig:
        if value is None:
            return MLDatasetConfig()
        if isinstance(value, MLDatasetConfig):
            return value
        if not isinstance(value, Mapping):
            raise MLExperimentConfigError(
                "dataset must be a Mapping, MLDatasetConfig, or None"
            )
        unknown = [key for key in value if key != "label_col"]
        if unknown:
            raise MLExperimentConfigError(
                f"unknown dataset field(s): {unknown!r}"
            )
        return MLDatasetConfig(
            label_col=value.get("label_col", "forward_return")  # type: ignore[arg-type]
        )

    @staticmethod
    def _walk_forward_config(value: object) -> WalkForwardConfig:
        if isinstance(value, WalkForwardConfig):
            return value
        if not isinstance(value, Mapping):
            raise MLExperimentConfigError(
                "walk_forward must be a Mapping or WalkForwardConfig"
            )
        allowed = {
            "train_window_periods",
            "validation_periods",
            "window_type",
            "retrain_frequency",
            "embargo_periods",
        }
        unknown = [key for key in value if key not in allowed]
        if unknown:
            raise MLExperimentConfigError(
                f"unknown walk_forward field(s): {unknown!r}"
            )
        if not {"train_window_periods", "validation_periods"} <= set(value):
            raise MLExperimentConfigError(
                "walk_forward requires train_window_periods and "
                "validation_periods"
            )
        return WalkForwardConfig(
            train_window_periods=value["train_window_periods"],  # type: ignore[arg-type]
            validation_periods=value["validation_periods"],  # type: ignore[arg-type]
            window_type=value.get("window_type", "rolling"),  # type: ignore[arg-type]
            retrain_frequency=value.get("retrain_frequency", 1),  # type: ignore[arg-type]
            embargo_periods=value.get("embargo_periods", 0),  # type: ignore[arg-type]
        )

    @staticmethod
    def _training_config(value: object) -> WalkForwardTrainingConfig:
        if isinstance(value, WalkForwardTrainingConfig):
            return value
        if not isinstance(value, Mapping):
            raise MLExperimentConfigError(
                "training must be a Mapping or WalkForwardTrainingConfig"
            )
        return WalkForwardTrainingConfig.from_dict(value)

    @staticmethod
    def _evaluation_config(value: object) -> ModelEvaluationConfig:
        if value is None:
            return ModelEvaluationConfig()
        if isinstance(value, ModelEvaluationConfig):
            return value
        if not isinstance(value, Mapping):
            raise MLExperimentConfigError(
                "evaluation must be a Mapping, ModelEvaluationConfig, or None"
            )
        return ModelEvaluationConfig.from_dict(value)

    @staticmethod
    def _importance_config(
        value: object,
    ) -> PermutationImportanceOptionsConfig | None:
        if value is None:
            return None
        if isinstance(value, PermutationImportanceOptionsConfig):
            return value
        if not isinstance(value, Mapping):
            raise MLExperimentConfigError(
                "permutation_importance must be a Mapping, options config, "
                "or None"
            )
        return PermutationImportanceOptionsConfig.from_dict(value)

    def as_dict(self) -> dict[str, object]:
        result = {
            "dataset": {"label_col": self.dataset_config.label_col},
            "walk_forward": self.walk_forward_config.as_dict(),
            "training": self.training_config.as_dict(),
            "evaluation": self.evaluation_config.as_dict(),
            "permutation_importance": (
                None
                if self.permutation_importance is None
                else self.permutation_importance.as_dict()
            ),
        }
        return deepcopy(result)


def _iso(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class MLExperimentAudit:
    """Immutable summary of the complete in-memory stage chain."""

    model_name: str
    resolved_model_parameters: tuple[tuple[str, object], ...]
    input_rows: int
    dataset_rows: int
    n_features: int
    feature_names: tuple[str, ...]
    n_folds: int
    n_prediction_rows: int
    n_prediction_dates: int
    first_prediction_date: pd.Timestamp
    last_prediction_date: pd.Timestamp
    evaluation_completed: bool
    permutation_importance_enabled: bool
    permutation_importance_completed: bool
    permutation_importance_scoring: str | None
    permutation_importance_n_repeats: int | None
    stage_sequence: tuple[str, ...]
    config: MLExperimentConfig

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise MLExperimentIntegrityError("model_name must be non-empty")
        for name in (
            "input_rows",
            "dataset_rows",
            "n_features",
            "n_folds",
            "n_prediction_rows",
            "n_prediction_dates",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) <= 0
            ):
                raise MLExperimentIntegrityError(
                    f"{name} must be a positive integer"
                )
            object.__setattr__(self, name, int(value))
        names = tuple(self.feature_names)
        if len(names) != self.n_features or len(set(names)) != len(names):
            raise MLExperimentIntegrityError("feature_names are inconsistent")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(
            self,
            "resolved_model_parameters",
            tuple(self.resolved_model_parameters),
        )
        first = pd.Timestamp(self.first_prediction_date)
        last = pd.Timestamp(self.last_prediction_date)
        if pd.isna(first) or pd.isna(last) or first > last:
            raise MLExperimentIntegrityError(
                "prediction date range is invalid"
            )
        object.__setattr__(self, "first_prediction_date", first)
        object.__setattr__(self, "last_prediction_date", last)
        if not self.evaluation_completed:
            raise MLExperimentIntegrityError(
                "successful evaluation must be completed"
            )
        enabled = self.permutation_importance_enabled
        completed = self.permutation_importance_completed
        if enabled:
            if (
                not completed
                or self.permutation_importance_scoring not in {"rmse", "mae"}
                or self.permutation_importance_n_repeats is None
            ):
                raise MLExperimentIntegrityError(
                    "enabled importance audit is incomplete"
                )
        elif (
            completed
            or self.permutation_importance_scoring is not None
            or self.permutation_importance_n_repeats is not None
        ):
            raise MLExperimentIntegrityError(
                "disabled importance audit fields must be empty"
            )
        expected = (
            "dataset_build",
            "walk_forward_split",
            "training",
            "evaluation",
        )
        tail = ("integrity_validation", "result_build")
        expected_sequence = (
            expected + ("permutation_importance",) + tail
            if enabled
            else expected + tail
        )
        if tuple(self.stage_sequence) != expected_sequence:
            raise MLExperimentIntegrityError("stage_sequence is invalid")
        object.__setattr__(self, "stage_sequence", expected_sequence)
        if not isinstance(self.config, MLExperimentConfig):
            raise MLExperimentIntegrityError(
                "config must be MLExperimentConfig"
            )
        json.dumps(self.as_dict(), allow_nan=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "resolved_model_parameters": dict(
                self.resolved_model_parameters
            ),
            "input_rows": self.input_rows,
            "dataset_rows": self.dataset_rows,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names),
            "n_folds": self.n_folds,
            "n_prediction_rows": self.n_prediction_rows,
            "n_prediction_dates": self.n_prediction_dates,
            "first_prediction_date": _iso(self.first_prediction_date),
            "last_prediction_date": _iso(self.last_prediction_date),
            "evaluation_completed": self.evaluation_completed,
            "permutation_importance_enabled": (
                self.permutation_importance_enabled
            ),
            "permutation_importance_completed": (
                self.permutation_importance_completed
            ),
            "permutation_importance_scoring": (
                self.permutation_importance_scoring
            ),
            "permutation_importance_n_repeats": (
                self.permutation_importance_n_repeats
            ),
            "stage_sequence": list(self.stage_sequence),
            "config": self.config.as_dict(),
        }


class MLExperimentResult:
    """Hold the real stage outputs without raw inputs, datasets, or models."""

    def __init__(
        self,
        dataset_audit: MLDatasetAudit,
        walk_forward_plan: WalkForwardPlan,
        training_result: WalkForwardTrainingResult,
        evaluation_result: ModelEvaluationResult,
        permutation_importance_result: (
            WalkForwardPermutationImportanceResult | None
        ),
        audit: MLExperimentAudit,
    ) -> None:
        checks = (
            (dataset_audit, MLDatasetAudit, "dataset_audit"),
            (walk_forward_plan, WalkForwardPlan, "walk_forward_plan"),
            (
                training_result,
                WalkForwardTrainingResult,
                "training_result",
            ),
            (
                evaluation_result,
                ModelEvaluationResult,
                "evaluation_result",
            ),
            (audit, MLExperimentAudit, "audit"),
        )
        for value, expected, name in checks:
            if not isinstance(value, expected):
                raise MLExperimentIntegrityError(
                    f"{name} must be {expected.__name__}"
                )
        if permutation_importance_result is not None and not isinstance(
            permutation_importance_result,
            WalkForwardPermutationImportanceResult,
        ):
            raise MLExperimentIntegrityError(
                "permutation_importance_result has invalid type"
            )
        self._dataset_audit = dataset_audit
        self._walk_forward_plan = deepcopy(walk_forward_plan)
        self._training_result = training_result
        self._evaluation_result = evaluation_result
        self._permutation_importance_result = permutation_importance_result
        self._audit = audit

    @property
    def dataset_audit(self) -> MLDatasetAudit:
        return self._dataset_audit

    @property
    def walk_forward_plan(self) -> WalkForwardPlan:
        return deepcopy(self._walk_forward_plan)

    @property
    def training_result(self) -> WalkForwardTrainingResult:
        return self._training_result

    @property
    def evaluation_result(self) -> ModelEvaluationResult:
        return self._evaluation_result

    @property
    def permutation_importance_result(
        self,
    ) -> WalkForwardPermutationImportanceResult | None:
        return self._permutation_importance_result

    @property
    def audit(self) -> MLExperimentAudit:
        return self._audit


class MLExperimentRunner:
    """Run public V3 stages sequentially in memory."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        if registry is not None and not isinstance(registry, ModelRegistry):
            raise MLExperimentConfigError(
                "registry must be ModelRegistry or None"
            )
        self._registry = (
            create_default_model_registry() if registry is None else registry
        )

    def run(
        self, frame: pd.DataFrame, config: MLExperimentConfig
    ) -> MLExperimentResult:
        if not isinstance(frame, pd.DataFrame):
            raise MLExperimentDataError("frame must be a pandas DataFrame")
        if frame.empty:
            raise MLExperimentDataError("frame must not be empty")
        if not isinstance(config, MLExperimentConfig):
            raise MLExperimentConfigError(
                "config must be MLExperimentConfig"
            )
        input_frame = frame.copy(deep=True)
        input_rows = len(input_frame)
        stages: list[str] = []
        feature_names = self._feature_names(
            input_frame, config.dataset_config.label_col
        )

        stages.append("dataset_build")
        try:
            dataset = MLDatasetBuilder(config.dataset_config).build(
                input_frame,
                input_frame,
                feature_names,
            )
        except Exception as exc:
            raise MLExperimentStageError("dataset_build", exc) from exc
        if not isinstance(dataset, MLDataset):
            raise MLExperimentDataError(
                "dataset_build did not return MLDataset"
            )

        stages.append("walk_forward_split")
        try:
            plan = WalkForwardSplitter(
                config.walk_forward_config
            ).build(dataset)
        except Exception as exc:
            raise MLExperimentStageError(
                "walk_forward_split", exc
            ) from exc
        if not isinstance(plan, WalkForwardPlan) or not plan.splits:
            raise MLExperimentDataError(
                "walk_forward_split did not return a non-empty plan"
            )

        stages.append("training")
        try:
            training = WalkForwardTrainer(self._registry).run(
                dataset, plan, config.training_config
            )
        except Exception as exc:
            raise MLExperimentStageError("training", exc) from exc
        if not isinstance(training, WalkForwardTrainingResult):
            raise MLExperimentDataError(
                "training did not return WalkForwardTrainingResult"
            )

        stages.append("evaluation")
        try:
            evaluation = OOSModelEvaluator(
                config.evaluation_config
            ).evaluate(training)
        except Exception as exc:
            raise MLExperimentStageError("evaluation", exc) from exc
        if not isinstance(evaluation, ModelEvaluationResult):
            raise MLExperimentDataError(
                "evaluation did not return ModelEvaluationResult"
            )

        importance = None
        options = config.permutation_importance
        if options is not None:
            stages.append("permutation_importance")
            importance_config = WalkForwardPermutationImportanceConfig(
                model_name=config.training_config.model_name,
                model_params=config.training_config.model_params,
                scoring=options.scoring,
                n_repeats=options.n_repeats,
                random_state=options.random_state,
                permutation_scope=options.permutation_scope,
            )
            try:
                importance = WalkForwardPermutationImportanceRunner(
                    self._registry
                ).run(dataset, plan, importance_config)
            except Exception as exc:
                raise MLExperimentStageError(
                    "permutation_importance", exc
                ) from exc
            if not isinstance(
                importance, WalkForwardPermutationImportanceResult
            ):
                raise MLExperimentDataError(
                    "importance stage returned an invalid result"
                )

        stages.append("integrity_validation")
        self._validate_integrity(
            dataset, plan, training, evaluation, importance, config
        )
        stages.append("result_build")
        audit = MLExperimentAudit(
            model_name=training.audit.model_name,
            resolved_model_parameters=(
                training.audit.resolved_model_parameters
            ),
            input_rows=input_rows,
            dataset_rows=dataset.n_samples,
            n_features=dataset.n_features,
            feature_names=dataset.feature_names,
            n_folds=training.audit.n_folds,
            n_prediction_rows=training.audit.n_prediction_rows,
            n_prediction_dates=training.audit.n_prediction_dates,
            first_prediction_date=training.audit.first_prediction_date,
            last_prediction_date=training.audit.last_prediction_date,
            evaluation_completed=True,
            permutation_importance_enabled=options is not None,
            permutation_importance_completed=importance is not None,
            permutation_importance_scoring=(
                None if options is None else options.scoring
            ),
            permutation_importance_n_repeats=(
                None if options is None else options.n_repeats
            ),
            stage_sequence=tuple(stages),
            config=config,
        )
        try:
            return MLExperimentResult(
                dataset.audit,
                plan,
                training,
                evaluation,
                importance,
                audit,
            )
        except Exception as exc:
            raise MLExperimentStageError("result_build", exc) from exc

    @staticmethod
    def _feature_names(frame: pd.DataFrame, label_col: str) -> tuple[str, ...]:
        reserved = {
            *METADATA_COLUMNS,
            "entry_price",
            "exit_price",
            label_col,
        }
        names = tuple(column for column in frame.columns if column not in reserved)
        if not names:
            raise MLExperimentDataError(
                "frame must contain at least one feature column"
            )
        return names

    @staticmethod
    def _validate_integrity(
        dataset: MLDataset,
        plan: WalkForwardPlan,
        training: WalkForwardTrainingResult,
        evaluation: ModelEvaluationResult,
        importance: WalkForwardPermutationImportanceResult | None,
        config: MLExperimentConfig,
    ) -> None:
        train_audit = training.audit
        eval_audit: ModelEvaluationAudit = evaluation.audit
        if plan.n_splits != train_audit.n_folds:
            raise MLExperimentIntegrityError(
                "Plan and Training fold counts differ"
            )
        fold_features = tuple(
            fold.model_fit_audit.feature_names
            for fold in train_audit.fold_audits
        )
        if not fold_features or any(
            names != dataset.feature_names for names in fold_features
        ):
            raise MLExperimentIntegrityError(
                "Training feature names differ from Dataset"
            )
        if (
            eval_audit.model_name != train_audit.model_name
            or eval_audit.n_rows != train_audit.n_prediction_rows
            or eval_audit.n_dates != train_audit.n_prediction_dates
            or eval_audit.n_folds != train_audit.n_folds
            or eval_audit.first_date != train_audit.first_prediction_date
            or eval_audit.last_date != train_audit.last_prediction_date
            or eval_audit.row_coverage != 1.0
            or eval_audit.date_coverage != 1.0
        ):
            raise MLExperimentIntegrityError(
                "Evaluation and Training outputs differ"
            )
        options = config.permutation_importance
        if options is None:
            if importance is not None:
                raise MLExperimentIntegrityError(
                    "disabled importance must not produce a result"
                )
            return
        if importance is None:
            raise MLExperimentIntegrityError(
                "enabled importance did not produce a result"
            )
        importance_audit = importance.audit
        if (
            importance_audit.model_name != train_audit.model_name
            or importance_audit.resolved_model_parameters
            != train_audit.resolved_model_parameters
            or importance_audit.n_folds != train_audit.n_folds
            or importance_audit.feature_names != dataset.feature_names
            or importance_audit.first_prediction_date
            != train_audit.first_prediction_date
            or importance_audit.last_prediction_date
            != train_audit.last_prediction_date
            or importance_audit.scoring != options.scoring
            or importance_audit.n_repeats != options.n_repeats
            or importance_audit.permutation_scope
            != options.permutation_scope
        ):
            raise MLExperimentIntegrityError(
                "Importance and Training outputs differ"
            )
