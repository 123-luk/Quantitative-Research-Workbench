"""Pure helpers for the opt-in ML options of the unified Pipeline CLI."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math

from src.ml import MLArtifactExistsError
from src.pipeline.ml_config import (
    MLExperimentPipelineConfig,
    MLPipelineConfigError,
    MLPipelineError,
)
from src.pipeline.ml_execution import (
    MLPipelineArtifactError,
    MLPipelineExecutionError,
    MLPipelineIntegrityError,
    MLPipelinePanelError,
)


class MLCLIError(Exception):
    """Base error for ML-specific command-line configuration."""


class MLCLIConfigError(MLCLIError):
    """Raised when explicit ML command-line values are invalid."""


_OVERRIDE_KEYS = {
    "enabled",
    "panel_path",
    "model_name",
    "model_params",
    "permutation_importance_enabled",
    "importance_repeats",
    "importance_scoring",
    "minimum_cross_section_size",
    "save_artifacts",
    "artifact_root",
    "experiment_id",
    "parquet_compression",
}


def _reject_json_constant(value: str) -> object:
    raise MLCLIConfigError(
        f"model_params JSON contains forbidden constant {value}"
    )


def parse_ml_model_params(raw: str) -> dict[str, object]:
    """Parse a strict JSON object without evaluating Python expressions."""
    if not isinstance(raw, str) or not raw.strip():
        raise MLCLIConfigError(
            "model_params must be a non-empty JSON object"
        )
    try:
        parsed = json.loads(raw, parse_constant=_reject_json_constant)
    except MLCLIConfigError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MLCLIConfigError("model_params is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise MLCLIConfigError(
            "model_params JSON top level must be an object"
        )
    if any(not isinstance(key, str) for key in parsed):
        raise MLCLIConfigError("model_params keys must be strings")
    try:
        serialized = json.dumps(parsed, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MLCLIConfigError(
            "model_params must be strictly JSON serializable"
        ) from exc
    return json.loads(serialized)


def _experiment_mapping(
    merged: dict[str, object],
    *,
    required_for: str,
) -> dict[str, object]:
    experiment = merged.get("experiment")
    if not isinstance(experiment, Mapping):
        raise MLCLIConfigError(
            f"{required_for} requires an existing experiment configuration"
        )
    return deepcopy(dict(experiment))


def _nested_mapping(
    values: Mapping[str, object],
    key: str,
    *,
    required_for: str,
) -> dict[str, object]:
    nested = values.get(key)
    if not isinstance(nested, Mapping):
        raise MLCLIConfigError(
            f"{required_for} requires experiment.{key}"
        )
    return deepcopy(dict(nested))


def merge_ml_cli_overrides(
    config: MLExperimentPipelineConfig,
    overrides: Mapping[str, object],
) -> MLExperimentPipelineConfig:
    """Apply explicit leaf overrides and rebuild through the public config."""
    if not isinstance(config, MLExperimentPipelineConfig):
        raise MLCLIConfigError(
            "config must be MLExperimentPipelineConfig"
        )
    if not isinstance(overrides, Mapping):
        raise MLCLIConfigError("ML CLI overrides must be a Mapping")
    if any(not isinstance(key, str) for key in overrides):
        raise MLCLIConfigError("ML CLI override keys must be strings")
    unknown = sorted(set(overrides) - _OVERRIDE_KEYS)
    if unknown:
        raise MLCLIConfigError(
            "unknown ML CLI override field(s): " + ", ".join(unknown)
        )

    values = deepcopy(dict(overrides))
    merged = config.to_dict()
    for key in (
        "enabled",
        "panel_path",
        "save_artifacts",
        "artifact_root",
        "experiment_id",
        "parquet_compression",
    ):
        if key in values:
            merged[key] = deepcopy(values[key])

    experiment_leaf_keys = {
        "model_name",
        "model_params",
        "minimum_cross_section_size",
        "permutation_importance_enabled",
        "importance_repeats",
        "importance_scoring",
    }
    active_leaf_keys = experiment_leaf_keys.intersection(values)
    experiment: dict[str, object] | None = None
    if active_leaf_keys:
        experiment = _experiment_mapping(
            merged,
            required_for=", ".join(sorted(active_leaf_keys)),
        )

    if "model_name" in values or "model_params" in values:
        if experiment is None:
            raise MLCLIConfigError("model override requires experiment")
        training = _nested_mapping(
            experiment,
            "training",
            required_for="model override",
        )
        if "model_name" in values:
            training["model_name"] = deepcopy(values["model_name"])
        if "model_params" in values:
            training["model_params"] = deepcopy(values["model_params"])
        experiment["training"] = training

    if "minimum_cross_section_size" in values:
        if experiment is None:
            raise MLCLIConfigError(
                "minimum_cross_section_size requires experiment"
            )
        evaluation = _nested_mapping(
            experiment,
            "evaluation",
            required_for="minimum_cross_section_size",
        )
        evaluation["minimum_cross_section_size"] = deepcopy(
            values["minimum_cross_section_size"]
        )
        experiment["evaluation"] = evaluation

    importance_enabled = values.get("permutation_importance_enabled")
    leaf_requested = any(
        key in values for key in ("importance_repeats", "importance_scoring")
    )
    if (
        "permutation_importance_enabled" in values
        and not isinstance(importance_enabled, bool)
    ):
        raise MLCLIConfigError(
            "permutation_importance_enabled must be a bool"
        )
    if importance_enabled is False and leaf_requested:
        raise MLCLIConfigError(
            "importance leaf overrides conflict with explicit disable"
        )
    if experiment is not None and "permutation_importance_enabled" in values:
        if importance_enabled is False:
            experiment["permutation_importance"] = None
        elif experiment.get("permutation_importance") is None:
            experiment["permutation_importance"] = {
                "scoring": "rmse",
                "n_repeats": 5,
                "random_state": 42,
                "permutation_scope": "within_trade_date",
            }

    if leaf_requested:
        if experiment is None:
            raise MLCLIConfigError(
                "importance overrides require experiment configuration"
            )
        importance = experiment.get("permutation_importance")
        if not isinstance(importance, Mapping):
            raise MLCLIConfigError(
                "importance overrides require permutation importance "
                "to be enabled in YAML or CLI"
            )
        importance_values = deepcopy(dict(importance))
        if "importance_repeats" in values:
            importance_values["n_repeats"] = deepcopy(
                values["importance_repeats"]
            )
        if "importance_scoring" in values:
            importance_values["scoring"] = deepcopy(
                values["importance_scoring"]
            )
        experiment["permutation_importance"] = importance_values

    if experiment is not None:
        merged["experiment"] = experiment
    try:
        return MLExperimentPipelineConfig.from_dict(merged)
    except MLPipelineConfigError as exc:
        raise MLCLIConfigError("merged ML configuration is invalid") from exc


def _metric(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.6g}" if math.isfinite(number) else "N/A"


def format_ml_human_summary(
    summary: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """Format only compact public ML summary fields without side effects."""
    if summary is None:
        return ()
    if not isinstance(summary, Mapping):
        raise MLCLIConfigError("ML summary must be a Mapping or None")
    if summary.get("enabled") is not True:
        return ()
    if summary.get("r2_valid") is True:
        r2_text = _metric(summary.get("r2"))
    else:
        reason = summary.get("r2_invalid_reason")
        r2_text = "N/A" if not reason else f"N/A ({reason})"
    importance = (
        "completed"
        if summary.get("permutation_importance_completed") is True
        else "not run"
    )
    lines = [
        f"ML model: {summary.get('model_name')}",
        f"ML folds: {summary.get('n_folds')}",
        f"ML prediction rows: {summary.get('n_prediction_rows')}",
        f"ML prediction dates: {summary.get('n_prediction_dates')}",
        f"ML MAE: {_metric(summary.get('mae'))}",
        f"ML RMSE: {_metric(summary.get('rmse'))}",
        f"ML R²: {r2_text}",
        f"ML Pearson IC mean: {_metric(summary.get('pearson_ic_mean'))}",
        f"ML RankIC mean: {_metric(summary.get('rank_ic_mean'))}",
        f"ML permutation importance: {importance}",
    ]
    artifact_dir = summary.get("artifact_dir")
    lines.append(
        f"ML artifact directory: {artifact_dir}"
        if summary.get("artifacts_saved") is True and artifact_dir
        else "ML artifacts: not saved"
    )
    return tuple(lines)


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    pending = [error]
    seen: set[int] = set()
    result: list[BaseException] = []
    while pending and len(result) < 32:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(current)
        if isinstance(current.__cause__, BaseException):
            pending.append(current.__cause__)
        if isinstance(current.__context__, BaseException):
            pending.append(current.__context__)
    return tuple(result)


def exit_code_for_ml_error(error: BaseException) -> int:
    """Map public ML CLI/Pipeline exception types to stable exit codes."""
    if not isinstance(error, BaseException):
        return 1
    chain = _exception_chain(error)
    if any(isinstance(item, MLArtifactExistsError) for item in chain):
        return 5
    if any(
        isinstance(item, (MLCLIError, MLPipelineConfigError))
        for item in chain
    ):
        return 2
    if any(isinstance(item, MLPipelinePanelError) for item in chain):
        return 3
    if any(isinstance(item, MLPipelineArtifactError) for item in chain):
        return 6
    if any(
        isinstance(
            item,
            (
                MLPipelineExecutionError,
                MLPipelineIntegrityError,
                MLPipelineError,
            ),
        )
        for item in chain
    ):
        return 4
    return 1
