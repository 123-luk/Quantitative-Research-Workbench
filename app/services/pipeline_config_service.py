"""UI-independent configuration bridge for the canonical V5 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

from src.pipeline.config import PipelineConfig
from src.pipeline.holdings_config import HoldingsPipelineConfig


HIGH_SCORE_FIRST: Final = "分数越高越优"
LOW_SCORE_FIRST: Final = "分数越低越优"
ERROR_IF_INSUFFICIENT: Final = "报错"
USE_ALL_VALID: Final = "使用全部有效股票"
EQUAL_WEIGHT_LABEL: Final = "等权"

SIGNAL_DIRECTION_BY_LABEL: Final = {
    HIGH_SCORE_FIRST: "descending",
    LOW_SCORE_FIRST: "ascending",
}
INSUFFICIENT_POLICY_BY_LABEL: Final = {
    ERROR_IF_INSUFFICIENT: "error",
    USE_ALL_VALID: "allow_partial",
}


def get_default_holdings_top_n() -> int:
    """Return the canonical backend default used by the V5 UI widget."""
    return HoldingsPipelineConfig().top_n


def load_canonical_base_config(config_path: str | Path) -> PipelineConfig:
    """Load one direct-schema canonical PipelineConfig YAML file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError(f"Canonical UI config must be a YAML mapping: {path}")
    return PipelineConfig.from_dict(values)


def build_effective_pipeline_config(
    base_config: PipelineConfig,
    *,
    top_n: int | None = None,
    signal_direction_label: str = HIGH_SCORE_FIRST,
    insufficient_policy_label: str = ERROR_IF_INSUFFICIENT,
) -> PipelineConfig:
    """Apply the small V5 UI surface to a detached canonical config."""
    if not isinstance(base_config, PipelineConfig):
        raise TypeError("base_config must be a PipelineConfig.")
    try:
        signal_direction = SIGNAL_DIRECTION_BY_LABEL[signal_direction_label]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Signal direction option: {signal_direction_label!r}."
        ) from exc
    try:
        insufficient_policy = INSUFFICIENT_POLICY_BY_LABEL[
            insufficient_policy_label
        ]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported insufficient-universe option: {insufficient_policy_label!r}."
        ) from exc

    effective_top_n = get_default_holdings_top_n() if top_n is None else top_n
    values = base_config.to_dict()
    signal = dict(values["signal"])
    signal.update(
        {
            "enabled": True,
            "signal_direction": signal_direction,
        }
    )
    holdings = dict(values["holdings"])
    holdings.update(
        {
            "enabled": True,
            "top_n": effective_top_n,
            "insufficient_universe_policy": insufficient_policy,
            "weighting": "equal_weight",
        }
    )
    values["signal"] = signal
    values["holdings"] = holdings

    # PipelineConfig currently requires the legacy root field to equal enabled
    # holdings.top_n. This is a one-way compatibility mirror: UI input and V5
    # execution read only holdings.top_n, so the root never becomes a second truth.
    values["top_n"] = effective_top_n
    return PipelineConfig.from_dict(values)


def build_selection_summary(config: PipelineConfig) -> dict[str, object]:
    """Build the pre-run display strictly from the effective config."""
    if not config.signal.enabled or not config.holdings.enabled:
        raise ValueError("Effective V5 config must enable Signal and Holdings.")
    direction_labels = {
        backend: label for label, backend in SIGNAL_DIRECTION_BY_LABEL.items()
    }
    policy_labels = {
        backend: label for label, backend in INSUFFICIENT_POLICY_BY_LABEL.items()
    }
    return {
        "Top N": config.holdings.top_n,
        "Signal 排序": direction_labels[config.signal.signal_direction],
        "股票不足 N": policy_labels[
            config.holdings.insufficient_universe_policy
        ],
        "权重方式": EQUAL_WEIGHT_LABEL,
        "source mode": config.signal.source.mode,
    }
