"""Tests for same-execution Factor Research published outputs."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.factors.composition import FactorCompositionConfig
from src.factors.evaluation import FactorEvaluationConfig
from src.factors.forward_returns import ForwardReturnConfig
from src.factors.preprocessing import PreprocessingConfig
from src.factors.quantile_evaluation import QuantileEvaluationConfig
from src.factors.research_pipeline import FactorResearchConfig
from src.modeling_panel import ModelingPanelBuilder, ModelingPanelConfig
from src.pipeline import (
    FactorResearchPipelineConfig,
    FactorResearchPipelineExecutor,
    FactorResearchPublishedOutputs,
)
from src.pipeline.research_execution import _resolve_published_output_path


FEATURES = ("momentum_20d", "volatility_20d")


def _published_files(root: Path) -> tuple[Path, Path]:
    tables = root / "tables"
    tables.mkdir(parents=True)
    panel = tables / "final_factor_panel.parquet"
    returns = tables / "forward_returns.parquet"
    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"]),
            "ts_code": ["A"],
            "momentum_20d": [1.0],
            "volatility_20d": [2.0],
        }
    ).to_parquet(panel, index=False)
    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"]),
            "ts_code": ["A"],
            "entry_trade_date": pd.to_datetime(["2024-01-02"]),
            "exit_trade_date": pd.to_datetime(["2024-01-03"]),
            "entry_price": [10.0],
            "exit_price": [11.0],
            "forward_return": [0.1],
        }
    ).to_parquet(returns, index=False)
    return panel, returns


def _outputs(root: Path) -> FactorResearchPublishedOutputs:
    panel, returns = _published_files(root)
    return FactorResearchPublishedOutputs(
        root, panel, returns, FEATURES, "forward_return"
    )


def _panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=32)
    codes = [f"S{index:02d}" for index in range(10)]
    factors: list[dict[str, object]] = []
    prices: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(dates):
        for stock_index, code in enumerate(codes):
            close = (
                100.0
                * (1.0 + 0.0015 * (stock_index + 1)) ** date_index
                * (1.0 + 0.002 * np.sin(date_index + stock_index))
            )
            prices.append(
                {"trade_date": trade_date, "ts_code": code, "close": close}
            )
            if date_index < 30:
                factors.append(
                    {"trade_date": trade_date, "ts_code": code, "close": close}
                )
    scores = pd.DataFrame(
        [
            {"trade_date": trade_date, "ts_code": code}
            for trade_date in dates[22:26]
            for code in codes
        ]
    )
    return pd.DataFrame(factors), scores, pd.DataFrame(prices)


def _write_inputs(root: Path) -> dict[str, Path]:
    root.mkdir()
    paths: dict[str, Path] = {}
    for name, frame in zip(("factor_input", "score_panel", "price_panel"), _panels()):
        path = root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    return paths


def _config(
    paths: dict[str, Path], *, label: str = "forward_return"
) -> FactorResearchPipelineConfig:
    return FactorResearchPipelineConfig(
        enabled=True,
        factor_input_path=str(paths["factor_input"]),
        score_panel_path=str(paths["score_panel"]),
        price_panel_path=str(paths["price_panel"]),
        research=FactorResearchConfig(
            factor_names=FEATURES,
            composition_method="equal",
        ),
        preprocessing=PreprocessingConfig(
            missing_method="none",
            winsor_method="none",
            standardize_method="zscore",
            min_cross_section_size=5,
        ),
        evaluation=FactorEvaluationConfig(
            return_col=label,
            min_cross_section_size=5,
        ),
        quantile=QuantileEvaluationConfig(
            return_col=label,
            quantiles=5,
            min_cross_section_size=5,
            min_group_size=1,
        ),
        composition=FactorCompositionConfig(method="equal"),
        forward_returns=ForwardReturnConfig(
            return_col=label,
            entry_lag_periods=1,
            holding_periods=1,
        ),
    )


def _execute(tmp_path: Path, name: str = "run", label: str = "forward_return"):
    paths = _write_inputs(tmp_path / f"{name}-inputs")
    run = tmp_path / name
    run.mkdir()
    return FactorResearchPipelineExecutor(
        _config(paths, label=label), project_root=tmp_path
    ).execute(run)


def test_published_outputs_contract_is_frozen_absolute_and_detached(
    tmp_path: Path,
) -> None:
    outputs = _outputs(tmp_path / "artifact")
    assert outputs.artifact_dir.is_absolute()
    assert outputs.final_factor_panel_path.is_absolute()
    assert outputs.forward_returns_path.is_absolute()
    payload = outputs.as_dict()
    json.dumps(payload)
    payload["feature_names"].append("changed")
    assert outputs.feature_names == FEATURES
    with pytest.raises(FrozenInstanceError):
        outputs.label_column = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("features", "label"),
    [
        ((), "forward_return"),
        (("x", "x"), "forward_return"),
        (("trade_date",), "forward_return"),
        (("entry_price",), "forward_return"),
        (("forward_return",), "forward_return"),
        (("x",), " "),
    ],
)
def test_published_outputs_rejects_invalid_metadata(
    tmp_path: Path, features: tuple[str, ...], label: str
) -> None:
    root = tmp_path / "artifact"
    panel, returns = _published_files(root)
    with pytest.raises(ValueError):
        FactorResearchPublishedOutputs(root, panel, returns, features, label)


def test_published_outputs_rejects_missing_outside_same_and_non_parquet(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    panel, returns = _published_files(root)
    with pytest.raises(ValueError, match="existing"):
        FactorResearchPublishedOutputs(
            root, root / "missing.parquet", returns, FEATURES, "forward_return"
        )
    outside = tmp_path / "outside.parquet"
    panel.replace(outside)
    with pytest.raises(ValueError, match="inside"):
        FactorResearchPublishedOutputs(
            root, outside, returns, FEATURES, "forward_return"
        )
    panel = root / "tables" / "same.parquet"
    returns.replace(panel)
    with pytest.raises(ValueError, match="different"):
        FactorResearchPublishedOutputs(
            root, panel, panel, FEATURES, "forward_return"
        )
    text = root / "tables" / "data.txt"
    text.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="parquet"):
        FactorResearchPublishedOutputs(
            root, panel, text, FEATURES, "forward_return"
        )


def test_manifest_path_resolution_is_exact_and_safe(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    panel, _ = _published_files(root)
    assert _resolve_published_output_path(
        root,
        {"relative_path": "tables/final_factor_panel.parquet"},
        "final_factor_panel",
    ) == panel
    for value in (
        "../final_factor_panel.parquet",
        "tables/../final_factor_panel.parquet",
        "C:/final_factor_panel.parquet",
        "https://host/final_factor_panel.parquet",
        "tables/not-final_factor_panel.parquet",
    ):
        with pytest.raises(RuntimeError):
            _resolve_published_output_path(
                root, {"relative_path": value}, "final_factor_panel"
            )


def test_manifest_resolution_uses_only_referenced_file(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _published_files(root)
    unreferenced = root / "final_factor_panel.parquet"
    pd.DataFrame({"fake": [1]}).to_parquet(unreferenced)
    with pytest.raises(RuntimeError, match="regular published file"):
        _resolve_published_output_path(
            root,
            {"relative_path": "tables/missing/final_factor_panel.parquet"},
            "final_factor_panel",
        )


def test_executor_publishes_same_execution_paths_and_dict(tmp_path: Path) -> None:
    result = _execute(tmp_path)
    outputs = result.published_outputs
    assert outputs is not None
    assert outputs.artifact_dir == Path(result.artifact_dir)
    assert outputs.feature_names == result.factor_names == FEATURES
    assert outputs.label_column == "forward_return"
    assert outputs.final_factor_panel_path.is_file()
    assert outputs.forward_returns_path.is_file()
    payload = result.to_dict()
    assert payload["published_outputs"] == outputs.as_dict()
    payload["published_outputs"]["feature_names"].append("changed")
    assert result.to_dict()["published_outputs"]["feature_names"] == list(FEATURES)


def test_two_executions_do_not_cross_paths_or_use_newer_decoy(
    tmp_path: Path,
) -> None:
    first = _execute(tmp_path, "first")
    decoy = tmp_path / "latest-fake"
    _published_files(decoy)
    second = _execute(tmp_path, "second")
    assert first.published_outputs is not None
    assert second.published_outputs is not None
    assert first.published_outputs.artifact_dir != second.published_outputs.artifact_dir
    assert first.published_outputs.artifact_dir.name == "factor_research"
    assert second.published_outputs.artifact_dir.name == "factor_research"
    assert decoy not in first.published_outputs.final_factor_panel_path.parents
    assert decoy not in second.published_outputs.final_factor_panel_path.parents


@pytest.mark.parametrize("label", ["forward_return", "custom_forward_label"])
def test_published_outputs_feed_modeling_panel_builder(
    tmp_path: Path, label: str
) -> None:
    execution = _execute(tmp_path, f"run-{label}", label)
    outputs = execution.published_outputs
    assert outputs is not None
    factors = pd.read_parquet(outputs.final_factor_panel_path)
    returns = pd.read_parquet(outputs.forward_returns_path)
    result = ModelingPanelBuilder(
        ModelingPanelConfig(
            label_column=outputs.label_column,
            include_features=outputs.feature_names,
        )
    ).build(factors, returns)
    assert result.feature_names == outputs.feature_names
    assert result.label_column == outputs.label_column
    assert tuple(result.panel.columns) == (
        "trade_date",
        "ts_code",
        *outputs.feature_names,
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
        outputs.label_column,
    )


def test_public_import() -> None:
    from src.pipeline import FactorResearchPublishedOutputs as PublicType

    assert PublicType is FactorResearchPublishedOutputs
