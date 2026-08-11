"""Gate G tests for UI-only minimum-variance config mapping."""

from pathlib import Path

import pytest

from app.services.pipeline_config_service import (
    LEDOIT_WOLF_LABEL,
    MINIMUM_VARIANCE_LABEL,
    PORTFOLIO_METHOD_BY_LABEL,
    RISK_ESTIMATOR_BY_LABEL,
    SAMPLE_COVARIANCE_LABEL,
    SUGGESTED_RISK_MODEL_LOOKBACK,
    SUGGESTED_RISK_MODEL_MIN_OBSERVATIONS,
    build_portfolio_construction_ui_config,
)


@pytest.mark.parametrize(
    ("label", "estimator"),
    [
        (SAMPLE_COVARIANCE_LABEL, "sample_covariance"),
        (LEDOIT_WOLF_LABEL, "ledoit_wolf"),
    ],
)
def test_minimum_variance_estimator_controls_map_to_exact_nested_config(label, estimator):
    config = build_portfolio_construction_ui_config(
        method_label=MINIMUM_VARIANCE_LABEL,
        risk_model_estimator_label=label,
        risk_model_lookback=90,
        risk_model_min_observations=60,
        max_weight_enabled=True,
        max_weight_percent=20.0,
    )
    assert config.to_dict() == {
        "method": "minimum_variance",
        "params": {"risk_model": {
            "estimator": estimator,
            "params": {},
            "lookback_trading_days": 90,
            "min_observations": 60,
        }},
        "constraints": [
            {"type": "max_weight", "params": {"max_weight": 0.2}}
        ],
    }


def test_ui_suggestions_are_explicit_and_not_backend_defaults():
    assert SUGGESTED_RISK_MODEL_LOOKBACK == 120
    assert SUGGESTED_RISK_MODEL_MIN_OBSERVATIONS == 80
    assert PORTFOLIO_METHOD_BY_LABEL[MINIMUM_VARIANCE_LABEL] == "minimum_variance"
    assert RISK_ESTIMATOR_BY_LABEL == {
        SAMPLE_COVARIANCE_LABEL: "sample_covariance",
        LEDOIT_WOLF_LABEL: "ledoit_wolf",
    }


def test_unknown_risk_estimator_is_rejected_by_ui_bridge():
    with pytest.raises(ValueError, match="risk estimator"):
        build_portfolio_construction_ui_config(
            method_label=MINIMUM_VARIANCE_LABEL,
            risk_model_estimator_label="unknown",
        )


def test_streamlit_contains_controls_but_no_risk_or_optimizer_math():
    root = Path(__file__).parents[1] / "app"
    source = "\n".join((root / path).read_text(encoding="utf-8") for path in (
        "streamlit_app.py", "views/new_run.py", "i18n/catalog.py",
        "services/pipeline_config_service.py",
    ))
    for required in (
        "MINIMUM_VARIANCE_LABEL",
        "Risk Estimator",
        "Risk Lookback Trading Days",
        "Risk Minimum Observations",
    ):
        assert required in source
    for forbidden in (
        "np.cov",
        "DataFrame.cov",
        "LedoitWolf(",
        "scipy.optimize",
        "risk_aversion",
        "expected_return",
    ):
        assert forbidden not in source
