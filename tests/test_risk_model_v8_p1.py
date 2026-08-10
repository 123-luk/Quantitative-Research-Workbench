import json

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from src.portfolio_construction import HistoricalReturnWindow
from src.risk_model import (
    HistoricalCovarianceRiskModelService,
    LedoitWolfEstimator,
    RiskEstimate,
    RiskEstimatorRegistry,
    RiskModelConfig,
    RiskModelConfigError,
    RiskModelDataError,
    RiskModelRegistryError,
    RiskModelRequest,
    RiskModelResult,
    RiskModelValidationError,
    SampleCovarianceEstimator,
    build_default_risk_estimator_registry,
)


def config(estimator="sample_covariance", lookback=5, minimum=2):
    return RiskModelConfig(estimator, {}, lookback, minimum)


class Returns:
    def __init__(self, rows, cutoff="2024-01-05"):
        self.window = HistoricalReturnWindow(
            cutoff, pd.DataFrame(rows, columns=["trade_date", "ts_code", "return"])
        )
        self.calls = []

    def load_window(self, assets, formation, lookback):
        self.calls.append((assets, formation, lookback))
        return self.window


def rows():
    return [
        ("2024-01-02", "B", 0.02), ("2024-01-02", "A", 0.01),
        ("2024-01-03", "A", 0.03),
        ("2024-01-04", "A", -0.01), ("2024-01-04", "B", 0.00),
        ("2024-01-05", "A", 0.02), ("2024-01-05", "B", -0.01),
    ]


def test_config_roundtrip_is_detached_and_json_safe():
    raw = {"estimator": "sample_covariance", "params": {}, "lookback_trading_days": 120, "min_observations": 80}
    value = RiskModelConfig.from_dict(raw)
    encoded = value.to_dict(); encoded["params"]["x"] = 1
    assert value.to_dict() == raw
    json.dumps(value.to_dict(), allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("lookback_trading_days", True), ("lookback_trading_days", 1),
    ("min_observations", False), ("min_observations", 1), ("min_observations", 6),
])
def test_config_rejects_invalid_strict_integers(field, value):
    raw = {"estimator": "sample_covariance", "params": {}, "lookback_trading_days": 5, "min_observations": 2}
    raw[field] = value
    with pytest.raises(RiskModelConfigError): RiskModelConfig.from_dict(raw)


def test_config_rejects_unknown_fields_and_nonfinite_params():
    with pytest.raises(RiskModelConfigError): RiskModelConfig.from_dict({"estimator": "x", "params": {}, "lookback_trading_days": 5, "min_observations": 2, "x": 1})
    with pytest.raises(RiskModelConfigError): RiskModelConfig("x", {"a": np.inf}, 5, 2)


def test_request_preserves_exact_order_and_rejects_duplicates():
    request = RiskModelRequest("2024-01-05", ("B", "A"), config())
    assert request.assets == ("B", "A")
    with pytest.raises(RiskModelValidationError): RiskModelRequest("2024-01-05", ("A", "A"), config())


def test_sample_covariance_matches_hand_calculation_ddof_one():
    matrix = np.array([[1.0, 2.0], [2.0, 0.0], [3.0, 4.0]])
    actual = SampleCovarianceEstimator().estimate(matrix, None).covariance
    np.testing.assert_allclose(actual, np.array([[1.0, 1.0], [1.0, 4.0]]), atol=1e-15)


def test_ledoit_wolf_exact_library_parity_and_shrinkage():
    matrix = np.array([[.01, .02], [.03, -.01], [-.02, .04], [.01, .00]])
    actual = LedoitWolfEstimator().estimate(matrix, None)
    expected = LedoitWolf(assume_centered=False, store_precision=False).fit(matrix)
    np.testing.assert_allclose(actual.covariance, expected.covariance_, atol=1e-15)
    assert actual.diagnostics["shrinkage"] == pytest.approx(expected.shrinkage_)


@pytest.mark.parametrize("estimator", [SampleCovarianceEstimator(), LedoitWolfEstimator()])
def test_estimators_reject_hidden_params(estimator):
    with pytest.raises(RiskModelConfigError): estimator.parse_params({"hidden": 1})


def test_registry_is_fresh_strict_and_extensible():
    first, second = build_default_risk_estimator_registry(), build_default_risk_estimator_registry()
    assert first.names() == ("ledoit_wolf", "sample_covariance")
    class Plugin:
        name = "plugin"
        def parse_params(self, raw): return None
        def estimate(self, matrix, parsed): return RiskEstimate(np.cov(matrix, rowvar=False), {"plugin": True})
    first.register("plugin", Plugin())
    assert "plugin" not in second.names()
    with pytest.raises(RiskModelRegistryError): first.register("plugin", Plugin())
    with pytest.raises(RiskModelRegistryError): first.resolve("unknown")


def test_service_common_complete_case_order_and_suspension_zero():
    provider = Returns(rows())
    result = HistoricalCovarianceRiskModelService(provider).estimate(
        RiskModelRequest("2024-01-05", ("B", "A"), config())
    )
    expected = np.cov(np.array([[.02, .01], [.00, -.01], [-.01, .02]]), rowvar=False, ddof=1)
    assert result.assets == ("B", "A") and result.observation_count == 3
    np.testing.assert_allclose(result.covariance, expected, atol=1e-15)
    assert provider.calls[0][0] == ("B", "A")


def test_service_is_invariant_to_sparse_row_permutation():
    forward = HistoricalCovarianceRiskModelService(Returns(rows())).estimate(RiskModelRequest("2024-01-05", ("A", "B"), config()))
    reverse = HistoricalCovarianceRiskModelService(Returns(list(reversed(rows())))).estimate(RiskModelRequest("2024-01-05", ("A", "B"), config()))
    np.testing.assert_array_equal(forward.covariance, reverse.covariance)


def test_service_fails_on_insufficient_common_observations():
    with pytest.raises(RiskModelDataError, match="insufficient"):
        HistoricalCovarianceRiskModelService(Returns(rows())).estimate(RiskModelRequest("2024-01-05", ("A", "B"), config(minimum=4)))


def test_result_allows_singular_psd_and_is_defensive_json_safe():
    source = np.array([[1.0, 1.0], [1.0, 1.0]])
    result = RiskModelResult(formation_date="2024-01-05", risk_cutoff="2024-01-05", assets=("A", "B"), covariance=source, observation_count=2, estimator="x", diagnostics={"ok": 1.0})
    source[0, 0] = 9.0
    copy = result.covariance
    assert not copy.flags.writeable and result.covariance[0, 0] == 1.0
    json.dumps(result.diagnostics, allow_nan=False)


def test_result_rejects_obvious_non_psd_and_nonpositive_diagonal():
    with pytest.raises(RiskModelValidationError, match="semidefinite"):
        RiskModelResult(formation_date="2024-01-05", risk_cutoff="2024-01-05", assets=("A", "B"), covariance=[[1, 2], [2, 1]], observation_count=2, estimator="x", diagnostics={})
    with pytest.raises(RiskModelValidationError, match="diagonal"):
        RiskModelResult(formation_date="2024-01-05", risk_cutoff="2024-01-05", assets=("A", "B"), covariance=[[1, 0], [0, 0]], observation_count=2, estimator="x", diagnostics={})
