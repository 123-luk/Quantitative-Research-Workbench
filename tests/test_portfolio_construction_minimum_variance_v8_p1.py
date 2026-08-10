import numpy as np
import pandas as pd
import pytest

from src.portfolio_construction import (
    ConstraintSpec,
    PortfolioConstructionConfig,
    PortfolioConstructionConstraintError,
    PortfolioConstructionDataError,
    PortfolioConstructionEngine,
    PortfolioConstructionRegistry,
    PortfolioConstructionRequest,
    PortfolioConstructionServices,
    PortfolioConstructionValidationError,
    build_default_portfolio_construction_registry,
)
from src.portfolio_construction.strategies.minimum_variance import MinimumVarianceConstructor
from src.portfolio_optimization import OptimizationResult
from src.risk_model import RiskModelResult


def request(count=5, scores=None):
    codes = [f"A{i}" for i in range(count)]
    if scores is None: scores = np.arange(count, dtype=float)
    return PortfolioConstructionRequest(
        "2024-01-05",
        pd.DataFrame({
            "ts_code": codes,
            "score": scores,
            "rank": np.arange(1, count + 1),
            "selection_position": np.arange(1, count + 1),
        }),
    )


def config():
    return PortfolioConstructionConfig("minimum_variance", {"risk_model": {
        "estimator": "sample_covariance", "params": {},
        "lookback_trading_days": 120, "min_observations": 80,
    }})


class Risk:
    def __init__(self, covariance): self.covariance = covariance; self.requests = []
    def estimate(self, req):
        self.requests.append(req)
        return RiskModelResult(
            formation_date=req.formation_date, risk_cutoff=req.formation_date,
            assets=req.assets, covariance=self.covariance,
            observation_count=80, estimator=req.config.estimator, diagnostics={},
        )


@pytest.mark.parametrize("count", [2, 5, 10])
def test_minimum_variance_supports_factor_count_agnostic_candidate_sizes(count):
    risk = Risk(np.diag(np.arange(1, count + 1, dtype=float)))
    result = PortfolioConstructionEngine(services=PortfolioConstructionServices(risk_model=risk)).construct(request(count), config())
    expected = 1 / np.arange(1, count + 1, dtype=float); expected /= expected.sum()
    np.testing.assert_allclose(result.weights.target_weight, expected, atol=1e-8)
    assert tuple(result.weights.ts_code) == request(count).ts_codes


def test_scores_and_ranks_do_not_enter_risk_only_objective():
    risk = Risk(np.diag([1., 2., 3., 4., 5.]))
    engine = PortfolioConstructionEngine(services=PortfolioConstructionServices(risk_model=risk))
    first = engine.construct(request(scores=np.arange(5)), config()).weights
    second = engine.construct(request(scores=np.arange(5)[::-1]), config()).weights
    np.testing.assert_allclose(first.target_weight, second.target_weight, atol=1e-12)


def test_max_weight_binding_and_exact_feasible_cap():
    risk = Risk(np.diag([.01, 1., 1., 1.]))
    capped = PortfolioConstructionConfig(config().method, config().params, (ConstraintSpec("max_weight", {"max_weight": .3}),))
    result = PortfolioConstructionEngine(services=PortfolioConstructionServices(risk_model=risk)).construct(request(4), capped)
    assert result.weights.target_weight.max() <= .3 + 1e-12
    exact = PortfolioConstructionConfig(config().method, config().params, (ConstraintSpec("max_weight", {"max_weight": .25}),))
    equal = PortfolioConstructionEngine(services=PortfolioConstructionServices(risk_model=risk)).construct(request(4), exact)
    np.testing.assert_allclose(equal.weights.target_weight, .25, atol=1e-12)


def test_infeasible_cap_fails_before_risk_or_solver():
    risk = Risk(np.eye(4))
    invalid = PortfolioConstructionConfig(config().method, config().params, (ConstraintSpec("max_weight", {"max_weight": .2}),))
    with pytest.raises(PortfolioConstructionConstraintError):
        PortfolioConstructionEngine(services=PortfolioConstructionServices(risk_model=risk)).construct(request(4), invalid)
    assert risk.requests == []


def test_missing_risk_model_and_single_asset_fail_closed():
    with pytest.raises(PortfolioConstructionDataError, match="RiskModelService"):
        PortfolioConstructionEngine().construct(request(2), config())
    with pytest.raises(PortfolioConstructionDataError, match="at least two"):
        PortfolioConstructionEngine(services=PortfolioConstructionServices(risk_model=Risk([[1.]]))).construct(request(1), config())


def test_default_registry_declares_generic_risk_capability():
    registry = build_default_portfolio_construction_registry()
    assert "minimum_variance" in registry.names()
    assert registry.required_services("minimum_variance") == frozenset({"risk_model"})
    assert registry.required_services("inverse_volatility") == frozenset({"historical_returns"})


@pytest.mark.parametrize("params", [{}, {"risk_model": config().params["risk_model"], "solver": {}}, {"risk_model": {"estimator": "sample_covariance", "params": {"x": 1}, "lookback_trading_days": 5, "min_observations": 2}}])
def test_strategy_params_are_strict(params):
    strategy = MinimumVarianceConstructor()
    if params and set(params) == {"risk_model"}:
        parsed = strategy.parse_params(params)
        assert parsed.params["x"] == 1  # estimator registry rejects this later
    else:
        with pytest.raises(ValueError): strategy.parse_params(params)


class Backend:
    def __init__(self, result): self.result = result; self.problem = None
    def solve(self, problem): self.problem = problem; return self.result


def engine_with_backend(result):
    registry = PortfolioConstructionRegistry(); registry.register(MinimumVarianceConstructor(Backend(result)))
    return PortfolioConstructionEngine(strategy_registry=registry, services=PortfolioConstructionServices(risk_model=Risk(np.eye(2))))


@pytest.mark.parametrize("result", [
    OptimizationResult(weights=[.5, .5], success=False, status=1, message="fail", objective_value=.1, iterations=1),
    OptimizationResult(weights=[1.0], success=True, status=0, message="ok", objective_value=.1, iterations=1),
    OptimizationResult(weights=[-.01, 1.01], success=True, status=0, message="ok", objective_value=.1, iterations=1),
    OptimizationResult(weights=[.4, .4], success=True, status=0, message="ok", objective_value=.1, iterations=1),
])
def test_fake_backend_contract_violations_fail_closed(result):
    with pytest.raises(PortfolioConstructionValidationError): engine_with_backend(result).construct(request(2), config())


def test_backend_injection_uses_problem_without_engine_dispatch_changes():
    result = OptimizationResult(weights=[.25, .75], success=True, status=0, message="ok", objective_value=.1, iterations=1)
    backend = Backend(result)
    registry = PortfolioConstructionRegistry(); registry.register(MinimumVarianceConstructor(backend))
    actual = PortfolioConstructionEngine(strategy_registry=registry, services=PortfolioConstructionServices(risk_model=Risk(np.eye(2)))).construct(request(2), config())
    np.testing.assert_array_equal(actual.weights.target_weight, [.25, .75])
    assert backend.problem.name == "minimum_variance"
