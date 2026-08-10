import numpy as np
import pytest

from src.portfolio_optimization import (
    MinimumVarianceProblem, OptimizationResult, PortfolioOptimizationValidationError, ScipySLSQPBackend
)


def problem(covariance, cap=1.0):
    count = len(covariance)
    return MinimumVarianceProblem(covariance, np.full(count, 1/count), np.zeros(count), np.full(count, cap))


def test_diagonal_gmv_matches_inverse_variance_ground_truth():
    variances = np.array([1.0, 2.0, 4.0, 8.0])
    result = ScipySLSQPBackend().solve(problem(np.diag(variances)))
    expected = (1 / variances) / np.sum(1 / variances)
    assert result.success
    np.testing.assert_allclose(result.weights, expected, atol=1e-8)


def test_positive_analytical_gmv_ground_truth():
    covariance = np.array([[.04, .006, .004], [.006, .09, .01], [.004, .01, .16]])
    result = ScipySLSQPBackend().solve(problem(covariance))
    expected = np.linalg.solve(covariance, np.ones(3)); expected /= expected.sum()
    np.testing.assert_allclose(result.weights, expected, rtol=0.0, atol=1e-7)


def test_binding_cap_is_encoded_in_solver_bounds():
    result = ScipySLSQPBackend().solve(problem(np.diag([.01, 1, 1, 1]), cap=.4))
    assert result.success and result.weights[0] == pytest.approx(.4, abs=1e-9)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_backend_diagnostics_freeze_release_settings():
    result = ScipySLSQPBackend().solve(problem(np.eye(2)))
    assert result.diagnostics == {"disp": False, "ftol": 1e-12, "maxiter": 1000, "method": "SLSQP"}


@pytest.mark.parametrize("kwargs", [
    {"weights": [np.nan], "success": True, "status": 0, "message": "", "objective_value": 0.0, "iterations": 1},
    {"weights": [1.0], "success": True, "status": 0, "message": "", "objective_value": np.inf, "iterations": 1},
])
def test_optimization_result_rejects_nonfinite_fields(kwargs):
    with pytest.raises(PortfolioOptimizationValidationError): OptimizationResult(**kwargs)


def test_problem_defensive_arrays_and_analytic_gradient():
    covariance = np.array([[2.0, .5], [.5, 1.0]])
    item = problem(covariance)
    covariance[0, 0] = 99
    np.testing.assert_allclose(item.gradient(np.array([.25, .75])), np.array([.875, .875]))
    assert item.objective(np.array([.25, .75])) == pytest.approx(.4375)
