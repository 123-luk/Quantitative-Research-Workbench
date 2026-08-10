"""V8-P3 release E2E and adversarial numerical contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.holdings import (
    HOLDINGS_ARTIFACT_FILENAMES,
    HOLDINGS_OUTPUT_COLUMNS,
    HoldingsArtifactConfig,
    HoldingsArtifactStore,
    HoldingsBuilder,
    SignalArtifactProvenance,
)
from src.pipeline.portfolio_services import (
    PortfolioServiceFactoryRegistry,
    PortfolioServiceResolver,
)
from src.portfolio_construction import (
    ConstraintSpec,
    HistoricalReturnWindow,
    PortfolioConstructionConfig,
    PortfolioConstructionConstraintError,
    PortfolioConstructionDataError,
    PortfolioConstructionEngine,
    PortfolioConstructionRegistry,
    PortfolioConstructionServices,
)
from src.portfolio_construction.strategies.minimum_variance import (
    MinimumVarianceConstructor,
)
from src.portfolio_optimization import (
    MinimumVarianceProblem,
    OptimizationResult,
    ScipySLSQPBackend,
)
from src.risk_model import (
    HistoricalCovarianceRiskModelService,
    LedoitWolfEstimator,
    RiskModelConfig,
    RiskModelDataError,
    RiskModelRequest,
    RiskModelResult,
    SampleCovarianceEstimator,
)


def _problem(covariance: np.ndarray, cap: float = 1.0) -> MinimumVarianceProblem:
    count = len(covariance)
    return MinimumVarianceProblem(
        covariance,
        np.full(count, 1.0 / count),
        np.zeros(count),
        np.full(count, cap),
    )


def _gmv(covariance: np.ndarray, cap: float = 1.0) -> np.ndarray:
    result = ScipySLSQPBackend().solve(_problem(covariance, cap))
    assert result.success
    return result.weights


class _Returns:
    def __init__(self, rows: list[tuple[object, str, float]], cutoff: str = "2024-01-10"):
        self.window = HistoricalReturnWindow(
            cutoff,
            pd.DataFrame(rows, columns=["trade_date", "ts_code", "return"]),
        )

    def load_window(self, assets, formation_date, lookback):
        del assets, formation_date, lookback
        return self.window


def _risk_config(estimator: str = "sample_covariance", minimum: int = 5):
    return RiskModelConfig(estimator, {}, 8, minimum)


def test_sample_covariance_uses_exact_four_asset_common_matrix_daily_ddof_one():
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    values = {
        "A": [.01, .02, -.01, .03, .00, .01],
        "B": [None, None, .02, -.01, .01, .04],  # pre-listing absence
        "C": [.00, .01, 0.0, -.02, .03, .01],   # resolved suspension zero
        "D": [.02, -.01, .01, .00, .02, -.03],
    }
    rows = [
        (date, asset, value)
        for asset, series in values.items()
        for date, value in zip(dates, series, strict=True)
        if value is not None
    ]
    request = RiskModelRequest("2024-01-10", ("D", "B", "A", "C"), _risk_config(minimum=4))
    result = HistoricalCovarianceRiskModelService(_Returns(rows)).estimate(request)
    expected_matrix = np.array([
        [values[asset][index] for asset in request.assets]
        for index in range(2, 6)
    ], dtype=float)
    assert result.assets == request.assets
    assert result.observation_count == 4
    assert result.diagnostics["aligned_start"] == "2024-01-04"
    assert result.diagnostics["aligned_end"] == "2024-01-09"
    np.testing.assert_allclose(result.covariance, np.cov(expected_matrix, rowvar=False, ddof=1), atol=1e-15)
    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=0.0)
    assert np.linalg.eigvalsh(result.covariance).min() >= -1e-12


def test_common_observation_boundary_fails_at_m_minus_one_and_succeeds_at_m():
    rows = [(f"2024-01-0{i}", asset, float(i + offset)) for i in range(2, 7) for asset, offset in (("A", 0), ("B", 1))]
    service = HistoricalCovarianceRiskModelService(_Returns(rows, "2024-01-06"))
    success = service.estimate(RiskModelRequest("2024-01-06", ("A", "B"), _risk_config(minimum=5)))
    assert success.observation_count == 5
    missing = [row for row in rows if not (row[0] == "2024-01-02" and row[1] == "B")]
    with pytest.raises(RiskModelDataError, match="insufficient common"):
        HistoricalCovarianceRiskModelService(_Returns(missing, "2024-01-06")).estimate(
            RiskModelRequest("2024-01-06", ("A", "B"), _risk_config(minimum=5))
        )


@pytest.mark.parametrize("scale", [1.0, 252.0, 10_000.0])
def test_analytical_gmv_and_scale_invariance(scale: float):
    covariance = np.diag([1.0, 2.0, 4.0, 8.0])
    expected = np.linalg.solve(covariance, np.ones(4))
    expected /= expected.sum()
    np.testing.assert_allclose(_gmv(scale * covariance), expected, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("kind", ["diagonal", "correlated", "near_singular"])
def test_expanded_solver_ground_truth(kind: str):
    if kind == "diagonal":
        covariance = np.diag([1.0, 2.0, 4.0, 8.0, 16.0])
    elif kind == "correlated":
        loadings = np.array([[.4, .1], [.3, -.2], [.2, .4], [.1, -.3], [.25, .2]])
        covariance = 100.0 * (loadings @ loadings.T + np.diag([.2, .25, .3, .35, .4]))
    else:
        vector = np.array([1.0, .99, 1.01, 1.02, .98])[:, None]
        covariance = vector @ vector.T + 1e-5 * np.eye(5)
    expected = np.linalg.solve(covariance, np.ones(5))
    expected /= expected.sum()
    actual = _gmv(covariance)
    if bool((expected >= 0.0).all()):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
    else:
        equal = np.full(5, .2)
        assert actual.min() >= -1e-10 and actual.sum() == pytest.approx(1.0, abs=1e-10)
        assert actual @ covariance @ actual <= equal @ covariance @ equal + 1e-9


def test_fixed_seed_sixty_spd_cases_are_legal_and_dominate_equal_weight():
    rng = np.random.default_rng(90210)
    for case in range(60):
        count = 2 + case % 19
        sample = rng.normal(size=(count + 5, count))
        covariance = 100.0 * (sample.T @ sample / sample.shape[0] + .05 * np.eye(count))
        actual = _gmv(covariance)
        assert np.isfinite(actual).all() and actual.min() >= -1e-10
        assert actual.sum() == pytest.approx(1.0, abs=1e-10)
        analytical = np.linalg.solve(covariance, np.ones(count))
        analytical /= analytical.sum()
        if bool((analytical >= 0.0).all()):
            np.testing.assert_allclose(actual, analytical, rtol=0.0, atol=1e-7)
        equal = np.full(count, 1.0 / count)
        assert actual @ covariance @ actual <= equal @ covariance @ equal + 1e-9


def test_asset_and_long_row_permutations_preserve_covariance_and_weights():
    rng = np.random.default_rng(44)
    matrix = rng.normal(size=(12, 5))
    assets = tuple("ABCDE")
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    rows = [(date, asset, matrix[i, j]) for i, date in enumerate(dates) for j, asset in enumerate(assets)]
    shuffled = list(rows)
    rng.shuffle(shuffled)
    permutation = ("D", "A", "E", "B", "C")
    first = HistoricalCovarianceRiskModelService(_Returns(rows, "2024-01-12")).estimate(
        RiskModelRequest("2024-01-12", assets, RiskModelConfig("sample_covariance", {}, 12, 5))
    )
    second = HistoricalCovarianceRiskModelService(_Returns(shuffled, "2024-01-12")).estimate(
        RiskModelRequest("2024-01-12", permutation, RiskModelConfig("sample_covariance", {}, 12, 5))
    )
    indices = [assets.index(asset) for asset in permutation]
    np.testing.assert_allclose(second.covariance, first.covariance[np.ix_(indices, indices)], atol=1e-15)
    first_weights = dict(zip(assets, _gmv(first.covariance), strict=True))
    second_weights = dict(zip(permutation, _gmv(second.covariance), strict=True))
    np.testing.assert_allclose([first_weights[a] for a in assets], [second_weights[a] for a in assets], atol=1e-8)


def test_sample_and_ledoit_wolf_share_identity_but_differ_and_are_deterministic():
    rng = np.random.default_rng(123)
    matrix = rng.normal(size=(30, 6)) @ np.diag([.01, .02, .04, .07, .11, .16])
    sample = SampleCovarianceEstimator().estimate(matrix, None)
    first = LedoitWolfEstimator().estimate(matrix, None)
    second = LedoitWolfEstimator().estimate(matrix, None)
    assert first.diagnostics["shrinkage"] > 0.0
    assert not np.allclose(sample.covariance, first.covariance)
    np.testing.assert_array_equal(first.covariance, second.covariance)
    assert first.diagnostics == second.diagnostics
    for covariance in (sample.covariance, first.covariance):
        assert np.isfinite(covariance).all()
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-12
        weights = _gmv(covariance)
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)


def _diagonal_capped_reference(variances: np.ndarray, cap: float) -> np.ndarray:
    low, high = 0.0, float(variances.max())
    while np.minimum(cap, high / variances).sum() < 1.0:
        high *= 2.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if np.minimum(cap, middle / variances).sum() < 1.0:
            low = middle
        else:
            high = middle
    result = np.minimum(cap, high / variances)
    return result / result.sum()


@pytest.mark.parametrize("count,cap", [(5, .20), (5, .25), (10, .10), (10, .15)])
def test_max_weight_matrix_matches_independent_diagonal_kkt_reference(count: int, cap: float):
    variances = np.geomspace(.01, 1.0, count)
    actual = _gmv(np.diag(variances), cap)
    expected = _diagonal_capped_reference(variances, cap)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    assert actual @ np.diag(variances) @ actual <= expected @ np.diag(variances) @ expected + 1e-10
    assert actual.max() <= cap + 1e-9
    assert actual.sum() == pytest.approx(1.0, abs=1e-10)
    if cap > 1.0 / count:
        assert actual[0] == pytest.approx(cap, abs=1e-8)


def test_infeasible_five_asset_cap_fails_closed():
    risk = _Risk(np.eye(5))
    with pytest.raises(PortfolioConstructionConstraintError):
        HoldingsBuilder(PortfolioConstructionEngine(services=PortfolioConstructionServices(risk_model=risk))).build(
            _signals(5), top_n=5, insufficient_universe_policy="error", weighting="equal_weight",
            portfolio_construction=_minimum_variance_config(.19),
        )
    assert risk.requests == []


class _Risk:
    def __init__(self, covariance: np.ndarray):
        self.covariance = covariance
        self.requests: list[RiskModelRequest] = []
        self.results: list[RiskModelResult] = []

    def estimate(self, request: RiskModelRequest) -> RiskModelResult:
        self.requests.append(request)
        result = RiskModelResult(
            formation_date=request.formation_date,
            risk_cutoff=request.formation_date,
            assets=request.assets,
            covariance=self.covariance,
            observation_count=20,
            estimator=request.config.estimator,
            diagnostics={},
        )
        self.results.append(result)
        return result


def _signals(count: int) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": pd.Series([pd.Timestamp("2024-01-10")] * count, dtype="datetime64[ns]"),
        "ts_code": pd.Series([f"S{i:02d}" for i in range(count)], dtype="string"),
        "score": np.linspace(100.0, -100.0, count),
        "rank": np.arange(1, count + 1, dtype=np.int64),
    })


def _minimum_variance_config(cap: float | None = None) -> PortfolioConstructionConfig:
    constraints = () if cap is None else (ConstraintSpec("max_weight", {"max_weight": cap}),)
    return PortfolioConstructionConfig("minimum_variance", {"risk_model": {
        "estimator": "sample_covariance", "params": {},
        "lookback_trading_days": 20, "min_observations": 10,
    }}, constraints)


@pytest.mark.parametrize("count", [2, 5, 10, 20])
def test_selection_identity_is_exact_through_risk_optimizer_and_holdings(count: int):
    selected = _signals(count)
    risk = _Risk(np.diag(np.arange(1, count + 1, dtype=float)))
    built = HoldingsBuilder(PortfolioConstructionEngine(services=PortfolioConstructionServices(risk_model=risk))).build(
        selected, top_n=count, insufficient_universe_policy="error", weighting="equal_weight",
        portfolio_construction=_minimum_variance_config(),
    )
    expected = tuple(selected.ts_code)
    assert risk.requests[0].assets == expected
    assert risk.results[0].assets == expected
    assert tuple(built.holdings.ts_code) == expected
    assert tuple(built.holdings.columns) == HOLDINGS_OUTPUT_COLUMNS


def test_single_asset_minimum_variance_fails_but_equal_and_rank_remain_valid():
    signal = _signals(1)
    with pytest.raises(PortfolioConstructionDataError, match="at least two"):
        HoldingsBuilder(PortfolioConstructionEngine(services=PortfolioConstructionServices(risk_model=_Risk(np.eye(1))))).build(
            signal, top_n=1, insufficient_universe_policy="error", weighting="equal_weight",
            portfolio_construction=_minimum_variance_config(),
        )
    for method in ("equal_weight", "rank_weight"):
        result = HoldingsBuilder().build(
            signal, top_n=1, insufficient_universe_policy="error", weighting="equal_weight",
            portfolio_construction=PortfolioConstructionConfig(method, {}),
        )
        assert result.holdings.target_weight.iloc[0] == 1.0


class _Backend:
    def solve(self, problem):
        weights = np.zeros(len(problem.initial_weights))
        weights[-1] = 1.0
        return OptimizationResult(
            weights=weights, success=True, status=0, message="ok",
            objective_value=float(problem.objective(weights)), iterations=1,
        )


def test_zero_weight_selected_rows_survive_canonical_artifact_roundtrip(tmp_path: Path):
    count = 5
    registry = PortfolioConstructionRegistry()
    registry.register(MinimumVarianceConstructor(_Backend()))
    engine = PortfolioConstructionEngine(
        strategy_registry=registry,
        services=PortfolioConstructionServices(risk_model=_Risk(np.eye(count))),
    )
    portfolio = _minimum_variance_config()
    built = HoldingsBuilder(engine).build(
        _signals(count), top_n=count, insufficient_universe_policy="error",
        weighting="equal_weight", portfolio_construction=portfolio,
    )
    assert len(built.holdings) == count and int((built.holdings.target_weight == 0.0).sum()) == count - 1
    signal_dir = tmp_path / "signal"
    signal_dir.mkdir()
    signal_path = signal_dir / "signals.parquet"
    signal_path.write_bytes(b"v8-p3-signal")
    provenance = SignalArtifactProvenance(
        signal_dir, signal_path, "1.0", hashlib.sha256(signal_path.read_bytes()).hexdigest()
    )
    written = HoldingsArtifactStore().write(
        built, provenance, HoldingsArtifactConfig(tmp_path / "holdings"), portfolio_construction=portfolio
    )
    assert set(path.name for path in written.artifact_dir.iterdir()) == set(HOLDINGS_ARTIFACT_FILENAMES)
    persisted = pd.read_parquet(written.holdings_path)
    pdt.assert_frame_equal(persisted, built.holdings, check_dtype=False)
    assert tuple(persisted.columns) == HOLDINGS_OUTPUT_COLUMNS


def test_service_graph_diamond_builds_shared_dependency_once():
    events: list[str] = []
    registry = PortfolioServiceFactoryRegistry()
    registry.register("D", factory=lambda resolved: events.append("D") or object())
    registry.register("B", dependencies={"D"}, factory=lambda resolved: events.append("B") or resolved["D"])
    registry.register("C", dependencies={"D"}, factory=lambda resolved: events.append("C") or resolved["D"])
    registry.register("A", dependencies={"B", "C"}, factory=lambda resolved: events.append("A") or (resolved["B"], resolved["C"]))
    resolved = PortfolioServiceResolver(registry).resolve({"A"})
    assert events == ["D", "B", "C", "A"]
    assert resolved["A"][0] is resolved["A"][1]
