"""Gate E tests for generic transitive portfolio service resolution."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from src.pipeline.holdings_execution import HoldingsPipelineExecutionError
from src.pipeline.portfolio_services import (
    PortfolioServiceFactoryRegistry,
    PortfolioServiceGraphError,
    PortfolioServiceResolver,
)
from src.pipeline.runner import _portfolio_engine
from src.pipeline.research_backtest_config import ResearchBacktestPipelineConfig
from src.pipeline.research_backtest_executor import ResearchBacktestPipelineExecutor
from src.portfolio_construction import (
    PortfolioConstructionRegistry,
    StrategyConstructionOutput,
)
from src.risk_model import HistoricalCovarianceRiskModelService


class Client:
    def get_trade_cal(self, *args, **kwargs): ...
    def get_daily(self, *args, **kwargs): ...
    def get_stock_basic(self, *args, **kwargs): ...
    def get_suspend_d(self, *args, **kwargs): ...


def test_transitive_dependencies_build_once_in_deterministic_order() -> None:
    events: list[str] = []
    registry = PortfolioServiceFactoryRegistry()
    registry.register(
        "B", factory=lambda resolved: events.append("B") or object()
    )
    registry.register(
        "A",
        dependencies={"B"},
        factory=lambda resolved: events.append("A") or ("A", resolved["B"]),
    )
    resolved = PortfolioServiceResolver(registry).resolve({"A", "B"})
    assert events == ["B", "A"]
    assert resolved["A"][1] is resolved["B"]


def test_cycle_duplicate_unknown_and_none_factory_fail_closed() -> None:
    cycle = PortfolioServiceFactoryRegistry()
    cycle.register("A", dependencies={"B"}, factory=lambda resolved: object())
    cycle.register("B", dependencies={"A"}, factory=lambda resolved: object())
    with pytest.raises(PortfolioServiceGraphError, match="A -> B -> A"):
        PortfolioServiceResolver(cycle).resolve({"A"})
    with pytest.raises(PortfolioServiceGraphError, match="already registered"):
        cycle.register("A", factory=lambda resolved: object())
    with pytest.raises(PortfolioServiceGraphError, match="unknown"):
        PortfolioServiceResolver(PortfolioServiceFactoryRegistry()).resolve({"X"})
    empty = PortfolioServiceFactoryRegistry()
    empty.register("empty", factory=lambda resolved: None)
    with pytest.raises(PortfolioServiceGraphError, match="returned None"):
        PortfolioServiceResolver(empty).resolve({"empty"})


@pytest.mark.parametrize("method", ["equal_weight", "rank_weight"])
def test_market_independent_methods_create_no_client_or_services(method: str) -> None:
    def forbidden():
        raise AssertionError("market client must not be created")

    engine, client = _portfolio_engine(method, forbidden, shared_client=None)
    assert client is None
    assert engine._services.historical_returns is None
    assert engine._services.risk_model is None


def test_inverse_builds_historical_only_once() -> None:
    client = Client()
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return client

    engine, actual = _portfolio_engine(
        "inverse_volatility", factory, shared_client=None
    )
    assert actual is client and calls == 1
    assert engine._services.historical_returns is not None
    assert engine._services.risk_model is None


def test_minimum_variance_builds_transitive_services_once_and_reuses_client() -> None:
    client = Client()
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return client

    engine, actual = _portfolio_engine(
        "minimum_variance", factory, shared_client=None
    )
    assert actual is client and calls == 1
    assert engine._services.historical_returns is not None
    assert isinstance(engine._services.risk_model, HistoricalCovarianceRiskModelService)

    def forbidden():
        raise AssertionError("shared client must be reused")

    _, reused = _portfolio_engine(
        "minimum_variance", forbidden, shared_client=client
    )
    assert reused is client
    backtest = ResearchBacktestPipelineExecutor(
        ResearchBacktestPipelineConfig(), actual
    )
    assert backtest.client is actual


class CustomStrategy:
    name = "custom"
    supported_constraint_types = frozenset()
    required_services = frozenset({"custom_A"})

    def parse_params(self, raw_params: Mapping[str, object]) -> None:
        return None

    def construct(self, request, parsed_params, constraints, services):
        assert services.capability("custom_A")[0] == "A"
        count = len(request.ts_codes)
        return StrategyConstructionOutput(
            pd.DataFrame(
                {
                    "ts_code": request.ts_codes,
                    "target_weight": np.full(count, 1.0 / count),
                }
            )
        )


def test_custom_strategy_resolves_custom_transitive_capability_without_branch() -> None:
    strategies = PortfolioConstructionRegistry()
    strategies.register(CustomStrategy())
    factories = PortfolioServiceFactoryRegistry()
    factories.register("custom_B", factory=lambda resolved: ("B",))
    factories.register(
        "custom_A",
        dependencies={"custom_B"},
        factory=lambda resolved: ("A", resolved["custom_B"]),
    )
    engine, client = _portfolio_engine(
        "custom",
        lambda: (_ for _ in ()).throw(AssertionError("no market client")),
        shared_client=None,
        strategy_registry=strategies,
        service_registry=factories,
    )
    assert client is None
    assert engine._services.capability("custom_A")[0] == "A"


def test_runner_wraps_graph_errors_at_holdings_boundary() -> None:
    strategies = PortfolioConstructionRegistry()
    strategies.register(CustomStrategy())
    with pytest.raises(HoldingsPipelineExecutionError, match="unknown"):
        _portfolio_engine(
            "custom",
            lambda: Client(),
            shared_client=None,
            strategy_registry=strategies,
            service_registry=PortfolioServiceFactoryRegistry(),
        )
