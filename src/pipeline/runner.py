"""Pipeline skeleton runner that wires data checks and experiment artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

from src.data.data_manager import DataManager
from src.data.tushare_client import TushareClient
from src.pipeline.config import PipelineConfig
from src.pipeline.experiment import ExperimentManager
from src.pipeline.ml_execution import MLExperimentPipelineExecutor
from src.pipeline.modeling_panel_config import (
    ModelingPanelPipelineExecutionError,
)
from src.pipeline.modeling_panel_execution import (
    ModelingPanelPipelineExecutor,
    ModelingPanelPipelineResult,
)
from src.pipeline.research_execution import (
    FactorResearchExecutionResult,
    FactorResearchPipelineExecutor,
)
from src.pipeline.signal_execution import (
    SignalPipelineExecutionError,
    SignalPipelineExecutor,
)
from src.pipeline.holdings_execution import (
    HoldingsPipelineExecutionError,
    HoldingsPipelineExecutor,
    HoldingsPipelineResult,
)
from src.pipeline.research_backtest_executor import (
    ResearchBacktestPipelineExecutor,
)
from src.portfolio_construction import (
    PortfolioConstructionEngine,
    PortfolioConstructionServices,
    build_default_portfolio_construction_registry,
)
from src.portfolio_construction.adapters import (
    ResearchBacktestHistoricalReturnService,
)
from src.risk_model import HistoricalCovarianceRiskModelService
from src.pipeline.portfolio_services import (
    PortfolioServiceFactoryRegistry,
    PortfolioServiceGraphError,
    PortfolioServiceResolver,
)


def _portfolio_engine(
    method: str,
    market_client_factory: Callable[[], object],
    *,
    shared_client: object | None,
    strategy_registry=None,
    service_registry: PortfolioServiceFactoryRegistry | None = None,
) -> tuple[PortfolioConstructionEngine, object | None]:
    registry = strategy_registry or build_default_portfolio_construction_registry()
    required = registry.required_services(method)
    client = shared_client

    def market_client() -> object:
        nonlocal client
        if client is None:
            client = market_client_factory()
        return client

    factories = service_registry or PortfolioServiceFactoryRegistry()
    if service_registry is None:
        factories.register(
            "historical_returns",
            factory=lambda resolved: ResearchBacktestHistoricalReturnService(
                market_client()
            ),
        )
        factories.register(
            "risk_model",
            dependencies={"historical_returns"},
            factory=lambda resolved: HistoricalCovarianceRiskModelService(
                resolved["historical_returns"]  # type: ignore[arg-type]
            ),
        )
    try:
        resolved = PortfolioServiceResolver(factories).resolve(required)
    except PortfolioServiceGraphError as exc:
        raise HoldingsPipelineExecutionError(str(exc)) from exc
    services = PortfolioConstructionServices(
        historical_returns=resolved.get("historical_returns"),  # type: ignore[arg-type]
        risk_model=resolved.get("risk_model"),  # type: ignore[arg-type]
        extensions={
            name: value
            for name, value in resolved.items()
            if name not in {"historical_returns", "risk_model"}
        },
    )
    return PortfolioConstructionEngine(
        strategy_registry=registry, services=services
    ), client


def run_pipeline(
    config: PipelineConfig,
    *,
    market_client_factory: Callable[[], object] | None = None,
) -> dict[str, Any]:
    """Run the V1 pipeline skeleton and return a concise run summary.

    The V1 runner only checks local data cache readiness and creates experiment
    artifacts. It does not download TuShare data, train models, or run backtests.
    """
    client_factory = TushareClient if market_client_factory is None else (
        market_client_factory
    )
    required_start_date = config.required_start_date
    required_end_date = config.required_end_date

    data_manager = DataManager()
    data_status = data_manager.prepare_data(
        {
            "required_start_date": required_start_date,
            "backtest_end": required_end_date,
            "required_datasets": config.required_datasets,
        }
    )

    experiment_manager = ExperimentManager(config.output_dir)
    run_dir = experiment_manager.create_run_dir(
        strategy_name=config.strategy_name,
        stock_pool=config.stock_pool,
    )

    cache_status = str(data_status["cache_status"])
    missing_ranges = data_status["missing_ranges"]
    status = "ready" if cache_status == "ready" else "missing_data"

    summary = {
        "status": status,
        "run_dir": str(run_dir),
        "required_start_date": required_start_date,
        "required_end_date": required_end_date,
        "cache_status": cache_status,
        "missing_ranges": missing_ranges,
        "strategy_name": config.strategy_name,
        "stock_pool": config.stock_pool,
    }

    factor_research_result = FactorResearchExecutionResult.disabled()
    if config.factor_research.enabled:
        executor = FactorResearchPipelineExecutor(config.factor_research)
        factor_research_result = executor.execute(
            run_dir,
            metadata={
                "pipeline_status": status,
                "strategy_name": config.strategy_name,
                "stock_pool": config.stock_pool,
                "required_start_date": required_start_date,
                "required_end_date": required_end_date,
            },
        )
        summary["factor_research"] = factor_research_result.to_dict()

    modeling_panel_result = ModelingPanelPipelineResult.disabled()
    if config.modeling_panel.enabled:
        modeling_executor = ModelingPanelPipelineExecutor(
            config.modeling_panel
        )
        research_input = (
            factor_research_result
            if config.modeling_panel.source.mode == "factor_research"
            else None
        )
        modeling_panel_result = modeling_executor.execute(
            run_dir,
            factor_research_result=research_input,
        )
        summary["modeling_panel"] = modeling_panel_result.as_dict()

    ml_result = None
    if config.ml_experiment.enabled:
        if config.modeling_panel.enabled:
            if (
                not modeling_panel_result.enabled
                or modeling_panel_result.panel_path is None
            ):
                raise ModelingPanelPipelineExecutionError(
                    "enabled Modeling Panel stage returned no panel_path"
                )
            ml_executor = MLExperimentPipelineExecutor(
                config.ml_experiment
            )
            ml_result = ml_executor.execute(
                run_dir,
                panel_path_override=modeling_panel_result.panel_path,
            )
        else:
            ml_executor = MLExperimentPipelineExecutor(
                config.ml_experiment
            )
            ml_result = ml_executor.execute(run_dir)
        if (
            config.signal.enabled
            and config.signal.source.mode == "ml"
            and ml_result is None
        ):
            raise SignalPipelineExecutionError(
                "enabled ML stage returned no result for Signal handoff."
            )
        summary["ml_experiment"] = ml_result.to_dict()

    signal_result = None
    if config.signal.enabled:
        signal_executor = SignalPipelineExecutor(config.signal)
        signal_result = signal_executor.execute(
            run_dir,
            ml_result=(
                ml_result if config.signal.source.mode == "ml" else None
            ),
        )
        if config.holdings.enabled and signal_result is None:
            raise HoldingsPipelineExecutionError(
                "enabled Signal stage returned no result for Holdings handoff."
            )
        summary["signal"] = signal_result.as_dict()

    holdings_result = HoldingsPipelineResult.disabled()
    market_client: object | None = None
    if config.holdings.enabled:
        holdings_engine, market_client = _portfolio_engine(
            config.holdings.portfolio_construction.method,
            client_factory,
            shared_client=market_client,
        )
        holdings_executor = HoldingsPipelineExecutor(
            config.holdings, holdings_engine
        )
        holdings_result = holdings_executor.execute(
            run_dir,
            signal_result=signal_result,
        )
        summary["holdings"] = holdings_result.as_dict()

    if config.research_backtest.enabled:
        if market_client is None:
            market_client = client_factory()
        backtest_executor = ResearchBacktestPipelineExecutor(
            config.research_backtest,
            market_client,  # type: ignore[arg-type]
        )
        artifact_dir = Path(run_dir) / config.research_backtest.artifact_subdir
        backtest_result = backtest_executor.execute(
            artifact_dir=artifact_dir,
            end_date=required_end_date,
            holdings_result=(
                holdings_result
                if config.research_backtest.source.mode == "pipeline"
                else None
            ),
        )
        summary["research_backtest"] = backtest_result.to_dict()

    experiment_manager.save_config_snapshot(run_dir, config)
    experiment_manager.save_run_info(
        run_dir,
        {
            "status": status,
            "created_at": datetime.now().replace(microsecond=0).isoformat(),
            "strategy_name": config.strategy_name,
            "stock_pool": config.stock_pool,
            "required_start_date": required_start_date,
            "required_end_date": required_end_date,
            "cache_status": cache_status,
            "missing_ranges": missing_ranges,
        },
    )
    experiment_manager.save_metrics(
        run_dir,
        {
            "status": "placeholder",
            "metrics_ready": False,
            "message": "Pipeline skeleton has not run strategy, model, or backtest metrics.",
        },
    )

    return summary
