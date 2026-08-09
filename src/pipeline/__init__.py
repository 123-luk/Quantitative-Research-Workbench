"""Pipeline configuration and experiment run helpers."""

from src.pipeline.config import PipelineConfig
from src.pipeline.experiment import ExperimentManager
from src.pipeline.ml_cli import (
    MLCLIConfigError,
    MLCLIError,
    exit_code_for_ml_error,
    format_ml_human_summary,
    merge_ml_cli_overrides,
    parse_ml_model_params,
)
from src.pipeline.ml_config import (
    MLExperimentPipelineConfig,
    MLPipelineConfigError,
    MLPipelineError,
)
from src.pipeline.modeling_panel_config import (
    ModelingPanelOutputConfig,
    ModelingPanelPipelineConfig,
    ModelingPanelPipelineConfigError,
    ModelingPanelPipelineError,
    ModelingPanelPipelineExecutionError,
    ModelingPanelSourceConfig,
)
from src.pipeline.modeling_panel_execution import (
    ModelingPanelPipelineExecutor,
    ModelingPanelPipelineResult,
)
from src.pipeline.ml_execution import (
    MLExperimentPipelineExecutor,
    MLExperimentPipelineResult,
    MLPipelineArtifactError,
    MLPipelineExecutionError,
    MLPipelineIntegrityError,
    MLPipelinePanelError,
    read_ml_modeling_panel,
)
from src.pipeline.research_config import FactorResearchPipelineConfig
from src.pipeline.signal_config import (
    PredictionSourceConfig,
    SignalConfigError,
    SignalPipelineConfig,
)
from src.pipeline.signal_execution import (
    SignalPipelineExecutionError,
    SignalPipelineExecutor,
    SignalPipelineResult,
)
from src.pipeline.holdings_config import HoldingsConfigError, HoldingsPipelineConfig
from src.pipeline.holdings_execution import (
    HoldingsPipelineExecutionError,
    HoldingsPipelineExecutor,
    HoldingsPipelineResult,
)
from src.pipeline.research_backtest_config import (
    BacktestScheduleConfig,
    BacktestSourceConfig,
    BenchmarkConfig,
    PerformanceConfig,
    PortfolioAccountingConfig,
    ResearchBacktestConfigError,
    ResearchBacktestPipelineConfig,
    ReturnAlignmentConfig,
    TransactionCostConfig,
)
from src.pipeline.research_execution import (
    FactorResearchExecutionResult,
    FactorResearchPublishedOutputs,
    FactorResearchPipelineExecutor,
)
from src.pipeline.runner import run_pipeline

__all__ = [
    "ExperimentManager",
    "MLCLIConfigError",
    "MLCLIError",
    "MLExperimentPipelineConfig",
    "MLExperimentPipelineExecutor",
    "MLExperimentPipelineResult",
    "MLPipelineArtifactError",
    "MLPipelineConfigError",
    "MLPipelineError",
    "MLPipelineExecutionError",
    "MLPipelineIntegrityError",
    "MLPipelinePanelError",
    "ModelingPanelOutputConfig",
    "ModelingPanelPipelineConfig",
    "ModelingPanelPipelineConfigError",
    "ModelingPanelPipelineError",
    "ModelingPanelPipelineExecutionError",
    "ModelingPanelPipelineExecutor",
    "ModelingPanelPipelineResult",
    "ModelingPanelSourceConfig",
    "read_ml_modeling_panel",
    "exit_code_for_ml_error",
    "format_ml_human_summary",
    "merge_ml_cli_overrides",
    "parse_ml_model_params",
    "FactorResearchExecutionResult",
    "FactorResearchPublishedOutputs",
    "FactorResearchPipelineExecutor",
    "FactorResearchPipelineConfig",
    "PipelineConfig",
    "PredictionSourceConfig",
    "SignalConfigError",
    "SignalPipelineConfig",
    "SignalPipelineExecutionError",
    "SignalPipelineExecutor",
    "SignalPipelineResult",
    "HoldingsConfigError",
    "HoldingsPipelineConfig",
    "HoldingsPipelineExecutionError",
    "HoldingsPipelineExecutor",
    "HoldingsPipelineResult",
    "BacktestScheduleConfig",
    "BacktestSourceConfig",
    "BenchmarkConfig",
    "PerformanceConfig",
    "PortfolioAccountingConfig",
    "ResearchBacktestConfigError",
    "ResearchBacktestPipelineConfig",
    "ReturnAlignmentConfig",
    "TransactionCostConfig",
    "run_pipeline",
]
