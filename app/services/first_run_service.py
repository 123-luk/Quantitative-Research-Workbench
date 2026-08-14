"""Offline-first Workbench orchestration across P4B, P4C3, and the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
import json
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Callable, Iterable, Mapping

import pandas as pd

from app.services.canonical_market_client import CanonicalPipelineMarketClient
from app.services.credential_service import ProviderErrorKind, classify_provider_error
from app.services.result_service import ResultService
from app.services.run_service import RunOutcome, RunService, SafeRunError
from src.data import CoverageLedger, PartitionedParquetStore, RawParquetStore
from src.data.provider_registry import ProviderClientFactory, ProviderId
from src.data.provider_quality import sanitize_identifier_evidence
from src.data.coverage_ledger import CoverageRecord
from src.data.coverage_planner import scope_key
from src.data.contracts import CoverageKind, DataRequirement, ResearchFrequency, canonical_date
from src.data.dataset_registry import DatasetRegistry, create_default_dataset_registry
from src.data.canonical_store import CanonicalDataError
from src.data.preparation import CuratedTradingCalendarResolver, DataPreparationResult, DataPreparationService, DataUnavailableError, MissingCredentialError
from src.factors import FactorRegistry
from src.factors.examples import register_example_factors
from src.factors.financial_factors import register_financial_factors
from src.factors.forward_returns import ForwardReturnConfig
from src.factors.preprocessing import PreprocessingConfig
from src.factors.price_volume import register_price_volume_factors
from src.factors.research_pipeline import FactorResearchConfig, FactorResearchRunner
from src.factors.valuation import register_valuation_factors
from src.pipeline.config import PipelineConfig
from src.pipeline.experiment import ExperimentManager
from src.data.provider_contracts import ProviderContractRegistry
from src.pipeline.runner import run_pipeline
from src.research_data import AdjustedPriceService, CanonicalAdjustedPriceDataSource, ForwardReturnSpec, HistoryRequirement, ResearchCalendar, ResearchInputBuilder, ResearchInputMaterialization, ResearchInputPlan, ResearchInputPlanner, ResearchMaterializationStore, compose_requirements
from src.research_data.adjusted_prices import AdjustedPriceError
from src.research_data.planning import ResearchInputError
from src.universe import CanonicalUniverseDataSource, UniverseService, UniverseSpec
from src.universe.contracts import UnsupportedLegacySecurityIdentifier
from src.universe.data import STOCK_BASIC_SCOPE


class WorkbenchErrorCode(str, Enum):
    CREDENTIAL_MISSING = "CREDENTIAL_MISSING"
    AUTHENTICATION_INVALID = "AUTHENTICATION_INVALID"
    PERMISSION_INSUFFICIENT = "PERMISSION_INSUFFICIENT"
    POINTS_INSUFFICIENT = "POINTS_INSUFFICIENT"
    NETWORK_ERROR = "NETWORK_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_EMPTY = "PROVIDER_EMPTY"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    PROVIDER_RESPONSE_INVALID = "PROVIDER_RESPONSE_INVALID"
    PROVIDER_DATA_QUALITY = "PROVIDER_DATA_QUALITY"
    UNSUPPORTED_LEGACY_SECURITY_IDENTIFIER = "UNSUPPORTED_LEGACY_SECURITY_IDENTIFIER"
    DATA_INCOMPLETE = "DATA_INCOMPLETE"
    COVERAGE_VALIDATION = "COVERAGE_VALIDATION"
    UNSUPPORTED = "UNSUPPORTED"
    PIPELINE_ERROR = "PIPELINE_ERROR"


def classify_data_unavailable_error(exc: DataUnavailableError) -> WorkbenchErrorCode:
    """Map one safe preparation failure without exposing provider text."""
    if exc.origin == "local":
        return WorkbenchErrorCode.COVERAGE_VALIDATION
    if exc.origin == "provider_quality":
        if exc.diagnosis_code and "UNSUPPORTED_LEGACY_SECURITY_IDENTIFIER" in exc.diagnosis_code:
            return WorkbenchErrorCode.UNSUPPORTED_LEGACY_SECURITY_IDENTIFIER
        return WorkbenchErrorCode.PROVIDER_DATA_QUALITY
    kind = classify_provider_error(exc.safe_cause or exc)
    cause_text = " ".join(
        f"{type(item).__name__} {item}"
        for item in (exc.safe_cause, getattr(exc.safe_cause, "__cause__", None))
        if item is not None
    ).lower()
    if "empty" in cause_text or "no rows" in cause_text or "returned no" in cause_text:
        return WorkbenchErrorCode.PROVIDER_EMPTY
    if isinstance(exc.safe_cause, sqlite3.Error):
        return WorkbenchErrorCode.COVERAGE_VALIDATION
    if isinstance(exc.safe_cause, CanonicalDataError):
        provider_contract_terms = (
            "missing required columns", "provider result", "requested trade date",
            "requested scope", "result does not", "snapshot does not match",
        )
        return (
            WorkbenchErrorCode.PROVIDER_RESPONSE_INVALID
            if any(term in cause_text for term in provider_contract_terms)
            else WorkbenchErrorCode.COVERAGE_VALIDATION
        )
    return {
        ProviderErrorKind.AUTHENTICATION_INVALID: WorkbenchErrorCode.AUTHENTICATION_INVALID,
        ProviderErrorKind.PERMISSION_INSUFFICIENT: WorkbenchErrorCode.PERMISSION_INSUFFICIENT,
        ProviderErrorKind.POINTS_INSUFFICIENT: WorkbenchErrorCode.POINTS_INSUFFICIENT,
        ProviderErrorKind.RATE_LIMITED: WorkbenchErrorCode.RATE_LIMITED,
        ProviderErrorKind.NETWORK_ERROR: WorkbenchErrorCode.NETWORK_ERROR,
        ProviderErrorKind.RESPONSE_INVALID: WorkbenchErrorCode.PROVIDER_RESPONSE_INVALID,
        ProviderErrorKind.PROVIDER_ERROR: WorkbenchErrorCode.DATA_INCOMPLETE,
    }[kind]


def classify_data_unavailable_stage(exc: DataUnavailableError, active_stage: str) -> str:
    """Expose provider quality as its true stage without relabeling local faults."""
    return "quality_validation" if exc.origin == "provider_quality" else active_stage


@dataclass(frozen=True)
class FailureDiagnostic:
    ledger_status: str | None = None
    canonical_status: str | None = None
    consistency_issue: str | None = None
    repair_action: str | None = None
    provider_attempts: int | None = None
    network_category: str | None = None
    transaction_fetch_id: str | None = None
    transaction_state: str | None = None
    transaction_operation: str | None = None
    transaction_error_code: str | None = None
    transaction_exception_type: str | None = None
    transaction_cause_type: str | None = None
    transaction_message: str | None = None
    transaction_rows: int | None = None
    transaction_fields: tuple[str, ...] = ()
    transaction_quality_evidence: Mapping[str, object] = field(default_factory=dict)


def _network_category(exc: BaseException | None) -> str | None:
    values: list[str] = []
    current = exc
    while current is not None and len(values) < 8:
        values.append(f"{type(current).__name__} {current}".lower())
        current = current.__cause__ or current.__context__
    text = " ".join(values)
    if "readtimeout" in text or "read timed out" in text or "read timeout" in text:
        return "READ_TIMEOUT"
    if "connecttimeout" in text or "connect timed out" in text or "connect timeout" in text:
        return "CONNECT_TIMEOUT"
    if "dns" in text or "name resolution" in text or "getaddrinfo" in text:
        return "DNS_FAILURE"
    if "connection" in text or "proxy" in text or "socket" in text:
        return "CONNECTION_FAILURE"
    return None


class WorkbenchRunError(RuntimeError):
    def __init__(
        self,
        code: WorkbenchErrorCode,
        stage: str,
        run_id: str | None = None,
        pipeline_error: SafeRunError | None = None,
        dataset_id: str | None = None,
        missing_range: tuple[str, str] | None = None,
        user_message: str | None = None,
        diagnostic: FailureDiagnostic | None = None,
    ) -> None:
        super().__init__(f"{code.value} at {stage}")
        self.code = code
        self.stage = stage
        self.run_id = run_id
        self.pipeline_error = pipeline_error
        self.dataset_id = dataset_id
        self.missing_range = missing_range
        self.user_message = user_message or f"{code.value} at {stage}"
        self.diagnostic = diagnostic

    def __str__(self) -> str:
        return f"{self.code.value} at {self.stage}"


@dataclass(frozen=True)
class WorkbenchRunDraft:
    pipeline_config: PipelineConfig
    universe_spec: UniverseSpec
    research_frequency: ResearchFrequency

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline_config, PipelineConfig):
            raise TypeError("pipeline_config must be a PipelineConfig.")
        if not isinstance(self.universe_spec, UniverseSpec):
            raise TypeError("universe_spec must be a UniverseSpec.")
        if not isinstance(self.research_frequency, ResearchFrequency):
            raise TypeError("research_frequency must be a ResearchFrequency.")


@dataclass(frozen=True)
class ReadinessRow:
    dataset_id: str
    scope: tuple[tuple[str, str], ...]
    required_start: str
    required_end: str
    required_units: int
    missing_units: tuple[str, ...]
    status: str
    action: str
    endpoint: str = ""
    official_minimum_points: int | str = "OFFICIAL_NOT_STATED"
    provider_id: str = "tushare_official"


@dataclass(frozen=True)
class DataReadinessPreview:
    rows: tuple[ReadinessRow, ...]
    research_plan: ResearchInputPlan | None
    research_inputs_reusable: bool
    calendar_bootstrap_required: bool = False

    @property
    def ready(self) -> bool:
        return bool(self.rows) and all(not row.missing_units for row in self.rows)


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    status: str
    dataset_id: str | None = None
    completed: int | None = None
    total: int | None = None
    detail: str | None = None


@dataclass(frozen=True)
class FirstRunResult:
    run: RunOutcome
    plan: ResearchInputPlan
    materialization: ResearchInputMaterialization
    prepared: DataPreparationResult
    provider_calls: int
    elapsed_seconds: float
    stages: tuple[ProgressEvent, ...]


class ReadOnlyCoverageLedger:
    """Coverage query adapter that never creates or mutates SQLite state."""

    def __init__(self, path: str | Path, *, provider_id: str = "tushare_official") -> None:
        self.path = Path(path)
        self.provider_id = provider_id

    def _has_provider_column(self) -> bool:
        if not self.path.is_file():
            return False
        with self._connect() as connection:
            return "provider_id" in {row[1] for row in connection.execute("PRAGMA table_info(coverage_units)")}

    def _connect(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def complete_units(self, dataset_id: str, scope_value: str, units: Iterable[str]) -> frozenset[str]:
        values = tuple(dict.fromkeys(units))
        if not values or not self.path.is_file():
            return frozenset()
        placeholders = ",".join("?" for _ in values)
        with self._connect() as connection:
            provider_clause = "provider_id=? AND " if self._has_provider_column() else ""
            params = (self.provider_id, dataset_id, scope_value, *values) if provider_clause else (dataset_id, scope_value, *values)
            rows = connection.execute(f"SELECT unit_key FROM coverage_units WHERE {provider_clause}dataset_id=? AND scope_key=? AND status='COMPLETE' AND unit_key IN ({placeholders})", params).fetchall()
        return frozenset(str(row["unit_key"]) for row in rows)

    def records(self, dataset_id: str | None = None) -> tuple[CoverageRecord, ...]:
        if not self.path.is_file():
            return ()
        has_provider = self._has_provider_column()
        sql, params = ("SELECT * FROM coverage_units WHERE provider_id=?", (self.provider_id,)) if has_provider else ("SELECT * FROM coverage_units", ())
        if dataset_id is not None:
            sql, params = (sql + (" AND dataset_id=?" if has_provider else " WHERE dataset_id=?"), (*params, dataset_id))
        with self._connect() as connection:
            values = []
            for row in connection.execute(sql + " ORDER BY dataset_id,scope_key,unit_key", params):
                item = dict(row)
                item.setdefault("provider_id", self.provider_id)
                values.append(CoverageRecord(**item))
            return tuple(values)


def create_workbench_factor_registry() -> FactorRegistry:
    """Mirror the existing Factor Research executor registry without formulas."""
    registry = FactorRegistry()
    register_example_factors(registry)
    register_price_volume_factors(registry)
    register_valuation_factors(registry)
    register_financial_factors(registry)
    return registry


def _calendar_days(start: str, end: str) -> tuple[str, ...]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return tuple((first + timedelta(days=index)).isoformat() for index in range((last - first).days + 1))


def _bootstrap_bounds(draft: WorkbenchRunDraft, registry: FactorRegistry) -> tuple[str, str]:
    start = date.fromisoformat(draft.pipeline_config.backtest_start)
    end = date.fromisoformat(draft.pipeline_config.backtest_end)
    lookback = max((registry.get(name).metadata.lookback_days for name in draft.pipeline_config.selected_factors), default=0)
    history_periods = _ml_history_periods(draft)
    period_days = 3 if draft.research_frequency is ResearchFrequency.DAILY else 35
    before = max(31, lookback * 3 + 14, history_periods * period_days + 14)
    forward = draft.pipeline_config.factor_research.forward_returns
    after = max(31 if draft.pipeline_config.research_backtest.enabled else 14, (forward.entry_lag_periods + forward.holding_periods) * 3 + 7)
    return (start - timedelta(days=before)).isoformat(), (end + timedelta(days=after)).isoformat()


def _ml_history_periods(draft: WorkbenchRunDraft) -> int:
    """Return the formation history needed for one leakage-safe ML split."""
    experiment = draft.pipeline_config.ml_experiment.experiment
    if not draft.pipeline_config.ml_experiment.enabled or experiment is None:
        return 1
    split = experiment.walk_forward_config
    forward = draft.pipeline_config.factor_research.forward_returns
    return (
        split.train_window_periods
        + split.validation_periods
        + split.embargo_periods
        + forward.entry_lag_periods
        + forward.holding_periods
        + 1
    )


def _bootstrap_requirement(draft: WorkbenchRunDraft, registry: FactorRegistry) -> DataRequirement:
    start, end = _bootstrap_bounds(draft, registry)
    return DataRequirement.create(
        "trade_cal", scope={"exchange": "SSE"}, required_start=start, required_end=end,
        required_fields=("cal_date", "is_open"), reason="Workbench first-run calendar bootstrap", as_of_cutoff=end,
    )


def _read_calendar(runtime: "WorkbenchRuntime", requirement: DataRequirement) -> ResearchCalendar:
    client = CanonicalPipelineMarketClient(
        registry=runtime.registry, ledger=runtime.ledger, store=runtime.curated,
        stock_basic_as_of=runtime.end_date, calendar_scope=requirement.scope,
    )
    frame = client.get_trade_cal(requirement.required_start, requirement.required_end)
    return ResearchCalendar(frame)


def _pipeline_requirements(draft: WorkbenchRunDraft, plan: ResearchInputPlan, calendar: ResearchCalendar) -> tuple[DataRequirement, ...]:
    if not draft.pipeline_config.research_backtest.enabled and draft.pipeline_config.holdings.portfolio_construction.method not in {"inverse_volatility", "minimum_variance"}:
        return ()
    config = draft.pipeline_config
    starts = [plan.formation_dates[0]]
    portfolio = config.holdings.portfolio_construction
    if portfolio.method == "inverse_volatility":
        starts.append(calendar.resolve_history(plan.formation_dates[0], HistoryRequirement.trading_days(int(portfolio.params["lookback_trading_days"]))).start_date)
    elif portfolio.method == "minimum_variance":
        risk = portfolio.params["risk_model"]
        starts.append(calendar.resolve_history(plan.formation_dates[0], HistoryRequirement.trading_days(int(risk["lookback_trading_days"]))).start_date)
    start = min(starts)
    end = config.backtest_end
    requirements = [DataRequirement.create("daily", scope="CN_A", required_start=start, required_end=end, required_fields=("ts_code", "trade_date", "pct_chg"), reason="existing Portfolio/Research Backtest security returns", as_of_cutoff=end)]
    if config.research_backtest.enabled:
        requirements.extend((
            DataRequirement.create("stock_basic", scope=STOCK_BASIC_SCOPE, required_start=end, required_end=end, required_fields=("ts_code", "list_status", "list_date", "delist_date"), reason="existing Research Backtest security lifecycle"),
            DataRequirement.create("index_daily", scope={"index_code": config.research_backtest.benchmark.benchmark_code}, required_start=plan.formation_dates[0], required_end=end, required_fields=("ts_code", "trade_date", "pct_chg"), reason="existing Research Backtest benchmark returns", as_of_cutoff=end),
        ))
        if config.research_backtest.suspension_mode == "STRICT_EVENT":
            requirements.append(DataRequirement.create("suspend_d", scope="CN_A", required_start=plan.formation_dates[0], required_end=end, required_fields=("ts_code", "trade_date", "suspend_type"), reason="strict Research Backtest suspension-event evidence", as_of_cutoff=end))
    return tuple(requirements)


def bind_materialized_inputs(config: PipelineConfig, materialization: ResearchInputMaterialization) -> PipelineConfig:
    """Map all five generated P4C3 inputs onto existing explicit file contracts."""
    values = config.to_dict()
    research = dict(values["factor_research"])
    research.update({
        "enabled": True,
        "factor_input_path": str(materialization.paths["factor_input.parquet"]),
        "score_panel_path": str(materialization.paths["score_panel.parquet"]),
        "price_panel_path": str(materialization.paths["price_panel.parquet"]),
        "exposure_panel_path": None,
    })
    values["factor_research"] = research
    modeling = dict(values["modeling_panel"])
    modeling["source"] = {
        "mode": "files",
        "factor_panel_path": str(materialization.paths["modeling_factor_panel.parquet"]),
        "forward_returns_path": str(materialization.paths["modeling_forward_returns.parquet"]),
    }
    values["modeling_panel"] = modeling
    values["required_datasets"] = []
    return PipelineConfig.from_dict(values)


class WorkbenchRuntime:
    def __init__(self, config: PipelineConfig, *, root: str | Path | None = None, factor_registry: FactorRegistry | None = None, client_factory: Callable[[str], object] | None = None, read_only: bool = False) -> None:
        self.project_root = Path(root or Path(__file__).resolve().parents[2]).resolve()
        self.config = config
        self.end_date = config.backtest_end
        data_root = Path(config.data_root)
        if not data_root.is_absolute():
            data_root = self.project_root / data_root
        base_data_root = data_root.resolve()
        self.provider_id = ProviderId(config.provider_id).value
        self.data_root = (
            base_data_root
            if self.provider_id == ProviderId.TUSHARE_OFFICIAL.value
            else base_data_root / "providers" / self.provider_id
        )
        self.registry = create_default_dataset_registry()
        self.factor_registry = factor_registry or create_workbench_factor_registry()
        ledger_path = self.data_root / "metadata" / "catalog.sqlite"
        self.ledger = ReadOnlyCoverageLedger(ledger_path, provider_id=self.provider_id) if read_only else CoverageLedger(ledger_path, provider_id=self.provider_id)
        self.curated = PartitionedParquetStore(self.data_root / "curated", engine="pyarrow", provider_id=self.provider_id)
        self.raw = RawParquetStore(self.data_root / "raw", engine="pyarrow", provider_id=self.provider_id)
        self.client_factory = client_factory or (lambda token: ProviderClientFactory().create(self.provider_id, token))

    def preparation(self, *, open_dates: Callable[[str, str], Iterable[object]] | None = None) -> DataPreparationService:
        return DataPreparationService(registry=self.registry, ledger=self.ledger, curated_store=self.curated, raw_store=self.raw, open_dates=open_dates, client_factory=self.client_factory)

    def research_builder(self, draft: WorkbenchRunDraft, plan: ResearchInputPlan, calendar: ResearchCalendar) -> ResearchInputBuilder:
        required_start = min(item.required_start for item in plan.requirements)
        required_end = max(item.required_end for item in plan.requirements)
        universe = CanonicalUniverseDataSource(
            registry=self.registry, ledger=self.ledger, store=self.curated,
            stock_basic_as_of=plan.end_date,
            index_weight_start=(pd.Timestamp(plan.start_date).to_period("M") - 1).start_time.date().isoformat(),
            stock_basic_required_start=required_start,
            stock_basic_required_end=required_end,
        )
        market = CanonicalAdjustedPriceDataSource(registry=self.registry, ledger=self.ledger, store=self.curated, scope="CN_A")
        research = draft.pipeline_config.factor_research
        if research.research is None:
            raise WorkbenchRunError(WorkbenchErrorCode.UNSUPPORTED, "validate")
        runner = FactorResearchRunner(
            self.factor_registry, research.research,
            preprocessing_config=research.preprocessing,
            neutralization_config=research.neutralization,
            evaluation_config=research.evaluation,
            quantile_config=research.quantile,
            composition_config=research.composition,
            rolling_config=research.rolling,
            forward_return_config=research.forward_returns,
        )
        return ResearchInputBuilder(
            calendar=calendar, universe_service=UniverseService(), universe_data=universe,
            factor_registry=self.factor_registry, dataset_source=market,
            adjusted_prices=AdjustedPriceService(market), factor_runner=runner,
            store=ResearchMaterializationStore(self.data_root / "research_inputs"),
        )

    def pipeline_service(self, draft: WorkbenchRunDraft) -> RunService:
        client = CanonicalPipelineMarketClient(
            registry=self.registry, ledger=self.ledger, store=self.curated,
            stock_basic_as_of=draft.pipeline_config.backtest_end,
        )
        def execute(config: PipelineConfig, *, run_created_callback=None, stage_callback=None):
            return run_pipeline(config, market_client_factory=lambda: client, run_created_callback=run_created_callback, stage_callback=stage_callback)
        return RunService(execute, supports_identity_hook=True)


class FirstRunOrchestrator:
    STAGES = ("validate", "plan", "check", "download", "build", "pipeline", "artifacts", "complete")

    def __init__(self, runtime_factory: Callable[[PipelineConfig], WorkbenchRuntime] | None = None, preview_runtime_factory: Callable[[PipelineConfig], WorkbenchRuntime] | None = None) -> None:
        self.runtime_factory = runtime_factory or (lambda config: WorkbenchRuntime(config))
        self.preview_runtime_factory = preview_runtime_factory or (lambda config: WorkbenchRuntime(config, read_only=True))

    @staticmethod
    def _event(
        events: list[ProgressEvent],
        callback: Callable[[ProgressEvent], None] | None,
        stage: str,
        status: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        detail: str | None = None,
    ) -> None:
        event = ProgressEvent(stage, status, completed=completed, total=total, detail=detail)
        events.append(event)
        if callback is not None:
            callback(event)

    @staticmethod
    def _rows(plans: Iterable[object], provider_id: str = "tushare_official") -> tuple[ReadinessRow, ...]:
        from src.data.provider_contracts import ProviderContractRegistry
        contracts = ProviderContractRegistry()
        rows = []
        for item in plans:
            missing = tuple(item.missing_units)
            complete = len(item.complete_units)
            status = "READY" if not missing else ("PARTIAL" if complete else "MISSING")
            contract = contracts.get(provider_id, item.requirement.dataset_id)
            rows.append(ReadinessRow(item.requirement.dataset_id, item.requirement.scope, item.requirement.required_start, item.requirement.required_end, len(item.required_units), missing, status, "REUSE_LOCAL" if not missing else "DOWNLOAD_MISSING", contract.api_name, contract.minimum_points, provider_id))
        return tuple(rows)

    def _plan(self, draft: WorkbenchRunDraft, runtime: WorkbenchRuntime, calendar: ResearchCalendar) -> ResearchInputPlan:
        forward = ForwardReturnSpec.from_config(draft.pipeline_config.factor_research.forward_returns)
        requested = calendar.formation_dates(
            draft.research_frequency,
            draft.pipeline_config.backtest_start,
            draft.pipeline_config.backtest_end,
        )
        first = requested[0]
        research_end = draft.pipeline_config.backtest_end
        if (
            draft.pipeline_config.research_backtest.enabled
            and requested[-1] == canonical_date(research_end)
        ):
            if len(requested) < 2:
                raise WorkbenchRunError(WorkbenchErrorCode.UNSUPPORTED, "plan")
            research_end = requested[-2]
        periods = _ml_history_periods(draft)
        history = (
            HistoryRequirement.trading_days(periods)
            if draft.research_frequency is ResearchFrequency.DAILY
            else HistoryRequirement.calendar_months(periods)
        )
        research_start = min(
            calendar.resolve_history(first, history).start_date,
            draft.pipeline_config.required_start_date,
        )
        return ResearchInputPlanner(calendar=calendar, universe_service=UniverseService(), factor_registry=runtime.factor_registry).build(
            research_frequency=draft.research_frequency,
            start_date=research_start,
            end_date=research_end,
            universe_spec=draft.universe_spec,
            factor_ids=draft.pipeline_config.selected_factors,
            forward_return_spec=forward,
        )

    def preview(self, draft: WorkbenchRunDraft) -> DataReadinessPreview:
        runtime = self.preview_runtime_factory(draft.pipeline_config)
        bootstrap = _bootstrap_requirement(draft, runtime.factor_registry)
        bootstrap_service = runtime.preparation()
        calendar_plan = bootstrap_service.inspect((bootstrap,))[0]
        if not calendar_plan.ready:
            return DataReadinessPreview(self._rows((calendar_plan,), runtime.provider_id), None, False, True)
        calendar = _read_calendar(runtime, bootstrap)
        plan = self._plan(draft, runtime, calendar)
        resolver = CuratedTradingCalendarResolver(runtime.registry, runtime.ledger, runtime.curated, scope={"exchange": "SSE"})
        preparation = runtime.preparation(open_dates=resolver)
        requirements = compose_requirements((*plan.requirements, *_pipeline_requirements(draft, plan, calendar)))
        missing = preparation.inspect(requirements)
        reusable = False
        if all(item.ready for item in missing):
            reusable = runtime.research_builder(draft, plan, calendar).inspect(plan).reusable
        return DataReadinessPreview(self._rows(missing, runtime.provider_id), plan, reusable)

    def run(self, draft: WorkbenchRunDraft, *, credential: str | None, progress: Callable[[ProgressEvent], None] | None = None) -> FirstRunResult:
        started = perf_counter()
        events: list[ProgressEvent] = []
        self._event(events, progress, "validate", "STARTED")
        if draft.pipeline_config.factor_research.research is None or draft.pipeline_config.factor_research.research.use_neutralization:
            raise WorkbenchRunError(WorkbenchErrorCode.UNSUPPORTED, "validate")
        runtime = self.runtime_factory(draft.pipeline_config)
        active_stage = "check"
        bootstrap = _bootstrap_requirement(draft, runtime.factor_registry)
        bootstrap_service = runtime.preparation()
        calendar_missing = bootstrap_service.inspect((bootstrap,))
        if any(not item.ready for item in calendar_missing) and not credential:
            raise WorkbenchRunError(WorkbenchErrorCode.CREDENTIAL_MISSING, "check")
        self._event(events, progress, "plan", "STARTED")
        try:
            active_stage = "download"
            bootstrap_result = bootstrap_service.ensure((bootstrap,), credential)
            active_stage = "plan"
            calendar = _read_calendar(runtime, bootstrap)
            plan = self._plan(draft, runtime, calendar)
            resolver = CuratedTradingCalendarResolver(runtime.registry, runtime.ledger, runtime.curated, scope={"exchange": "SSE"})
            preparation = runtime.preparation(open_dates=resolver)
            requirements = compose_requirements((*plan.requirements, *_pipeline_requirements(draft, plan, calendar)))
            local = preparation.inspect(requirements)
            self._event(events, progress, "check", "COMPLETE")
            if any(not item.ready for item in local) and not credential:
                raise WorkbenchRunError(WorkbenchErrorCode.CREDENTIAL_MISSING, "check")
            active_stage = "download"
            missing_total = sum(len(item.missing_units) for item in local)
            required_total = sum(len(item.required_units) for item in local)
            complete_total = sum(len(item.complete_units) for item in local)
            self._event(
                events,
                progress,
                "download",
                "STARTED" if missing_total else "SKIPPED",
                completed=complete_total if required_total else None,
                total=required_total or None,
                detail="Downloading only ledger-missing coverage." if missing_total else "All required coverage is COMPLETE; provider calls are skipped.",
            )
            prepared = preparation.ensure(
                requirements,
                credential,
                progress=lambda dataset_id, unit, completed, total: self._event(
                    events,
                    progress,
                    "download",
                    "STARTED",
                    completed=completed,
                    total=total,
                    detail=f"{dataset_id} · {unit}",
                ),
            )
            self._event(events, progress, "download", "COMPLETE", completed=required_total or None, total=required_total or None)
            self._event(events, progress, "build", "STARTED")
            active_stage = "build"
            materialization = runtime.research_builder(draft, plan, calendar).build(plan)
            self._event(events, progress, "build", "COMPLETE")
            bound = bind_materialized_inputs(draft.pipeline_config, materialization)
            self._event(events, progress, "pipeline", "STARTED")
            active_stage = "pipeline"
            outcome = runtime.pipeline_service(draft).run(
                bound,
                stage_callback=lambda stage, status: self._event(events, progress, stage, status),
            )
            if not outcome.success or not outcome.run_id:
                raise WorkbenchRunError(
                    WorkbenchErrorCode.PIPELINE_ERROR,
                    "pipeline",
                    outcome.run_id,
                    outcome.error,
                )
            run_dir = ExperimentManager(bound.output_dir).resolve_run_dir(outcome.run_id)
            contracts = ProviderContractRegistry()
            datasets = sorted({item.dataset_id for item in requirements})
            provenance = {
                "provider_id": bound.provider_id,
                "datasets": [{
                    "dataset_id": dataset_id,
                    "endpoint": contracts.get(bound.provider_id, dataset_id).api_name,
                    "schema_version": runtime.registry.get(dataset_id).schema_version,
                } for dataset_id in datasets],
                "coverage_start": min(item.required_start for item in requirements),
                "coverage_end": max(item.required_end for item in requirements),
                "quality_conclusion": "canonical and provider quality contracts passed",
                "degraded": bound.research_backtest.suspension_mode == "STANDARD_ROBUST",
                "degradation_reason": "unconfirmed missing daily rows freeze trades" if bound.research_backtest.suspension_mode == "STANDARD_ROBUST" else None,
                "cross_provider_comparison": "NOT_RUN",
                "point_in_time_rule": "explicit as-of cutoffs and historical index membership",
            }
            (run_dir / "data_provenance.json").write_text(
                json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            self._event(events, progress, "pipeline", "COMPLETE")
            self._event(events, progress, "artifacts", "STARTED")
            active_stage = "artifacts"
            ResultService(bound.output_dir).load(outcome.run_id)
            self._event(events, progress, "artifacts", "COMPLETE")
            self._event(events, progress, "complete", "COMPLETE")
            return FirstRunResult(outcome, plan, materialization, prepared, bootstrap_result.provider_calls + prepared.provider_calls, perf_counter() - started, tuple(events))
        except WorkbenchRunError:
            raise
        except UnsupportedLegacySecurityIdentifier:
            raise WorkbenchRunError(
                WorkbenchErrorCode.UNSUPPORTED_LEGACY_SECURITY_IDENTIFIER,
                "quality_validation",
                user_message=(
                    "A cached provider reference overlaps the complete required interval without a verified tradable mapping."
                ),
            ) from None
        except MissingCredentialError as exc:
            raise WorkbenchRunError(WorkbenchErrorCode.CREDENTIAL_MISSING, "download") from None
        except DataUnavailableError as exc:
            code = classify_data_unavailable_error(exc)
            missing_range = (
                None
                if exc.dataset_id
                and runtime.registry.get(exc.dataset_id).coverage_kind is CoverageKind.GLOBAL_SNAPSHOT
                else (exc.units[0], exc.units[-1]) if exc.units else None
            )
            diagnostic = None
            if exc.dataset_id and exc.units:
                unit = exc.units[0]
                scope = exc.scope or (('scope', 'CN_A'),)
                spec = runtime.registry.get(exc.dataset_id)
                matching = [
                    record for record in runtime.ledger.records(exc.dataset_id)
                    if record.scope_key == scope_key(scope) and record.unit_key == unit
                ]
                ledger_status = matching[0].status if len(matching) == 1 else "MISSING"
                try:
                    rows = runtime.curated.rows_for_unit(spec, unit=unit, scope=scope)
                    if len(rows):
                        canonical_status = f"READABLE_ROWS:{len(rows)}"
                    elif runtime.curated.has_empty_marker(spec, unit=unit, scope=scope):
                        canonical_status = "EMPTY_MARKER"
                    else:
                        canonical_status = "MISSING"
                except Exception:
                    canonical_status = "UNREADABLE"
                attempts = getattr(exc.safe_cause, "provider_attempts", None)
                transition = None
                try:
                    transition = runtime.ledger.latest_transition(
                        exc.dataset_id, scope_key(scope), unit
                    )
                except (AttributeError, sqlite3.Error, ValueError):
                    transition = None
                transition_fields: tuple[str, ...] = ()
                transition_quality_evidence: dict[str, object] = {}
                if transition is not None:
                    try:
                        parsed_fields = json.loads(transition.fields)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        parsed_fields = ()
                    if isinstance(parsed_fields, list):
                        transition_fields = tuple(
                            str(value) for value in parsed_fields if isinstance(value, str)
                        )
                    try:
                        parsed_evidence = json.loads(transition.quality_evidence)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        parsed_evidence = {}
                    transition_quality_evidence = sanitize_identifier_evidence(
                        parsed_evidence
                    )
                diagnostic = FailureDiagnostic(
                    ledger_status=ledger_status,
                    canonical_status=canonical_status,
                    consistency_issue=(
                        "ledger/canonical mismatch"
                        if ledger_status == "COMPLETE" and canonical_status in {"MISSING", "UNREADABLE"}
                        else None
                    ),
                    repair_action=(
                        "review quarantined provider identifiers without assuming a canonical mapping"
                        if exc.origin == "provider_quality"
                        else "refetch missing unit and publish canonical proof before marking COMPLETE"
                    ),
                    provider_attempts=attempts,
                    network_category=_network_category(exc.safe_cause),
                    transaction_fetch_id=transition.fetch_id if transition else None,
                    transaction_state=transition.state if transition else None,
                    transaction_operation=transition.operation if transition else None,
                    transaction_error_code=transition.error_code if transition else None,
                    transaction_exception_type=transition.exception_type if transition else None,
                    transaction_cause_type=transition.exception_cause_type if transition else None,
                    transaction_message=transition.safe_message if transition else None,
                    transaction_rows=transition.rows if transition else None,
                    transaction_fields=transition_fields,
                    transaction_quality_evidence=transition_quality_evidence,
                )
            failure_stage = classify_data_unavailable_stage(exc, active_stage)
            raise WorkbenchRunError(
                code,
                failure_stage,
                dataset_id=exc.dataset_id,
                missing_range=missing_range,
                user_message=(
                    "Provider data failed the quality gate before canonical publication. Review the bounded identifier evidence; no mapping was assumed."
                    if exc.origin == "provider_quality"
                    else "Data preparation failed before research calculation started. Review the dataset, missing range, and recommended recovery action."
                ),
                diagnostic=diagnostic,
            ) from None
        except (CanonicalDataError, AdjustedPriceError, ResearchInputError):
            raise WorkbenchRunError(WorkbenchErrorCode.COVERAGE_VALIDATION, active_stage) from None
        except Exception as exc:
            if active_stage != "download":
                raise WorkbenchRunError(WorkbenchErrorCode.COVERAGE_VALIDATION, active_stage) from None
            kind = classify_provider_error(exc)
            mapping = {
                ProviderErrorKind.AUTHENTICATION_INVALID: WorkbenchErrorCode.AUTHENTICATION_INVALID,
                ProviderErrorKind.PERMISSION_INSUFFICIENT: WorkbenchErrorCode.PERMISSION_INSUFFICIENT,
                ProviderErrorKind.POINTS_INSUFFICIENT: WorkbenchErrorCode.POINTS_INSUFFICIENT,
                ProviderErrorKind.RATE_LIMITED: WorkbenchErrorCode.RATE_LIMITED,
                ProviderErrorKind.NETWORK_ERROR: WorkbenchErrorCode.NETWORK_ERROR,
                ProviderErrorKind.RESPONSE_INVALID: WorkbenchErrorCode.PROVIDER_RESPONSE_INVALID,
                ProviderErrorKind.PROVIDER_ERROR: WorkbenchErrorCode.PROVIDER_ERROR,
            }
            raise WorkbenchRunError(mapping[kind], active_stage) from None
