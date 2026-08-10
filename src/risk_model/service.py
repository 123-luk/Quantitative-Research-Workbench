"""Historical covariance service with exact common complete-case alignment."""

from __future__ import annotations

import math

import numpy as np

from src.portfolio_construction.contracts import HistoricalReturnService, HistoricalReturnWindow
from src.portfolio_construction.errors import PortfolioConstructionDataError

from .contracts import RiskEstimate, RiskModelRequest, RiskModelResult
from .errors import RiskModelDataError, RiskModelValidationError
from .registry import RiskEstimatorRegistry, build_default_risk_estimator_registry


class HistoricalCovarianceRiskModelService:
    def __init__(self, historical_returns: HistoricalReturnService, registry: RiskEstimatorRegistry | None = None) -> None:
        if not callable(getattr(historical_returns, "load_window", None)):
            raise RiskModelValidationError("historical_returns must provide load_window.")
        self._historical_returns = historical_returns
        self._registry = build_default_risk_estimator_registry() if registry is None else registry
        if not isinstance(self._registry, RiskEstimatorRegistry):
            raise RiskModelValidationError("registry must be RiskEstimatorRegistry.")

    def estimate(self, request: RiskModelRequest) -> RiskModelResult:
        if not isinstance(request, RiskModelRequest):
            raise RiskModelValidationError("request must be RiskModelRequest.")
        try:
            window = self._historical_returns.load_window(
                request.assets, request.formation_date, request.config.lookback_trading_days
            )
        except PortfolioConstructionDataError as exc:
            raise RiskModelDataError(str(exc)) from exc
        except Exception as exc:
            raise RiskModelDataError("historical return service failed.") from exc
        if not isinstance(window, HistoricalReturnWindow):
            raise RiskModelDataError("historical return service must return HistoricalReturnWindow.")
        returns = window.returns
        if window.risk_cutoff > request.formation_date or (
            not returns.empty and bool((returns["trade_date"] > window.risk_cutoff).any())
        ) or (not returns.empty and bool((returns["trade_date"] > request.formation_date).any())):
            raise RiskModelDataError("historical return window contains future data.")
        observed = set(str(item) for item in returns["ts_code"])
        if not observed.issubset(set(request.assets)):
            raise RiskModelDataError("historical return window contains unrequested assets.")
        wide = returns.pivot(index="trade_date", columns="ts_code", values="return")
        wide = wide.reindex(columns=list(request.assets)).sort_index()
        aligned = wide.dropna(axis=0, how="any")
        observation_count = len(aligned)
        if observation_count > request.config.lookback_trading_days:
            raise RiskModelDataError("common return window exceeds configured lookback.")
        if observation_count < request.config.min_observations:
            raise RiskModelDataError("insufficient common return observations.")
        values = aligned.to_numpy(dtype=np.float64, copy=True)
        if values.shape != (observation_count, len(request.assets)) or not np.isfinite(values).all():
            raise RiskModelDataError("aligned return matrix must be dense and finite.")
        estimator = self._registry.resolve(request.config.estimator)
        params = estimator.parse_params(request.config.params)
        estimate = estimator.estimate(values, params)
        if not isinstance(estimate, RiskEstimate):
            raise RiskModelValidationError("risk estimator must return RiskEstimate.")
        covariance = np.asarray(estimate.covariance, dtype=np.float64)
        if covariance.shape != (len(request.assets), len(request.assets)):
            raise RiskModelValidationError("estimator covariance has the wrong shape.")
        eigenvalues = np.linalg.eigvalsh(covariance)
        condition = float(np.linalg.cond(covariance))
        matrix_diagnostics: dict[str, object] = {
            "min_eigenvalue": float(eigenvalues[0]),
            "max_eigenvalue": float(eigenvalues[-1]),
            "condition_number": condition if math.isfinite(condition) else None,
            "condition_status": "finite" if math.isfinite(condition) else "singular",
        }
        diagnostics = {
            "aligned_start": aligned.index[0].strftime("%Y-%m-%d"),
            "aligned_end": aligned.index[-1].strftime("%Y-%m-%d"),
            "matrix": matrix_diagnostics,
            "estimator": dict(estimate.diagnostics),
        }
        return RiskModelResult(
            formation_date=request.formation_date,
            risk_cutoff=window.risk_cutoff,
            assets=request.assets,
            covariance=covariance,
            observation_count=observation_count,
            estimator=estimator.name,
            diagnostics=diagnostics,
        )
