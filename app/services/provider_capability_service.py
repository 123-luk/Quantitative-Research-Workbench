"""Minimal, serial provider capability probes with sanitized outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from app.services.credential_service import ProviderErrorKind, classify_provider_error
from src.data.provider_contracts import ProviderContractRegistry
from src.data.provider_registry import ProviderClientFactory, ProviderId


@dataclass(frozen=True)
class CapabilityProbe:
    dataset_id: str
    status: str
    error_kind: str | None
    dates: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityReport:
    provider_id: str
    probes: tuple[CapabilityProbe, ...]

    def status_for(self, dataset_id: str) -> str | None:
        matches = tuple(item.status for item in self.probes if item.dataset_id == dataset_id)
        return "AVAILABLE" if matches and all(value == "AVAILABLE" for value in matches) else (matches[0] if matches else None)


class ProviderCapabilityService:
    """Probe without concurrency, frequency discovery, persistence, or failover."""

    def __init__(self, factory: ProviderClientFactory | None = None) -> None:
        self.factory = factory or ProviderClientFactory()

    @staticmethod
    def _calls(client: object) -> tuple[tuple[str, tuple[str, ...], Callable[[], object]], ...]:
        return (
            ("trade_cal", ("2020-03-12",), lambda: client.get_trade_cal("20200312", "20200312")),
            ("stock_basic", ("REFERENCE",), lambda: client.get_stock_basic("L")),
            ("daily", ("2020-03-12",), lambda: client.get_daily(trade_date="20200312")),
            ("daily_basic", ("2020-03-12",), lambda: client.get_daily_basic(trade_date="20200312")),
            ("adj_factor", ("2020-03-12",), lambda: client.get_adj_factor(trade_date="20200312")),
            ("stk_limit", ("2020-03-12",), lambda: client.get_stk_limit(trade_date="20200312")),
            ("suspend_d", ("2020-03-12",), lambda: client.get_suspend_d(trade_date="20200312")),
            ("suspend_d", ("2023-11-16",), lambda: client.get_suspend_d(trade_date="20231116")),
            ("index_weight", ("2020-03",), lambda: client.get_index_weight("000300.SH", "20200301", "20200331")),
            ("index_daily", ("2020-03-12",), lambda: client.get_index_daily("000300.SH", trade_date="20200312")),
            ("monthly", ("2020-03",), lambda: client.get_monthly("000001.SZ", "20200301", "20200331")),
        )

    def run(self, provider_id: str, token: str) -> CapabilityReport:
        provider = ProviderId(provider_id).value
        client = self.factory.create(provider, token)
        contracts = ProviderContractRegistry()
        results: list[CapabilityProbe] = []
        for dataset_id, dates, call in self._calls(client):
            try:
                value = call()
                contract = contracts.get(provider, dataset_id)
                if not isinstance(value, pd.DataFrame):
                    raise TypeError("provider result is not a DataFrame")
                if value.empty and dataset_id != "suspend_d":
                    status = "EMPTY_UNVERIFIED"
                elif not set(contract.output_fields).issubset(value.columns):
                    status = "SCHEMA_INCOMPATIBLE"
                else:
                    status = "AVAILABLE"
                results.append(CapabilityProbe(dataset_id, status, None, dates))
            except Exception as exc:
                kind = classify_provider_error(exc)
                results.append(CapabilityProbe(dataset_id, {
                    ProviderErrorKind.PERMISSION_INSUFFICIENT: "PERMISSION_INSUFFICIENT",
                    ProviderErrorKind.POINTS_INSUFFICIENT: "POINTS_INSUFFICIENT",
                    ProviderErrorKind.RATE_LIMITED: "RATE_LIMITED",
                    ProviderErrorKind.NETWORK_ERROR: "TEMPORARY_TIMEOUT_OR_NETWORK",
                    ProviderErrorKind.RESPONSE_INVALID: "SCHEMA_INCOMPATIBLE",
                    ProviderErrorKind.AUTHENTICATION_INVALID: "AUTHENTICATION_INVALID",
                    ProviderErrorKind.CREDENTIAL_MISSING: "CREDENTIAL_MISSING",
                    ProviderErrorKind.PROVIDER_ERROR: "PROVIDER_ERROR_OR_ENDPOINT_UNAVAILABLE",
                }[kind], kind.value, dates))
        return CapabilityReport(provider, tuple(results))
