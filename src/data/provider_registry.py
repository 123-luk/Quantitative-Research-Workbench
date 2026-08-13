"""Central provider identities and safe TuShare client construction.

The third-party proxy is deliberately represented as a distinct provider.  It
is not an official TuShare service and it never participates in automatic
failover.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from src.data.tushare_client import TushareClient


class ProviderId(str, Enum):
    TUSHARE_OFFICIAL = "tushare_official"
    TUSHARE_PROXY = "tushare_proxy"


PROXY_HTTPS_ENDPOINT = "https://tuaremax.top"


class ProviderCompatibilityError(RuntimeError):
    """Raised when the proxy's documented SDK integration is unavailable."""


class TushareProxyClient(TushareClient):
    """TuShare-SDK compatible client pinned to the configured HTTPS proxy.

    The two private SDK fields are intentionally confined to this adapter.
    TuShare's global token state is never mutated.
    """

    provider_id = ProviderId.TUSHARE_PROXY.value

    def _configure_client(self, pro: object, token: str) -> object:
        token_field = "_DataApi__token"
        url_field = "_DataApi__http_url"
        if not hasattr(pro, token_field) or not hasattr(pro, url_field):
            raise ProviderCompatibilityError(
                "代理接口与当前 TuShare SDK 版本不兼容。"
            )
        setattr(pro, token_field, token)
        setattr(pro, url_field, PROXY_HTTPS_ENDPOINT)
        if getattr(pro, token_field) != token or getattr(pro, url_field) != PROXY_HTTPS_ENDPOINT:
            raise ProviderCompatibilityError(
                "代理接口与当前 TuShare SDK 版本不兼容。"
            )
        return pro


@dataclass(frozen=True)
class ProviderDefinition:
    provider_id: ProviderId
    display_name_zh: str
    display_name_en: str
    third_party: bool
    endpoint: str | None
    advertised_access: str


class ProviderRegistry:
    """Closed registry for providers approved by the application."""

    def __init__(self) -> None:
        self._items = {
            ProviderId.TUSHARE_OFFICIAL: ProviderDefinition(
                ProviderId.TUSHARE_OFFICIAL,
                "官方接口",
                "Official API",
                False,
                None,
                "TuShare official endpoint-specific rules",
            ),
            ProviderId.TUSHARE_PROXY: ProviderDefinition(
                ProviderId.TUSHARE_PROXY,
                "代理接口",
                "Proxy API",
                True,
                PROXY_HTTPS_ENDPOINT,
                "PROXY_ADVERTISED_5000_LEVEL_NOT_OFFICIALLY_VERIFIED",
            ),
        }

    def get(self, provider_id: str | ProviderId) -> ProviderDefinition:
        try:
            key = provider_id if isinstance(provider_id, ProviderId) else ProviderId(provider_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("Unsupported provider_id.") from exc
        return self._items[key]

    def list(self) -> tuple[ProviderDefinition, ...]:
        return tuple(self._items[key] for key in ProviderId)


class ProviderClientFactory:
    """Construct exactly one explicitly selected provider client."""

    def __init__(
        self,
        *,
        official_factory: Callable[[str], object] | None = None,
        proxy_factory: Callable[[str], object] | None = None,
    ) -> None:
        self._factories = {
            ProviderId.TUSHARE_OFFICIAL: official_factory or (lambda token: TushareClient(token)),
            ProviderId.TUSHARE_PROXY: proxy_factory or (lambda token: TushareProxyClient(token)),
        }

    def create(self, provider_id: str | ProviderId, token: str) -> object:
        definition = ProviderRegistry().get(provider_id)
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Provider token must be a non-empty string.")
        return self._factories[definition.provider_id](token.strip())
