"""Session-first TuShare credential resolution and sanitized connection checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Callable

from src.data.credentials import CredentialProvider, EnvironmentCredentialProvider
from src.data.tushare_client import TushareClient


class ProviderErrorKind(str, Enum):
    CREDENTIAL_MISSING = "CREDENTIAL_MISSING"
    AUTHENTICATION_INVALID = "AUTHENTICATION_INVALID"
    PERMISSION_INSUFFICIENT = "PERMISSION_INSUFFICIENT"
    POINTS_INSUFFICIENT = "POINTS_INSUFFICIENT"
    RATE_LIMITED = "RATE_LIMITED"
    NETWORK_ERROR = "NETWORK_ERROR"
    RESPONSE_INVALID = "RESPONSE_INVALID"
    PROVIDER_ERROR = "PROVIDER_ERROR"


@dataclass(frozen=True, repr=False)
class ResolvedCredential:
    _token: str | None
    source: str

    def __post_init__(self) -> None:
        token = self._token
        if token is not None and (not isinstance(token, str) or not token.strip()):
            raise ValueError("credential token must be a non-empty string or None.")
        if self.source not in {"session", "environment", "none"}:
            raise ValueError("credential source is invalid.")
        object.__setattr__(self, "_token", token.strip() if token else None)

    @property
    def available(self) -> bool:
        return self._token is not None

    def reveal_for_provider(self) -> str | None:
        """Return the secret only at the provider-call boundary."""
        return self._token


@dataclass(frozen=True)
class ConnectionResult:
    success: bool
    error_kind: ProviderErrorKind | None = None


def classify_provider_error(exc: BaseException) -> ProviderErrorKind:
    """Classify without returning provider text, which may contain a credential."""
    chain: list[str] = []
    current: BaseException | None = exc
    while current is not None and len(chain) < 6:
        chain.append(f"{type(current).__name__} {current}")
        current = current.__cause__ or current.__context__
    text = " ".join(chain).lower()
    if re.search(r"points?|积分", text):
        return ProviderErrorKind.POINTS_INSUFFICIENT
    if re.search(r"rate.?limit|too many|frequency|频率|限流", text):
        return ProviderErrorKind.RATE_LIMITED
    if re.search(r"permission|quota|权限", text):
        return ProviderErrorKind.PERMISSION_INSUFFICIENT
    if re.search(r"auth|invalid.*token|token.*invalid|credential|认证", text):
        return ProviderErrorKind.AUTHENTICATION_INVALID
    if re.search(r"timeout|connection|network|dns|socket|网络", text):
        return ProviderErrorKind.NETWORK_ERROR
    if re.search(r"missing required columns|structure|schema|json|decode|结构|字段", text):
        return ProviderErrorKind.RESPONSE_INVALID
    return ProviderErrorKind.PROVIDER_ERROR


class CredentialService:
    def __init__(
        self,
        *,
        environment: CredentialProvider | None = None,
        client_factory: Callable[[str], object] | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self.environment = environment or EnvironmentCredentialProvider(root)
        self.client_factory = client_factory or (lambda token: TushareClient(token))

    def resolve(self, session_token: object = None) -> ResolvedCredential:
        if isinstance(session_token, str) and session_token.strip():
            return ResolvedCredential(session_token, "session")
        value = self.environment.tushare_token()
        if value:
            return ResolvedCredential(value, "environment")
        return ResolvedCredential(None, "none")

    def test_connection(self, session_token: object = None) -> ConnectionResult:
        credential = self.resolve(session_token)
        token = credential.reveal_for_provider()
        if token is None:
            return ConnectionResult(False, ProviderErrorKind.CREDENTIAL_MISSING)
        try:
            client = self.client_factory(token)
            method = getattr(client, "get_trade_cal")
            method(start_date="20240102", end_date="20240102")
        except Exception as exc:
            return ConnectionResult(False, classify_provider_error(exc))
        return ConnectionResult(True)
