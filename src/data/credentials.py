"""Credential resolution boundaries that never persist or expose token values."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv


class CredentialProvider(Protocol):
    def tushare_token(self) -> str | None: ...


class EnvironmentCredentialProvider:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)

    def tushare_token(self) -> str | None:
        load_dotenv(self.project_root / ".env", override=False)
        value = os.getenv("TUSHARE_TOKEN")
        return value.strip() if value and value.strip() else None
