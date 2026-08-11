"""Deterministic test-process limits for restricted Windows environments."""

from __future__ import annotations

import os


# Joblib otherwise invokes an operating-system helper to discover physical CPU
# topology. That probe is irrelevant to these deterministic tests and is
# denied in restricted Windows runners. This does not alter production code.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
