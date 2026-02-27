"""Shared rate limiter instance.

Centralised here to avoid circular imports when auth_routes or widget
routers need to apply per-endpoint limits while main.py owns the FastAPI
app setup.

Limits are configurable via environment variables:
    RATE_LIMIT_DEFAULT   – default per-IP limit  (default: "100/minute")
    RATE_LIMIT_AUTH      – auth endpoint limit    (default: "10/minute")
    RATE_LIMIT_UPLOAD    – file-upload limit      (default: "20/minute")
    RATE_LIMIT_STORAGE_URI – slowapi storage backend
                             (default: "memory://", use "redis://…" for
                              distributed deployments)
"""

import os
import re

from slowapi import Limiter
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Storage backend
# ---------------------------------------------------------------------------
# Use in-process memory by default.  Point to a Redis instance in production:
#   RATE_LIMIT_STORAGE_URI=redis://redis:6379
_storage_uri: str = os.environ.get("RATE_LIMIT_STORAGE_URI", "memory://")

# ---------------------------------------------------------------------------
# Limit strings (can be overridden via environment)
# ---------------------------------------------------------------------------
_LIMIT_PATTERN = re.compile(r"^\d+/(second|minute|hour|day)s?$", re.IGNORECASE)


def _validate_limit(value: str, env_var: str) -> str:
    """Validate a rate limit string.  Raises ValueError on bad format."""
    if not _LIMIT_PATTERN.match(value):
        raise ValueError(
            f"Invalid rate limit for {env_var}: {value!r}. "
            "Expected format: '<number>/<second|minute|hour|day>' "
            "(e.g. '10/minute')."
        )
    return value


LIMIT_DEFAULT: str = _validate_limit(
    os.environ.get("RATE_LIMIT_DEFAULT", "100/minute"), "RATE_LIMIT_DEFAULT"
)
LIMIT_AUTH: str = _validate_limit(
    os.environ.get("RATE_LIMIT_AUTH", "10/minute"), "RATE_LIMIT_AUTH"
)
LIMIT_UPLOAD: str = _validate_limit(
    os.environ.get("RATE_LIMIT_UPLOAD", "20/minute"), "RATE_LIMIT_UPLOAD"
)

# ---------------------------------------------------------------------------
# Shared limiter instance
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[LIMIT_DEFAULT],
    storage_uri=_storage_uri,
)
