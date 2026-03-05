"""
Authentication utilities for Orange3 Web Backend.

Provides:
- Password hashing with bcrypt via passlib
- JWT token creation and verification (access + refresh)
- Token payload extraction helpers
"""

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------

_jwt_secret_default = "CHANGE_ME_IN_PRODUCTION_use_openssl_rand_hex_32"
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", _jwt_secret_default)

# Warn loudly if using the insecure default
import logging as _logging

_auth_logger = _logging.getLogger(__name__)
if JWT_SECRET_KEY == _jwt_secret_default:
    _env = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
    if _env in ("production", "prod"):
        raise RuntimeError(
            "JWT_SECRET_KEY must be set to a secure random value in production. "
            "Generate one with: openssl rand -hex 32"
        )
    else:
        _auth_logger.warning(
            "JWT_SECRET_KEY is using the insecure default. "
            "Set JWT_SECRET_KEY environment variable before deploying to production."
        )
JWT_ALGORITHM: str = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
    os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)
JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = int(
    os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7")
)
JWT_ISSUER: str = os.getenv("JWT_ISSUER", "orange3-web")

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True if plain_password matches the bcrypt hash."""
    return _pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(
    user_id: str,
    email: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a short-lived JWT access token.

    Payload fields:
        sub  – user ID (standard JWT claim)
        email, role, type="access", exp, iat, iss
    """
    expire = _utcnow() + (
        expires_delta or timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "type": "access",
        "exp": expire,
        "iat": _utcnow(),
        "iss": JWT_ISSUER,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a long-lived JWT refresh token.

    Payload fields:
        sub, type="refresh", exp, iat, iss
    """
    expire = _utcnow() + (
        expires_delta or timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    )
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
        "iat": _utcnow(),
        "iss": JWT_ISSUER,
        # jti provides uniqueness so that two tokens issued for the same user
        # in the same second produce different hashes in the database.
        "jti": secrets.token_hex(16),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and verify a JWT token.

    Raises:
        JWTError – if token is invalid, expired, or has wrong issuer.
    """
    return jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM],
        issuer=JWT_ISSUER,
    )


def get_token_user_id(token: str) -> Optional[str]:
    """Return the user_id (sub) from a token, or None on any error."""
    try:
        payload = decode_token(token)
        return payload.get("sub")
    except JWTError:
        return None


def hash_token(token: str) -> str:
    """Return a SHA-256 hex digest of a token for safe DB storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def get_access_token_expire_minutes() -> int:
    return JWT_ACCESS_TOKEN_EXPIRE_MINUTES


def get_refresh_token_expire_days() -> int:
    return JWT_REFRESH_TOKEN_EXPIRE_DAYS
