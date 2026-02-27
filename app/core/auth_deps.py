"""
FastAPI dependencies for JWT authentication and role-based access control.

Usage:
    from app.core.auth_deps import require_auth, require_admin

    @router.get("/protected")
    async def protected_route(claims: dict = Depends(require_auth)):
        ...
"""

import logging
from typing import Any, Callable, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import decode_token
from .database import get_db
from .db_models import UserDB, UserRole

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


async def get_current_user_from_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> Optional[UserDB]:
    """
    Extract and verify Bearer token from Authorization header.
    Returns the UserDB row, or None if no valid token is present.
    """
    if not credentials:
        return None

    token = credentials.credentials
    try:
        payload = decode_token(token)
    except JWTError:
        return None

    if payload.get("type") != "access":
        return None

    user_id: str = payload.get("sub", "")
    if not user_id:
        return None

    from sqlalchemy import select

    result = await db.execute(select(UserDB).where(UserDB.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        return None

    return user


async def require_auth(
    user: Optional[UserDB] = Depends(get_current_user_from_token),
) -> UserDB:
    """Dependency that requires a valid authenticated user. Raises 401 if missing."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_role(*roles: str) -> Callable[..., Any]:
    """
    Factory that returns a FastAPI dependency enforcing one of the given roles.

    Example:
        @router.delete("/users/{id}")
        async def delete_user(current_user = Depends(require_role("admin"))):
            ...
    """

    async def _dependency(user: UserDB = Depends(require_auth)) -> UserDB:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user

    return _dependency


# Convenience aliases
require_admin = require_role(UserRole.ADMIN)
