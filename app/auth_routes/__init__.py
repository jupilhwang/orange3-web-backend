"""JWT Authentication API endpoints.

Routes (all under /api/v1/auth):
    POST /register         – create account
    POST /login            – email+password login
    POST /refresh          – exchange refresh token for new access token
    POST /logout           – revoke refresh token
    GET  /me               – current user profile
    PUT  /me               – update profile
    PUT  /password         – change password
    POST /google           – Google OAuth login/register
    POST /github           – GitHub OAuth login/register
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError
from sqlalchemy import select, update as sql_update
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_access_token_expire_minutes,
    get_refresh_token_expire_days,
    hash_password,
    hash_token,
    verify_password,
)
from ..core.auth_deps import require_auth
from ..core.auth_schemas import (
    AuthResponse,
    ChangePasswordRequest,
    LoginRequest,
    MessageResponse,
    OAuthRequest,
    RefreshRequest,
    RegisterRequest,
    UpdateProfileRequest,
    UserResponse,
)
from ..core.database import get_db
from ..core.db_models import RefreshTokenDB, UserDB, UserRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# ---------------------------------------------------------------------------
# OAuth provider endpoints
# ---------------------------------------------------------------------------

_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GITHUB_USER_URL = "https://api.github.com/user"
_GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_to_response(user: UserDB) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login_at=user.last_login_at,
    )


async def _store_refresh_token(
    db: AsyncSession,
    user_id: str,
    token: str,
    expire_days: int,
) -> None:
    """Persist a hashed refresh token to the database."""
    token_hash = hash_token(token)
    expires_at = datetime.now(timezone.utc) + timedelta(days=expire_days)
    db_token = RefreshTokenDB(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    db.add(db_token)
    await db.flush()


def _build_auth_response(user: UserDB, access_token: str, refresh_token: str) -> AuthResponse:
    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="Bearer",
        expires_in=get_access_token_expire_minutes() * 60,
        user=_user_to_response(user),
    )


# ---------------------------------------------------------------------------
# POST /register
# ---------------------------------------------------------------------------


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """Register a new user account."""
    result = await db.execute(select(UserDB).where(UserDB.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "user_exists", "message": "User with this email already exists"},
        )

    user = UserDB(
        email=body.email,
        password_hash=hash_password(body.password),
        name=body.name,
        role=UserRole.USER,
        is_active=True,
    )
    db.add(user)
    await db.flush()

    access_token = create_access_token(user.id, user.email, user.role)
    refresh_token = create_refresh_token(user.id)
    await _store_refresh_token(db, user.id, refresh_token, get_refresh_token_expire_days())

    logger.info(f"[Auth] New user registered: {user.email} ({user.id})")
    return _build_auth_response(user, access_token, refresh_token)


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------


@router.post("/login", response_model=AuthResponse)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """Authenticate with email and password."""
    result = await db.execute(select(UserDB).where(UserDB.email == body.email))
    user = result.scalar_one_or_none()

    # Constant-time path to prevent user enumeration
    if not user or not user.password_hash or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_credentials", "message": "Invalid email or password"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "user_inactive", "message": "Account is disabled"},
        )

    user.last_login_at = datetime.now(timezone.utc)
    db.add(user)
    await db.flush()

    access_token = create_access_token(user.id, user.email, user.role)
    refresh_token = create_refresh_token(user.id)
    await _store_refresh_token(db, user.id, refresh_token, get_refresh_token_expire_days())

    logger.info(f"[Auth] User logged in: {user.email}")
    return _build_auth_response(user, access_token, refresh_token)


# ---------------------------------------------------------------------------
# POST /refresh
# ---------------------------------------------------------------------------


@router.post("/refresh", response_model=AuthResponse)
async def refresh_token(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """Exchange a valid refresh token for a new access + refresh token pair (rotation)."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "invalid_token", "message": "Invalid or expired refresh token"},
    )

    try:
        payload = decode_token(body.refresh_token)
    except JWTError:
        raise credentials_error

    if payload.get("type") != "refresh":
        raise credentials_error

    user_id: str = payload.get("sub", "")
    token_hash = hash_token(body.refresh_token)

    # Verify token is in DB and not revoked
    result = await db.execute(
        select(RefreshTokenDB).where(
            RefreshTokenDB.token_hash == token_hash,
            RefreshTokenDB.revoked == False,  # noqa: E712
        )
    )
    stored_token = result.scalar_one_or_none()
    if not stored_token:
        raise credentials_error

    # Rotate: revoke used token
    stored_token.revoked = True
    db.add(stored_token)

    result = await db.execute(select(UserDB).where(UserDB.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise credentials_error

    access_token = create_access_token(user.id, user.email, user.role)
    new_refresh_token = create_refresh_token(user.id)
    await _store_refresh_token(db, user.id, new_refresh_token, get_refresh_token_expire_days())

    logger.info(f"[Auth] Token refreshed for user: {user.email}")
    return _build_auth_response(user, access_token, new_refresh_token)


# ---------------------------------------------------------------------------
# POST /logout
# ---------------------------------------------------------------------------


@router.post("/logout", response_model=MessageResponse)
async def logout(
    body: Optional[RefreshRequest] = None,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Revoke the refresh token. Client should discard both tokens."""
    if body and body.refresh_token:
        token_hash = hash_token(body.refresh_token)
        result = await db.execute(
            select(RefreshTokenDB).where(RefreshTokenDB.token_hash == token_hash)
        )
        stored = result.scalar_one_or_none()
        if stored:
            stored.revoked = True
            db.add(stored)

    logger.info("[Auth] User logged out")
    return MessageResponse(message="Logged out successfully")


# ---------------------------------------------------------------------------
# GET /me
# ---------------------------------------------------------------------------


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserDB = Depends(require_auth)) -> UserResponse:
    """Return the authenticated user's profile."""
    return _user_to_response(current_user)


# ---------------------------------------------------------------------------
# PUT /me
# ---------------------------------------------------------------------------


@router.put("/me", response_model=UserResponse)
async def update_me(
    body: UpdateProfileRequest,
    current_user: UserDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Update the authenticated user's profile."""
    if body.name is not None:
        current_user.name = body.name
        current_user.updated_at = datetime.now(timezone.utc)
        db.add(current_user)
        await db.flush()

    logger.info(f"[Auth] Profile updated for user: {current_user.email}")
    return _user_to_response(current_user)


# ---------------------------------------------------------------------------
# PUT /password
# ---------------------------------------------------------------------------


@router.put("/password", response_model=MessageResponse)
async def change_password(
    body: ChangePasswordRequest,
    current_user: UserDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Change password. Revokes all refresh tokens for security."""
    if not current_user.password_hash or not verify_password(
        body.current_password, current_user.password_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_password", "message": "Current password is incorrect"},
        )

    current_user.password_hash = hash_password(body.new_password)
    current_user.updated_at = datetime.now(timezone.utc)
    db.add(current_user)
    await db.flush()

    # Revoke all refresh tokens (forces re-login on other devices)
    await db.execute(
        sql_update(RefreshTokenDB)
        .where(RefreshTokenDB.user_id == current_user.id)
        .values(revoked=True)
    )

    logger.info(f"[Auth] Password changed for user: {current_user.email}")
    return MessageResponse(message="Password changed successfully")


# ---------------------------------------------------------------------------
# POST /google  (OAuth 2.0)
# ---------------------------------------------------------------------------


@router.post("/google", response_model=AuthResponse)
async def google_oauth(
    body: OAuthRequest,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """
    Exchange a Google authorization code for Orange3 tokens.
    Requires GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables.
    """
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"error": "oauth_not_configured", "message": "Google OAuth is not configured"},
        )

    async with httpx.AsyncClient() as http_client:
        token_resp = await http_client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "code": body.code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": body.redirect_uri or "",
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "oauth_failed", "message": "Google token exchange failed"},
            )
        google_tokens = token_resp.json()

        userinfo_resp = await http_client.get(
            _GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {google_tokens['access_token']}"},
        )
        if userinfo_resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "oauth_failed", "message": "Failed to fetch Google user info"},
            )
        google_user = userinfo_resp.json()

    google_id: str = google_user.get("sub", "")
    email: str = google_user.get("email", "")
    name: str = google_user.get("name", email)

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "oauth_failed", "message": "Google account has no email"},
        )

    result = await db.execute(select(UserDB).where(UserDB.google_id == google_id))
    user = result.scalar_one_or_none()
    if not user:
        result = await db.execute(select(UserDB).where(UserDB.email == email))
        user = result.scalar_one_or_none()

    if not user:
        user = UserDB(email=email, name=name, role=UserRole.USER, google_id=google_id, is_active=True)
        db.add(user)
        await db.flush()
    else:
        if not user.google_id:
            user.google_id = google_id
        user.last_login_at = datetime.now(timezone.utc)
        db.add(user)
        await db.flush()

    access_token = create_access_token(user.id, user.email, user.role)
    refresh_token = create_refresh_token(user.id)
    await _store_refresh_token(db, user.id, refresh_token, get_refresh_token_expire_days())

    logger.info(f"[Auth] Google OAuth login: {user.email}")
    return _build_auth_response(user, access_token, refresh_token)


# ---------------------------------------------------------------------------
# POST /github  (OAuth 2.0)
# ---------------------------------------------------------------------------


@router.post("/github", response_model=AuthResponse)
async def github_oauth(
    body: OAuthRequest,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """
    Exchange a GitHub authorization code for Orange3 tokens.
    Requires GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET environment variables.
    """
    client_id = os.getenv("GITHUB_CLIENT_ID")
    client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"error": "oauth_not_configured", "message": "GitHub OAuth is not configured"},
        )

    async with httpx.AsyncClient() as http_client:
        token_resp = await http_client.post(
            _GITHUB_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": body.code,
            },
            headers={"Accept": "application/json"},
        )
        if token_resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "oauth_failed", "message": "GitHub token exchange failed"},
            )
        github_tokens = token_resp.json()
        gh_access_token = github_tokens.get("access_token", "")
        if not gh_access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "oauth_failed", "message": "GitHub returned no access token"},
            )

        user_resp = await http_client.get(
            _GITHUB_USER_URL,
            headers={"Authorization": f"token {gh_access_token}", "Accept": "application/json"},
        )
        if user_resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "oauth_failed", "message": "Failed to fetch GitHub user info"},
            )
        gh_user = user_resp.json()

        email: str = gh_user.get("email", "")
        if not email:
            emails_resp = await http_client.get(
                _GITHUB_EMAILS_URL,
                headers={"Authorization": f"token {gh_access_token}", "Accept": "application/json"},
            )
            if emails_resp.status_code == 200:
                for entry in emails_resp.json():
                    if entry.get("primary") and entry.get("verified"):
                        email = entry["email"]
                        break

    github_id: str = str(gh_user.get("id", ""))
    name: str = gh_user.get("name") or gh_user.get("login", email)

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "oauth_failed", "message": "GitHub account has no accessible email"},
        )

    result = await db.execute(select(UserDB).where(UserDB.github_id == github_id))
    user = result.scalar_one_or_none()
    if not user:
        result = await db.execute(select(UserDB).where(UserDB.email == email))
        user = result.scalar_one_or_none()

    if not user:
        user = UserDB(email=email, name=name, role=UserRole.USER, github_id=github_id, is_active=True)
        db.add(user)
        await db.flush()
    else:
        if not user.github_id:
            user.github_id = github_id
        user.last_login_at = datetime.now(timezone.utc)
        db.add(user)
        await db.flush()

    access_token = create_access_token(user.id, user.email, user.role)
    refresh_token = create_refresh_token(user.id)
    await _store_refresh_token(db, user.id, refresh_token, get_refresh_token_expire_days())

    logger.info(f"[Auth] GitHub OAuth login: {user.email}")
    return _build_auth_response(user, access_token, refresh_token)
