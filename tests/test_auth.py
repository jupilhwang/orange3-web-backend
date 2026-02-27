"""
Tests for JWT Authentication API endpoints.

Covers:
- POST /api/v1/auth/register
- POST /api/v1/auth/login
- POST /api/v1/auth/refresh
- POST /api/v1/auth/logout
- GET  /api/v1/auth/me
- PUT  /api/v1/auth/me
- PUT  /api/v1/auth/password
"""

import uuid

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AUTH_PREFIX = "/api/v1/auth"

TEST_PASSWORD = "securepassword123"
TEST_NAME = "Test Auth User"


def unique_email(prefix: str = "test") -> str:
    """Return a unique email address to avoid collisions across test runs."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}@example.com"


async def register_user(client: AsyncClient, email: str | None = None) -> dict:
    if email is None:
        email = unique_email()
    resp = await client.post(
        f"{AUTH_PREFIX}/register",
        json={"email": email, "password": TEST_PASSWORD, "name": TEST_NAME},
    )
    return resp


async def login_user(client: AsyncClient, email: str) -> dict:
    resp = await client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": email, "password": TEST_PASSWORD},
    )
    return resp


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient):
    email = unique_email("register_ok")
    resp = await register_user(client, email)
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "Bearer"
    assert data["user"]["email"] == email
    assert data["user"]["role"] == "user"


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    email = unique_email("dup_register")
    await register_user(client, email)
    resp = await register_user(client, email)
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_register_invalid_email(client: AsyncClient):
    resp = await client.post(
        f"{AUTH_PREFIX}/register",
        json={"email": "not-an-email", "password": "pass123", "name": "Bad Email"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_register_short_password(client: AsyncClient):
    resp = await client.post(
        f"{AUTH_PREFIX}/register",
        json={"email": "short_pass@example.com", "password": "12", "name": "Short"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient):
    email = unique_email("login_ok")
    await register_user(client, email)
    resp = await login_user(client, email)
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == email


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient):
    email = unique_email("wrong_pass")
    await register_user(client, email)
    resp = await client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": email, "password": "wrongpassword"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_unknown_email(client: AsyncClient):
    resp = await client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": "nobody@example.com", "password": TEST_PASSWORD},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_token(client: AsyncClient):
    email = unique_email("refresh_ok")
    await register_user(client, email)
    login_resp = await login_user(client, email)
    refresh_token = login_resp.json()["refresh_token"]

    resp = await client.post(
        f"{AUTH_PREFIX}/refresh",
        json={"refresh_token": refresh_token},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    # New refresh token should differ (token rotation)
    assert data["refresh_token"] != refresh_token


@pytest.mark.asyncio
async def test_refresh_invalid_token(client: AsyncClient):
    resp = await client.post(
        f"{AUTH_PREFIX}/refresh",
        json={"refresh_token": "not.a.valid.jwt"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_refresh_used_token_rejected(client: AsyncClient):
    """After using a refresh token once (rotation), reusing it must fail."""
    email = unique_email("refresh_rotate")
    await register_user(client, email)
    login_resp = await login_user(client, email)
    old_refresh = login_resp.json()["refresh_token"]

    # First use — should succeed
    resp1 = await client.post(
        f"{AUTH_PREFIX}/refresh", json={"refresh_token": old_refresh}
    )
    assert resp1.status_code == 200

    # Second use of old token — must be rejected
    resp2 = await client.post(
        f"{AUTH_PREFIX}/refresh", json={"refresh_token": old_refresh}
    )
    assert resp2.status_code == 401


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logout(client: AsyncClient):
    email = unique_email("logout_ok")
    await register_user(client, email)
    login_resp = await login_user(client, email)
    refresh_token = login_resp.json()["refresh_token"]

    resp = await client.post(
        f"{AUTH_PREFIX}/logout",
        json={"refresh_token": refresh_token},
    )
    assert resp.status_code == 200
    assert "message" in resp.json()

    # After logout the refresh token must not work
    resp2 = await client.post(
        f"{AUTH_PREFIX}/refresh", json={"refresh_token": refresh_token}
    )
    assert resp2.status_code == 401


# ---------------------------------------------------------------------------
# GET /me
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_me(client: AsyncClient):
    email = unique_email("getme_ok")
    reg = await register_user(client, email)
    access_token = reg.json()["access_token"]

    resp = await client.get(
        f"{AUTH_PREFIX}/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == email


@pytest.mark.asyncio
async def test_get_me_unauthorized(client: AsyncClient):
    resp = await client.get(f"{AUTH_PREFIX}/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_me_invalid_token(client: AsyncClient):
    resp = await client.get(
        f"{AUTH_PREFIX}/me",
        headers={"Authorization": "Bearer invalidtoken"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# PUT /me
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_profile(client: AsyncClient):
    email = unique_email("update_profile")
    reg = await register_user(client, email)
    access_token = reg.json()["access_token"]

    resp = await client.put(
        f"{AUTH_PREFIX}/me",
        json={"name": "Updated Name"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "Updated Name"


# ---------------------------------------------------------------------------
# PUT /password
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_change_password(client: AsyncClient):
    email = unique_email("change_pass")
    reg = await register_user(client, email)
    access_token = reg.json()["access_token"]

    resp = await client.put(
        f"{AUTH_PREFIX}/password",
        json={"current_password": TEST_PASSWORD, "new_password": "newSecure456"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 200

    # Old password must no longer work
    resp2 = await client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": email, "password": TEST_PASSWORD},
    )
    assert resp2.status_code == 401

    # New password must work
    resp3 = await client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": email, "password": "newSecure456"},
    )
    assert resp3.status_code == 200


@pytest.mark.asyncio
async def test_change_password_wrong_current(client: AsyncClient):
    email = unique_email("change_pass_wrong")
    reg = await register_user(client, email)
    access_token = reg.json()["access_token"]

    resp = await client.put(
        f"{AUTH_PREFIX}/password",
        json={"current_password": "wrongpassword", "new_password": "newSecure456"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resp.status_code == 401
