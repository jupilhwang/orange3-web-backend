"""
Pydantic schemas for JWT Authentication API endpoints.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


# Custom EmailStr that allows local domains like orange3.local
class _CustomEmailStr(str):
    """Custom email that allows special/local domains."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, info=None):
        from pydantic import EmailStr

        # Try standard validation first
        try:
            EmailStr.validate(v, info)
            return v
        except:
            # Allow if it looks like email@domain (has @ and a domain part)
            if isinstance(v, str) and "@" in v:
                parts = v.rsplit("@", 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    return v
            raise ValueError("Invalid email format")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: _CustomEmailStr
    password: str = Field(min_length=6, max_length=128)
    name: str = Field(min_length=1, max_length=255)

    @field_validator("name")
    @classmethod
    def name_strip(cls, v: str) -> str:
        return v.strip()


class LoginRequest(BaseModel):
    email: _CustomEmailStr
    password: str = Field(min_length=1)


class RefreshRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=6, max_length=128)


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)

    @field_validator("name")
    @classmethod
    def name_strip(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v else v


class OAuthRequest(BaseModel):
    """OAuth token exchange request (Google / GitHub)."""

    code: str  # authorization code from provider
    redirect_uri: Optional[str] = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int  # seconds until access token expires
    user: UserResponse


class MessageResponse(BaseModel):
    message: str
