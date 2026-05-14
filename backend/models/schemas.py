# ============================================================
# NOIZE Backend — models/schemas.py
# Pydantic request/response models with full validation.
# ============================================================

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
import re


# ── Auth ──────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64,
                          description="Alphanumeric username, 3-64 chars")
    password: str = Field(..., min_length=8, max_length=128,
                          description="Password, 8-128 chars")

    @field_validator("username")
    @classmethod
    def username_safe(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-\.@]+$", v):
            raise ValueError(
                "Username may only contain letters, digits, _, -, ., @"
            )
        return v.strip()

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class GoogleLoginRequest(BaseModel):
    token: str = Field(..., min_length=10,
                       description="Google ID token from client SDK")


# ── File / Analysis ───────────────────────────────────────────

# Allowed CSV column name characters (prevent injection)
_SAFE_COL = re.compile(r"^[a-zA-Z0-9 _\-\.]+$")

class AnalyzeRequest(BaseModel):
    filename: str = Field(
        ...,
        description=(
            "Filename previously returned by /upload. "
            "Must end with .csv and contain no path separators."
        ),
    )
    target_column: str    = Field(..., min_length=1, max_length=64)
    protected_column: str | None = Field(
        default=None, min_length=1, max_length=64
    )

    @field_validator("filename")
    @classmethod
    def filename_safe(cls, v: str) -> str:
        # Strip any leading path — prevents path traversal
        basename = v.replace("\\", "/").split("/")[-1]
        if not basename.endswith(".csv"):
            raise ValueError("filename must be a .csv file")
        if ".." in basename or "/" in basename or "\\" in basename:
            raise ValueError("filename must not contain path separators")
        return basename

    @field_validator("target_column", "protected_column")
    @classmethod
    def column_name_safe(cls, v: str | None) -> str | None:
        if v is not None and not _SAFE_COL.match(v):
            raise ValueError(
                "Column names may only contain letters, digits, spaces, _, -, ."
            )
        return v


class CompareRequest(BaseModel):
    id1: str = Field(..., min_length=36, max_length=36,
                     description="UUID of first result")
    id2: str = Field(..., min_length=36, max_length=36,
                     description="UUID of second result")


# ── Response envelope ─────────────────────────────────────────

class APIResponse(BaseModel):
    """Standard response envelope used by all endpoints."""
    status: str
    data:   object | None
    error:  str | None


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10, description="Refresh token from /auth/login")
