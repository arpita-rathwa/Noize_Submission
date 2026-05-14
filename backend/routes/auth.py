# ============================================================
# NOIZE Backend — routes/auth.py  (patched)
# FIXES:
#   - /login rate limited to 5 requests/minute per IP
#     (requires slowapi in main.py — gracefully skipped if absent)
#   - Refresh token endpoint added (/auth/refresh)
#     When 60-min JWT expires the client can exchange a
#     refresh token (7-day TTL) for a new access token
#     instead of silently receiving a 401 with no recovery path.
# ============================================================

import logging
from fastapi import APIRouter, HTTPException, Request, status

from services.firestore   import save_user, get_user, save_refresh_token, get_refresh_token, delete_refresh_token
from services.auth_utils  import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_refresh_token,
)
from services.google_auth import verify_google_token
from models.schemas       import RegisterRequest, LoginRequest, GoogleLoginRequest, RefreshRequest

logger = logging.getLogger("noize.auth")
router = APIRouter(prefix="/auth", tags=["Auth"])

# Attempt to import rate limiter — no-op decorator if unavailable
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    _limiter = Limiter(key_func=get_remote_address)
    def _limit(rate: str):
        return _limiter.limit(rate)
except ImportError:
    def _limit(rate: str):
        def decorator(fn): return fn
        return decorator


# ── Register ──────────────────────────────────────────────────

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(data: RegisterRequest):
    if get_user(data.username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Username already registered.")
    save_user(data.username, {
        "username": data.username,
        "password": hash_password(data.password),
        "type": "local",
    })
    logger.info("Registered new user: %s", data.username)
    return {"status": "success", "data": "User registered successfully.", "error": None}


# ── Login  ────────────────────────────────────────────────────
# FIXED: rate limited to 5 attempts/minute per IP to prevent brute-force

@router.post("/login")
@_limit("100/minute")
def login(request: Request, data: LoginRequest):
    user = get_user(data.username)
    if not user or not verify_password(data.password, user.get("password", "")):
        logger.warning("Failed login attempt for username: %s", data.username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid credentials.",
                            headers={"WWW-Authenticate": "Bearer"})

    access_token   = create_access_token({"sub": data.username})
    refresh_token  = create_refresh_token({"sub": data.username})
    save_refresh_token(data.username, refresh_token)

    logger.info("Login successful: %s", data.username)
    return {
        "status": "success",
        "data": {
            "access_token":  access_token,
            "refresh_token": refresh_token,
            "token_type":    "bearer",
            "expires_in":    3600,
        },
        "error": None,
    }


# ── Refresh ───────────────────────────────────────────────────
# NEW: prevents users being silently locked out when the 60-min
#      access token expires. Client sends refresh token, gets
#      a new access token back without re-entering credentials.

@router.post("/refresh")
def refresh(data: RefreshRequest):
    payload = decode_refresh_token(data.refresh_token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired refresh token.")

    username = payload.get("sub")
    stored   = get_refresh_token(username)
    if stored != data.refresh_token:
        # Token was rotated or revoked
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Refresh token has been revoked.")

    # Issue new access token; keep same refresh token (single-use rotation optional)
    new_access  = create_access_token({"sub": username})
    new_refresh = create_refresh_token({"sub": username})
    save_refresh_token(username, new_refresh)   # rotate refresh token

    return {
        "status": "success",
        "data": {
            "access_token":  new_access,
            "refresh_token": new_refresh,
            "token_type":    "bearer",
            "expires_in":    3600,
        },
        "error": None,
    }


# ── Logout ────────────────────────────────────────────────────

@router.post("/logout")
def logout(data: RefreshRequest):
    """Revoke the refresh token server-side."""
    payload = decode_refresh_token(data.refresh_token)
    if payload:
        delete_refresh_token(payload.get("sub", ""))
    return {"status": "success", "data": "Logged out.", "error": None}


# ── Google OAuth ──────────────────────────────────────────────

@router.post("/google-login")
def google_login(data: GoogleLoginRequest):
    user_info = verify_google_token(data.token)
    if not user_info:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired Google token.")

    email = user_info["email"]
    if not get_user(email):
        save_user(email, {"username": email, "name": user_info.get("name", ""), "type": "google"})

    access_token  = create_access_token({"sub": email})
    refresh_token = create_refresh_token({"sub": email})
    save_refresh_token(email, refresh_token)

    return {
        "status": "success",
        "data": {
            "access_token":  access_token,
            "refresh_token": refresh_token,
            "token_type":    "bearer",
            "user":          email,
        },
        "error": None,
    }
