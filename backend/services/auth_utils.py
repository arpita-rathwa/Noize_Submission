# ============================================================
# NOIZE Backend — services/auth_utils.py  (patched)
# FIXES:
#   - create_refresh_token() + decode_refresh_token() added
#     (7-day TTL, separate secret to prevent token confusion)
#   - Logging replaces warnings.warn
# ============================================================

import os, logging
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger("noize.auth_utils")

SECRET_KEY              = os.getenv("SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION_ENV_DO_NOT_USE_DEFAULT")
REFRESH_SECRET_KEY      = os.getenv("REFRESH_SECRET_KEY", SECRET_KEY + "_refresh")
ALGORITHM               = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES  = int(os.getenv("TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS    = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

if SECRET_KEY == "CHANGE_THIS_IN_PRODUCTION_ENV_DO_NOT_USE_DEFAULT":
    logger.warning("SECRET_KEY is using the insecure default. Set it via environment variable.")

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
security    = HTTPBearer()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"]  = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload["type"] = "access"
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """7-day refresh token signed with a separate secret."""
    payload = data.copy()
    payload["exp"]  = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload["type"] = "refresh"
    return jwt.encode(payload, REFRESH_SECRET_KEY, algorithm=ALGORITHM)


def decode_refresh_token(token: str) -> dict | None:
    """Return payload dict or None if invalid/expired."""
    try:
        payload = jwt.decode(token, REFRESH_SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            return None
        return payload
    except JWTError:
        return None


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None or payload.get("type") != "access":
            raise exc
    except JWTError:
        raise exc
    return username
