# ============================================================
# NOIZE Backend — services/google_auth.py  (patched)
# FIXES:
#   - warnings.warn replaced with logging.warning
#     (consistent with all other services; captured by Gunicorn)
# ============================================================

import os
import logging

logger = logging.getLogger("noize.google_auth")

GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")

if not GOOGLE_CLIENT_ID:
    logger.warning(
        "GOOGLE_CLIENT_ID env var is not set. "
        "Google OAuth login will reject all tokens."
    )

try:
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests
    _HAS_GOOGLE_AUTH = True
except ImportError:
    _HAS_GOOGLE_AUTH = False


def verify_google_token(token: str) -> dict | None:
    """
    Verify a Google ID token and return user info.

    Returns
    -------
    dict with keys {email, name, sub} on success, or None on failure.
    """
    if not _HAS_GOOGLE_AUTH:
        raise RuntimeError(
            "google-auth is not installed. Run: pip install google-auth"
        )

    if not GOOGLE_CLIENT_ID:
        logger.error("Cannot verify Google token — GOOGLE_CLIENT_ID is not set.")
        return None

    try:
        idinfo = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
        return {
            "email": idinfo["email"],
            "name":  idinfo.get("name", ""),
            "sub":   idinfo.get("sub", ""),
        }
    except Exception as exc:
        logger.warning("Google token verification failed: %s", exc)
        return None
