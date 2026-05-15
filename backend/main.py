# ============================================================
# NOIZE Backend — main.py  (patched)
# FIXES:
#   - Rate limiter on /auth/login (slowapi, 5/minute per IP)
#   - Proper logging module (captured by Gunicorn, print() is not)
#   - ALLOWED_ORIGINS actually enforced + warning if wildcard
# ============================================================

import sys, os, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("noize.main")

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

from routes import upload, analyze, auth
from routes.misc import router as misc_router
from routes.data_routes import router as data_router
from services.error_handler import global_exception_handler, http_exception_handler

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=get_remote_address)
    _rl = True
except ImportError:
    limiter = None
    _rl = False
    logger.warning("slowapi not installed — login rate limiting DISABLED. pip install slowapi")

app = FastAPI(title="NOIZE — AI Fairness Backend", version="1.2.0",
              docs_url="/docs", redoc_url="/redoc")

if _rl:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)

# ── CORS — Blocker 3 fix ──────────────────────────────────────
# ALLOWED_ORIGINS must be set in .env — no wildcard in production.
# Flutter web build runs on :8080 by default.
# Firebase Hosting uses https://your-app.web.app
# Add BOTH to ALLOWED_ORIGINS before demo day.
_raw     = os.getenv("ALLOWED_ORIGINS", "")
_origins = [o.strip() for o in _raw.split(",") if o.strip()]
if not _origins:
    # Safe fallback for local dev — never wildcard
    _origins = ["http://localhost:8080", "http://localhost:3000",
                "http://localhost:8001"]
    logger.warning(
        "ALLOWED_ORIGINS not set — defaulting to localhost only. "
        "Set ALLOWED_ORIGINS=https://your-app.web.app in .env for production."
    )
if "*" in _origins:
    logger.error(
        "ALLOWED_ORIGINS='*' is insecure and will be rejected in production. "
        "Replace with your actual Flutter/web app URL."
    )

app.add_middleware(CORSMiddleware, allow_origins=_origins,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(misc_router)
app.include_router(data_router)

@app.get("/", tags=["Health"])
def root():
    return {"status": "success", "data": {"message": "NOIZE backend running", "docs": "/docs"}, "error": None}

@app.get("/health", tags=["Health"])
def health():
    checks = {"rate_limiting": _rl}
    for lib in ("google.auth", "jose", "pandas", "passlib"):
        try: __import__(lib); checks[lib] = True
        except ImportError: checks[lib] = False
    return {"status": "success", "data": {"version": "1.2.0", "checks": checks}, "error": None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
