# ============================================================
# NOIZE — api/main.py
# FIX 2: CORS locked to ALLOWED_ORIGINS env var (no wildcard)
# FIX 3: Gunicorn-ready logging
# ============================================================

import sys, os, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("noize.ml")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.pre_audit  import router as pre_router
from api.routes.post_audit import router as post_router
from api.routes.reports    import router as report_router

app = FastAPI(
    title       = "NOIZE — AI Fairness Audit API",
    description = "Pre- and post-model bias detection, mitigation, and reporting.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS — FIX 2: no wildcard ────────────────────────────────
_raw     = os.getenv("ALLOWED_ORIGINS", "")
_origins = [o.strip() for o in _raw.split(",") if o.strip()]
if not _origins:
    _origins = ["http://localhost:8080", "http://localhost:3000"]
    logger.warning(
        "ALLOWED_ORIGINS not set — defaulting to localhost only. "
        "Set ALLOWED_ORIGINS=https://your-app.web.app in production."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(pre_router)
app.include_router(post_router)
app.include_router(report_router)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "NOIZE Fairness API is running", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health():
    modules = {}
    for lib in ("google.generativeai", "reportlab", "plotly", "fairlearn"):
        try: __import__(lib); modules[lib] = True
        except ImportError: modules[lib] = False
    return {"status": "ok", "version": "1.0.0", "modules": modules}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
