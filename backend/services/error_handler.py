# ============================================================
# NOIZE Backend — services/error_handler.py  (patched)
# FIXES:
#   - traceback.print_exc() replaced with logger.exception()
#     which is captured by Gunicorn/uvicorn log workers
# ============================================================

import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("noize.error_handler")


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for unhandled exceptions.
    Logs the full traceback via the logging module (captured by Gunicorn).
    Returns a safe envelope — never exposes raw exception internals to clients.
    """
    logger.exception(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "data":   None,
            "error":  "An unexpected server error occurred.",
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Convert FastAPI's HTTPException into the standard NOIZE envelope.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "data":   None,
            "error":  exc.detail,
        },
        headers=getattr(exc, "headers", None),
    )
