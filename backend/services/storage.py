import logging
logger = logging.getLogger("noize.storage")

# ============================================================
# NOIZE Backend — services/storage.py
# File upload handling.
#
# FIXES vs original:
#   - Upload dir changed from "mkdir_uploads" (accidental name)
#     to "uploads" configurable via UPLOAD_DIR env var
#   - mkdir is called lazily inside save_file() so import-time
#     side effects don't break unit tests
#   - Path traversal: filename is sanitised before use
#   - File size limit enforced (configurable via MAX_UPLOAD_MB)
#   - save_file() returns just the base filename (not full path)
#     so the client stores a safe reference, not an fs path
# ============================================================

import os
import re
import uuid

UPLOAD_DIR    = os.getenv("UPLOAD_DIR", "uploads")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_BYTES     = MAX_UPLOAD_MB * 1024 * 1024

# Only CSV files accepted
ALLOWED_EXTENSIONS = {".csv"}


def _ensure_upload_dir() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def _safe_filename(filename: str) -> str:
    """
    Return a safe basename.
    - Strips directory components (prevents path traversal)
    - Keeps only alphanumeric, dot, dash, underscore
    - Prepends a UUID4 prefix to avoid collisions
    """
    basename = os.path.basename(filename.replace("\\", "/"))
    # Remove everything except safe chars
    clean    = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", basename)
    prefix   = uuid.uuid4().hex[:8]
    return f"{prefix}_{clean}"


def save_file(content: bytes, original_filename: str) -> str:
    """
    Persist uploaded bytes to the uploads directory.

    Parameters
    ----------
    content           : raw file bytes
    original_filename : filename supplied by the client

    Returns
    -------
    str : safe stored filename (not a full path — use get_upload_path()
          server-side to reconstruct the full path)

    Raises
    ------
    ValueError : if the file is too large or has a disallowed extension
    """
    # ── Extension check ───────────────────────────────────────
    _, ext = os.path.splitext(original_filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"File type '{ext}' is not allowed. "
            f"Only {ALLOWED_EXTENSIONS} are accepted."
        )

    # ── Size check ────────────────────────────────────────────
    if len(content) > MAX_BYTES:
        raise ValueError(
            f"File exceeds the {MAX_UPLOAD_MB} MB upload limit "
            f"({len(content) // (1024*1024)} MB received)."
        )

    # ── Write ─────────────────────────────────────────────────
    _ensure_upload_dir()
    safe_name = _safe_filename(original_filename)
    full_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(full_path, "wb") as fh:
        fh.write(content)

    return safe_name   # ← safe reference returned to client


def get_upload_path(filename: str) -> str:
    """
    Reconstruct the full server path for a previously uploaded file.
    Raises ValueError if the path would escape UPLOAD_DIR.
    """
    # Sanitise again — client-supplied value must not traverse paths
    safe_name = os.path.basename(filename.replace("\\", "/"))
    full_path = os.path.realpath(os.path.join(UPLOAD_DIR, safe_name))
    upload_real = os.path.realpath(UPLOAD_DIR)

    if not full_path.startswith(upload_real + os.sep) and full_path != upload_real:
        raise ValueError("Invalid filename — path traversal attempt detected.")

    return full_path


def list_uploads() -> list[str]:
    """Return a list of uploaded filenames."""
    _ensure_upload_dir()
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")]
