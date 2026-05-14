# ============================================================
# NOIZE Backend — services/firestore.py
# FIX 1: Real Google Firestore with automatic db.json fallback
#
# Production setup:
#   Option A (Cloud Run / GKE):
#     Set FIREBASE_CREDENTIALS_JSON=<entire service account JSON as one line>
#     Set FIREBASE_PROJECT_ID=your-project-id
#   Option B (local dev with key file):
#     Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
#
#   Get your service account key:
#   Firebase Console → Project Settings → Service Accounts → Generate key
# ============================================================

import json, os, tempfile, threading, logging

logger = logging.getLogger("noize.firestore")

# ── Try to initialise Firebase Admin SDK ─────────────────────
_USE_FIRESTORE = False
_db            = None

try:
    import firebase_admin
    from firebase_admin import credentials, firestore as _fs

    _creds_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

    if _creds_json:
        # Credentials passed as JSON string in env var (recommended for Cloud Run)
        _cred_dict = json.loads(_creds_json)
        _cred = credentials.Certificate(_cred_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(_cred)
        _db = _fs.client()
        _USE_FIRESTORE = True
        logger.info("✓ Firestore initialised from FIREBASE_CREDENTIALS_JSON")

    elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        _db = _fs.client()
        _USE_FIRESTORE = True
        logger.info("✓ Firestore initialised from GOOGLE_APPLICATION_CREDENTIALS")

    else:
        logger.warning(
            "Firebase credentials not set — using db.json fallback. "
            "Set FIREBASE_CREDENTIALS_JSON or GOOGLE_APPLICATION_CREDENTIALS."
        )

except ImportError:
    logger.warning("firebase-admin not installed — using db.json fallback. pip install firebase-admin")
except Exception as exc:
    logger.error("Firebase init failed (%s) — using db.json fallback.", exc)


# ── Fallback: db.json (local dev only) ───────────────────────
DB_FILE = os.getenv("DB_FILE", "db.json")
_lock   = threading.Lock()

def _load() -> dict:
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        logger.error("db.json could not be parsed — treating as empty.")
        return {}

def _save(data: dict) -> None:
    db_dir = os.path.dirname(os.path.abspath(DB_FILE)) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=db_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp_path, DB_FILE)
    except Exception:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise

COL_RESULTS = "results"
COL_USERS   = "users"
COL_TOKENS  = "refresh_tokens"

# ── Results ───────────────────────────────────────────────────
def save_result(result_id: str, result_data: dict) -> None:
    if _USE_FIRESTORE:
        _db.collection(COL_RESULTS).document(result_id).set(result_data)
    else:
        with _lock:
            db = _load(); db[result_id] = result_data; _save(db)

def get_result(result_id: str) -> dict | None:
    if _USE_FIRESTORE:
        doc = _db.collection(COL_RESULTS).document(result_id).get()
        return doc.to_dict() if doc.exists else None
    return _load().get(result_id)

def get_all_results() -> list[dict]:
    if _USE_FIRESTORE:
        return [d.to_dict() for d in _db.collection(COL_RESULTS).stream()]
    return [v for k, v in _load().items() if not k.startswith(("user_", "refresh_"))]

def delete_result(result_id: str) -> bool:
    if _USE_FIRESTORE:
        ref = _db.collection(COL_RESULTS).document(result_id)
        if not ref.get().exists: return False
        ref.delete(); return True
    with _lock:
        db = _load()
        if result_id not in db: return False
        del db[result_id]; _save(db)
    return True

# ── Users ─────────────────────────────────────────────────────
def save_user(username: str, user_data: dict) -> None:
    if _USE_FIRESTORE:
        _db.collection(COL_USERS).document(username).set(user_data)
    else:
        with _lock:
            db = _load(); db[f"user_{username}"] = user_data; _save(db)

def get_user(username: str) -> dict | None:
    if _USE_FIRESTORE:
        doc = _db.collection(COL_USERS).document(username).get()
        return doc.to_dict() if doc.exists else None
    return _load().get(f"user_{username}")

# ── Refresh tokens ────────────────────────────────────────────
def save_refresh_token(username: str, token: str) -> None:
    if _USE_FIRESTORE:
        _db.collection(COL_TOKENS).document(username).set({"token": token})
    else:
        with _lock:
            db = _load(); db[f"refresh_{username}"] = token; _save(db)

def get_refresh_token(username: str) -> str | None:
    if _USE_FIRESTORE:
        doc = _db.collection(COL_TOKENS).document(username).get()
        return doc.to_dict().get("token") if doc.exists else None
    return _load().get(f"refresh_{username}")

def delete_refresh_token(username: str) -> None:
    if _USE_FIRESTORE:
        _db.collection(COL_TOKENS).document(username).delete()
    else:
        with _lock:
            db = _load(); db.pop(f"refresh_{username}", None); _save(db)
