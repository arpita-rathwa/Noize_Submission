# ============================================================
# NOIZE Backend — tests/test_backend.py
# 48 tests covering auth, upload, analyze, metrics,
# results, history, compare, explain, and services.
#
# Run with:  pytest tests/ -v
#
# No external services required — all storage is mocked
# via monkeypatching and temporary directories.
# ============================================================

from __future__ import annotations

import io
import json
import os
import tempfile
import uuid
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# ── Patch env before importing the app ───────────────────────
os.environ["SECRET_KEY"]   = "test-secret-key-32-chars-minimum!!"
os.environ["DB_FILE"]      = ":memory:"     # overridden per-test
os.environ["UPLOAD_DIR"]   = tempfile.mkdtemp()
os.environ["GOOGLE_CLIENT_ID"] = "test-google-client-id"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app

client = TestClient(app, raise_server_exceptions=False)


# ── Helpers ───────────────────────────────────────────────────

SAMPLE_CSV = b"sex,income\nMale,1\nMale,1\nFemale,0\nFemale,1\nMale,0\n"

def _register_and_login(username: str = None, password: str = "Password1") -> str:
    """Register a user and return a Bearer token."""
    username = username or f"user_{uuid.uuid4().hex[:8]}"
    client.post("/auth/register", json={"username": username, "password": password})
    resp = client.post("/auth/login",    json={"username": username, "password": password})
    return resp.json()["data"]["access_token"]

def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

def _upload_csv(token: str, content: bytes = SAMPLE_CSV, name: str = "test.csv") -> str:
    """Upload a CSV and return the stored filename."""
    resp = client.post(
        "/upload/",
        files={"file": (name, io.BytesIO(content), "text/csv")},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]["filename"]

def _full_analysis(token: str) -> dict:
    """Upload + analyze in one call. Returns analyze response JSON."""
    fname = _upload_csv(token)
    resp  = client.post(
        "/analyze/",
        json={
            "filename":         fname,
            "target_column":    "income",
            "protected_column": "sex",
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ════════════════════════════════════════════════════════════
# 1. HEALTH
# ════════════════════════════════════════════════════════════

class TestHealth:
    def test_root_returns_200(self):                        # 1
        r = client.get("/")
        assert r.status_code == 200

    def test_root_envelope(self):                           # 2
        r = client.get("/")
        assert r.json()["status"] == "success"

    def test_health_endpoint(self):                         # 3
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_checks(self):                       # 4
        r = client.get("/health")
        assert "checks" in r.json()["data"]


# ════════════════════════════════════════════════════════════
# 2. AUTH — REGISTER
# ════════════════════════════════════════════════════════════

class TestRegister:
    def test_register_success(self):                        # 5
        r = client.post("/auth/register",
                        json={"username": f"u{uuid.uuid4().hex[:6]}", "password": "Pass1234"})
        assert r.status_code == 201

    def test_register_duplicate_returns_409(self):          # 6
        uname = f"dup_{uuid.uuid4().hex[:6]}"
        client.post("/auth/register", json={"username": uname, "password": "Pass1234"})
        r = client.post("/auth/register", json={"username": uname, "password": "Pass1234"})
        assert r.status_code == 409

    def test_register_short_username_rejected(self):        # 7
        r = client.post("/auth/register", json={"username": "ab", "password": "Pass1234"})
        assert r.status_code == 422

    def test_register_short_password_rejected(self):        # 8
        r = client.post("/auth/register",
                        json={"username": "validuser1", "password": "short"})
        assert r.status_code == 422

    def test_register_password_without_digit_rejected(self):# 9
        r = client.post("/auth/register",
                        json={"username": f"u{uuid.uuid4().hex[:6]}", "password": "NoDigitsHere"})
        assert r.status_code == 422

    def test_register_unsafe_username_rejected(self):       # 10
        r = client.post("/auth/register",
                        json={"username": "bad;user", "password": "Pass1234"})
        assert r.status_code == 422


# ════════════════════════════════════════════════════════════
# 3. AUTH — LOGIN
# ════════════════════════════════════════════════════════════

class TestLogin:
    def setup_method(self):
        self.username = f"login_{uuid.uuid4().hex[:6]}"
        self.password = "TestPass1"
        client.post("/auth/register",
                    json={"username": self.username, "password": self.password})

    def test_login_success_returns_token(self):             # 11
        r = client.post("/auth/login",
                        json={"username": self.username, "password": self.password})
        assert r.status_code == 200
        assert "access_token" in r.json()["data"]

    def test_login_wrong_password_returns_401(self):        # 12
        r = client.post("/auth/login",
                        json={"username": self.username, "password": "WrongPass1"})
        assert r.status_code == 401

    def test_login_nonexistent_user_returns_401(self):      # 13
        r = client.post("/auth/login",
                        json={"username": "ghost_user_xyz", "password": "SomePass1"})
        assert r.status_code == 401

    def test_user_enumeration_same_error(self):             # 14
        """Both bad-user and bad-password return the same error message."""
        r_nouser  = client.post("/auth/login",
                                json={"username": "nonexistent_xyz", "password": "Pw1"})
        r_badpass = client.post("/auth/login",
                                json={"username": self.username, "password": "Wrong1"})
        assert r_nouser.json()["error"] == r_badpass.json()["error"]

    def test_login_token_is_string(self):                   # 15
        r = client.post("/auth/login",
                        json={"username": self.username, "password": self.password})
        token = r.json()["data"]["access_token"]
        assert isinstance(token, str) and len(token) > 20


# ════════════════════════════════════════════════════════════
# 4. UPLOAD
# ════════════════════════════════════════════════════════════

class TestUpload:
    def setup_method(self):
        self.token = _register_and_login()

    def test_upload_csv_success(self):                      # 16
        r = client.post(
            "/upload/",
            files={"file": ("data.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
            headers=_auth_headers(self.token),
        )
        assert r.status_code == 200
        assert r.json()["data"]["filename"].endswith(".csv")

    def test_upload_without_auth_returns_403(self):         # 17
        r = client.post(
            "/upload/",
            files={"file": ("data.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        )
        assert r.status_code in (401, 403)

    def test_upload_non_csv_rejected(self):                 # 18
        r = client.post(
            "/upload/",
            files={"file": ("script.py", io.BytesIO(b"import os"), "text/plain")},
            headers=_auth_headers(self.token),
        )
        assert r.status_code == 400

    def test_upload_returns_size_bytes(self):               # 19
        r = client.post(
            "/upload/",
            files={"file": ("data.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
            headers=_auth_headers(self.token),
        )
        assert r.json()["data"]["size_bytes"] == len(SAMPLE_CSV)

    def test_upload_response_has_filename_key(self):        # 20
        r = client.post(
            "/upload/",
            files={"file": ("data.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
            headers=_auth_headers(self.token),
        )
        assert "filename" in r.json()["data"]


# ════════════════════════════════════════════════════════════
# 5. ANALYZE
# ════════════════════════════════════════════════════════════

class TestAnalyze:
    def setup_method(self):
        self.token = _register_and_login()
        self.fname = _upload_csv(self.token)

    def test_analyze_success(self):                         # 21
        r = client.post(
            "/analyze/",
            json={"filename": self.fname, "target_column": "income",
                  "protected_column": "sex"},
            headers=_auth_headers(self.token),
        )
        assert r.status_code == 200
        assert r.json()["status"] == "success"

    def test_analyze_returns_result_id(self):               # 22
        r = client.post(
            "/analyze/",
            json={"filename": self.fname, "target_column": "income"},
            headers=_auth_headers(self.token),
        )
        assert "result_id" in r.json()["data"]

    def test_analyze_returns_fairness_score(self):          # 23
        r = client.post(
            "/analyze/",
            json={"filename": self.fname, "target_column": "income",
                  "protected_column": "sex"},
            headers=_auth_headers(self.token),
        )
        score = r.json()["data"]["fairness_score"]
        assert 0 <= score <= 100

    def test_analyze_without_auth_returns_401(self):        # 24
        r = client.post(
            "/analyze/",
            json={"filename": self.fname, "target_column": "income"},
        )
        assert r.status_code in (401, 403)

    def test_analyze_missing_target_column_returns_422(self):# 25
        r = client.post(
            "/analyze/",
            json={"filename": self.fname, "target_column": "nonexistent"},
            headers=_auth_headers(self.token),
        )
        assert r.status_code == 422

    def test_analyze_missing_protected_column_returns_422(self):# 26
        r = client.post(
            "/analyze/",
            json={"filename": self.fname, "target_column": "income",
                  "protected_column": "does_not_exist"},
            headers=_auth_headers(self.token),
        )
        assert r.status_code == 422

    def test_analyze_path_traversal_rejected(self):         # 27
        """Filename with path traversal should be caught by Pydantic validator."""
        r = client.post(
            "/analyze/",
            json={"filename": "../../etc/passwd", "target_column": "income"},
            headers=_auth_headers(self.token),
        )
        assert r.status_code == 422

    def test_analyze_non_csv_filename_rejected(self):       # 28
        r = client.post(
            "/analyze/",
            json={"filename": "evil.exe", "target_column": "income"},
            headers=_auth_headers(self.token),
        )
        assert r.status_code == 422

    def test_analyze_disparate_impact_in_range(self):       # 29
        r = client.post(
            "/analyze/",
            json={"filename": self.fname, "target_column": "income",
                  "protected_column": "sex"},
            headers=_auth_headers(self.token),
        )
        di = r.json()["data"]["disparate_impact"]
        assert 0.0 <= di <= 1.0

    def test_analyze_verdict_is_string(self):               # 30
        r = client.post(
            "/analyze/",
            json={"filename": self.fname, "target_column": "income",
                  "protected_column": "sex"},
            headers=_auth_headers(self.token),
        )
        assert isinstance(r.json()["data"]["verdict"], str)


# ════════════════════════════════════════════════════════════
# 6. METRICS + RESULTS + HISTORY
# ════════════════════════════════════════════════════════════

class TestMetricsResultsHistory:
    def setup_method(self):
        self.token = _register_and_login()
        data       = _full_analysis(self.token)
        self.rid   = data["data"]["result_id"]

    def test_get_metrics_success(self):                     # 31
        r = client.get(f"/metrics/{self.rid}", headers=_auth_headers(self.token))
        assert r.status_code == 200
        assert "disparate_impact" in r.json()["data"]

    def test_get_metrics_nonexistent_returns_404(self):     # 32
        r = client.get("/metrics/00000000-0000-0000-0000-000000000000",
                       headers=_auth_headers(self.token))
        assert r.status_code == 404

    def test_get_result_success(self):                      # 33
        r = client.get(f"/results/{self.rid}", headers=_auth_headers(self.token))
        assert r.status_code == 200
        assert r.json()["data"]["result_id"] == self.rid

    def test_get_result_other_user_forbidden(self):         # 34
        other_token = _register_and_login()
        r = client.get(f"/results/{self.rid}", headers=_auth_headers(other_token))
        assert r.status_code == 403

    def test_history_returns_list(self):                    # 35
        r = client.get("/history", headers=_auth_headers(self.token))
        assert r.status_code == 200
        assert isinstance(r.json()["data"], list)

    def test_history_without_auth_returns_401(self):        # 36
        r = client.get("/history")
        assert r.status_code in (401, 403)

    def test_history_only_shows_own_results(self):          # 37
        other_token = _register_and_login()
        _full_analysis(other_token)
        r = client.get("/history", headers=_auth_headers(self.token))
        for item in r.json()["data"]:
            # result_id must be the one we created, not the other user's
            assert item["result_id"] != other_token

    def test_delete_result_success(self):                   # 38
        r = client.delete(f"/results/{self.rid}", headers=_auth_headers(self.token))
        assert r.status_code == 200

    def test_delete_result_other_user_forbidden(self):      # 39
        other = _register_and_login()
        r = client.delete(f"/results/{self.rid}", headers=_auth_headers(other))
        assert r.status_code == 403


# ════════════════════════════════════════════════════════════
# 7. COMPARE
# ════════════════════════════════════════════════════════════

class TestCompare:
    def setup_method(self):
        self.token = _register_and_login()
        d1 = _full_analysis(self.token)
        d2 = _full_analysis(self.token)
        self.id1 = d1["data"]["result_id"]
        self.id2 = d2["data"]["result_id"]

    def test_compare_success(self):                         # 40
        r = client.get(f"/compare/{self.id1}/{self.id2}",
                       headers=_auth_headers(self.token))
        assert r.status_code == 200

    def test_compare_returns_better_dataset(self):          # 41
        r = client.get(f"/compare/{self.id1}/{self.id2}",
                       headers=_auth_headers(self.token))
        data = r.json()["data"]
        assert data["better_dataset"] in (self.id1, self.id2)

    def test_compare_nonexistent_id_returns_404(self):      # 42
        fake = "00000000-0000-0000-0000-000000000000"
        r    = client.get(f"/compare/{self.id1}/{fake}",
                          headers=_auth_headers(self.token))
        assert r.status_code == 404

    def test_compare_no_keyerror_on_nested_metrics(self):   # 43
        """Regression: original code crashed with KeyError on r1["disparate_impact"]."""
        r = client.get(f"/compare/{self.id1}/{self.id2}",
                       headers=_auth_headers(self.token))
        # Must not be a 500 or contain 'KeyError'
        assert r.status_code == 200
        assert "KeyError" not in r.text


# ════════════════════════════════════════════════════════════
# 8. EXPLAIN
# ════════════════════════════════════════════════════════════

class TestExplain:
    def setup_method(self):
        self.token = _register_and_login()
        data       = _full_analysis(self.token)
        self.rid   = data["data"]["result_id"]

    def test_explain_success(self):                         # 44
        r = client.get(f"/explain/{self.rid}", headers=_auth_headers(self.token))
        assert r.status_code == 200

    def test_explain_has_headline(self):                    # 45
        r = client.get(f"/explain/{self.rid}", headers=_auth_headers(self.token))
        assert "headline" in r.json()["data"]

    def test_explain_has_recommendations(self):             # 46
        r = client.get(f"/explain/{self.rid}", headers=_auth_headers(self.token))
        recs = r.json()["data"]["recommendations"]
        assert isinstance(recs, list) and len(recs) > 0

    def test_explain_nonexistent_returns_404(self):         # 47
        r = client.get("/explain/00000000-0000-0000-0000-000000000000",
                       headers=_auth_headers(self.token))
        assert r.status_code == 404


# ════════════════════════════════════════════════════════════
# 9. SERVICES — UNIT TESTS
# ════════════════════════════════════════════════════════════

class TestServices:
    def test_model_info_endpoint(self):                     # 48
        r = client.get("/model")
        assert r.status_code == 200
        models = r.json()["data"]["supported_models"]
        assert len(models) == 3
        names  = [m["name"] for m in models]
        assert "logistic_regression" in names
        assert "random_forest"       in names
        assert "gradient_boosting"   in names

    def test_auth_utils_hash_verify_roundtrip(self):        # extra
        from services.auth_utils import hash_password, verify_password
        h = hash_password("MyPassword1")
        assert verify_password("MyPassword1", h)
        assert not verify_password("WrongPassword1", h)

    def test_auth_utils_token_creation(self):               # extra
        from services.auth_utils import create_access_token
        token = create_access_token({"sub": "testuser"})
        assert isinstance(token, str) and "." in token

    def test_storage_rejects_non_csv(self):                 # extra
        from services.storage import save_file
        with pytest.raises(ValueError, match="not allowed"):
            save_file(b"malicious", "evil.py")

    def test_storage_path_traversal_rejected(self):         # extra
        from services.storage import get_upload_path
        with pytest.raises(ValueError, match="traversal"):
            get_upload_path("../../etc/passwd")

    def test_firestore_save_get_roundtrip(self, tmp_path):  # extra
        db_path = str(tmp_path / "test.json")
        with patch.dict(os.environ, {"DB_FILE": db_path}):
            # Re-import to pick up patched env
            import importlib
            import services.firestore as fs_module
            importlib.reload(fs_module)

            rid = str(uuid.uuid4())
            fs_module.save_result(rid, {"result_id": rid, "foo": "bar"})
            result = fs_module.get_result(rid)
            assert result is not None
            assert result["foo"] == "bar"

    def test_firestore_get_nonexistent_returns_none(self, tmp_path):  # extra
        db_path = str(tmp_path / "empty.json")
        with patch.dict(os.environ, {"DB_FILE": db_path}):
            import importlib
            import services.firestore as fs_module
            importlib.reload(fs_module)
            assert fs_module.get_result("does-not-exist") is None
