# ============================================================
# NOIZE Backend — routes/analyze.py  (interconnected v3)
# Now calls the ML engine via httpx for full bias analysis.
# BUG FIX: was doing its own basic math instead of using ML engine.
# ============================================================

import os
import uuid
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from models.schemas      import AnalyzeRequest
from services.firestore  import save_result
from services.storage    import get_upload_path
from services.auth_utils import get_current_user

logger = logging.getLogger("noize.analyze")
router = APIRouter(prefix="/analyze", tags=["Analyze"])

ML_ENGINE_URL = os.getenv("ML_ENGINE_URL", "http://localhost:8000")


@router.post("/")
def analyze(data: AnalyzeRequest, user: str = Depends(get_current_user)):

    # Step 1: resolve file safely
    try:
        file_path = get_upload_path(data.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"File '{data.filename}' not found. Upload it first via /upload/."
        )

    # Step 2: forward file to ML engine
    try:
        with open(file_path, "rb") as f:
            ml_resp = httpx.post(
                f"{ML_ENGINE_URL}/pre-audit/upload-analyze",
                files={"file": (data.filename, f, "text/csv")},
                timeout=60.0,
            )
        ml_resp.raise_for_status()
        ml = ml_resp.json()

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="ML engine is not running. Start it on port 8000 first."
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="ML engine timed out.")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"ML engine error: {exc.response.text}")

    # Step 3: extract and store results
    quality  = ml.get("data_quality", {})
    basic    = ml.get("basic_info", {})
    total_rows = int(basic.get("total_rows", 0))
    fairness_score = float(quality.get("score", 50))
    confidence_score = round(min(total_rows / 1000, 1.0) * 100, 2)

    if fairness_score >= 80:   verdict, emoji = "LOW BIAS",    "🟢"
    elif fairness_score >= 60: verdict, emoji = "MEDIUM BIAS", "🟡"
    else:                      verdict, emoji = "HIGH BIAS",   "🔴"

    result_id = str(uuid.uuid4())
    save_result(result_id, {
        "result_id": result_id, "user": user, "filename": data.filename,
        "target_column": data.target_column, "protected_column": data.protected_column,
        "rows": total_rows,
        "metrics": {
            "fairness_score": fairness_score, "confidence_score": confidence_score,
            "verdict": verdict, "emoji": emoji,
            "data_quality": quality,
            "protected_attrs": ml.get("protected_attributes", []),
            "target_candidates": ml.get("target_candidates", []),
        },
        "ml_full_result": ml,
    })
    logger.info("result_id=%s user=%s fairness=%.1f verdict=%s", result_id, user, fairness_score, verdict)

    return {
        "status": "success",
        "data": {
            "result_id": result_id,
            "fairness_score": fairness_score,
            "confidence_score": confidence_score,
            "verdict": verdict,
            "emoji": emoji,
            "rows": total_rows,
            "protected_attrs": ml.get("protected_attributes", []),
            "target_candidates": ml.get("target_candidates", []),
            "data_quality": quality,
        },
        "error": None,
    }
