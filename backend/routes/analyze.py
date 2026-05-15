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

    # Step 1b: validate requested columns exist before calling ML engine
    # Returns 422 (not 503) so tests that omit columns get the right status code.
    try:
        import pandas as _pd
        _header = _pd.read_csv(file_path, nrows=0)
        _cols   = set(_header.columns.str.strip())
        if data.target_column not in _cols:
            raise HTTPException(
                status_code=422,
                detail=f"Column '{data.target_column}' not found in dataset. "
                       f"Available columns: {sorted(_cols)}",
            )
        if data.protected_column and data.protected_column not in _cols:
            raise HTTPException(
                status_code=422,
                detail=f"Column '{data.protected_column}' not found in dataset. "
                       f"Available columns: {sorted(_cols)}",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read CSV headers: {exc}")

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
    bias_metrics = ml.get("bias_metrics", {})
    total_rows = int(basic.get("total_rows", 0))
    fairness_score = float(quality.get("score", 50))
    confidence_score = round(min(total_rows / 1000, 1.0) * 100, 2)
    disparate_impact = float(bias_metrics.get("disparate_impact", 0.0))
    statistical_parity_gap = float(bias_metrics.get("statistical_parity_gap", 0.0))

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
            "disparate_impact": disparate_impact,
            "statistical_parity_gap": statistical_parity_gap,
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
            "disparate_impact": disparate_impact,
            "statistical_parity_gap": statistical_parity_gap,
            "protected_attrs": ml.get("protected_attributes", []),
            "target_candidates": ml.get("target_candidates", []),
            "data_quality": quality,
        },
        "error": None,
    }
