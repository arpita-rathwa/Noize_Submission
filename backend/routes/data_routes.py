# ============================================================
# NOIZE Backend — routes/data_routes.py
# Routes that were missing from main.py:
#   GET  /metrics/{result_id}
#   GET  /results/{result_id}
#   GET  /history/
#   DELETE /results/{result_id}
#   POST /compare/
#   GET  /explain/{result_id}
# ============================================================

import os
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from services.firestore  import get_result, get_all_results, delete_result
from services.auth_utils import get_current_user

logger = logging.getLogger("noize.data_routes")
router = APIRouter(tags=["Data"])

ML_ENGINE_URL = os.getenv("ML_ENGINE_URL", "http://localhost:8000")


def _get_own_result(result_id: str, user: str) -> dict:
    """Fetch a result and verify ownership. Raises 404/403 appropriately."""
    result = get_result(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Result '{result_id}' not found.")
    if result.get("user") != user:
        raise HTTPException(status_code=403, detail="Access denied.")
    return result


# ── GET /metrics/{result_id} ─────────────────────────────────

@router.get("/metrics/{result_id}")
def get_metrics(result_id: str, user: str = Depends(get_current_user)):
    result  = _get_own_result(result_id, user)
    metrics = result.get("metrics", {})
    return {"status": "success", "data": metrics, "error": None}


# ── GET /results/{result_id} ─────────────────────────────────

@router.get("/results/{result_id}")
def get_result_detail(result_id: str, user: str = Depends(get_current_user)):
    result = _get_own_result(result_id, user)
    return {"status": "success", "data": result, "error": None}


# ── DELETE /results/{result_id} ──────────────────────────────

@router.delete("/results/{result_id}")
def delete_result_endpoint(result_id: str, user: str = Depends(get_current_user)):
    _get_own_result(result_id, user)   # ownership check
    delete_result(result_id)
    return {"status": "success", "data": {"deleted": result_id}, "error": None}


# ── GET /history/ ────────────────────────────────────────────

@router.get("/history/")
def get_history(user: str = Depends(get_current_user)):
    all_results = get_all_results()
    own = [r for r in all_results if r.get("user") == user]
    return {"status": "success", "data": own, "error": None}


# ── POST /compare/ ───────────────────────────────────────────

@router.post("/compare/")
def compare_results(
    payload: dict,
    user: str = Depends(get_current_user),
):
    """
    Compare two analysis results.
    Body: {"result_id_1": "...", "result_id_2": "..."}
    """
    id1 = payload.get("result_id_1") or payload.get("id1")
    id2 = payload.get("result_id_2") or payload.get("id2")

    if not id1 or not id2:
        raise HTTPException(status_code=422, detail="Provide result_id_1 and result_id_2.")

    r1 = _get_own_result(id1, user)
    r2 = _get_own_result(id2, user)

    def _score(r):
        return r.get("metrics", {}).get("fairness_score", 0)

    s1, s2 = _score(r1), _score(r2)

    better = id1 if s1 >= s2 else id2
    delta  = round(abs(s1 - s2), 2)

    return {
        "status": "success",
        "data": {
            "result_1": {"result_id": id1, "fairness_score": s1,
                         "metrics": r1.get("metrics", {})},
            "result_2": {"result_id": id2, "fairness_score": s2,
                         "metrics": r2.get("metrics", {})},
            "better_dataset": better,
            "score_delta":    delta,
        },
        "error": None,
    }


# ── GET /explain/{result_id} ─────────────────────────────────

@router.get("/explain/{result_id}")
def explain(result_id: str, user: str = Depends(get_current_user)):
    """
    Call the ML engine's Gemini-powered explanation endpoint for this result.
    Falls back to a deterministic explanation if the ML engine is unavailable.
    """
    result  = _get_own_result(result_id, user)
    metrics = result.get("metrics", {})

    # Try the ML engine first
    try:
        resp = httpx.post(
            f"{ML_ENGINE_URL}/explain",
            json={"metrics": metrics, "result_id": result_id},
            timeout=30.0,
        )
        if resp.status_code == 200:
            explanation = resp.json()
            return {"status": "success", "data": explanation, "error": None}
    except (httpx.ConnectError, httpx.TimeoutException):
        logger.warning("ML engine unavailable for /explain — using fallback.")

    # Deterministic fallback (no Gemini required)
    score   = metrics.get("fairness_score", 50)
    verdict = metrics.get("verdict", "UNKNOWN")
    di      = metrics.get("data_quality", {}).get("disparate_impact", None)

    headline = (
        f"This dataset shows {verdict} with a fairness score of {score:.1f}/100."
    )

    if score >= 80:
        summary = "The model appears largely fair across protected groups."
        recs = [
            "Continue monitoring for bias as the model is retrained on new data.",
            "Conduct periodic fairness audits every quarter.",
        ]
    elif score >= 60:
        summary = "Moderate bias detected. Some groups receive noticeably different outcomes."
        recs = [
            "Apply Reweighing or Disparate Impact Remover before retraining.",
            "Review feature engineering for proxy variables correlated with the protected attribute.",
            "Consider threshold optimisation in post-processing.",
        ]
    else:
        summary = "Significant bias detected. Immediate mitigation is recommended."
        recs = [
            "Apply Reweighing (Kamiran & Calders 2012) to rebalance training data.",
            "Use Disparate Impact Remover on correlated numeric features.",
            "After retraining, apply Reject Option Classification to predictions.",
            "Consult your legal/compliance team before deploying this model.",
        ]

    return {
        "status": "success",
        "data": {
            "headline":        headline,
            "summary":         summary,
            "recommendations": recs,
            "fairness_score":  score,
            "verdict":         verdict,
            "disparate_impact": di,
            "source":          "fallback",
        },
        "error": None,
    }
