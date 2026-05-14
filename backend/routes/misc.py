# ============================================================
# NOIZE Backend — routes/misc.py
# Consolidates: metrics, results, history, compare,
#               explain, model into one router file.
#
# FIXES vs original:
#   - compare.py KeyError crash: r1["disparate_impact"] assumed
#     metrics were at top level — they are nested under "metrics"
#   - explain.py: richer output — verdict, group_rates, all
#     fairness scores, recommendations (not just one string)
#   - history.py: auth required (was public)
#   - results.py: auth required + ownership check
#   - All endpoints return proper HTTP 404 instead of 200+error
# ============================================================

from fastapi import APIRouter, Depends, HTTPException, status

from services.firestore  import get_result, get_all_results, delete_result
from services.auth_utils import get_current_user

router = APIRouter(tags=["Audit"])


# ── GET /metrics/{result_id} ──────────────────────────────────

@router.get("/metrics/{result_id}")
def get_metrics(result_id: str, user: str = Depends(get_current_user)):
    """Return the fairness metrics stored for a result."""
    result = get_result(result_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result '{result_id}' not found.",
        )

    metrics = result.get("metrics", {})
    return {
        "status": "success",
        "data": {
            "result_id":          result_id,
            "disparate_impact":   metrics.get("disparate_impact"),
            "statistical_parity": metrics.get("statistical_parity"),
            "theil_index":        metrics.get("theil_index"),
            "fairness_score":     metrics.get("fairness_score"),
            "confidence_score":   metrics.get("confidence_score"),
            "group_rates":        metrics.get("group_rates", {}),
            "verdict":            metrics.get("verdict"),
            "emoji":              metrics.get("emoji"),
        },
        "error": None,
    }


# ── GET /results/{result_id} ──────────────────────────────────

@router.get("/results/{result_id}")
def get_result_endpoint(result_id: str, user: str = Depends(get_current_user)):
    """Return the full stored result object."""
    result = get_result(result_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result '{result_id}' not found.",
        )

    # Ownership check: only the user who ran the analysis can view it
    if result.get("user") != user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this result.",
        )

    return {"status": "success", "data": result, "error": None}


# ── DELETE /results/{result_id} ───────────────────────────────

@router.delete("/results/{result_id}", status_code=status.HTTP_200_OK)
def delete_result_endpoint(result_id: str, user: str = Depends(get_current_user)):
    """Delete a stored result (owner only)."""
    result = get_result(result_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result '{result_id}' not found.",
        )
    if result.get("user") != user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this result.",
        )

    delete_result(result_id)
    return {"status": "success", "data": "Result deleted.", "error": None}


# ── GET /history ──────────────────────────────────────────────

@router.get("/history")
def history(user: str = Depends(get_current_user)):
    """
    Return all past audit results for the authenticated user.
    FIX: was public (no auth) — now requires a valid token.
    """
    all_results = get_all_results()

    # Filter to only this user's results
    user_results = [r for r in all_results if r.get("user") == user]

    # Return a summary list (not full raw data for brevity)
    summaries = []
    for r in user_results:
        m = r.get("metrics", {})
        summaries.append({
            "result_id":       r.get("result_id"),
            "filename":        r.get("filename"),
            "fairness_score":  m.get("fairness_score"),
            "verdict":         m.get("verdict"),
            "emoji":           m.get("emoji"),
            "protected_column": r.get("protected_column"),
            "target_column":   r.get("target_column"),
            "rows":            r.get("rows"),
        })

    return {
        "status": "success",
        "data":   summaries,
        "error":  None,
    }


# ── GET /compare/{id1}/{id2} ──────────────────────────────────

@router.get("/compare/{id1}/{id2}")
def compare(id1: str, id2: str, user: str = Depends(get_current_user)):
    """
    Compare two audit results and identify the fairer dataset.

    FIX — KeyError crash: original code accessed r1["disparate_impact"]
    directly, but metrics are stored under r1["metrics"]["disparate_impact"].
    """
    r1 = get_result(id1)
    r2 = get_result(id2)

    missing = []
    if not r1:
        missing.append(id1)
    if not r2:
        missing.append(id2)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result(s) not found: {missing}",
        )

    # FIX: drill into nested "metrics" dict
    di1 = r1.get("metrics", {}).get("disparate_impact", 0.0)
    di2 = r2.get("metrics", {}).get("disparate_impact", 0.0)
    fs1 = r1.get("metrics", {}).get("fairness_score", 0.0)
    fs2 = r2.get("metrics", {}).get("fairness_score", 0.0)

    # Closer to 1.0 = fairer
    better_id    = id1 if abs(1 - di1) <= abs(1 - di2) else id2
    worse_id     = id2 if better_id == id1 else id1
    di_better    = di1 if better_id == id1 else di2
    di_worse     = di2 if better_id == id1 else di1

    improvement  = round(abs(di_better - di_worse) * 100, 2)

    return {
        "status": "success",
        "data": {
            "better_dataset": better_id,
            "worse_dataset":  worse_id,
            "comparison": {
                id1: {
                    "disparate_impact": di1,
                    "fairness_score":   fs1,
                    "filename":         r1.get("filename"),
                },
                id2: {
                    "disparate_impact": di2,
                    "fairness_score":   fs2,
                    "filename":         r2.get("filename"),
                },
            },
            "di_improvement_pct": improvement,
            "summary": (
                f"Dataset '{r1.get('filename')}' is fairer by {improvement}% "
                f"on Disparate Impact."
                if better_id == id1 else
                f"Dataset '{r2.get('filename')}' is fairer by {improvement}% "
                f"on Disparate Impact."
            ),
        },
        "error": None,
    }


# ── GET /explain/{result_id} ──────────────────────────────────

@router.get("/explain/{result_id}")
def explain(result_id: str, user: str = Depends(get_current_user)):
    """
    Return a structured plain-English explanation of an audit result.

    FIX — richer output: original returned one sentence.
    Now returns: verdict, severity, affected groups, metric values,
    legal compliance flag, and actionable recommendations.
    """
    result = get_result(result_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result '{result_id}' not found.",
        )

    metrics = result.get("metrics", {})
    di      = metrics.get("disparate_impact")
    sp      = metrics.get("statistical_parity", 0.0)
    theil   = metrics.get("theil_index", 0.0)
    fs      = metrics.get("fairness_score", 0.0)
    groups  = metrics.get("group_rates", {})

    if di is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Metrics are incomplete — re-run analysis.",
        )

    # ── Severity ──────────────────────────────────────────────
    if di >= 0.9:
        severity, emoji_sev, headline = (
            "LOW",
            "🟢",
            "This dataset shows low bias. It is likely safe to train a model on it.",
        )
    elif di >= 0.8:
        severity, emoji_sev, headline = (
            "MEDIUM",
            "🟡",
            "Moderate bias detected. Review group outcome rates before training.",
        )
    else:
        severity, emoji_sev, headline = (
            "HIGH",
            "🔴",
            "High bias detected. The 80% legal threshold is breached. "
            "Apply mitigation before training.",
        )

    # ── Affected groups ───────────────────────────────────────
    privileged   = max(groups, key=groups.get) if groups else None
    unprivileged = min(groups, key=groups.get) if groups else None
    affected_msg = (
        f"The '{unprivileged}' group receives positive outcomes at a significantly "
        f"lower rate ({groups.get(unprivileged, 0):.1%}) compared to "
        f"'{privileged}' ({groups.get(privileged, 0):.1%})."
        if len(groups) >= 2 else
        "No protected column was specified — group analysis not available."
    )

    # ── Legal compliance ──────────────────────────────────────
    legal_ok  = di >= 0.8
    legal_msg = (
        "✅ Meets the legal 80% Disparate Impact threshold."
        if legal_ok else
        "❌ Violates the 80% Disparate Impact rule (US Equal Employment law). "
        "Deployment without mitigation creates legal liability."
    )

    # ── Recommendations ───────────────────────────────────────
    recommendations = []
    if di < 0.8:
        recommendations.append(
            "Apply Reweighing (Kamiran & Calders 2012) to balance group-outcome weights."
        )
        recommendations.append(
            "Apply Disparate Impact Remover (Feldman 2015) to repair feature distributions."
        )
    if sp > 0.15:
        recommendations.append(
            f"Statistical Parity gap of {sp:.1%} is large — "
            "consider oversampling the underrepresented group."
        )
    if theil > 0.2:
        recommendations.append(
            f"Theil Index of {theil:.3f} indicates high inequality — "
            "inspect the target variable distribution."
        )
    if not recommendations:
        recommendations.append("No immediate action needed. Monitor fairness metrics post-deployment.")

    return {
        "status": "success",
        "data": {
            "result_id":         result_id,
            "filename":          result.get("filename"),
            "protected_column":  result.get("protected_column"),
            "fairness_score":    fs,
            "severity":          severity,
            "emoji":             emoji_sev,
            "headline":          headline,
            "affected_groups":   affected_msg,
            "legal_compliance":  legal_msg,
            "is_legally_compliant": legal_ok,
            "metrics_summary": {
                "disparate_impact":   di,
                "statistical_parity": sp,
                "theil_index":        theil,
                "group_rates":        groups,
            },
            "recommendations":   recommendations,
        },
        "error": None,
    }


# ── GET /model ────────────────────────────────────────────────

@router.get("/model")
def model_info():
    """
    Model route — returns supported model types and their descriptions.
    (Stub promoted to informational endpoint.)
    """
    return {
        "status": "success",
        "data": {
            "supported_models": [
                {
                    "name":        "logistic_regression",
                    "description": "Fast linear baseline. Best for interpretability.",
                },
                {
                    "name":        "random_forest",
                    "description": "Ensemble of decision trees. Good general performance.",
                },
                {
                    "name":        "gradient_boosting",
                    "description": "Highest accuracy. Slower to train.",
                },
            ],
            "note": (
                "Model training is handled by the ML engine at /post-audit/train. "
                "This backend manages auth, storage, and audit history."
            ),
        },
        "error": None,
    }
