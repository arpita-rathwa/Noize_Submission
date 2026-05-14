# ============================================================
# NOIZE — api/routes/post_audit.py
# PURPOSE: FastAPI router for all post-audit endpoints.
# ============================================================

import os
from fastapi import APIRouter, HTTPException

from api.schemas.request import PostAuditRequest, TradeoffRequest
from shared.data_loader  import load_dataset, DATASET_CONFIG

from post_audit.model_trainer    import ModelTrainer
from post_audit.decision_auditor import DecisionAuditor
from post_audit.mitigation_post  import PostMitigation
from post_audit.tradeoff_analyzer import TradeoffAnalyzer
from pre_audit.mitigation_pre    import PreMitigation

router = APIRouter(prefix="/post-audit", tags=["Post-Audit"])

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _dataset_path(dataset: str) -> str:
    path = os.path.join(DATA_DIR, f"{dataset}.csv")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset}.csv' not found. Download it first (see README)."
        )
    return path


def _resolve_cols(dataset: str, protected_col, target_col):
    cfg = DATASET_CONFIG.get(dataset, {})
    p   = protected_col or cfg.get("protected")
    t   = target_col    or cfg.get("target")
    if not p or not t:
        raise HTTPException(
            status_code=422,
            detail="Could not infer protected_col / target_col. Supply them explicitly."
        )
    return p, t


# ── Endpoints ────────────────────────────────────────────────

@router.post("/train")
def train_model(body: PostAuditRequest):
    """
    Train a model on the chosen dataset and return:
    - train / test performance metrics
    - top feature importances
    (Predictions are computed internally for /audit.)
    """
    path = _dataset_path(body.dataset)
    df, _ = load_dataset(path)
    p, t  = _resolve_cols(body.dataset, body.protected_col, body.target_col)

    trainer = ModelTrainer(df, protected_col=p, target_col=t)

    sample_weight = None
    if body.use_reweighing:
        mit_result    = PreMitigation(df, protected_col=p, target_col=t).apply_reweighing()
        rw_df         = mit_result["df_mitigated"]
        sample_weight = rw_df["sample_weight"].values

    result = trainer.train_and_evaluate(
        model_type    = body.model_type,
        sample_weight = sample_weight,
    )

    # Don't ship the predictions list (huge); keep metrics only
    return {
        "status":              result["status"],
        "model_type":          result["model_type"],
        "protected_col":       result["protected_col"],
        "target_col":          result["target_col"],
        "train_size":          result["train_size"],
        "test_size":           result["test_size"],
        "train_metrics":       result["train_metrics"],
        "test_metrics":        result["test_metrics"],
        "feature_importances": result["feature_importances"],
        "reweighing_applied":  body.use_reweighing,
    }


@router.post("/audit")
def audit_model(body: PostAuditRequest):
    """
    Full post-model audit:
    1. Train model
    2. Get predictions for every row
    3. Run DecisionAuditor (EOD, EO, per-group performance)
    Returns fairness metrics + per-group breakdown.
    """
    path = _dataset_path(body.dataset)
    df, _ = load_dataset(path)
    p, t  = _resolve_cols(body.dataset, body.protected_col, body.target_col)

    trainer = ModelTrainer(df, protected_col=p, target_col=t)

    sample_weight = None
    if body.use_reweighing:
        mit_result    = PreMitigation(df, protected_col=p, target_col=t).apply_reweighing()
        rw_df         = mit_result["df_mitigated"]
        sample_weight = rw_df["sample_weight"].values

    train_result = trainer.train_and_evaluate(
        model_type    = body.model_type,
        sample_weight = sample_weight,
    )
    predictions  = train_result["predictions"]

    auditor = DecisionAuditor(df, protected_col=p, target_col=t, predictions=predictions)
    result  = auditor.run_full_audit()

    result["model_metrics"] = {
        "train": train_result["train_metrics"],
        "test":  train_result["test_metrics"],
    }
    return result


@router.post("/mitigate")
def mitigate_post(body: PostAuditRequest):
    """
    Post-processing mitigation:
    Train a model → get probability scores →
    apply Threshold Optimisation + Reject Option Classification.
    Returns adjusted predictions and DI before/after.
    """
    path = _dataset_path(body.dataset)
    df, _ = load_dataset(path)
    p, t  = _resolve_cols(body.dataset, body.protected_col, body.target_col)

    # Train model and get probabilities
    trainer      = ModelTrainer(df, protected_col=p, target_col=t)
    train_result = trainer.train_and_evaluate(model_type=body.model_type)

    # Get probability scores from the trained model
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    X_sc   = trainer.scaler.transform(trainer.X)
    if hasattr(trainer.model, "predict_proba"):
        probs = trainer.model.predict_proba(X_sc)[:, 1]
    else:
        # fallback: use hard predictions as pseudo-probs
        probs = trainer.model.predict(X_sc).astype(float)

    mit    = PostMitigation(df, protected_col=p, target_col=t, predicted_probs=probs)
    result = mit.run_all_mitigations()

    # Strip large prediction arrays from response
    for key in ("threshold_optimisation", "reject_option"):
        if key in result:
            result[key].pop("adjusted_predictions", None)

    return result


@router.post("/tradeoff")
def tradeoff_analysis(body: TradeoffRequest):
    """
    Sweep decision thresholds and return the accuracy-fairness
    tradeoff curve + optimal operating point.
    """
    path = _dataset_path(body.dataset)
    df, _ = load_dataset(path)
    p, t  = _resolve_cols(body.dataset, body.protected_col, body.target_col)

    trainer      = ModelTrainer(df, protected_col=p, target_col=t)
    train_result = trainer.train_and_evaluate(model_type=body.model_type)

    import numpy as np
    X_sc  = trainer.scaler.transform(trainer.X)
    if hasattr(trainer.model, "predict_proba"):
        probs = trainer.model.predict_proba(X_sc)[:, 1]
    else:
        probs = trainer.model.predict(X_sc).astype(float)

    ta     = TradeoffAnalyzer(df, protected_col=p, target_col=t, predicted_probs=probs)
    result = ta.run_analysis()
    return result
