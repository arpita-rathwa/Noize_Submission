# ============================================================
# NOIZE — api/routes/pre_audit.py
# PURPOSE: FastAPI router for all pre-audit endpoints.
# ============================================================

import os
import tempfile
import shutil

from fastapi           import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from api.schemas.request  import (
    PreAuditRequest, ProxyDetectRequest,
    MitigationRequest,
)

from shared.data_loader      import load_dataset, DATASET_CONFIG
from pre_audit.data_analyzer import DataAnalyzer
from pre_audit.bias_detector import BiasDetector
from pre_audit.fairness_metrics import FairnessMetrics
from pre_audit.proxy_detector   import ProxyDetector
from pre_audit.mitigation_pre   import PreMitigation

router = APIRouter(prefix="/pre-audit", tags=["Pre-Audit"])

# ── Helper: resolve dataset path ────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

def _dataset_path(dataset: str) -> str:
    path = os.path.join(DATA_DIR, f"{dataset}.csv")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset}.csv' not found in {DATA_DIR}. "
                   "Download it first (see README)."
        )
    return path


def _resolve_cols(dataset: str, protected_col: str | None, target_col: str | None):
    cfg = DATASET_CONFIG.get(dataset, {})
    p   = protected_col or cfg.get("protected")
    t   = target_col    or cfg.get("target")
    if not p or not t:
        raise HTTPException(
            status_code=422,
            detail="Could not infer protected_col / target_col. "
                   "Please supply them explicitly."
        )
    return p, t


# ── Endpoints ────────────────────────────────────────────────

@router.post("/analyze")
def analyze_dataset(body: PreAuditRequest):
    """
    Step 1 — Profile a dataset: shape, types, missing values,
    protected attributes, target candidates, quality score.
    """
    path     = _dataset_path(body.dataset)
    analyzer = DataAnalyzer(path)
    result   = analyzer.run_full_analysis()

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))

    return result


@router.post("/detect")
def detect_bias(body: PreAuditRequest):
    """
    Step 2 — Run bias detection: DI, DP gap, SPD, verdict.
    """
    path = _dataset_path(body.dataset)
    df, cfg = load_dataset(path)
    p, t = _resolve_cols(body.dataset, body.protected_col, body.target_col)

    detector = BiasDetector(df, protected_col=p, target_col=t)
    return detector.run_full_detection()


@router.post("/metrics")
def fairness_metrics(body: PreAuditRequest):
    """
    Step 3 — Calculate full pre-audit fairness metric suite
    (DI, DP, SPD, Theil Index) + overall fairness score.
    """
    path = _dataset_path(body.dataset)
    df, cfg = load_dataset(path)
    p, t = _resolve_cols(body.dataset, body.protected_col, body.target_col)

    fm = FairnessMetrics(df, protected_col=p, target_col=t)
    return fm.get_all_metrics()


@router.post("/proxy")
def proxy_detection(body: ProxyDetectRequest):
    """
    Step 4 — Detect proxy variables correlated with
    protected attributes (correlation, MI, Cramér's V).
    """
    path = _dataset_path(body.dataset)
    df, _ = load_dataset(path)

    # Validate protected_cols exist in the df
    missing = [c for c in body.protected_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Columns not found: {missing}. Available: {list(df.columns)}"
        )

    detector = ProxyDetector(df, protected_cols=body.protected_cols)
    return detector.run_full_proxy_detection()


@router.post("/mitigate")
def mitigate(body: MitigationRequest):
    """
    Step 5 — Apply pre-processing bias mitigation:
    Reweighing + Disparate Impact Remover.
    Returns per-algorithm DI improvement (not the full df,
    which is too large to serialise over HTTP).
    """
    path = _dataset_path(body.dataset)
    df, cfg = load_dataset(path)
    p, t = _resolve_cols(body.dataset, body.protected_col, body.target_col)

    mit    = PreMitigation(df, protected_col=p, target_col=t)
    result = mit.run_all_mitigations()

    # Strip df objects before returning (not JSON-serialisable)
    for key in ("reweighing", "dir"):
        if key in result:
            result[key].pop("df_mitigated", None)

    return result


@router.post("/upload-analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    Upload any CSV and get a full pre-audit analysis.
    Useful for datasets beyond the 4 built-in ones.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Save to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()

        analyzer = DataAnalyzer(tmp.name)
        result   = analyzer.run_full_analysis()
    finally:
        os.unlink(tmp.name)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))

    return result
