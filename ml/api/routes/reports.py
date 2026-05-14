# ============================================================
# NOIZE — api/routes/reports.py
# PURPOSE: FastAPI router for report generation and
#          Gemini AI explanation endpoints.
# ============================================================

import os
import tempfile

from fastapi           import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.schemas.request import ExplainRequest, ReportRequest

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.post("/explain")
def explain(body: ExplainRequest):
    """
    Generate a plain-English explanation of audit results
    using Google Gemini.
    """
    try:
        from shared.gemini_explainer import GeminiExplainer
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))

    try:
        # Key read from GEMINI_API_KEY env var — do not pass in request body
        explainer = GeminiExplainer()
        if body.audit_type == "pre":
            text = explainer.explain_pre_audit(body.results)
        else:
            text = explainer.explain_post_audit(body.results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini error: {exc}")

    return {"status": "success", "explanation": text}


@router.post("/generate")
def generate_report(body: ReportRequest):
    """
    Generate a PDF audit report and return it as a download.
    """
    try:
        from shared.report_generator import ReportGenerator
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))

    tmp_path = os.path.join(tempfile.gettempdir(), body.output_filename)

    try:
        rg = ReportGenerator(
            pre_audit_results  = body.pre_audit_results,
            post_audit_results = body.post_audit_results,
            explanation        = body.explanation,
        )
        rg.generate(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")

    return FileResponse(
        path        = tmp_path,
        filename    = body.output_filename,
        media_type  = "application/pdf",
    )
