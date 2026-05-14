# ============================================================
# NOIZE Backend — routes/upload.py
# CSV file upload endpoint.
#
# FIXES vs original:
#   - Auth is now required (Depends(get_current_user))
#   - File type validated via storage.save_file() (.csv only)
#   - File size limit enforced in storage layer
#   - Returns safe filename (not server filesystem path)
#   - HTTP 400 on validation errors instead of 200 + message
# ============================================================

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from services.storage  import save_file
from services.auth_utils import get_current_user

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("/")
async def upload_file(
    file: UploadFile = File(...),
    user: str = Depends(get_current_user),
):
    """
    Upload a CSV dataset for analysis.

    - Requires a valid Bearer token.
    - Only .csv files accepted (max 50 MB by default).
    - Returns a `filename` reference to use in /analyze.
    """
    content = await file.read()

    try:
        stored_name = save_file(content, file.filename or "upload.csv")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return {
        "status": "success",
        "data": {
            "filename":     stored_name,
            "original":     file.filename,
            "size_bytes":   len(content),
            "uploaded_by":  user,
        },
        "error": None,
    }
