# ============================================================
# NOIZE — api/schemas/request.py
# PURPOSE: Pydantic models for all API request bodies.
# ============================================================

from pydantic import BaseModel, Field
from typing   import Literal


class PreAuditRequest(BaseModel):
    """Common parameters for any pre-audit endpoint."""
    dataset:       Literal["adult", "compas", "german", "home_credit"] = Field(
        ..., description="Which built-in dataset to use."
    )
    protected_col: str | None = Field(None, description="Override default protected attribute.")
    target_col:    str | None = Field(None, description="Override default target column.")


class ProxyDetectRequest(BaseModel):
    dataset:        Literal["adult", "compas", "german", "home_credit"]
    protected_cols: list[str] = Field(
        ..., description="List of protected attribute column names."
    )


class MitigationRequest(BaseModel):
    dataset:       Literal["adult", "compas", "german", "home_credit"]
    protected_col: str | None = None
    target_col:    str | None = None
    repair_level:  float      = Field(0.8, ge=0.0, le=1.0)


class PostAuditRequest(BaseModel):
    dataset:       Literal["adult", "compas", "german", "home_credit"]
    protected_col: str | None  = None
    target_col:    str | None  = None
    model_type:    Literal["logistic", "random_forest", "gradient_boost"] = "logistic"
    use_reweighing: bool       = Field(False, description="Apply Reweighing before training.")


class TradeoffRequest(BaseModel):
    dataset:       Literal["adult", "compas", "german", "home_credit"]
    protected_col: str | None = None
    target_col:    str | None = None
    model_type:    Literal["logistic", "random_forest", "gradient_boost"] = "logistic"


class ExplainRequest(BaseModel):
    audit_type:    Literal["pre", "post"] = "pre"
    results:       dict = Field(..., description="Raw audit result dict.")
    # FIXED: gemini_api_key removed from request body — it appeared in server access logs.
    # Set GEMINI_API_KEY environment variable on the server instead.


class ReportRequest(BaseModel):
    pre_audit_results:  dict
    post_audit_results: dict | None = None
    explanation:        str  | None = None
    output_filename:    str         = "noize_audit_report.pdf"
