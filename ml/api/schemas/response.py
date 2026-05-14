# ============================================================
# NOIZE — api/schemas/response.py
# PURPOSE: Pydantic models for all API responses.
#          Keeps the API contract explicit and documented.
# ============================================================

from pydantic import BaseModel
from typing   import Any


class StatusResponse(BaseModel):
    status:  str
    message: str


class AnalysisResponse(BaseModel):
    status:               str
    basic_info:           dict
    protected_attributes: list
    target_candidates:    list
    data_quality:         dict
    suggested_protected:  str | None
    suggested_target:     str | None


class BiasDetectionResponse(BaseModel):
    status:             str
    protected_column:   str
    target_column:      str
    groups:             list
    representation:     dict
    disparate_impact:   dict
    demographic_parity: dict
    statistical_parity: dict
    verdict:            dict


class FairnessMetricsResponse(BaseModel):
    status:         str
    protected_col:  str
    privileged:     str
    unprivileged:   str
    group_rates:    dict
    metrics:        dict
    fairness_score: int
    score_emoji:    str
    score_label:    str
    all_passed:     bool
    audit_mode:     str


class ProxyResponse(BaseModel):
    status:          str
    protected_cols:  list
    proxies:         dict
    total_found:     int
    recommendations: list
    warnings:        list
    safe_to_train:   bool


class MitigationResponse(BaseModel):
    status:        str
    protected_col: str
    recommended:   str
    reweighing:    dict
    dir:           dict


class PostAuditResponse(BaseModel):
    status:                  str
    protected_col:           str
    target_col:              str
    groups:                  list
    per_group_performance:   dict
    fairness_metrics:        dict
    prediction_distribution: dict
    model_metrics:           dict


class TradeoffResponse(BaseModel):
    status:               str
    curve:                list
    default_point:        dict
    optimal_point:        dict
    accuracy_cost:        float
    n_legal_thresholds:   int


class ExplainResponse(BaseModel):
    status:      str
    explanation: str


class ReportResponse(BaseModel):
    status:      str
    filename:    str
    message:     str


class HealthResponse(BaseModel):
    status:  str
    version: str
    modules: dict[str, bool]
