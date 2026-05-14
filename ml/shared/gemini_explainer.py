# ============================================================
# NOIZE — shared/gemini_explainer.py  (patched)
# FIXES:
#   - Gemini API key read from GEMINI_API_KEY env var
#     (was passed in request body → appeared in access logs)
#   - 30-second timeout on every API call
#   - Graceful fallback if google-generativeai not installed
# ============================================================

import json, os, logging
import logging
logger = logging.getLogger("noize.gemini")

try:
    import google.generativeai as genai
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False


class GeminiExplainer:
    MODEL   = "gemini-1.5-flash"
    TIMEOUT = 30   # seconds

    def __init__(self, api_key: str | None = None):
        if not _HAS_GENAI:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        # FIXED: prefer env var; fall back to explicit arg for backward compat
        resolved_key = os.getenv("GEMINI_API_KEY") or api_key
        if not resolved_key:
            raise ValueError(
                "Gemini API key not found. Set the GEMINI_API_KEY environment variable "
                "(do not pass it in the request body — it will appear in server logs)."
            )
        genai.configure(api_key=resolved_key)
        self._model = genai.GenerativeModel(self.MODEL)

    def _call(self, prompt: str) -> str:
        # FIXED: added request_options timeout so a slow/hung Gemini call
        # does not block the API worker indefinitely.
        try:
            from google.api_core import retry as api_retry
            response = self._model.generate_content(
                prompt,
                request_options={"timeout": self.TIMEOUT},
            )
            return response.text.strip()
        except Exception as exc:
            logger.error("Gemini API error: %s", exc)
            return f"Explanation unavailable (Gemini error: {exc})"

    def explain_pre_audit(self, fairness_results: dict) -> str:
        summary = {
            "fairness_score": fairness_results.get("fairness_score"),
            "privileged":     fairness_results.get("privileged"),
            "unprivileged":   fairness_results.get("unprivileged"),
            "metrics": {
                k: {"value": v.get("value"), "passed": v.get("passed")}
                for k, v in fairness_results.get("metrics", {}).items()
                if "error" not in v
            },
        }
        prompt = f"""
You are an AI fairness expert writing for a non-technical business audience.

Here are the fairness audit results for a machine-learning dataset:
{json.dumps(summary, indent=2)}

Please provide a concise explanation (4-6 sentences) covering:
1. Whether bias exists and how serious it is.
2. Which group is disadvantaged and why this matters.
3. The most important metric to focus on.
4. One concrete action the team should take before training the model.

Write in plain English. Avoid jargon. Be direct and actionable.
"""
        return self._call(prompt)

    def explain_post_audit(self, audit_results: dict) -> str:
        fm = audit_results.get("fairness_metrics", {})
        summary = {
            "fairness_score":  fm.get("fairness_score"),
            "privileged":      fm.get("privileged"),
            "unprivileged":    fm.get("unprivileged"),
            "predicted_di":    audit_results.get("prediction_distribution", {}).get("predicted_di"),
            "metrics": {
                k: {"value": v.get("value"), "passed": v.get("passed")}
                for k, v in fm.get("metrics", {}).items()
                if "error" not in v
            },
        }
        prompt = f"""
You are an AI fairness expert writing for a non-technical business audience.

A machine-learning model has been trained and audited. Here are the results:
{json.dumps(summary, indent=2)}

Please explain (4-6 sentences):
1. Whether the trained model is biased in its decisions.
2. Which group is being discriminated against and how.
3. Whether this creates legal risk under anti-discrimination law.
4. The recommended mitigation strategy.

Write in plain English. Be direct, clear, and actionable.
"""
        return self._call(prompt)

    def explain_metric(self, metric_name: str, metric_value: float) -> str:
        prompt = f"""
Explain what the fairness metric "{metric_name}" means and whether
a value of {metric_value:.4f} indicates a problem.

Keep it to 2-3 sentences in plain English for a business audience.
"""
        return self._call(prompt)
