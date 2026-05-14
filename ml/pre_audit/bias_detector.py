# ============================================================
# NOIZE — pre_audit/bias_detector.py
# PURPOSE: Core pre-model bias detection.
#          Computes DI, DP, SPD and overall verdict.
# ============================================================

import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger("noize.pre_audit_bias_detector")

from shared.data_loader import binarize_target


class BiasDetector:
    """
    Detects bias in a dataset *before* model training.

    Key metrics
    -----------
    - Disparate Impact Ratio  (legal 80 % rule)
    - Demographic Parity Gap  (> 10 % = bias)
    - Statistical Parity Diff (signed gap)
    - Representation analysis (group sizes)

    Usage
    -----
    detector = BiasDetector(df, protected_col="sex", target_col="income")
    result   = detector.run_full_detection()
    """

    DI_THRESHOLD  = 0.8    # 80 % rule — US equal-employment law
    DP_THRESHOLD  = 0.1    # max acceptable 10 % outcome gap
    REP_THRESHOLD = 20.0   # group must be ≥ 20 % of dataset

    def __init__(self, df: pd.DataFrame, protected_col: str, target_col: str):
        # ── Validate inputs ──────────────────────────────────
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty.")
        if protected_col not in df.columns:
            raise ValueError(
                f"Protected column '{protected_col}' not found. "
                f"Available: {list(df.columns)}"
            )
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found. "
                f"Available: {list(df.columns)}"
            )

        self.df            = binarize_target(df, target_col)
        self.protected_col = protected_col
        self.target_col    = target_col
        self.groups        = (
            self.df[protected_col].dropna().unique().tolist()
        )

    # ── Helpers ──────────────────────────────────────────────

    def _group_rates(self) -> dict[str, float]:
        """Positive-outcome rate for each group."""
        rates: dict[str, float] = {}
        for group in self.groups:
            mask  = self.df[self.protected_col] == group
            gdf   = self.df[mask]
            if len(gdf):
                rates[str(group)] = round(float(gdf[self.target_col].mean()), 4)
        return rates

    # ── Metrics ──────────────────────────────────────────────

    def analyze_representation(self) -> dict:
        """
        Check group sizes relative to dataset total.
        Groups below REP_THRESHOLD (20 %) are flagged.
        """
        counts = self.df[self.protected_col].value_counts()
        total  = len(self.df)
        rep: dict[str, dict] = {}

        for group in self.groups:
            count = int(counts.get(group, 0))
            pct   = round(count / total * 100, 2)
            under = pct < self.REP_THRESHOLD
            rep[str(group)] = {
                "count":          count,
                "percentage":     pct,
                "underrepresented": under,
                "status":         "UNDERREPRESENTED" if under else "OK",
                "emoji":          "🔴" if under else "✅",
            }

        pcts  = [v["percentage"] for v in rep.values()]
        ratio = round(max(pcts) / min(pcts), 2) if min(pcts) > 0 else float("inf")

        return {
            "groups":           rep,
            "imbalance_ratio":  ratio,
            "is_imbalanced":    ratio > 2.0,
            "total_samples":    total,
            "protected_column": self.protected_col,
        }

    def calculate_disparate_impact(self) -> dict:
        """
        DI = min_group_rate / max_group_rate
        DI < 0.8 → legal violation (80 % rule).
        """
        rates = self._group_rates()
        if len(rates) < 2:
            return {"error": "Need at least 2 groups."}

        max_rate     = max(rates.values())
        min_rate     = min(rates.values())
        privileged   = max(rates, key=rates.get)   # type: ignore[arg-type]
        unprivileged = min(rates, key=rates.get)   # type: ignore[arg-type]
        di           = round(min_rate / max_rate, 4) if max_rate > 0 else 0.0

        if di >= 0.9:
            severity, emoji, status = "LOW",    "🟢", "PASS"
        elif di >= self.DI_THRESHOLD:
            severity, emoji, status = "MEDIUM", "🟡", "PASS"
        else:
            severity, emoji, status = "HIGH",   "🔴", "FAIL"

        return {
            "disparate_impact":     di,
            "group_rates":          rates,
            "privileged_group":     privileged,
            "unprivileged_group":   unprivileged,
            "privileged_rate":      round(max_rate, 4),
            "unprivileged_rate":    round(min_rate, 4),
            "threshold":            self.DI_THRESHOLD,
            "passes_threshold":     di >= self.DI_THRESHOLD,
            "is_legally_compliant": di >= self.DI_THRESHOLD,
            "severity":             severity,
            "emoji":                emoji,
            "status":               status,
            "interpretation": (
                f"{unprivileged} receives positive outcomes at "
                f"{di*100:.1f}% the rate of {privileged}. "
                + ("Legal threshold met ✅" if di >= self.DI_THRESHOLD
                   else "BELOW legal threshold! 🔴")
            ),
        }

    def calculate_demographic_parity(self) -> dict:
        """
        DP gap = max_rate - min_rate
        Gap > 0.10 → significant bias.
        """
        rates = self._group_rates()
        if len(rates) < 2:
            return {"error": "Need at least 2 groups."}

        max_rate = max(rates.values())
        min_rate = min(rates.values())
        dp_gap   = round(max_rate - min_rate, 4)

        if dp_gap <= 0.05:
            severity, emoji, status = "LOW",    "🟢", "PASS"
        elif dp_gap <= self.DP_THRESHOLD:
            severity, emoji, status = "MEDIUM", "🟡", "PASS"
        else:
            severity, emoji, status = "HIGH",   "🔴", "FAIL"

        return {
            "demographic_parity_gap": dp_gap,
            "group_rates":            rates,
            "max_rate":               round(max_rate, 4),
            "min_rate":               round(min_rate, 4),
            "threshold":              self.DP_THRESHOLD,
            "passes_threshold":       dp_gap <= self.DP_THRESHOLD,
            "severity":               severity,
            "emoji":                  emoji,
            "status":                 status,
            "interpretation": (
                f"Outcome rates differ by {dp_gap*100:.1f}% across groups. "
                + ("Within acceptable range ✅"
                   if dp_gap <= self.DP_THRESHOLD
                   else "Exceeds acceptable gap! 🔴")
            ),
        }

    def calculate_statistical_parity(self) -> dict:
        """
        SPD = min_rate - max_rate  (signed, negative = disadvantaged)
        |SPD| > 0.10 → biased.
        """
        rates = self._group_rates()
        if len(rates) < 2:
            return {"error": "Need at least 2 groups."}

        spd = round(min(rates.values()) - max(rates.values()), 4)

        if abs(spd) < 0.05:
            severity, emoji = "LOW",    "🟢"
        elif abs(spd) < 0.1:
            severity, emoji = "MEDIUM", "🟡"
        else:
            severity, emoji = "HIGH",   "🔴"

        return {
            "statistical_parity_diff": spd,
            "group_rates":             rates,
            "passes_threshold":        abs(spd) < 0.1,
            "severity":                severity,
            "emoji":                   emoji,
            "interpretation": (
                f"SPD = {spd:.4f}. "
                + ("Fair distribution ✅" if abs(spd) < 0.1
                   else "Unfair distribution! 🔴")
            ),
        }

    def generate_verdict(self, di: dict, dp: dict, sp: dict, rep: dict) -> dict:
        """Roll up all metrics into a single PASS / FAIL verdict."""
        failures:     list[str] = []
        warnings_out: list[str] = []

        if not di.get("passes_threshold", True):
            failures.append(
                f"Disparate Impact = {di.get('disparate_impact', 0):.3f} "
                f"(below {self.DI_THRESHOLD} legal threshold)"
            )
        if not dp.get("passes_threshold", True):
            failures.append(
                f"Demographic Parity Gap = {dp.get('demographic_parity_gap', 0):.3f} "
                f"(above {self.DP_THRESHOLD} threshold)"
            )
        if rep.get("is_imbalanced", False):
            warnings_out.append(
                f"Group imbalance detected! Ratio: {rep.get('imbalance_ratio')}x"
            )

        severities = [
            di.get("severity", "LOW"),
            dp.get("severity", "LOW"),
            sp.get("severity", "LOW"),
        ]
        if "HIGH" in severities:
            overall_sev, overall_emoji = "HIGH",   "🔴"
        elif "MEDIUM" in severities:
            overall_sev, overall_emoji = "MEDIUM", "🟡"
        else:
            overall_sev, overall_emoji = "LOW",    "🟢"

        audit_passed = len(failures) == 0

        recs: list[str] = []
        if not audit_passed:
            recs.append("Apply Reweighing to balance group outcome rates.")
            recs.append("Use Disparate Impact Remover on feature values.")
        if rep.get("is_imbalanced"):
            recs.append("Oversample underrepresented groups before training.")
        if not recs:
            recs.append("Dataset passes all fairness checks — safe to train. ✅")

        return {
            "audit_passed":         audit_passed,
            "overall_severity":     overall_sev,
            "overall_emoji":        overall_emoji,
            "verdict": (
                "✅ AUDIT PASSED — Dataset is relatively fair!"
                if audit_passed else
                "❌ AUDIT FAILED — Bias detected! Fix before training!"
            ),
            "failures":             failures,
            "warnings":             warnings_out,
            "recommendations":      recs,
            "is_legally_compliant": di.get("is_legally_compliant", True),
            "metrics_summary": {
                "disparate_impact": {
                    "value":  di.get("disparate_impact", 0),
                    "status": di.get("status", "N/A"),
                    "emoji":  di.get("emoji", ""),
                },
                "demographic_parity": {
                    "value":  dp.get("demographic_parity_gap", 0),
                    "status": dp.get("status", "N/A"),
                    "emoji":  dp.get("emoji", ""),
                },
                "statistical_parity": {
                    "value":  sp.get("statistical_parity_diff", 0),
                    "status": "PASS" if sp.get("passes_threshold", True) else "FAIL",
                    "emoji":  sp.get("emoji", ""),
                },
            },
        }

    # ── Main entry point ─────────────────────────────────────

    def run_full_detection(self) -> dict:
        """
        Run all 5 detection steps and return a complete bias report.
        Called by /pre-audit/detect endpoint.
        """
        logger.info(f"\n{'='*55}")
        logger.info("NOIZE Bias Detection")
        logger.info(f"Protected: {self.protected_col}  |  Target: {self.target_col}")
        logger.info(f"Groups: {self.groups}")
        logger.info(f"{'='*55}")

        rep     = self.analyze_representation()
        di      = self.calculate_disparate_impact()
        dp      = self.calculate_demographic_parity()
        sp      = self.calculate_statistical_parity()
        verdict = self.generate_verdict(di, dp, sp, rep)

        logger.info(f"\n  DI  : {di.get('emoji','')} {di.get('disparate_impact','N/A')} ({di.get('status','N/A')})")
        logger.info(f"  DP  : {dp.get('emoji','')} {dp.get('demographic_parity_gap','N/A')} ({dp.get('status','N/A')})")
        logger.info(f"  SPD : {sp.get('emoji','')} {sp.get('statistical_parity_diff','N/A')}")
        logger.info(f"\n  {verdict['verdict']}")
        logger.info(f"{'='*55}\n")

        return {
            "status":           "success",
            "protected_column": self.protected_col,
            "target_column":    self.target_col,
            "groups":           self.groups,
            "representation":   rep,
            "disparate_impact": di,
            "demographic_parity": dp,
            "statistical_parity": sp,
            "verdict":          verdict,
        }
