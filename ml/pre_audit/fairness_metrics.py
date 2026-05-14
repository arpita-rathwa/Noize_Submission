# ============================================================
# NOIZE — pre_audit/fairness_metrics.py
# PURPOSE: All fairness metrics in one class.
#          Works for BOTH pre-audit (data only)
#          and post-audit (with model predictions).
# ============================================================

import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger("noize.pre_audit_fairness_metrics")

from shared.data_loader import binarize_target


class FairnessMetrics:
    """
    Calculates the full NOIZE fairness metric suite.

    Pre-audit mode  (predicted_col=None):
        - Disparate Impact, Demographic Parity,
          Statistical Parity, Theil Index

    Post-audit mode (predicted_col provided):
        - All of the above PLUS
          Equal Opportunity, Equalized Odds

    Usage
    -----
    fm = FairnessMetrics(df, protected_col="sex", target_col="income")
    result = fm.get_all_metrics()
    """

    THRESHOLDS = {
        "disparate_impact":    0.80,
        "demographic_parity":  0.10,
        "statistical_parity":  0.10,
        "equal_opportunity":   0.10,
        "equalized_odds":      0.10,
        "theil_index":         0.20,
    }

    def __init__(
        self,
        df: pd.DataFrame,
        protected_col: str,
        target_col: str,
        predicted_col: str | None = None,
    ):
        self.df            = binarize_target(df.copy(), target_col)
        self.protected_col = protected_col
        self.target_col    = target_col
        self.predicted_col = predicted_col

        self.groups = self.df[protected_col].dropna().unique().tolist()

        # Binarize predictions if provided
        if predicted_col and predicted_col in self.df.columns:
            pred = self.df[predicted_col]
            if not set(pred.dropna().unique()) <= {0, 1}:
                threshold = pred.median()
                self.df[predicted_col] = pred.apply(
                    lambda x: 1 if x >= threshold else 0
                )

        self._find_privilege()

    # ── Internal helpers ─────────────────────────────────────

    def _find_privilege(self):
        """Identify privileged (highest rate) and unprivileged groups."""
        rates = {}
        for g in self.groups:
            mask  = self.df[self.protected_col] == g
            rates[str(g)] = float(self.df[mask][self.target_col].mean())
        self.privileged   = max(rates, key=rates.get)   # type: ignore[arg-type]
        self.unprivileged = min(rates, key=rates.get)   # type: ignore[arg-type]
        self.group_rates  = {k: round(v, 4) for k, v in rates.items()}

    def _tpr_fpr(self, group: str) -> tuple[float, float]:
        """True-positive rate and false-positive rate for one group."""
        mask = self.df[self.protected_col] == group
        g    = self.df[mask]
        y    = g[self.target_col].values
        yhat = g[self.predicted_col].values if self.predicted_col else None
        if yhat is None:
            return 0.0, 0.0
        tp = int(((yhat == 1) & (y == 1)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return round(tpr, 4), round(fpr, 4)

    def _severity(self, value: float, lo: float, hi: float, lower_is_bad: bool = False) -> tuple[str, str]:
        """Return (severity_label, emoji) for a metric value."""
        if lower_is_bad:
            # Higher is better (e.g. DI)
            if value >= hi:
                return "LOW",    "🟢"
            if value >= lo:
                return "MEDIUM", "🟡"
            return "HIGH", "🔴"
        else:
            # Lower is better (e.g. DP gap)
            if value <= lo:
                return "LOW",    "🟢"
            if value <= hi:
                return "MEDIUM", "🟡"
            return "HIGH", "🔴"

    # ── Pre-audit metrics ────────────────────────────────────

    def disparate_impact(self) -> dict:
        """DI = P(positive|unprivileged) / P(positive|privileged)."""
        p_unpriv = self.group_rates.get(self.unprivileged, 0)
        p_priv   = self.group_rates.get(self.privileged, 1)
        di       = round(p_unpriv / p_priv, 4) if p_priv > 0 else 0.0
        t        = self.THRESHOLDS["disparate_impact"]
        sev, em  = self._severity(di, t, 0.9, lower_is_bad=True)

        return {
            "metric":      "Disparate Impact",
            "value":       di,
            "threshold":   t,
            "passed":      di >= t,
            "severity":    sev,
            "emoji":       em,
            "legal":       di >= t,
            "group_rates": self.group_rates,
            "formula":     "P(positive|unprivileged) / P(positive|privileged)",
        }

    def demographic_parity(self) -> dict:
        """DP gap = P(positive|privileged) - P(positive|unprivileged)."""
        p_priv   = self.group_rates.get(self.privileged, 0)
        p_unpriv = self.group_rates.get(self.unprivileged, 0)
        dp       = round(p_priv - p_unpriv, 4)
        t        = self.THRESHOLDS["demographic_parity"]
        sev, em  = self._severity(dp, 0.05, t)

        return {
            "metric":      "Demographic Parity Gap",
            "value":       dp,
            "threshold":   t,
            "passed":      dp <= t,
            "severity":    sev,
            "emoji":       em,
            "group_rates": self.group_rates,
            "formula":     "P(positive|privileged) - P(positive|unprivileged)",
        }

    def statistical_parity(self) -> dict:
        """SPD = P(positive|unprivileged) - P(positive|privileged) [signed]."""
        p_priv   = self.group_rates.get(self.privileged, 0)
        p_unpriv = self.group_rates.get(self.unprivileged, 0)
        spd      = round(p_unpriv - p_priv, 4)
        t        = self.THRESHOLDS["statistical_parity"]
        sev, em  = self._severity(abs(spd), 0.05, t)

        return {
            "metric":    "Statistical Parity Diff",
            "value":     spd,
            "threshold": t,
            "passed":    abs(spd) <= t,
            "severity":  sev,
            "emoji":     em,
            "formula":   "P(positive|unprivileged) - P(positive|privileged)",
        }

    def theil_index(self) -> dict:
        """
        Theil Index T = (1/n) Σ (yi/ȳ) ln(yi/ȳ).
        Measures inequality in outcome distribution.
        """
        vals   = self.df[self.target_col].fillna(0).values.astype(float) + 1e-10
        mu     = vals.mean()
        ratios = vals / mu
        theil  = float(np.mean(ratios * np.log(ratios)))
        theil  = round(max(0.0, theil), 4)
        t      = self.THRESHOLDS["theil_index"]
        sev, em = self._severity(theil, 0.1, t)

        return {
            "metric":    "Theil Index",
            "value":     theil,
            "threshold": t,
            "passed":    theil <= t,
            "severity":  sev,
            "emoji":     em,
            "formula":   "(1/n) * Σ (yi/ȳ) * ln(yi/ȳ)",
        }

    # ── Post-audit metrics ───────────────────────────────────

    def equal_opportunity(self) -> dict:
        """
        EOD = TPR(unprivileged) - TPR(privileged).
        Post-audit only — requires model predictions.
        """
        if not self.predicted_col:
            return {"metric": "Equal Opportunity Diff",
                    "error":  "Requires model predictions (post-audit only)."}

        tpr_rates = {str(g): self._tpr_fpr(g)[0] for g in self.groups}
        tpr_priv   = tpr_rates.get(self.privileged, 0)
        tpr_unpriv = tpr_rates.get(self.unprivileged, 0)
        eod        = round(tpr_unpriv - tpr_priv, 4)
        t          = self.THRESHOLDS["equal_opportunity"]
        sev, em    = self._severity(abs(eod), 0.05, t)

        return {
            "metric":    "Equal Opportunity Diff",
            "value":     eod,
            "tpr_rates": tpr_rates,
            "threshold": t,
            "passed":    abs(eod) <= t,
            "severity":  sev,
            "emoji":     em,
            "formula":   "TPR(unprivileged) - TPR(privileged)",
        }

    def equalized_odds(self) -> dict:
        """
        EO = max(|TPR gap|, |FPR gap|).
        Post-audit only — requires model predictions.
        """
        if not self.predicted_col:
            return {"metric": "Equalized Odds Diff",
                    "error":  "Requires model predictions (post-audit only)."}

        tpr_rates, fpr_rates = {}, {}
        for g in self.groups:
            tpr, fpr = self._tpr_fpr(g)
            tpr_rates[str(g)] = tpr
            fpr_rates[str(g)] = fpr

        tpr_gap = abs(tpr_rates.get(self.unprivileged, 0) - tpr_rates.get(self.privileged, 0))
        fpr_gap = abs(fpr_rates.get(self.unprivileged, 0) - fpr_rates.get(self.privileged, 0))
        eo      = round(max(tpr_gap, fpr_gap), 4)
        t       = self.THRESHOLDS["equalized_odds"]
        sev, em = self._severity(eo, 0.05, t)

        return {
            "metric":    "Equalized Odds Diff",
            "value":     eo,
            "tpr_rates": tpr_rates,
            "fpr_rates": fpr_rates,
            "tpr_gap":   round(tpr_gap, 4),
            "fpr_gap":   round(fpr_gap, 4),
            "threshold": t,
            "passed":    eo <= t,
            "severity":  sev,
            "emoji":     em,
            "formula":   "max(|TPR gap|, |FPR gap|)",
        }

    # ── Aggregation ──────────────────────────────────────────

    def get_all_metrics(self) -> dict:
        """
        Compute all available metrics.
        Returns a full report with per-metric results + overall score.
        """
        mode = "post-audit" if self.predicted_col else "pre-audit"
        logger.info(f"\nCalculating fairness metrics [{mode}] ...")
        print(f"Protected: {self.protected_col}  |  "
              f"Privileged: {self.privileged}  |  "
              f"Unprivileged: {self.unprivileged}")
        logger.info("─" * 50)

        metrics: dict[str, dict] = {
            "disparate_impact":   self.disparate_impact(),
            "demographic_parity": self.demographic_parity(),
            "statistical_parity": self.statistical_parity(),
            "theil_index":        self.theil_index(),
        }

        if self.predicted_col:
            metrics["equal_opportunity"] = self.equal_opportunity()
            metrics["equalized_odds"]    = self.equalized_odds()

        # Print per-metric summary
        for m in metrics.values():
            if "error" not in m:
                logger.info(f"  {m.get('emoji','')} {m.get('metric',''):<30}: {m.get('value', 'N/A')}")

        # Overall score: percentage of metrics that passed
        valid   = [m for m in metrics.values() if "error" not in m]
        passed  = [m for m in valid if m.get("passed", True)]
        score   = round(len(passed) / len(valid) * 100) if valid else 0

        if score >= 80:
            score_emoji, score_label = "🟢", "FAIR"
        elif score >= 60:
            score_emoji, score_label = "🟡", "CONCERN"
        else:
            score_emoji, score_label = "🔴", "BIASED"

        print(f"\n  {score_emoji} Overall Fairness Score: {score}%  "
              f"({len(passed)}/{len(valid)} metrics passed)")

        return {
            "status":         "success",
            "protected_col":  self.protected_col,
            "privileged":     self.privileged,
            "unprivileged":   self.unprivileged,
            "group_rates":    self.group_rates,
            "metrics":        metrics,
            "fairness_score": score,
            "score_emoji":    score_emoji,
            "score_label":    score_label,
            "all_passed":     score == 100,
            "audit_mode":     mode,
        }
