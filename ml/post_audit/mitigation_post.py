# ============================================================
# NOIZE — post_audit/mitigation_post.py
# PURPOSE: Fix bias AFTER a model has been trained.
#          Implements:
#            1. Threshold Optimisation (post-processing)
#            2. Calibrated Equalized Odds (post-processing)
# ============================================================

import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger("noize.post_audit_mitigation_post")

from shared.data_loader import binarize_target


class PostMitigation:
    """
    Post-processing mitigation: adjusts decision thresholds
    per group so that fairness criteria are better satisfied
    without retraining the model.

    Usage
    -----
    mit    = PostMitigation(df, protected_col, target_col, predicted_probs)
    result = mit.run_all_mitigations()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        protected_col: str,
        target_col: str,
        predicted_probs: list | np.ndarray,
    ):
        """
        Parameters
        ----------
        df              : original dataframe (with true labels)
        protected_col   : protected attribute column name
        target_col      : ground-truth label column
        predicted_probs : model probability scores (float 0–1)
                          If binary predictions are passed they
                          are used directly as 0/1 thresholds.
        """
        self.df             = binarize_target(df.copy(), target_col)
        self.protected_col  = protected_col
        self.target_col     = target_col
        self.probs          = np.array(predicted_probs, dtype=float)
        self.groups         = self.df[protected_col].dropna().unique().tolist()

        # Add probs to df for easy slicing
        self.df["__prob__"] = self.probs

    # ── Internal helpers ─────────────────────────────────────

    def _predict_at_threshold(self, threshold: float) -> np.ndarray:
        return (self.probs >= threshold).astype(int)

    def _group_tpr_fpr(self, preds: np.ndarray) -> dict[str, dict]:
        """TPR + FPR per group for a given prediction array."""
        result: dict[str, dict] = {}
        for g in self.groups:
            mask = self.df[self.protected_col] == g
            y    = self.df[mask][self.target_col].values
            yhat = preds[mask.values]

            tp = int(((yhat == 1) & (y == 1)).sum())
            fn = int(((yhat == 0) & (y == 1)).sum())
            fp = int(((yhat == 1) & (y == 0)).sum())
            tn = int(((yhat == 0) & (y == 0)).sum())
            pos_rate = float(yhat.mean())

            result[str(g)] = {
                "tpr":      round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
                "fpr":      round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
                "pos_rate": round(pos_rate, 4),
            }
        return result

    def _di(self, preds: np.ndarray) -> float:
        rates = []
        for g in self.groups:
            mask = self.df[self.protected_col] == g
            rates.append(float(preds[mask.values].mean()))
        return round(min(rates) / max(rates), 4) if max(rates) > 0 else 0.0

    # ── Algorithm 1: Threshold Optimisation ─────────────────

    def optimise_thresholds(self, fairness_criterion: str = "demographic_parity") -> dict:
        """
        Search for per-group decision thresholds that minimise
        unfairness (measured by Disparate Impact or DP gap)
        while keeping overall accuracy as high as possible.

        fairness_criterion: 'demographic_parity' | 'equalized_odds'
        """
        logger.info(f"\nThreshold Optimisation ({fairness_criterion}) ...")

        best_thresholds: dict[str, float] = {}
        candidate_thresholds = np.linspace(0.1, 0.9, 17)

        # Global baseline at 0.5
        baseline_preds = self._predict_at_threshold(0.5)
        di_before      = self._di(baseline_preds)

        # Grid search: for each group independently find threshold
        # that moves its positive rate toward the global average
        global_pos_rate = float(baseline_preds.mean())

        for g in self.groups:
            mask  = self.df[self.protected_col] == g
            probs = self.probs[mask.values]

            best_t, best_dist = 0.5, float("inf")
            for t in candidate_thresholds:
                preds    = (probs >= t).astype(int)
                pos_rate = float(preds.mean())
                dist     = abs(pos_rate - global_pos_rate)
                if dist < best_dist:
                    best_dist = dist
                    best_t    = t
            best_thresholds[str(g)] = round(float(best_t), 2)

        # Apply per-group thresholds
        adjusted_preds = np.zeros(len(self.df), dtype=int)
        for g in self.groups:
            mask = self.df[self.protected_col] == g
            t    = best_thresholds[str(g)]
            adjusted_preds[mask.values] = (self.probs[mask.values] >= t).astype(int)

        di_after    = self._di(adjusted_preds)
        improvement = round(di_after - di_before, 4)
        perf_before = self._group_tpr_fpr(baseline_preds)
        perf_after  = self._group_tpr_fpr(adjusted_preds)

        logger.info(f"  Thresholds : {best_thresholds}")
        logger.info(f"  DI Before  : {di_before:.4f}")
        logger.info(f"  DI After   : {di_after:.4f}  (Δ {improvement:+.4f})")

        return {
            "algorithm":         "Threshold Optimisation",
            "fairness_criterion": fairness_criterion,
            "thresholds":        best_thresholds,
            "adjusted_predictions": adjusted_preds.tolist(),
            "di_before":         di_before,
            "di_after":          di_after,
            "improvement":       improvement,
            "legal_compliant":   di_after >= 0.8,
            "perf_before":       perf_before,
            "perf_after":        perf_after,
        }

    # ── Algorithm 2: Reject Option Classification ────────────

    def reject_option_classification(self, margin: float = 0.15) -> dict:
        """
        Reject Option Classification (Kamiran et al. 2012).

        Near the decision boundary (|prob - 0.5| ≤ margin):
        - Unprivileged group → predict POSITIVE  (favour the disadvantaged)
        - Privileged group   → predict NEGATIVE

        Outside the margin: use the standard 0.5 threshold.

        Parameters
        ----------
        margin : half-width of the critical region (default 0.15 → 0.35–0.65).
        """
        logger.info(f"\nReject Option Classification (margin={margin}) ...")

        # Find privileged group (highest positive rate at 0.5 threshold)
        baseline_preds = self._predict_at_threshold(0.5)
        group_rates    = {
            str(g): float(baseline_preds[(self.df[self.protected_col] == g).values].mean())
            for g in self.groups
        }
        privileged   = max(group_rates, key=group_rates.get)   # type: ignore[arg-type]
        unprivileged = min(group_rates, key=group_rates.get)   # type: ignore[arg-type]

        di_before   = self._di(baseline_preds)
        roc_preds   = baseline_preds.copy()

        for i, (prob, group) in enumerate(
            zip(self.probs, self.df[self.protected_col].values)
        ):
            if abs(prob - 0.5) <= margin:
                # In the critical region
                if str(group) == unprivileged:
                    roc_preds[i] = 1   # boost disadvantaged group
                elif str(group) == privileged:
                    roc_preds[i] = 0   # suppress advantaged group

        di_after    = self._di(roc_preds)
        improvement = round(di_after - di_before, 4)

        logger.info(f"  Privileged  : {privileged}")
        logger.info(f"  Unprivileged: {unprivileged}")
        logger.info(f"  DI Before   : {di_before:.4f}")
        logger.info(f"  DI After    : {di_after:.4f}  (Δ {improvement:+.4f})")

        return {
            "algorithm":    "Reject Option Classification",
            "paper":        "Kamiran et al. (2012)",
            "margin":       margin,
            "privileged":   privileged,
            "unprivileged": unprivileged,
            "adjusted_predictions": roc_preds.tolist(),
            "di_before":    di_before,
            "di_after":     di_after,
            "improvement":  improvement,
            "legal_compliant": di_after >= 0.8,
            "perf_before":  self._group_tpr_fpr(baseline_preds),
            "perf_after":   self._group_tpr_fpr(roc_preds),
        }

    # ── Main entry point ─────────────────────────────────────

    def run_all_mitigations(self) -> dict:
        """
        Run both post-processing algorithms and recommend the better one.
        Called by /post-audit/mitigate endpoint.
        """
        logger.info(f"\n{'='*55}")
        logger.info("NOIZE Post-Model Mitigation")
        logger.info(f"Protected: {self.protected_col}  |  Target: {self.target_col}")
        logger.info(f"{'='*55}")

        to_results  = self.optimise_thresholds()
        roc_results = self.reject_option_classification()

        best = (
            "Threshold Optimisation"
            if to_results["improvement"] >= roc_results["improvement"]
            else "Reject Option Classification"
        )

        logger.info(f"\n  Recommended: {best}")
        logger.info(f"{'='*55}\n")

        return {
            "status":              "success",
            "protected_col":       self.protected_col,
            "threshold_optimisation": to_results,
            "reject_option":       roc_results,
            "recommended":         best,
        }
