# ============================================================
# NOIZE — post_audit/decision_auditor.py
# PURPOSE: Audit a trained model's predictions for
#          post-model fairness violations.
# ============================================================

import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger("noize.post_audit_decision_auditor")

from pre_audit.fairness_metrics import FairnessMetrics


class DecisionAuditor:
    """
    Runs post-model fairness analysis on a trained model's predictions.

    Compares fairness BEFORE mitigation (baseline model)
    vs AFTER mitigation (reweighed or other corrected model).

    Usage
    -----
    auditor = DecisionAuditor(
        df, protected_col="sex", target_col="income",
        predictions=model_preds
    )
    result  = auditor.run_full_audit()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        protected_col: str,
        target_col: str,
        predictions: list | np.ndarray,
    ):
        if len(predictions) != len(df):
            raise ValueError(
                f"predictions length ({len(predictions)}) must match df length ({len(df)})."
            )

        self.df            = df.copy()
        self.protected_col = protected_col
        self.target_col    = target_col
        self.predictions   = np.array(predictions)

        # Add prediction column
        self.df["__prediction__"] = self.predictions

        self.groups = self.df[protected_col].dropna().unique().tolist()

    # ── Per-group performance ────────────────────────────────

    def per_group_performance(self) -> dict:
        """
        Accuracy, precision, recall, F1 broken down by group.
        Reveals whether the model performs worse for some groups.
        """
        results: dict[str, dict] = {}

        for group in self.groups:
            mask  = self.df[self.protected_col] == group
            gdf   = self.df[mask]
            y     = gdf[self.target_col].values
            yhat  = gdf["__prediction__"].values

            if len(y) == 0:
                continue

            # Binarize y if needed
            if not set(y).issubset({0, 1}):
                most_common = pd.Series(y).value_counts().index[0]
                y = (y == most_common).astype(int)

            tp = int(((yhat == 1) & (y == 1)).sum())
            tn = int(((yhat == 0) & (y == 0)).sum())
            fp = int(((yhat == 1) & (y == 0)).sum())
            fn = int(((yhat == 0) & (y == 1)).sum())

            acc  = (tp + tn) / len(y) if len(y) > 0 else 0
            prec = tp / (tp + fp)     if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn)     if (tp + fn) > 0 else 0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
            fpr  = fp / (fp + tn)     if (fp + tn) > 0 else 0

            results[str(group)] = {
                "n_samples":  int(len(y)),
                "accuracy":   round(acc, 4),
                "precision":  round(prec, 4),
                "recall":     round(rec, 4),
                "f1":         round(f1, 4),
                "tpr":        round(rec, 4),   # TPR = recall
                "fpr":        round(fpr, 4),
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            }

        return results

    # ── Fairness metrics ─────────────────────────────────────

    def compute_fairness_metrics(self) -> dict:
        """
        Delegates to FairnessMetrics in post-audit mode.
        Returns the full metric report including EOD and EO.
        """
        fm = FairnessMetrics(
            df            = self.df,
            protected_col = self.protected_col,
            target_col    = self.target_col,
            predicted_col = "__prediction__",
        )
        return fm.get_all_metrics()

    # ── Decision analysis ─────────────────────────────────────

    def analyze_prediction_distribution(self) -> dict:
        """
        Check whether positive predictions are distributed fairly
        across groups (predicted demographic parity).
        """
        dist: dict[str, dict] = {}
        for group in self.groups:
            mask      = self.df[self.protected_col] == group
            preds     = self.df[mask]["__prediction__"].values
            pos_rate  = round(float(preds.mean()), 4)
            dist[str(group)] = {
                "n_positive":   int(preds.sum()),
                "n_total":      int(len(preds)),
                "positive_rate": pos_rate,
            }

        rates = [v["positive_rate"] for v in dist.values()]
        gap   = round(max(rates) - min(rates), 4)
        di    = round(min(rates) / max(rates), 4) if max(rates) > 0 else 0.0

        return {
            "group_prediction_rates": dist,
            "prediction_gap":         gap,
            "predicted_di":           di,
            "passes_di_threshold":    di >= 0.8,
            "passes_dp_threshold":    gap <= 0.1,
        }

    # ── Main entry point ─────────────────────────────────────

    def run_full_audit(self) -> dict:
        """
        Run the complete post-model audit.
        Called by /post-audit/audit endpoint.
        """
        logger.info(f"\n{'='*55}")
        logger.info("NOIZE Post-Model Audit")
        logger.info(f"Protected: {self.protected_col}  |  Target: {self.target_col}")
        logger.info(f"Groups: {self.groups}")
        logger.info(f"{'='*55}")

        perf       = self.per_group_performance()
        fairness   = self.compute_fairness_metrics()
        pred_dist  = self.analyze_prediction_distribution()

        # Quick summary
        score  = fairness.get("fairness_score", 0)
        emoji  = fairness.get("score_emoji", "")
        logger.info(f"\n  {emoji} Post-audit fairness score: {score}%")
        print(f"  Predicted DI : {pred_dist['predicted_di']:.4f} "
              f"({'✅ PASS' if pred_dist['passes_di_threshold'] else '❌ FAIL'})")
        logger.info(f"{'='*55}\n")

        return {
            "status":                  "success",
            "protected_col":           self.protected_col,
            "target_col":              self.target_col,
            "groups":                  self.groups,
            "per_group_performance":   perf,
            "fairness_metrics":        fairness,
            "prediction_distribution": pred_dist,
        }
