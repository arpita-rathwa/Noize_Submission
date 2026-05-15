# ============================================================
# NOIZE — pre_audit/mitigation_pre.py
# PURPOSE: Fix bias BEFORE model training.
#          Implements:
#            1. Reweighing            (Kamiran & Calders 2012)
#            2. Disparate Impact Remover (Feldman et al. 2015)
# ============================================================

import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger("noize.pre_audit_mitigation_pre")

from shared.data_loader import binarize_target


class PreMitigation:
    """
    Pre-processing bias mitigation.

    Usage
    -----
    mit     = PreMitigation(df, protected_col="sex", target_col="income")
    results = mit.run_all_mitigations()
    df_rw   = results["reweighing"]["df_mitigated"]   # has sample_weight col
    df_dir  = results["dir"]["df_mitigated"]           # has repaired features
    """

    def __init__(self, df: pd.DataFrame, protected_col: str, target_col: str):
        self.df            = binarize_target(df.copy(), target_col)
        self.protected_col = protected_col
        self.target_col    = target_col
        self.groups        = self.df[protected_col].dropna().unique().tolist()

    # ── Algorithm 1: Reweighing ──────────────────────────────

    def apply_reweighing(self) -> dict:
        """
        Assign sample weights so that each (group, outcome) combination
        has the same effective probability it would have in a fair dataset.

        Weight = P(group) × P(outcome)  /  P(group AND outcome)

        Returns the original df with an added 'sample_weight' column.
        Pass that column to sklearn's `fit(…, sample_weight=weights)`.
        """
        logger.info("\nApplying Reweighing (Kamiran & Calders 2012) ...")
        df_rw   = self.df.copy()
        n       = len(df_rw)
        weights = np.ones(n)

        for group in self.groups:
            for outcome in [0, 1]:
                mask      = (
                    (df_rw[self.protected_col] == group) &
                    (df_rw[self.target_col]    == outcome)
                )
                n_combo   = int(mask.sum())
                if n_combo == 0:
                    continue

                n_group   = int((df_rw[self.protected_col] == group).sum())
                n_outcome = int((df_rw[self.target_col]    == outcome).sum())

                p_expected = (n_group / n) * (n_outcome / n)
                p_actual   = n_combo / n
                weight     = p_expected / p_actual if p_actual > 0 else 1.0

                weights[mask.values] = weight
                arrow = "↑ upweighted" if weight > 1 else "↓ downweighted"
                logger.info(f"  group={group}, outcome={outcome}: weight={weight:.4f} ({arrow})")

        df_rw["sample_weight"] = weights

        # Metrics before / after
        def _group_rates(frame, weight_col=None):
            rates = {}
            for g in self.groups:
                m = frame[self.protected_col] == g
                if weight_col:
                    gd = frame[m]
                    w  = gd[weight_col]
                    rates[str(g)] = round(float((gd[self.target_col] * w).sum() / w.sum()), 4)
                else:
                    rates[str(g)] = round(float(frame[m][self.target_col].mean()), 4)
            return rates

        orig_rates = _group_rates(df_rw)
        w_rates    = _group_rates(df_rw, "sample_weight")

        def _di(rates):
            v = list(rates.values())
            return round(min(v) / max(v), 4) if max(v) > 0 else 0.0

        di_before  = _di(orig_rates)
        di_after   = _di(w_rates)
        improvement = round(di_after - di_before, 4)

        logger.info(f"\n  DI Before : {di_before:.4f}")
        logger.info(f"  DI After  : {di_after:.4f}  (Δ {improvement:+.4f})")
        logger.info(f"  Legal ≥0.8: {'✅ YES' if di_after >= 0.8 else '❌ Not yet, but improved'}")

        return {
            "algorithm":       "Reweighing",
            "paper":           "Kamiran & Calders (2012)",
            "df_mitigated":    df_rw,
            "weight_column":   "sample_weight",
            "original_rates":  orig_rates,
            "weighted_rates":  w_rates,
            "di_before":       di_before,
            "di_after":        di_after,
            "improvement":     improvement,
            "legal_compliant": di_after >= 0.8,
            "how_to_use": (
                "Pass the 'sample_weight' column to sklearn's fit() as "
                "the sample_weight parameter."
            ),
        }

    # ── Algorithm 2: Disparate Impact Remover ────────────────

    def apply_disparate_impact_remover(self, repair_level: float = 0.8) -> dict:
        """
        Repair numeric feature distributions so all groups share the same
        marginal distribution while preserving within-group rank.

        repair_level: 0.0 = no change, 1.0 = full repair.
        """
        logger.info("\nApplying Disparate Impact Remover (Feldman et al. 2015) ...")
        logger.info(f"Repair level: {repair_level}")

        df_rep     = self.df.copy()
        repaired_cols: list[str] = []

        numeric_cols = [
            c for c in df_rep.select_dtypes(include=[np.number]).columns
            if c not in [self.protected_col, self.target_col]
        ]

        if not numeric_cols:
            logger.info("  ⚠️ No numeric features to repair!")
            return {
                "algorithm":    "Disparate Impact Remover",
                "df_mitigated": df_rep,
                "error":        "No numeric features found.",
            }

        for col in numeric_cols:
            try:
                group_vals: dict[str, np.ndarray] = {}
                for g in self.groups:
                    mask = df_rep[self.protected_col] == g
                    vals = df_rep[mask][col].dropna().sort_values().values
                    if len(vals) > 0:
                        group_vals[str(g)] = vals

                if len(group_vals) < 2:
                    continue

                max_len = max(len(v) for v in group_vals.values())

                # Interpolate each group to a common length
                interpolated = [
                    np.interp(
                        np.linspace(0, 1, max_len),
                        np.linspace(0, 1, len(v)),
                        v,
                    )
                    for v in group_vals.values()
                ]
                median_dist = np.median(interpolated, axis=0)

                # Repair each group
                orig_dtype = df_rep[col].dtype   # preserve int/float dtype
                for g in self.groups:
                    mask     = df_rep[self.protected_col] == g
                    idx      = df_rep[mask].index
                    orig     = df_rep.loc[idx, col].values
                    ranks    = np.argsort(np.argsort(orig))
                    pos      = ranks / max(len(orig) - 1, 1)
                    repaired = np.interp(pos, np.linspace(0, 1, max_len), median_dist)
                    blended  = (1 - repair_level) * orig + repair_level * repaired
                    # Cast back to original dtype to avoid pandas 2.x LossySetitemError
                    df_rep.loc[idx, col] = blended.astype(orig_dtype)

                repaired_cols.append(col)

            except Exception as exc:
                logger.info(f"  ⚠️ Could not repair '{col}': {exc}")

        logger.info(f"  ✅ Repaired {len(repaired_cols)} features: {repaired_cols}")

        # DI before vs after
        def _di_from_df(frame):
            rates = {}
            for g in self.groups:
                m = frame[self.protected_col] == g
                rates[str(g)] = float(frame[m][self.target_col].mean())
            v = list(rates.values())
            return round(min(v) / max(v), 4) if max(v) > 0 else 0.0

        di_before  = _di_from_df(self.df)
        di_after   = _di_from_df(df_rep)
        improvement = round(di_after - di_before, 4)

        logger.info(f"  DI Before : {di_before:.4f}")
        logger.info(f"  DI After  : {di_after:.4f}  (Δ {improvement:+.4f})")

        return {
            "algorithm":    "Disparate Impact Remover",
            "paper":        "Feldman et al. (2015)",
            "df_mitigated": df_rep,
            "repair_level": repair_level,
            "repaired_cols": repaired_cols,
            "di_before":    di_before,
            "di_after":     di_after,
            "improvement":  improvement,
            "legal_compliant": di_after >= 0.8,
        }

    # ── Main entry point ─────────────────────────────────────

    def run_all_mitigations(self) -> dict:
        """
        Run both algorithms and recommend the better one.
        Called by /pre-audit/mitigate endpoint.
        """
        logger.info(f"\n{'='*55}")
        logger.info("NOIZE Pre-Model Mitigation")
        logger.info(f"Protected: {self.protected_col}  |  Target: {self.target_col}")
        logger.info(f"{'='*55}")

        rw_results  = self.apply_reweighing()
        dir_results = self.apply_disparate_impact_remover(repair_level=0.8)

        best = (
            "Reweighing"
            if rw_results["improvement"] >= dir_results["improvement"]
            else "Disparate Impact Remover"
        )

        logger.info(f"\n{'='*55}")
        logger.info(f"  Recommended: {best}")
        logger.info(f"{'='*55}\n")

        return {
            "status":        "success",
            "protected_col": self.protected_col,
            "reweighing":    rw_results,
            "dir":           dir_results,
            "recommended":   best,
        }
