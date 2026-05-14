# ============================================================
# NOIZE — pre_audit/proxy_detector.py
# PURPOSE: Detect features that are indirect proxies
#          for protected attributes (proxy discrimination).
# ============================================================

import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings("ignore")


class ProxyDetector:
    """
    Finds features that are NOT protected attributes themselves
    but are statistically correlated with them.

    Three detection methods
    -----------------------
    1. Pearson correlation   — numeric vs numeric
    2. Mutual Information    — any type vs any type
    3. Cramér's V            — categorical vs categorical
    """

    CORRELATION_THRESHOLD = 0.5
    MI_THRESHOLD          = 0.1
    CRAMERS_THRESHOLD     = 0.3

    def __init__(self, df: pd.DataFrame, protected_cols: list[str]):
        self.df             = df.copy()
        self.protected_cols = protected_cols
        self._df_encoded    = self._encode_data()

    # ── Encoding ─────────────────────────────────────────────

    def _encode_data(self) -> pd.DataFrame:
        """Label-encode all categorical columns for MI computation."""
        enc = self.df.copy()
        le  = LabelEncoder()
        for col in enc.columns:
            if enc[col].dtype == "object":
                enc[col] = le.fit_transform(
                    enc[col].fillna("MISSING").astype(str)
                )
        return enc

    # ── Cramér's V ───────────────────────────────────────────

    def _cramers_v(self, col1: str, col2: str) -> float:
        """Chi-squared based association strength for two categorical columns."""
        ct   = pd.crosstab(self.df[col1], self.df[col2])
        n    = ct.values.sum()
        r, c = ct.shape
        if min(r, c) <= 1:
            return 0.0

        expected = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / n
        chi2     = float(np.sum(((ct.values - expected) ** 2) / np.where(expected > 0, expected, 1)))
        v        = np.sqrt(chi2 / (n * (min(r, c) - 1))) if n > 0 else 0.0
        return round(float(v), 4)

    # ── Detection methods ────────────────────────────────────

    def detect_correlation_proxies(self, protected_col: str) -> list[dict]:
        """Numeric features with |Pearson r| ≥ CORRELATION_THRESHOLD."""
        le = LabelEncoder()
        if self.df[protected_col].dtype == "object":
            prot_enc = le.fit_transform(
                self.df[protected_col].fillna("MISSING").astype(str)
            )
        else:
            prot_enc = self.df[protected_col].fillna(0).values

        proxies: list[dict] = []
        numeric_cols = [
            c for c in self.df.select_dtypes(include=[np.number]).columns
            if c not in self.protected_cols
        ]

        for col in numeric_cols:
            col_vals = self.df[col].fillna(0).values
            try:
                corr     = float(np.corrcoef(prot_enc, col_vals)[0, 1])
                abs_corr = abs(corr)
                if abs_corr >= self.CORRELATION_THRESHOLD:
                    proxies.append({
                        "feature":         col,
                        "protected_attr":  protected_col,
                        "method":          "correlation",
                        "score":           round(abs_corr, 4),
                        "direction":       "positive" if corr > 0 else "negative",
                        "risk_level":      "HIGH" if abs_corr >= 0.7 else "MEDIUM",
                        "emoji":           "🔴" if abs_corr >= 0.7 else "🟡",
                        "interpretation":  (
                            f"{col} correlates {abs_corr:.2f} with "
                            f"{protected_col}. Strong proxy risk!"
                        ),
                    })
            except Exception:
                continue

        return sorted(proxies, key=lambda x: x["score"], reverse=True)

    def detect_mutual_info_proxies(self, protected_col: str) -> list[dict]:
        """All feature types via Mutual Information ≥ MI_THRESHOLD."""
        prot_enc    = self._df_encoded[protected_col].values
        feature_cols = [c for c in self._df_encoded.columns if c not in self.protected_cols]
        if not feature_cols:
            return []

        X = self._df_encoded[feature_cols].fillna(0).values
        try:
            mi_scores = mutual_info_classif(X, prot_enc, random_state=42)
        except Exception as exc:
            print(f"  ⚠️ MI calculation failed: {exc}")
            return []

        proxies: list[dict] = []
        for col, mi in zip(feature_cols, mi_scores):
            if mi >= self.MI_THRESHOLD:
                risk  = "HIGH" if mi >= 0.3 else "MEDIUM"
                emoji = "🔴"  if mi >= 0.3 else "🟡"
                proxies.append({
                    "feature":        col,
                    "protected_attr": protected_col,
                    "method":         "mutual_info",
                    "score":          round(float(mi), 4),
                    "risk_level":     risk,
                    "emoji":          emoji,
                    "interpretation": (
                        f"{col} shares {mi:.3f} mutual info with "
                        f"{protected_col}. Potential proxy!"
                    ),
                })

        return sorted(proxies, key=lambda x: x["score"], reverse=True)

    def detect_categorical_proxies(self, protected_col: str) -> list[dict]:
        """Categorical features with Cramér's V ≥ CRAMERS_THRESHOLD."""
        cat_cols = [
            c for c in self.df.select_dtypes(include=["object"]).columns
            if c not in self.protected_cols
        ]
        proxies: list[dict] = []

        for col in cat_cols:
            try:
                v = self._cramers_v(protected_col, col)
                if v >= self.CRAMERS_THRESHOLD:
                    proxies.append({
                        "feature":        col,
                        "protected_attr": protected_col,
                        "method":         "cramers_v",
                        "score":          v,
                        "risk_level":     "HIGH" if v >= 0.5 else "MEDIUM",
                        "emoji":          "🔴" if v >= 0.5 else "🟡",
                        "interpretation": (
                            f"{col} has Cramér's V = {v:.3f} with "
                            f"{protected_col}. Categorical proxy risk!"
                        ),
                    })
            except Exception:
                continue

        return sorted(proxies, key=lambda x: x["score"], reverse=True)

    # ── Main entry point ─────────────────────────────────────

    def run_full_proxy_detection(self) -> dict:
        """
        Run all three methods for every protected attribute.
        Called by /pre-audit/proxy endpoint.
        """
        print(f"\n{'='*55}")
        print(f"NOIZE Proxy Variable Detection")
        print(f"Protected attrs: {self.protected_cols}")
        print(f"{'='*55}")

        all_proxies:  dict[str, list[dict]] = {}
        all_removals: list[str]             = []
        all_warnings: list[str]             = []

        for protected in self.protected_cols:
            print(f"\nAnalysing: {protected}")
            print("─" * 40)

            proxies_found: list[dict] = []

            # Method 1: Correlation
            print("  [1/3] Correlation analysis ...")
            corr_proxies = self.detect_correlation_proxies(protected)
            proxies_found.extend(corr_proxies)
            print(f"        Found: {len(corr_proxies)} features")

            # Method 2: Mutual Information
            print("  [2/3] Mutual Information ...")
            mi_proxies  = self.detect_mutual_info_proxies(protected)
            existing    = {p["feature"] for p in proxies_found}
            proxies_found.extend(p for p in mi_proxies if p["feature"] not in existing)
            print(f"        Found: {len(mi_proxies)} features")

            # Method 3: Cramér's V
            print("  [3/3] Cramér's V ...")
            cat_proxies = self.detect_categorical_proxies(protected)
            existing    = {p["feature"] for p in proxies_found}
            proxies_found.extend(p for p in cat_proxies if p["feature"] not in existing)
            print(f"        Found: {len(cat_proxies)} features")

            proxies_found.sort(key=lambda x: x["score"], reverse=True)
            all_proxies[protected] = proxies_found

            high_risk = [p for p in proxies_found if p["risk_level"] == "HIGH"]
            if high_risk:
                for p in high_risk:
                    rec = (
                        f"REMOVE or transform '{p['feature']}' — "
                        f"HIGH proxy risk for '{protected}' "
                        f"(score: {p['score']:.3f})"
                    )
                    if rec not in all_removals:
                        all_removals.append(rec)
                all_warnings.append(
                    f"{len(high_risk)} HIGH-risk proxy feature(s) for {protected}!"
                )

        total = sum(len(v) for v in all_proxies.values())
        print(f"\n{'='*55}")
        print(f"Total proxy features found: {total}")
        print(f"{'='*55}\n")

        return {
            "status":          "success",
            "protected_cols":  self.protected_cols,
            "proxies":         all_proxies,
            "total_found":     total,
            "recommendations": all_removals,
            "warnings":        all_warnings,
            "safe_to_train":   total == 0,
        }
