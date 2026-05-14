# ============================================================
# NOIZE — pre_audit/data_analyzer.py
# PURPOSE: Step 1 of Pre-Model Audit.
#          Load, profile, and score any dataset.
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from shared.data_loader import load_dataset, binarize_target


class DataAnalyzer:
    """
    Profiles a dataset before bias detection.

    Usage
    -----
    analyzer = DataAnalyzer("/path/to/file.csv")
    result   = analyzer.run_full_analysis()
    """

    PROTECTED_KEYWORDS = [
        "gender", "sex", "race", "ethnicity", "age",
        "religion", "nationality", "disability", "marital",
        "color", "origin", "caste", "tribe",
    ]

    TARGET_KEYWORDS = [
        "target", "label", "outcome", "result", "income",
        "default", "status", "loan", "approve", "hired",
        "decision", "output", "recidivism", "prediction",
    ]

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.df: pd.DataFrame | None = None
        self._config: dict = {}

    # ── Loading ──────────────────────────────────────────────

    def load_data(self) -> bool:
        """Load dataset using the shared loader. Returns True on success."""
        try:
            self.df, self._config = load_dataset(self.filepath)
            print(f"✅ Loaded: {self.filename}  "
                  f"({self.df.shape[0]:,} rows × {self.df.shape[1]} cols)")
            return True
        except (FileNotFoundError, ValueError) as exc:
            print(f"❌ {exc}")
            return False

    # ── Basic info ───────────────────────────────────────────

    def get_basic_info(self) -> dict:
        """Return shape, types, missing values, duplicates."""
        if self.df is None:
            return {"error": "Call load_data() first."}

        missing      = self.df.isnull().sum()
        missing      = missing[missing > 0]
        missing_pct  = (missing / len(self.df) * 100).round(2)
        duplicates   = int(self.df.duplicated().sum())
        memory_mb    = round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2)

        return {
            "filename":       self.filename,
            "total_rows":     int(len(self.df)),
            "total_columns":  int(len(self.df.columns)),
            "column_names":   list(self.df.columns),
            "missing_values": missing.to_dict(),
            "missing_pct":    missing_pct.to_dict(),
            "duplicate_rows": duplicates,
            "memory_mb":      memory_mb,
            "column_types":   self.df.dtypes.astype(str).to_dict(),
        }

    # ── Column analysis ──────────────────────────────────────

    def get_column_analysis(self) -> dict:
        """Per-column statistics (numeric + categorical)."""
        if self.df is None:
            return {"error": "Call load_data() first."}

        analysis: dict = {}
        for col in self.df.columns:
            col_data = self.df[col].dropna()

            if pd.api.types.is_numeric_dtype(self.df[col]):
                analysis[col] = {
                    "type":     "numeric",
                    "mean":     round(float(col_data.mean()), 3),
                    "median":   round(float(col_data.median()), 3),
                    "std":      round(float(col_data.std()), 3),
                    "min":      round(float(col_data.min()), 3),
                    "max":      round(float(col_data.max()), 3),
                    "skewness": round(float(col_data.skew()), 3),
                    "missing":  int(self.df[col].isnull().sum()),
                }
            else:
                vc   = col_data.value_counts()
                vpct = (col_data.value_counts(normalize=True) * 100).round(2)
                analysis[col] = {
                    "type":             "categorical",
                    "unique_count":     int(col_data.nunique()),
                    "top_5_values":     vc.head(5).to_dict(),
                    "distribution_pct": vpct.head(10).to_dict(),
                    "missing":          int(self.df[col].isnull().sum()),
                }

        return analysis

    # ── Protected attributes ─────────────────────────────────

    def detect_protected_attributes(self) -> list[dict]:
        """Heuristic scan for protected-attribute columns."""
        if self.df is None:
            return []

        found: list[dict] = []
        for col in self.df.columns:
            col_lower = col.lower()
            for kw in self.PROTECTED_KEYWORDS:
                if kw in col_lower:
                    vd = (
                        self.df[col].value_counts(normalize=True) * 100
                    ).round(2).to_dict()
                    found.append({
                        "column":          col,
                        "matched_keyword": kw,
                        "unique_values":   int(self.df[col].nunique()),
                        "sample_values":   list(self.df[col].dropna().unique()[:5]),
                        "distribution":    vd,
                        "is_imbalanced":   (max(vd.values()) > 70 if vd else False),
                    })
                    break   # one match per column is enough

        return found

    # ── Target detection ─────────────────────────────────────

    def detect_target_variable(self) -> list[dict]:
        """Score each column as a target-variable candidate."""
        if self.df is None:
            return []

        candidates: list[dict] = []
        for col in self.df.columns:
            score, reasons = 0, []
            col_lower = col.lower()

            for kw in self.TARGET_KEYWORDS:
                if kw in col_lower:
                    score += 3
                    reasons.append(f"name contains '{kw}'")
                    break

            if self.df[col].nunique() == 2:
                score += 2
                reasons.append("binary column (2 values)")

            if col == self.df.columns[-1]:
                score += 1
                reasons.append("last column in dataset")

            if score > 0:
                vd = (
                    self.df[col].value_counts(normalize=True) * 100
                ).round(2).to_dict()
                candidates.append({
                    "column":          col,
                    "likelihood_score": score,
                    "reasons":         reasons,
                    "unique_values":   int(self.df[col].nunique()),
                    "distribution":    vd,
                })

        candidates.sort(key=lambda x: x["likelihood_score"], reverse=True)
        return candidates

    # ── Quality score ────────────────────────────────────────

    def calculate_data_quality_score(self) -> dict:
        """
        0-100 quality score.
        Deductions: missing values, duplicates, imbalanced groups.
        """
        if self.df is None:
            return {"error": "Call load_data() first."}

        score  = 100
        issues = []

        # Missing values
        total_cells   = self.df.shape[0] * self.df.shape[1]
        missing_cells = int(self.df.isnull().sum().sum())
        missing_pct   = missing_cells / total_cells * 100
        ded_missing   = min(missing_pct * 2, 30)
        score        -= ded_missing
        if missing_pct > 0:
            issues.append(
                f"Missing data: {missing_pct:.1f}% of cells (-{ded_missing:.0f} pts)"
            )

        # Duplicates
        dup_count = int(self.df.duplicated().sum())
        dup_pct   = dup_count / len(self.df) * 100
        ded_dup   = min(dup_pct * 2, 20)
        score    -= ded_dup
        if dup_count > 0:
            issues.append(
                f"Duplicates: {dup_count} rows ({dup_pct:.1f}%) (-{ded_dup:.0f} pts)"
            )

        # Imbalanced protected attributes
        protected        = self.detect_protected_attributes()
        imbal_count      = sum(1 for p in protected if p.get("is_imbalanced"))
        ded_imbal        = min(imbal_count * 10, 20)
        score           -= ded_imbal
        if imbal_count > 0:
            issues.append(
                f"Imbalanced groups: {imbal_count} protected attribute(s) (-{ded_imbal} pts)"
            )

        score = max(0, round(score))
        if score >= 80:
            label, emoji = "GOOD",    "🟢"
        elif score >= 60:
            label, emoji = "FAIR",    "🟡"
        else:
            label, emoji = "POOR",    "🔴"

        return {
            "score":   score,
            "label":   label,
            "emoji":   emoji,
            "issues":  issues,
            "details": {
                "missing_pct":      round(missing_pct, 2),
                "duplicate_pct":    round(dup_pct, 2),
                "imbalanced_attrs": imbal_count,
            },
        }

    # ── Main entry point ─────────────────────────────────────

    def run_full_analysis(self) -> dict:
        """
        Load + profile the dataset in one call.
        Called by the FastAPI /pre-audit/analyze endpoint.
        """
        print(f"\n{'='*55}")
        print(f"NOIZE Pre-Audit — {self.filename}")
        print(f"{'='*55}")

        if not self.load_data():
            return {"status": "error", "message": "Failed to load data."}

        basic_info   = self.get_basic_info()
        col_analysis = self.get_column_analysis()
        protected    = self.detect_protected_attributes()
        targets      = self.detect_target_variable()
        quality      = self.calculate_data_quality_score()

        print(f"  ✅ Quality: {quality['emoji']} {quality['score']}/100  "
              f"| Protected attrs: {len(protected)}  "
              f"| Target candidates: {len(targets)}")

        return {
            "status":               "success",
            "basic_info":           basic_info,
            "column_analysis":      col_analysis,
            "protected_attributes": protected,
            "target_candidates":    targets,
            "data_quality":         quality,
            # Expose the default columns inferred by the loader
            "suggested_protected":  self._config.get("protected"),
            "suggested_target":     self._config.get("target"),
        }
