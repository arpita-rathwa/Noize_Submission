# ============================================================
# NOIZE Backend — services/binarize.py
# Single canonical binarize_target() used by both the backend
# routes/analyze.py and (optionally) the ML engine.
# FIXES the code duplication between _binarize() in analyze.py
# and binarize_target() in the ML engine's data_loader.py.
# ============================================================

import pandas as pd


def binarize_target(series: pd.Series) -> pd.Series:
    """
    Convert a target column to binary 0/1.
    Handles: already 0/1, Yes/No, Y/N, >50K/<=50K, 1/2 (German credit).
    Fallback: most-frequent value → 1.
    """
    uniq       = series.dropna().unique().tolist()
    uniq_lower = {str(v).strip().lower() for v in uniq}

    if set(uniq) <= {0, 1}:
        return series.astype(int)
    if uniq_lower <= {"yes", "no"}:
        return series.str.strip().str.lower().map({"yes": 1, "no": 0})
    if uniq_lower <= {"y", "n"}:
        return series.str.strip().str.lower().map({"y": 1, "n": 0})
    if any(">50k" in str(v).lower() for v in uniq):
        return series.str.strip().str.lower().apply(lambda x: 1 if ">50k" in str(x) else 0)
    if set(uniq) <= {1, 2}:
        return series.map({1: 1, 2: 0})
    most_common = series.value_counts().index[0]
    return series.apply(lambda x: 1 if x == most_common else 0)
