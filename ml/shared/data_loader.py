# ============================================================
# NOIZE — shared/data_loader.py
# PURPOSE: Single place to load any of the 4 datasets.
#          Both pre_audit and post_audit import from here.
# ============================================================

import os
import logging
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("noize.data_loader")

# Maximum dataset size (rows × columns). Prevents OOM on huge uploads.
# Override via MAX_DATASET_CELLS env var.
MAX_DATASET_CELLS = int(os.getenv("MAX_DATASET_CELLS", str(50_000 * 200)))


# ── Column schema for datasets that have no header ──────────
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income",
]

# ── Default protected + target columns per dataset ──────────
DATASET_CONFIG = {
    "adult": {
        "protected": "sex",
        "target": "income",
        "sep": ",",
        "header": None,
        "columns": ADULT_COLUMNS,
        "skipinitialspace": True,
    },
    "compas": {
        "protected": "race",
        "target": "two_year_recid",
        "sep": ",",
        "header": 0,
        "columns": None,
    },
    "german": {
        "protected": "col_8",   # personal status / gender proxy
        "target": "col_20",     # credit risk (1=good, 2=bad)
        "sep": " ",
        "header": None,
        "columns": None,
    },
    "home_credit": {
        "protected": "Gender",
        "target": "Loan_Status",
        "sep": ",",
        "header": 0,
        "columns": None,
    },
}


def _detect_dataset_type(filename: str) -> str:
    """Guess dataset type from filename."""
    name = os.path.basename(filename).lower()
    if "adult" in name:
        return "adult"
    if "compas" in name:
        return "compas"
    if "german" in name:
        return "german"
    if "home" in name or "credit" in name or "loan" in name:
        return "home_credit"
    return "unknown"


def load_dataset(filepath: str) -> tuple[pd.DataFrame, dict]:
    """
    Load any of the 4 NOIZE datasets from a filepath.

    Returns
    -------
    df      : cleaned pandas DataFrame
    config  : dict with 'protected' and 'target' column names
              (plus other loader metadata)

    Raises
    ------
    FileNotFoundError if the file does not exist.
    ValueError        if the file cannot be parsed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    dtype = _detect_dataset_type(filepath)
    cfg   = DATASET_CONFIG.get(dtype, {
        "protected": None,
        "target": None,
        "sep": ",",
        "header": 0,
        "columns": None,
        "skipinitialspace": False,
    })

    read_kwargs = {
        "sep": cfg.get("sep", ","),
        "header": cfg.get("header", 0),
        "skipinitialspace": cfg.get("skipinitialspace", False),
    }
    if cfg.get("columns"):
        read_kwargs["names"] = cfg["columns"]

    try:
        df = pd.read_csv(filepath, **read_kwargs)
    except Exception as exc:
        raise ValueError(f"Could not parse {filepath}: {exc}") from exc

    # FIXED: guard against OOM from huge CSV files
    cells = df.shape[0] * df.shape[1]
    if cells > MAX_DATASET_CELLS:
        raise ValueError(
            f"Dataset too large: {df.shape[0]:,} rows × {df.shape[1]} columns = "
            f"{cells:,} cells. Limit is {MAX_DATASET_CELLS:,} cells. "
            "Reduce dataset size or increase MAX_DATASET_CELLS env var."
        )
    logger.info("Loaded %s: %d rows × %d cols (type=%s)", filepath, df.shape[0], df.shape[1], dtype)

    # ── Generic column names for headerless datasets ─────────
    if cfg.get("header") is None and not cfg.get("columns"):
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    # ── Strip whitespace everywhere ──────────────────────────
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    return df, cfg


def binarize_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Convert the target column to 0 / 1 in-place (on a copy).

    Handles:
      - Already 0/1
      - Yes/No  or  Y/N
      - >50K / <=50K  (Adult income)
      - 1 / 2         (German credit: 1=good → 1, 2=bad → 0)
      - Anything else: most-frequent value → 1
    """
    df = df.copy()
    col   = df[target_col]
    uniq  = col.dropna().unique().tolist()
    uniq_lower = {str(v).strip().lower() for v in uniq}

    if set(uniq) <= {0, 1}:
        pass  # already binary

    elif uniq_lower <= {"yes", "no"}:
        df[target_col] = col.str.strip().str.lower().map({"yes": 1, "no": 0})

    elif uniq_lower <= {"y", "n"}:
        df[target_col] = col.str.strip().str.lower().map({"y": 1, "n": 0})

    elif any(">50k" in str(v).lower() for v in uniq):
        df[target_col] = col.str.strip().str.lower().apply(
            lambda x: 1 if ">50k" in str(x) else 0
        )

    elif set(uniq) <= {1, 2}:
        df[target_col] = col.map({1: 1, 2: 0})

    else:
        most_common = col.value_counts().index[0]
        df[target_col] = col.apply(lambda x: 1 if x == most_common else 0)

    return df


def create_synthetic_home_credit(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic loan dataset with deliberate gender bias.
    Used when the real Home Credit CSV is unavailable.
    Male approval: 72 %  |  Female approval: 58 %
    """
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], n, p=[0.65, 0.35])
    approval_prob = np.where(gender == "Male", 0.72, 0.58)

    df = pd.DataFrame({
        "Loan_ID":            [f"LP{i:04d}" for i in range(n)],
        "Gender":             gender,
        "Married":            rng.choice(["Yes", "No"], n),
        "Dependents":         rng.choice(["0", "1", "2", "3+"], n),
        "Education":          rng.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22]),
        "Self_Employed":      rng.choice(["Yes", "No"], n, p=[0.14, 0.86]),
        "ApplicantIncome":    rng.integers(0, 15000, n),
        "CoapplicantIncome":  rng.integers(0, 5000, n),
        "LoanAmount":         rng.integers(10, 500, n),
        "Loan_Amount_Term":   rng.choice([360, 180, 120, 240], n),
        "Credit_History":     rng.choice([1.0, 0.0], n, p=[0.84, 0.16]),
        "Property_Area":      rng.choice(["Urban", "Rural", "Semiurban"], n),
        "Loan_Status":        np.where(rng.random(n) < approval_prob, "Y", "N"),
    })
    return df
