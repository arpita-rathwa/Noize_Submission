# ============================================================
# NOIZE — post_audit/model_trainer.py
# PURPOSE: Train a baseline model on any of the 4 datasets
#          and return predictions for post-audit analysis.
# ============================================================

import logging
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
import joblib
warnings.filterwarnings("ignore")
logger = logging.getLogger("noize.post_audit_model_trainer")

from shared.data_loader import binarize_target


# ── Available model types ────────────────────────────────────
# Factory function — returns a FRESH model instance every call.
# The old module-level dict held shared stateful instances, meaning
# concurrent /train requests would corrupt each other's weights.
def _make_model(model_type: str):
    """Return a new unfitted sklearn estimator for the given type."""
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000, random_state=42)
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    if model_type == "gradient_boost":
        return GradientBoostingClassifier(n_estimators=100, random_state=42)
    raise ValueError(
        f"model_type must be one of ['logistic', 'random_forest', 'gradient_boost'], got '{model_type}'"
    )

# Keep a frozenset for fast membership checks (validation only)
_VALID_MODEL_TYPES = frozenset(["logistic", "random_forest", "gradient_boost"])


class ModelTrainer:
    """
    Trains a classification model and returns predictions
    alongside standard performance metrics.

    Supports:
    - Baseline training (no fairness constraints)
    - Weighted training (using Reweighing sample weights)

    Usage
    -----
    trainer = ModelTrainer(df, protected_col="sex", target_col="income")
    result  = trainer.train_and_evaluate(model_type="logistic")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        protected_col: str,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        if df.empty or len(df) == 0:
            raise ValueError(
                "ModelTrainer received an empty DataFrame. "
                "Provide at least a few rows to train on."
            )
        self.df            = binarize_target(df.copy(), target_col)
        self.protected_col = protected_col
        self.target_col    = target_col
        self.test_size     = test_size
        self.random_state  = random_state
        self.model         = None

        # Prepare encoded feature matrix
        self.X, self.y, self.encoders = self._prepare_features()

    # ── Feature engineering ──────────────────────────────────

    def _prepare_features(self):
        """
        Encode categoricals with LabelEncoder and return (X, y, encoders).
        Drops the target column; keeps protected col as a feature
        (bias auditors want to see what the model learns from it).
        """
        df  = self.df.copy()
        y   = df[self.target_col].values
        X   = df.drop(columns=[self.target_col])

        encoders: dict[str, LabelEncoder] = {}
        for col in X.select_dtypes(include=["object"]).columns:
            le             = LabelEncoder()
            X[col]         = le.fit_transform(X[col].fillna("MISSING").astype(str))
            encoders[col]  = le

        X = X.fillna(0)
        return X, y, encoders

    # ── Metrics helper ───────────────────────────────────────

    def _compute_metrics(self, y_true, y_pred, y_prob=None) -> dict:
        cm    = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        metrics = {
            "accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
            "precision":   round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":      round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1":          round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
        if y_prob is not None:
            try:
                metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
            except ValueError:
                pass
        return metrics

    # ── Training ─────────────────────────────────────────────

    def train_and_evaluate(
        self,
        model_type: str = "logistic",
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        """
        Train the chosen model, evaluate on a held-out test set,
        and return predictions for every row in the dataset.

        Parameters
        ----------
        model_type    : one of 'logistic', 'random_forest', 'gradient_boost'
        sample_weight : array of per-sample weights (from Reweighing).
                        When None, all samples are weighted equally.

        Returns
        -------
        dict with:
          model_type, train_metrics, test_metrics,
          predictions (for the full df), feature_importances
        """
        if model_type not in _VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {sorted(_VALID_MODEL_TYPES)}")

        logger.info(f"\n{'='*55}")
        logger.info(f"NOIZE Model Trainer — {model_type}")
        logger.info(f"Protected: {self.protected_col}  |  Target: {self.target_col}")
        logger.info(f"{'='*55}")

        X, y = self.X, self.y

        # Train / test split
        idx = np.arange(len(X))
        tr_idx, te_idx = train_test_split(
            idx, test_size=self.test_size,
            random_state=self.random_state, stratify=y,
        )

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        sw_tr      = sample_weight[tr_idx] if sample_weight is not None else None

        # Build pipeline with scaling
        base_model = _make_model(model_type)  # fresh instance — thread-safe
        scaler     = StandardScaler()
        X_tr_sc    = scaler.fit_transform(X_tr)
        X_te_sc    = scaler.transform(X_te)

        # Fit
        fit_kwargs = {}
        if sw_tr is not None:
            fit_kwargs["sample_weight"] = sw_tr
        base_model.fit(X_tr_sc, y_tr, **fit_kwargs)
        self.model  = base_model
        self.scaler = scaler

        # Predict on train + test
        y_tr_pred = base_model.predict(X_tr_sc)
        y_te_pred = base_model.predict(X_te_sc)

        y_tr_prob = y_te_prob = None
        if hasattr(base_model, "predict_proba"):
            y_tr_prob = base_model.predict_proba(X_tr_sc)[:, 1]
            y_te_prob = base_model.predict_proba(X_te_sc)[:, 1]

        train_metrics = self._compute_metrics(y_tr, y_tr_pred, y_tr_prob)
        test_metrics  = self._compute_metrics(y_te, y_te_pred, y_te_prob)

        logger.info(
            "Train accuracy=%.4f F1=%.4f | Test accuracy=%.4f F1=%.4f",
            train_metrics['accuracy'], train_metrics['f1'],
            test_metrics['accuracy'],  test_metrics['f1'],
        )

        # Predict on full dataset for fairness audit
        X_all_sc   = scaler.transform(X)
        all_preds  = base_model.predict(X_all_sc)

        # Feature importances
        feat_imp: dict[str, float] = {}
        if hasattr(base_model, "feature_importances_"):
            feat_imp = dict(zip(
                X.columns,
                [round(float(v), 4) for v in base_model.feature_importances_],
            ))
        elif hasattr(base_model, "coef_"):
            feat_imp = dict(zip(
                X.columns,
                [round(float(v), 4) for v in base_model.coef_[0]],
            ))

        return {
            "status":               "success",
            "model_type":           model_type,
            "protected_col":        self.protected_col,
            "target_col":           self.target_col,
            "train_size":           len(tr_idx),
            "test_size":            len(te_idx),
            "train_metrics":        train_metrics,
            "test_metrics":         test_metrics,
            "feature_importances":  dict(sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:10]),
            # Predictions array (same length as df) — used for post-audit
            "predictions":          all_preds.tolist(),
            "test_indices":         te_idx.tolist(),
        }

    def save_model(self, path: str):
        """Persist the trained model to disk."""
        if self.model is None:
            raise RuntimeError("No model trained yet — call train_and_evaluate() first.")
        joblib.dump({"model": self.model, "scaler": self.scaler, "encoders": self.encoders}, path)
        logger.info(f"✅ Model saved to {path}")

    def load_model(self, path: str):
        """Load a previously saved model from disk."""
        bundle        = joblib.load(path)
        self.model    = bundle["model"]
        self.scaler   = bundle["scaler"]
        self.encoders = bundle["encoders"]
        logger.info(f"✅ Model loaded from {path}")
