# ============================================================
# NOIZE — tests/test_post_audit.py
# Run with: pytest tests/test_post_audit.py -v
# ============================================================

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.data_loader           import create_synthetic_home_credit
from post_audit.model_trainer     import ModelTrainer
from post_audit.decision_auditor  import DecisionAuditor
from post_audit.mitigation_post   import PostMitigation
from post_audit.tradeoff_analyzer import TradeoffAnalyzer


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df():
    return create_synthetic_home_credit(n=600, seed=7)


@pytest.fixture(scope="module")
def trained(sample_df):
    """Train once and reuse across tests in this module."""
    trainer = ModelTrainer(sample_df, protected_col="Gender", target_col="Loan_Status")
    result  = trainer.train_and_evaluate(model_type="logistic")
    return trainer, result


@pytest.fixture(scope="module")
def probs(sample_df, trained):
    trainer, _ = trained
    X_sc = trainer.scaler.transform(trainer.X)
    if hasattr(trainer.model, "predict_proba"):
        return trainer.model.predict_proba(X_sc)[:, 1]
    return trainer.model.predict(X_sc).astype(float)


# ── ModelTrainer ─────────────────────────────────────────────

class TestModelTrainer:

    def test_result_has_required_keys(self, trained):
        _, result = trained
        for key in ("status", "model_type", "train_metrics", "test_metrics",
                    "feature_importances", "predictions"):
            assert key in result

    def test_predictions_length_matches_df(self, sample_df, trained):
        _, result = trained
        assert len(result["predictions"]) == len(sample_df)

    def test_accuracy_between_0_and_1(self, trained):
        _, result = trained
        assert 0 <= result["test_metrics"]["accuracy"] <= 1.0

    def test_f1_between_0_and_1(self, trained):
        _, result = trained
        assert 0 <= result["test_metrics"]["f1"] <= 1.0

    @pytest.mark.parametrize("model_type", ["logistic", "random_forest"])
    def test_multiple_model_types(self, sample_df, model_type):
        trainer = ModelTrainer(sample_df, protected_col="Gender", target_col="Loan_Status")
        result  = trainer.train_and_evaluate(model_type=model_type)
        assert result["status"] == "success"

    def test_invalid_model_type_raises(self, sample_df):
        trainer = ModelTrainer(sample_df, protected_col="Gender", target_col="Loan_Status")
        with pytest.raises(ValueError):
            trainer.train_and_evaluate(model_type="magic_model")


# ── DecisionAuditor ──────────────────────────────────────────

class TestDecisionAuditor:

    def test_init_raises_on_length_mismatch(self, sample_df):
        with pytest.raises(ValueError, match="length"):
            DecisionAuditor(sample_df, "Gender", "Loan_Status", predictions=[0, 1])

    def test_per_group_performance_returns_all_groups(self, sample_df, trained):
        _, result = trained
        auditor   = DecisionAuditor(
            sample_df, "Gender", "Loan_Status",
            predictions=result["predictions"]
        )
        perf = auditor.per_group_performance()
        assert "Male"   in perf
        assert "Female" in perf

    def test_per_group_accuracy_valid(self, sample_df, trained):
        _, result = trained
        auditor   = DecisionAuditor(
            sample_df, "Gender", "Loan_Status",
            predictions=result["predictions"]
        )
        for group, metrics in auditor.per_group_performance().items():
            assert 0 <= metrics["accuracy"] <= 1.0

    def test_full_audit_has_required_keys(self, sample_df, trained):
        _, result = trained
        auditor   = DecisionAuditor(
            sample_df, "Gender", "Loan_Status",
            predictions=result["predictions"]
        )
        r = auditor.run_full_audit()
        for key in ("status", "per_group_performance",
                    "fairness_metrics", "prediction_distribution"):
            assert key in r

    def test_fairness_metrics_has_post_audit_mode(self, sample_df, trained):
        _, result = trained
        auditor   = DecisionAuditor(
            sample_df, "Gender", "Loan_Status",
            predictions=result["predictions"]
        )
        r  = auditor.run_full_audit()
        fm = r["fairness_metrics"]
        assert fm.get("audit_mode") == "post-audit"
        assert "equal_opportunity" in fm["metrics"]


# ── PostMitigation ───────────────────────────────────────────

class TestPostMitigation:

    def test_threshold_optimisation_returns_thresholds(self, sample_df, probs):
        mit = PostMitigation(sample_df, "Gender", "Loan_Status", predicted_probs=probs)
        r   = mit.optimise_thresholds()
        assert "thresholds"  in r
        assert "Male"        in r["thresholds"]
        assert "Female"      in r["thresholds"]

    def test_threshold_values_between_0_and_1(self, sample_df, probs):
        mit = PostMitigation(sample_df, "Gender", "Loan_Status", predicted_probs=probs)
        r   = mit.optimise_thresholds()
        for t in r["thresholds"].values():
            assert 0.0 <= t <= 1.0

    def test_roc_improves_di(self, sample_df, probs):
        mit = PostMitigation(sample_df, "Gender", "Loan_Status", predicted_probs=probs)
        r   = mit.reject_option_classification(margin=0.2)
        assert r["improvement"] >= 0.0

    def test_run_all_returns_recommended(self, sample_df, probs):
        mit = PostMitigation(sample_df, "Gender", "Loan_Status", predicted_probs=probs)
        r   = mit.run_all_mitigations()
        assert r["recommended"] in (
            "Threshold Optimisation",
            "Reject Option Classification",
        )


# ── TradeoffAnalyzer ─────────────────────────────────────────

class TestTradeoffAnalyzer:

    def test_curve_has_correct_length(self, sample_df, probs):
        ta     = TradeoffAnalyzer(sample_df, "Gender", "Loan_Status", probs)
        result = ta.run_analysis(n_thresholds=11)
        assert len(result["curve"]) == 11

    def test_each_point_has_required_keys(self, sample_df, probs):
        ta     = TradeoffAnalyzer(sample_df, "Gender", "Loan_Status", probs)
        result = ta.run_analysis(n_thresholds=5)
        for pt in result["curve"]:
            for key in ("threshold", "accuracy", "f1", "di", "dp_gap", "legal"):
                assert key in pt

    def test_default_point_is_at_0_5(self, sample_df, probs):
        ta     = TradeoffAnalyzer(sample_df, "Gender", "Loan_Status", probs)
        result = ta.run_analysis()
        assert result["default_point"]["threshold"] == 0.5

    def test_optimal_point_passes_di_if_possible(self, sample_df, probs):
        ta     = TradeoffAnalyzer(sample_df, "Gender", "Loan_Status", probs)
        result = ta.run_analysis()
        opt    = result["optimal_point"]
        if result["n_legal_thresholds"] > 0:
            assert opt["legal"] is True

    def test_accuracy_cost_is_float(self, sample_df, probs):
        ta     = TradeoffAnalyzer(sample_df, "Gender", "Loan_Status", probs)
        result = ta.run_analysis()
        assert isinstance(result["accuracy_cost"], float)
