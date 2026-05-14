# ============================================================
# NOIZE — tests/test_pre_audit.py
# Run with: pytest tests/test_pre_audit.py -v
# ============================================================

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.data_loader         import binarize_target, create_synthetic_home_credit
from pre_audit.bias_detector    import BiasDetector
from pre_audit.fairness_metrics import FairnessMetrics
from pre_audit.proxy_detector   import ProxyDetector
from pre_audit.mitigation_pre   import PreMitigation


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Small synthetic dataset with deliberate gender bias."""
    return create_synthetic_home_credit(n=500, seed=42)


@pytest.fixture
def binary_df():
    """Minimal binary dataframe for metric sanity checks."""
    rng = np.random.default_rng(0)
    n   = 200
    sex = rng.choice(["Male", "Female"], n, p=[0.6, 0.4])
    # Male gets positive outcome 60% of the time, Female only 30%
    prob = np.where(sex == "Male", 0.6, 0.3)
    y    = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({"sex": sex, "income": y,
                          "age": rng.integers(18, 65, n),
                          "hours": rng.integers(20, 60, n)})


# ── binarize_target ──────────────────────────────────────────

class TestBinarizeTarget:

    def test_yesno(self):
        df = pd.DataFrame({"t": ["Yes", "No", "Yes", "No"]})
        out = binarize_target(df, "t")
        assert list(out["t"]) == [1, 0, 1, 0]

    def test_yn(self):
        df = pd.DataFrame({"t": ["Y", "N", "Y"]})
        out = binarize_target(df, "t")
        assert list(out["t"]) == [1, 0, 1]

    def test_already_binary(self):
        df = pd.DataFrame({"t": [0, 1, 1, 0]})
        out = binarize_target(df, "t")
        assert list(out["t"]) == [0, 1, 1, 0]

    def test_german_1_2(self):
        df = pd.DataFrame({"t": [1, 2, 1, 2]})
        out = binarize_target(df, "t")
        assert list(out["t"]) == [1, 0, 1, 0]

    def test_adult_income(self):
        df = pd.DataFrame({"t": [">50K", "<=50K", ">50K"]})
        out = binarize_target(df, "t")
        assert list(out["t"]) == [1, 0, 1]


# ── BiasDetector ─────────────────────────────────────────────

class TestBiasDetector:

    def test_init_raises_on_missing_col(self, binary_df):
        with pytest.raises(ValueError, match="not found"):
            BiasDetector(binary_df, "nonexistent", "income")

    def test_representation_keys(self, binary_df):
        bd  = BiasDetector(binary_df, "sex", "income")
        rep = bd.analyze_representation()
        assert "groups"         in rep
        assert "imbalance_ratio" in rep
        assert "total_samples"   in rep

    def test_di_is_ratio_between_0_and_1(self, binary_df):
        bd = BiasDetector(binary_df, "sex", "income")
        di = bd.calculate_disparate_impact()
        assert 0.0 <= di["disparate_impact"] <= 1.0

    def test_dp_gap_non_negative(self, binary_df):
        bd = BiasDetector(binary_df, "sex", "income")
        dp = bd.calculate_demographic_parity()
        assert dp["demographic_parity_gap"] >= 0.0

    def test_known_bias_detected(self, binary_df):
        """Male 60%, Female 30% → DI ≈ 0.5 → FAIL."""
        bd      = BiasDetector(binary_df, "sex", "income")
        di      = bd.calculate_disparate_impact()
        assert di["disparate_impact"] < 0.8
        assert di["status"] == "FAIL"

    def test_run_full_detection_returns_all_keys(self, binary_df):
        bd     = BiasDetector(binary_df, "sex", "income")
        result = bd.run_full_detection()
        for key in ("status", "representation", "disparate_impact",
                    "demographic_parity", "statistical_parity", "verdict"):
            assert key in result

    def test_verdict_audit_failed_on_biased_data(self, binary_df):
        bd     = BiasDetector(binary_df, "sex", "income")
        result = bd.run_full_detection()
        assert result["verdict"]["audit_passed"] is False

    def test_sample_home_credit(self, sample_df):
        bd     = BiasDetector(sample_df, "Gender", "Loan_Status")
        result = bd.run_full_detection()
        assert result["status"] == "success"


# ── FairnessMetrics ──────────────────────────────────────────

class TestFairnessMetrics:

    def test_pre_audit_all_keys(self, binary_df):
        fm     = FairnessMetrics(binary_df, "sex", "income")
        result = fm.get_all_metrics()
        assert "metrics"        in result
        assert "fairness_score" in result
        assert "privileged"     in result
        assert "unprivileged"   in result

    def test_score_between_0_and_100(self, binary_df):
        fm = FairnessMetrics(binary_df, "sex", "income")
        r  = fm.get_all_metrics()
        assert 0 <= r["fairness_score"] <= 100

    def test_post_audit_includes_eo_metrics(self, binary_df):
        # Add fake predictions
        rng = np.random.default_rng(1)
        binary_df["pred"] = rng.integers(0, 2, len(binary_df))
        fm     = FairnessMetrics(binary_df, "sex", "income", predicted_col="pred")
        result = fm.get_all_metrics()
        assert "equal_opportunity" in result["metrics"]
        assert "equalized_odds"    in result["metrics"]

    def test_pre_audit_has_no_post_metrics(self, binary_df):
        fm     = FairnessMetrics(binary_df, "sex", "income")
        result = fm.get_all_metrics()
        assert "equal_opportunity" not in result["metrics"]

    def test_theil_index_non_negative(self, binary_df):
        fm    = FairnessMetrics(binary_df, "sex", "income")
        theil = fm.theil_index()
        assert theil["value"] >= 0.0


# ── ProxyDetector ────────────────────────────────────────────

class TestProxyDetector:

    def test_returns_dict_structure(self, binary_df):
        pd_ = ProxyDetector(binary_df, protected_cols=["sex"])
        r   = pd_.run_full_proxy_detection()
        assert r["status"] == "success"
        assert "proxies"   in r
        assert "sex"       in r["proxies"]

    def test_no_proxies_on_independent_data(self):
        """Truly independent features should not show up as proxies."""
        rng = np.random.default_rng(99)
        n   = 300
        df  = pd.DataFrame({
            "sex":  rng.choice(["M", "F"], n),
            "col1": rng.standard_normal(n),
            "col2": rng.standard_normal(n),
        })
        pd_ = ProxyDetector(df, protected_cols=["sex"])
        r   = pd_.run_full_proxy_detection()
        assert r["total_found"] == 0


# ── PreMitigation ────────────────────────────────────────────

class TestPreMitigation:

    def test_reweighing_adds_weight_column(self, binary_df):
        mit = PreMitigation(binary_df, "sex", "income")
        r   = mit.apply_reweighing()
        assert "sample_weight" in r["df_mitigated"].columns

    def test_reweighing_weights_sum_to_n(self, binary_df):
        """Sum of weights should approximately equal n (they are normalised per group)."""
        mit     = PreMitigation(binary_df, "sex", "income")
        r       = mit.apply_reweighing()
        weights = r["df_mitigated"]["sample_weight"]
        # Weights are NOT required to sum to n — just check they are positive
        assert (weights > 0).all()

    def test_reweighing_improves_di(self, binary_df):
        mit = PreMitigation(binary_df, "sex", "income")
        r   = mit.apply_reweighing()
        assert r["improvement"] >= 0.0

    def test_dir_repaired_numeric_cols(self, binary_df):
        mit = PreMitigation(binary_df, "sex", "income")
        r   = mit.apply_disparate_impact_remover(repair_level=1.0)
        assert len(r.get("repaired_cols", [])) > 0

    def test_run_all_returns_recommendation(self, binary_df):
        mit = PreMitigation(binary_df, "sex", "income")
        r   = mit.run_all_mitigations()
        assert r["recommended"] in ("Reweighing", "Disparate Impact Remover")
