# ============================================================
# NOIZE — post_audit/tradeoff_analyzer.py
# PURPOSE: Visualise the accuracy-fairness tradeoff curve
#          and find the optimal operating point.
# ============================================================

import logging
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
logger = logging.getLogger("noize.post_audit.tradeoff_analyzer")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

from shared.data_loader import binarize_target

COLORS = {
    "primary":   "#00D4FF",
    "secondary": "#7B2FFF",
    "danger":    "#FF4444",
    "success":   "#00FF88",
    "warning":   "#FFB800",
    "bg":        "#0A0A0F",
    "card":      "#1A1A2E",
}


class TradeoffAnalyzer:
    """
    Sweeps decision thresholds from 0.1 to 0.9 and records
    both performance (accuracy / F1) and fairness (DI, DP gap)
    at each threshold.

    Produces a tradeoff curve so you can pick the threshold
    that best balances fairness and performance.

    Usage
    -----
    ta     = TradeoffAnalyzer(df, protected_col, target_col, probs)
    result = ta.run_analysis()
    fig    = ta.plot_tradeoff()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        protected_col: str,
        target_col: str,
        predicted_probs: list | np.ndarray,
    ):
        self.df            = binarize_target(df.copy(), target_col)
        self.protected_col = protected_col
        self.target_col    = target_col
        self.probs         = np.array(predicted_probs, dtype=float)
        self.groups        = self.df[protected_col].dropna().unique().tolist()
        self._results: list[dict] = []

    # ── Helpers ──────────────────────────────────────────────

    def _metrics_at_threshold(self, t: float) -> dict:
        preds = (self.probs >= t).astype(int)
        y     = self.df[self.target_col].values

        # Overall accuracy + F1
        tp = int(((preds == 1) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())

        acc  = (tp + tn) / len(y) if len(y) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        # Fairness: DI + DP gap
        rates = []
        for g in self.groups:
            mask  = self.df[self.protected_col] == g
            rates.append(float(preds[mask.values].mean()))

        di     = round(min(rates) / max(rates), 4) if max(rates) > 0 else 0.0
        dp_gap = round(max(rates) - min(rates), 4)

        return {
            "threshold": round(t, 2),
            "accuracy":  round(acc, 4),
            "f1":        round(f1, 4),
            "di":        di,
            "dp_gap":    dp_gap,
            "legal":     di >= 0.8,
        }

    # ── Analysis ─────────────────────────────────────────────

    def run_analysis(self, n_thresholds: int = 33) -> dict:
        """
        Sweep thresholds and compute metrics at each point.

        Returns a dict with:
          - curve:           list of per-threshold metrics
          - optimal_point:   threshold with best F1 that still passes DI ≥ 0.8
          - default_point:   metrics at threshold = 0.5
          - fairness_cost:   accuracy drop from default to optimal-fair threshold
        """
        thresholds   = np.linspace(0.05, 0.95, n_thresholds)
        self._results = [self._metrics_at_threshold(t) for t in thresholds]

        default   = self._metrics_at_threshold(0.5)

        # Best fair point: highest F1 among legally compliant thresholds
        fair_pts  = [r for r in self._results if r["legal"]]
        if fair_pts:
            optimal = max(fair_pts, key=lambda x: x["f1"])
        else:
            # No threshold passes DI ≥ 0.8 — pick highest DI
            optimal = max(self._results, key=lambda x: x["di"])

        accuracy_cost = round(default["accuracy"] - optimal["accuracy"], 4)

        logger.info(
            "Tradeoff analysis | default@0.5: acc=%.4f DI=%.4f (%s) | "
            "optimal@%.2f: acc=%.4f DI=%.4f (%s) | accuracy_cost=%+.4f",
            default['accuracy'], default['di'], 'PASS' if default['legal'] else 'FAIL',
            optimal['threshold'], optimal['accuracy'], optimal['di'],
            'PASS' if optimal['legal'] else 'FAIL', accuracy_cost,
        )

        return {
            "status":        "success",
            "curve":         self._results,
            "default_point": default,
            "optimal_point": optimal,
            "accuracy_cost": accuracy_cost,
            "n_legal_thresholds": len(fair_pts),
        }

    # ── Visualisation ────────────────────────────────────────

    def plot_tradeoff(self) -> "go.Figure | None":
        """
        Plot accuracy vs fairness (DI) across all thresholds.
        Returns a Plotly Figure or None if Plotly is not installed.
        """
        if not _HAS_PLOTLY:
            logger.warning("plotly not installed — cannot generate tradeoff plot.")
            return None

        if not self._results:
            self.run_analysis()

        thresholds = [r["threshold"] for r in self._results]
        accuracies = [r["accuracy"]  for r in self._results]
        dis        = [r["di"]        for r in self._results]
        f1s        = [r["f1"]        for r in self._results]
        legal      = [r["legal"]     for r in self._results]

        fig = go.Figure()

        # Accuracy line
        fig.add_trace(go.Scatter(
            x=thresholds, y=accuracies,
            mode="lines+markers",
            name="Accuracy",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(color=[COLORS["success"] if l else COLORS["danger"] for l in legal], size=7),
        ))

        # DI line
        fig.add_trace(go.Scatter(
            x=thresholds, y=dis,
            mode="lines+markers",
            name="Disparate Impact",
            line=dict(color=COLORS["warning"], width=2),
            yaxis="y2",
        ))

        # DI threshold line
        fig.add_hline(
            y=0.8, line_dash="dash", line_color=COLORS["danger"],
            annotation_text="DI threshold (0.8)",
            annotation_font_color=COLORS["danger"],
            yref="y2",
        )

        fig.update_layout(
            title=dict(
                text=("Accuracy vs Fairness Tradeoff<br>"
                      "<sup>Green markers = legally compliant | Red = DI < 0.8</sup>"),
                font=dict(color="white"),
            ),
            xaxis =dict(title="Decision Threshold",   tickfont=dict(color="white"), titlefont=dict(color="white")),
            yaxis =dict(title="Accuracy",              tickfont=dict(color="white"), titlefont=dict(color="white"), range=[0, 1]),
            yaxis2=dict(title="Disparate Impact",      tickfont=dict(color="white"), titlefont=dict(color="white"),
                        overlaying="y", side="right",  range=[0, 1.1]),
            legend=dict(font=dict(color="white"), bgcolor=COLORS["card"]),
            plot_bgcolor  = COLORS["bg"],
            paper_bgcolor = COLORS["bg"],
            height=450,
        )
        return fig

    def plot_f1_vs_di(self) -> "go.Figure | None":
        """Scatter plot of F1 vs DI — the classic accuracy-fairness frontier."""
        if not _HAS_PLOTLY:
            return None

        if not self._results:
            self.run_analysis()

        colors = [COLORS["success"] if r["legal"] else COLORS["danger"] for r in self._results]
        fig = go.Figure(go.Scatter(
            x     = [r["di"] for r in self._results],
            y     = [r["f1"] for r in self._results],
            mode  = "markers+text",
            text  = [str(r["threshold"]) for r in self._results],
            textposition = "top center",
            textfont     = dict(size=9, color="white"),
            marker=dict(color=colors, size=8),
        ))
        fig.add_vline(x=0.8, line_dash="dash", line_color=COLORS["warning"],
                      annotation_text="Legal min DI",
                      annotation_font_color=COLORS["warning"])
        fig.update_layout(
            title=dict(text="F1 vs Disparate Impact<br><sup>Each point = a decision threshold</sup>",
                       font=dict(color="white")),
            xaxis=dict(title="Disparate Impact", tickfont=dict(color="white"), titlefont=dict(color="white")),
            yaxis=dict(title="F1 Score",         tickfont=dict(color="white"), titlefont=dict(color="white")),
            plot_bgcolor  = COLORS["bg"],
            paper_bgcolor = COLORS["bg"],
            height=450,
        )
        return fig
