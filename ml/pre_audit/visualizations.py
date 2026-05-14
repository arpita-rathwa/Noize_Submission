# ============================================================
# NOIZE — pre_audit/visualizations.py
# PURPOSE: All pre-audit bias visualizations.
#          Returns Plotly figures (JSON-serialisable).
# ============================================================

import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

from shared.data_loader import binarize_target

# ── NOIZE dark theme colours ─────────────────────────────────
COLORS = {
    "primary":   "#00D4FF",
    "secondary": "#7B2FFF",
    "danger":    "#FF4444",
    "success":   "#00FF88",
    "warning":   "#FFB800",
    "neutral":   "#888888",
    "bg":        "#0A0A0F",
    "card":      "#1A1A2E",
}
_LAYOUT = dict(
    plot_bgcolor  = COLORS["bg"],
    paper_bgcolor = COLORS["bg"],
    font          = dict(color="white"),
)
_AXIS = dict(tickfont=dict(color="white"), titlefont=dict(color="white"))


class BiasVisualizer:
    """
    Generates up to 6 Plotly figures for the pre-audit report.

    Usage
    -----
    viz   = BiasVisualizer(df, protected_col="sex", target_col="income")
    plots = viz.generate_all_plots(disparate_impact=0.63, demographic_parity=0.18)
    # each value in `plots` is a Plotly Figure or None
    """

    def __init__(self, df: pd.DataFrame, protected_col: str, target_col: str):
        self.df            = binarize_target(df.copy(), target_col)
        self.protected_col = protected_col
        self.target_col    = target_col
        self.groups        = self.df[protected_col].dropna().unique().tolist()

    # ── Helper ────────────────────────────────────────────────

    def _group_rates(self) -> dict[str, float]:
        rates: dict[str, float] = {}
        for g in self.groups:
            mask = self.df[self.protected_col] == g
            rates[str(g)] = round(float(self.df[mask][self.target_col].mean()) * 100, 2)
        return rates

    # ── Graph 1: Representation ──────────────────────────────

    def plot_group_representation(self) -> go.Figure:
        counts = self.df[self.protected_col].value_counts()
        total  = len(self.df)
        pcts   = (counts / total * 100).round(2)

        bar_colors = [
            COLORS["danger"]  if p < 20 else
            COLORS["warning"] if p < 35 else
            COLORS["success"]
            for p in pcts
        ]

        fig = go.Figure(go.Bar(
            x            = pcts.index.tolist(),
            y            = pcts.values.tolist(),
            marker_color = bar_colors,
            text         = [f"{p:.1f}%" for p in pcts],
            textposition = "outside",
            textfont     = dict(color="white"),
            customdata   = counts.values.tolist(),
            hovertemplate= "<b>%{x}</b><br>Percentage: %{y:.1f}%<br>Count: %{customdata}<extra></extra>",
        ))
        fig.add_hline(y=20, line_dash="dash", line_color=COLORS["danger"],
                      annotation_text="Min threshold (20%)",
                      annotation_font_color=COLORS["danger"])
        fig.update_layout(
            title=dict(
                text=(f"Group Representation: {self.protected_col}"
                      f"<br><sup>Red = underrepresented → sampling bias risk</sup>"),
                font=dict(color="white"),
            ),
            xaxis=dict(title="Group", **_AXIS),
            yaxis=dict(title="% of Dataset", range=[0, max(pcts) * 1.2], **_AXIS),
            showlegend=False, height=400, **_LAYOUT,
        )
        return fig

    # ── Graph 2: Outcome by group ────────────────────────────

    def plot_outcome_by_group(self) -> go.Figure:
        rates   = dict(sorted(self._group_rates().items(), key=lambda x: x[1]))
        groups  = list(rates.keys())
        vals    = list(rates.values())
        gap     = round(max(vals) - min(vals), 2)

        bar_colors = [COLORS["danger"]] + [COLORS["warning"]] * (len(vals) - 2) + [COLORS["success"]]
        if len(vals) == 2:
            bar_colors = [COLORS["danger"], COLORS["success"]]

        bias_label = "🔴 HIGH" if gap > 20 else "🟡 MEDIUM" if gap > 10 else "🟢 LOW"

        fig = go.Figure(go.Bar(
            x=groups, y=vals,
            marker_color=bar_colors,
            text=[f"{r:.1f}%" for r in vals],
            textposition="outside",
            textfont=dict(color="white"),
        ))
        fig.update_layout(
            title=dict(
                text=(f"Positive Outcome Rate by {self.protected_col}"
                      f"<br><sup>{bias_label} BIAS | Gap: {gap:.1f}%</sup>"),
                font=dict(color="white"),
            ),
            xaxis=dict(title="Group", **_AXIS),
            yaxis=dict(title="Positive Outcome Rate (%)", range=[0, max(vals) * 1.3], **_AXIS),
            showlegend=False, height=400, **_LAYOUT,
        )
        return fig

    # ── Graph 3: Fairness gauges ─────────────────────────────

    def plot_fairness_gauges(self, disparate_impact: float, demographic_parity: float) -> go.Figure:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=["Disparate Impact Ratio", "Demographic Parity Gap"],
        )

        def _gauge(value, ref, lo, hi, higher_better=True):
            ok_color  = COLORS["success"]
            bad_color = COLORS["danger"]
            good = (value >= ref) if higher_better else (value <= ref)
            return go.Indicator(
                mode="gauge+number+delta",
                value=round(value, 3),
                delta={"reference": ref,
                       "increasing": {"color": ok_color if higher_better else bad_color},
                       "decreasing": {"color": bad_color if higher_better else ok_color}},
                gauge={
                    "axis": {"range": [lo, hi], "tickfont": {"color": "white"}},
                    "bar":  {"color": ok_color if good else bad_color},
                    "steps": [
                        {"range": [lo, ref], "color": "#2D0000" if higher_better else "#002D00"},
                        {"range": [ref, hi], "color": "#002D00" if higher_better else "#2D0000"},
                    ],
                    "threshold": {"line": {"color": COLORS["warning"], "width": 4},
                                  "thickness": 0.75, "value": ref},
                },
                number={"font": {"color": ok_color if good else bad_color, "size": 40}},
            )

        fig.add_trace(_gauge(disparate_impact, 0.8, 0, 1, higher_better=True), row=1, col=1)
        fig.add_trace(_gauge(demographic_parity, 0.1, 0, 0.5, higher_better=False), row=1, col=2)
        fig.update_layout(
            title=dict(text="Fairness Metrics Dashboard<br><sup>Yellow line = threshold</sup>",
                       font=dict(color="white")),
            height=350, **_LAYOUT,
        )
        return fig

    # ── Graph 4: Distribution box plot ───────────────────────

    def plot_distribution(self, numeric_col: str | None = None) -> go.Figure | None:
        if numeric_col is None:
            num_cols = [
                c for c in self.df.select_dtypes(include=[np.number]).columns
                if c != self.target_col
            ]
            if not num_cols:
                return None
            numeric_col = num_cols[0]

        if numeric_col not in self.df.columns:
            return None

        seq = [COLORS["primary"], COLORS["secondary"], COLORS["warning"],
               COLORS["danger"], COLORS["success"]]
        fig = px.box(
            self.df, x=self.protected_col, y=numeric_col,
            color=self.protected_col, color_discrete_sequence=seq,
            points="outliers",
            title=(f"Distribution of {numeric_col} by {self.protected_col}"
                   f"<br><sup>Unequal spread = hidden structural bias</sup>"),
        )
        fig.update_layout(
            xaxis=dict(**_AXIS), yaxis=dict(**_AXIS),
            legend=dict(font=dict(color="white"), bgcolor=COLORS["card"]),
            height=400, **_LAYOUT,
        )
        return fig

    # ── Graph 5: Correlation heatmap ─────────────────────────

    def plot_correlation_heatmap(self) -> go.Figure | None:
        num_df = self.df.select_dtypes(include=[np.number])
        if len(num_df.columns) < 2:
            return None

        corr = num_df.corr().round(3)
        fig  = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale=[[0.0, COLORS["secondary"]], [0.5, "#111111"], [1.0, COLORS["primary"]]],
            zmid=0,
            text=corr.values, texttemplate="%{text:.2f}",
            textfont=dict(size=9, color="white"),
            colorbar=dict(tickfont=dict(color="white"),
                          title=dict(text="Correlation", font=dict(color="white"))),
        ))
        fig.update_layout(
            title=dict(
                text=("Feature Correlation Heatmap"
                      "<br><sup>Strong correlation with protected attrs = proxy bias risk</sup>"),
                font=dict(color="white"),
            ),
            xaxis=dict(tickfont=dict(color="white", size=9), tickangle=45),
            yaxis=dict(tickfont=dict(color="white", size=9)),
            height=500, **_LAYOUT,
        )
        return fig

    # ── Graph 6: Before vs After ─────────────────────────────

    def plot_before_after(self, before_metrics: dict, after_metrics: dict) -> go.Figure:
        metrics     = list(before_metrics.keys())
        before_vals = list(before_metrics.values())
        after_vals  = list(after_metrics.values())

        improvements = [
            f"{((a - b) / abs(b) * 100):+.1f}%" if b != 0 else "N/A"
            for b, a in zip(before_vals, after_vals)
        ]

        fig = go.Figure([
            go.Bar(name="Before", x=metrics, y=before_vals,
                   marker_color=COLORS["danger"],
                   text=[f"{v:.3f}" for v in before_vals], textposition="outside",
                   textfont=dict(color="white")),
            go.Bar(name="After",  x=metrics, y=after_vals,
                   marker_color=COLORS["success"],
                   text=[f"{v:.3f}" for v in after_vals], textposition="outside",
                   textfont=dict(color="white")),
        ])

        for i, (metric, imp) in enumerate(zip(metrics, improvements)):
            fig.add_annotation(
                x=metric, y=max(before_vals[i], after_vals[i]) * 1.15,
                text=f"Δ {imp}", showarrow=False,
                font=dict(color=COLORS["warning"], size=12),
            )

        fig.update_layout(
            title=dict(text="Before vs After Mitigation<br><sup>NOIZE reduced bias!</sup>",
                       font=dict(color="white")),
            barmode="group",
            xaxis=dict(**_AXIS), yaxis=dict(**_AXIS),
            legend=dict(font=dict(color="white"), bgcolor=COLORS["card"]),
            height=450, **_LAYOUT,
        )
        return fig

    # ── Main entry point ─────────────────────────────────────

    def generate_all_plots(
        self,
        disparate_impact: float | None   = None,
        demographic_parity: float | None = None,
    ) -> dict[str, go.Figure | None]:
        """
        Generate all five standard plots.
        Returns a dict of {name: Figure | None}.
        """
        print("Generating bias visualisations ...")
        plots: dict = {}

        plots["representation"] = self.plot_group_representation()
        plots["outcome"]        = self.plot_outcome_by_group()

        if disparate_impact is not None and demographic_parity is not None:
            plots["gauges"] = self.plot_fairness_gauges(disparate_impact, demographic_parity)
        else:
            plots["gauges"] = None

        plots["distribution"] = self.plot_distribution()
        plots["correlation"]  = self.plot_correlation_heatmap()

        done = sum(1 for v in plots.values() if v is not None)
        print(f"✅ {done}/5 graphs generated.")
        return plots
