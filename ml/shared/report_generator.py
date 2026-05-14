# ============================================================
# NOIZE — shared/report_generator.py
# PURPOSE: Generate a professional PDF audit report
#          from pre- or post-audit results.
# ============================================================

import os
import json
import warnings
import tempfile
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, Image as RLImage,
    )
    from reportlab.lib            import colors
    from reportlab.lib.styles     import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units      import inch
    from reportlab.lib.pagesizes  import A4
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False


class ReportGenerator:
    """
    Generates a PDF fairness audit report.

    Usage
    -----
    rg = ReportGenerator(
        pre_audit_results  = ...,   # output of BiasDetector.run_full_detection()
        post_audit_results = ...,   # output of DecisionAuditor.run_full_audit() [optional]
        explanation        = "..."  # Gemini explanation string [optional]
        logo_path          = "/path/to/logo.png"  # optional
    )
    rg.generate("report.pdf")
    """

    def __init__(
        self,
        pre_audit_results:  dict,
        post_audit_results: dict | None  = None,
        explanation:        str  | None  = None,
        logo_path:          str  | None  = None,
    ):
        if not _HAS_REPORTLAB:
            raise ImportError(
                "reportlab is not installed. Run: pip install reportlab"
            )
        self.pre    = pre_audit_results
        self.post   = post_audit_results
        self.expl   = explanation
        self.logo   = logo_path
        self._styles = self._build_styles()

    # ── Styles ────────────────────────────────────────────────

    def _build_styles(self):
        ss = getSampleStyleSheet()
        ss.add(ParagraphStyle("NOIZETitle",   fontSize=22, textColor=colors.white,  spaceAfter=12))
        ss.add(ParagraphStyle("NOIZEHeading", fontSize=14, textColor=colors.cyan,   spaceAfter=8))
        ss.add(ParagraphStyle("NOIZEBody",    fontSize=11, textColor=colors.whitesmoke, spaceAfter=6))
        ss.add(ParagraphStyle("NOIZEInsight", fontSize=11, textColor=colors.orange, spaceAfter=6))
        ss.add(ParagraphStyle("NOIZEPass",    fontSize=11, textColor=colors.green,  spaceAfter=4))
        ss.add(ParagraphStyle("NOIZEFail",    fontSize=11, textColor=colors.red,    spaceAfter=4))
        return ss

    # ── Chart generation ─────────────────────────────────────

    def _make_metrics_chart(self, metrics: dict, title: str) -> str | None:
        """Create a matplotlib bar chart and save to a temp file."""
        if not _HAS_MPL:
            return None

        names, values, colours = [], [], []
        for m in metrics.values():
            if "error" not in m and isinstance(m.get("value"), float):
                names.append(m.get("metric", "?")[:20])
                values.append(m.get("value", 0))
                colours.append("#00FF88" if m.get("passed", True) else "#FF4444")

        if not names:
            return None

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(names, values, color=colours)
        ax.set_title(title, color="white")
        ax.set_facecolor("#0A0A0F")
        fig.patch.set_facecolor("#0A0A0F")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        plt.tight_layout()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=120, bbox_inches="tight", facecolor="#0A0A0F")
        plt.close(fig)
        return tmp.name

    # ── Table builder ─────────────────────────────────────────

    def _metrics_table(self, metrics: dict) -> Table:
        rows = [["Metric", "Value", "Status"]]
        for m in metrics.values():
            if "error" not in m:
                status = "✅ PASS" if m.get("passed", True) else "❌ FAIL"
                val    = f"{m['value']:.4f}" if isinstance(m.get("value"), float) else str(m.get("value", "N/A"))
                rows.append([m.get("metric", "?"), val, status])

        t = Table(rows, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.black),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("BACKGROUND",  (0, 1), (-1, -1), colors.Color(0.1, 0.1, 0.1)),
            ("TEXTCOLOR",   (0, 1), (-1, -1), colors.whitesmoke),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.Color(0.12, 0.12, 0.12), colors.Color(0.08, 0.08, 0.08)]),
        ]))
        return t

    # ── Build PDF ────────────────────────────────────────────

    def generate(self, output_path: str = "noize_audit_report.pdf") -> str:
        """
        Build and save the PDF report.

        Returns the output path.
        """
        doc      = SimpleDocTemplate(output_path, pagesize=A4,
                                     leftMargin=0.75*inch, rightMargin=0.75*inch,
                                     topMargin=0.75*inch, bottomMargin=0.75*inch)
        elements = []
        S        = self._styles

        # ── Logo ────────────────────────────────────────────────
        if self.logo and os.path.exists(self.logo):
            elements.append(RLImage(self.logo, width=1.5*inch, height=1.5*inch))
            elements.append(Spacer(1, 6))

        # ── Title ────────────────────────────────────────────────
        elements.append(Paragraph("NOIZE — AI Fairness Audit Report", S["NOIZETitle"]))
        elements.append(Spacer(1, 4))

        # ── Pre-audit ────────────────────────────────────────────
        elements.append(Paragraph("Pre-Model Audit", S["NOIZEHeading"]))
        verdict = self.pre.get("verdict", {})
        protected = self.pre.get("protected_column", "?")
        target    = self.pre.get("target_column", "?")
        elements.append(Paragraph(
            f"Protected attribute: <b>{protected}</b>  |  Target: <b>{target}</b>",
            S["NOIZEBody"]
        ))
        elements.append(Paragraph(verdict.get("verdict", "N/A"),
                                  S["NOIZEPass"] if verdict.get("audit_passed") else S["NOIZEFail"]))
        elements.append(Spacer(1, 8))

        # Bias metrics table
        ms = {
            "di": self.pre.get("disparate_impact", {}),
            "dp": self.pre.get("demographic_parity", {}),
            "sp": self.pre.get("statistical_parity", {}),
        }
        # Normalise keys for _metrics_table
        norm = {}
        for k, v in ms.items():
            if isinstance(v, dict) and "disparate_impact" in v:
                norm[k] = {"metric": "Disparate Impact",    "value": v["disparate_impact"], "passed": v.get("passes_threshold", True)}
            elif isinstance(v, dict) and "demographic_parity_gap" in v:
                norm[k] = {"metric": "Demographic Parity Gap", "value": v["demographic_parity_gap"], "passed": v.get("passes_threshold", True)}
            elif isinstance(v, dict) and "statistical_parity_diff" in v:
                norm[k] = {"metric": "Statistical Parity Diff", "value": v["statistical_parity_diff"], "passed": v.get("passes_threshold", True)}

        if norm:
            elements.append(self._metrics_table(norm))
            elements.append(Spacer(1, 8))

        # Chart
        chart = self._make_metrics_chart(norm, "Pre-Audit Metrics")
        if chart:
            elements.append(RLImage(chart, width=5*inch, height=2.5*inch))
            elements.append(Spacer(1, 8))

        # Recommendations
        recs = verdict.get("recommendations", [])
        if recs:
            elements.append(Paragraph("Recommendations:", S["NOIZEHeading"]))
            for r in recs:
                elements.append(Paragraph(f"• {r}", S["NOIZEInsight"]))
            elements.append(Spacer(1, 8))

        # ── Post-audit ────────────────────────────────────────────
        if self.post:
            elements.append(Paragraph("Post-Model Audit", S["NOIZEHeading"]))
            fm      = self.post.get("fairness_metrics", {})
            score   = fm.get("fairness_score", 0)
            emoji   = fm.get("score_emoji", "")
            elements.append(Paragraph(
                f"Fairness Score: {emoji} {score}%  ({fm.get('score_label', '')})",
                S["NOIZEPass"] if score >= 80 else S["NOIZEFail"]
            ))
            elements.append(Spacer(1, 6))

            post_metrics = fm.get("metrics", {})
            if post_metrics:
                elements.append(self._metrics_table(post_metrics))
                elements.append(Spacer(1, 8))

        # ── Gemini explanation ────────────────────────────────────
        if self.expl:
            elements.append(Paragraph("AI Explanation (Gemini)", S["NOIZEHeading"]))
            elements.append(Paragraph(self.expl, S["NOIZEBody"]))
            elements.append(Spacer(1, 8))

        doc.build(elements)
        print(f"✅ Report saved: {output_path}")
        return output_path
