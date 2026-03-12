"""
AuraOptima - PDF Report Generator
Generates professional batch analysis PDF reports using fpdf2.
"""

import json
import os
from datetime import datetime
from fpdf import FPDF


def _safe_text(text):
    """Remove non-latin1 chars for Helvetica compatibility."""
    text = str(text)
    text = text.replace("\u2705", "[OK]")
    text = text.replace("\u26a0\ufe0f", "[!]")
    text = text.replace("\u26a0", "[!]")
    return text.encode('latin-1', errors='replace').decode('latin-1')


class AuraOptimaPDF(FPDF):
    """Custom PDF class with AuraOptima branding."""

    def header(self):
        self.set_fill_color(0, 150, 200)
        self.rect(0, 0, 210, 3, 'F')
        self.set_fill_color(245, 248, 252)
        self.rect(0, 3, 210, 22, 'F')
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 120, 180)
        self.set_y(7)
        self.cell(0, 8, "  AURAOPTIMA", ln=False)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 110, 120)
        self.cell(0, 8, "Golden Signature Intelligence", ln=True, align="R")
        self.set_y(28)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        ts = datetime.now().strftime('%Y-%m-%d %H:%M')
        self.cell(0, 10, f"AuraOptima Report  |  Generated {ts}  |  Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(0, 120, 180)
        self.cell(0, 8, f"  {_safe_text(title)}", ln=True)
        self.set_draw_color(0, 150, 200)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def kpi_row(self, items):
        box_w = (190 - (len(items) - 1) * 4) / len(items)
        start_x = 10
        y = self.get_y()
        max_h = 20
        for i, (label, value, color) in enumerate(items):
            x = start_x + i * (box_w + 4)
            self.set_fill_color(245, 248, 252)
            self.set_draw_color(200, 210, 220)
            self.rect(x, y, box_w, max_h, 'DF')
            self.set_xy(x + 2, y + 2)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(100, 110, 120)
            self.cell(box_w - 4, 4, label.upper(), align="L")
            self.set_xy(x + 2, y + 8)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*color)
            self.cell(box_w - 4, 8, str(value), align="L")
        self.set_y(y + max_h + 4)

    def status_badge(self, status):
        color_map = {
            "HEALTHY": (34, 160, 75),
            "WARNING": (200, 140, 0),
            "CRITICAL": (220, 50, 60),
        }
        color = color_map.get(status, (100, 100, 100))
        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9)
        badge_w = self.get_string_width(f"  {status}  ") + 4
        self.cell(badge_w, 7, f"  {status}  ", fill=True, ln=False)
        self.set_text_color(0, 0, 0)

    def data_table(self, headers, rows, col_widths=None):
        if not col_widths:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(13, 17, 23)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self.set_fill_color(255, 255, 255)
            else:
                self.set_fill_color(245, 248, 252)
            for i, cell_data in enumerate(row):
                text = _safe_text(cell_data)
                color = (0, 0, 0)
                if text in ("CRITICAL", "red"):
                    color = (220, 50, 60)
                elif text in ("WARNING", "orange"):
                    color = (200, 140, 0)
                elif text in ("OK", "HEALTHY", "green"):
                    color = (34, 160, 75)
                self.set_text_color(*color)
                self.cell(col_widths[i], 6, text, border=1, fill=True, align="C")
            self.ln()
        self.set_text_color(0, 0, 0)


def generate_batch_report(report, batch_data=None, mode="balanced"):
    """Generate a PDF report from a deviation analysis report dict. Returns PDF bytes."""
    pdf = AuraOptimaPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    batch_id = report["batch_id"]
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Batch Analysis Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 110, 120)
    pdf.cell(0, 6, f"Batch ID: {batch_id}  |  Mode: {_safe_text(report.get('mode_label', mode))}  |  Generated: {timestamp}", ln=True)
    pdf.ln(2)

    # Status badge
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(30, 7, "Status: ", ln=False)
    pdf.status_badge(report["overall_status"])
    pdf.ln(10)

    # Executive Summary
    pdf.section_title("Executive Summary")
    summary = report["summary"]
    savings = report["savings_potential"]
    kpis = [
        ("Critical Params", str(summary["critical_params"]),
         (220, 50, 60) if summary["critical_params"] > 0 else (34, 160, 75)),
        ("Warning Params", str(summary["warning_params"]),
         (200, 140, 0) if summary["warning_params"] > 0 else (34, 160, 75)),
        ("Energy Savings", f"{savings['energy_savings_kWh']:.1f} kWh", (0, 120, 180)),
        ("Carbon Savings", f"{savings['carbon_savings_kg']:.1f} kg CO2", (34, 160, 75)),
    ]
    pdf.kpi_row(kpis)

    # Golden Signature Reference
    pdf.section_title("Golden Signature Reference")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 5, f"Source Batch: {report.get('golden_source', 'N/A')}  |  "
                    f"Mode: {_safe_text(report.get('mode_label', mode))}", ln=True)
    pdf.ln(3)

    # Parameter Deviation Table
    pdf.section_title("Parameter Deviation Analysis")
    headers = ["Parameter", "Batch Value", "Golden Value", "Deviation %", "Direction", "Severity"]
    col_widths = [40, 25, 25, 25, 25, 22]
    if pdf.get_y() > 230:
        pdf.add_page()
    rows = []
    for param, info in report["param_analysis"].items():
        rows.append([
            param.replace("_", " "),
            f"{info['batch_value']:.1f}",
            f"{info['golden_value']:.1f}",
            f"{info['deviation_pct']:+.1f}%",
            info["direction"],
            info["severity"],
        ])
    pdf.data_table(headers, rows, col_widths)

    # Outcome Comparison
    pdf.ln(4)
    if pdf.get_y() > 230:
        pdf.add_page()
    pdf.section_title("Outcome vs Golden Signature")
    headers = ["Metric", "Batch", "Golden", "Gap", "Status"]
    col_widths = [45, 30, 30, 30, 35]
    rows = []
    for metric, info in report["outcome_comparison"].items():
        rows.append([
            metric.replace("_", " "),
            f"{info['batch']:.2f}",
            f"{info['golden']:.2f}",
            f"{info['gap']:+.2f}",
            _safe_text(info["status"]),
        ])
    pdf.data_table(headers, rows, col_widths)

    # Top Recommendations
    pdf.ln(4)
    if pdf.get_y() > 230:
        pdf.add_page()
    pdf.section_title("Top Recommendations")
    if report["top_actions"]:
        for i, action in enumerate(report["top_actions"], 1):
            sev = action["severity"]
            if sev == "CRITICAL":
                pdf.set_text_color(220, 50, 60)
                icon = "[!]"
            else:
                pdf.set_text_color(200, 140, 0)
                icon = "[*]"
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, f"  {icon}  {i}. {action['parameter'].replace('_', ' ')} "
                           f"(Deviation: {action['deviation']:+.1f}%)", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(0, 5, f"       Action: {_safe_text(action['action'])}", ln=True)
            pdf.ln(2)
    else:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(34, 160, 75)
        pdf.cell(0, 6, "  All parameters are within the optimal range.", ln=True)

    # Savings Potential
    pdf.ln(2)
    if pdf.get_y() > 250:
        pdf.add_page()
    pdf.set_fill_color(235, 248, 255)
    pdf.set_draw_color(0, 150, 200)
    y = pdf.get_y()
    pdf.rect(10, y, 190, 18, 'DF')
    pdf.set_xy(14, y + 3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(0, 120, 180)
    pdf.cell(0, 5, "Savings Potential", ln=True)
    pdf.set_x(14)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 5, _safe_text(savings["message"]), ln=True)
    pdf.set_y(y + 22)

    # Full Batch Data
    if batch_data:
        pdf.add_page()
        pdf.section_title("Full Batch Data")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(40, 40, 40)
        headers = ["Parameter", "Value"]
        col_widths = [80, 80]
        rows = []
        for key, val in batch_data.items():
            rows.append([str(key).replace("_", " "), str(round(val, 3) if isinstance(val, float) else val)])
        pdf.data_table(headers, rows, col_widths)

    # Footer note
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 5, "This report was auto-generated by AuraOptima AI - Golden Signature Intelligence Platform.", ln=True)
    pdf.cell(0, 5, "For questions, contact your manufacturing optimization team.", ln=True)

    return pdf.output()
