import os
from fpdf import FPDF
from datetime import datetime
from typing import List, Dict, Union

class ReportPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        font_path = os.path.join(font_dir, "DejaVuSans.ttf")
        # register the one file for both normal and bold
        self.add_font(family="DejaVu", style="",  fname=font_path, uni=True)
        self.add_font(family="DejaVu", style="B", fname=font_path, uni=True)
        self.add_font(family="DejaVu", style="I", fname=font_path, uni=True)
        self.add_font(family="DejaVu", style="BI", fname=font_path, uni=True)

    def header(self):
        # Report title
        self.set_font("DejaVu", "B", 16)
        self.cell(0, 10, "Credit Risk Model Monitoring Report", ln=True, align="C")
        # Subtitle
        self.set_font("DejaVu", "", 12)
        self.cell(0, 8, "Drift and Performance Analysis", ln=True, align="C")
        # Divider line
        self.ln(4)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        # Position footer at 15mm from bottom
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.cell(
            0, 10,
            f"Generated on {datetime.now():%Y-%m-%d %H:%M} | Page {self.page_no()}",
            0, 0, "C"
        )

    def write_summary(self, summary: str, title: str = "Executive Summary"):
        self.set_font("DejaVu", "B", 12)
        self.cell(0, 8, title, ln=True)
        self.set_font("DejaVu", "", 11)
        self.multi_cell(0, 8, summary.strip())
        self.ln(6)

    def write_drift_table(self, data: List[Dict[str, Union[str, float]]]):
        # Table title
        self.set_font("DejaVu", "B", 12)
        self.cell(0, 8, "Feature Drift Results", ln=True)
        self.ln(2)

        col_widths = [50, 35, 60, 45]
        headers    = ["Feature", "Drift Score", "Test Used", "Status"]

        # Header row
        self.set_font("DejaVu", "B", 11)
        for header, width in zip(headers, col_widths):
            self.cell(width, 8, header, border=1, align="C")
        self.ln()

        # Data rows
        self.set_font("DejaVu", "", 10)
        for row in data:
            self.cell(col_widths[0], 8, str(row.get("feature", "")), border=1)
            score = row.get("score")
            if isinstance(score, (int, float)):
                disp = "<0.001" if score < 0.001 else f"{score:.3f}"
            else:
                disp = str(score)
            self.cell(col_widths[1], 8, disp, border=1, align="R")

            raw_test = row.get("test_used", "")
            test_display = raw_test.split("(")[0] if raw_test else raw_test
            self.cell(col_widths[2], 8, test_display, border=1)

            status = row.get("status", "")
            self.cell(col_widths[3], 8, str(status), border=1, align="C")
            self.ln()
        self.ln(6)

    def write_shap_table(self, data: List[Dict[str, float]]):
        # Title
        self.set_font("DejaVu", "B", 12)
        self.cell(0, 8, "SHAP Feature Comparison", ln=True)
        self.ln(2)
        # Columns
        col_widths = [50, 50, 50, 40]
        headers    = ["Feature", "Mean v1", "Mean v2", "Δ"]
        self.set_font("DejaVu", "B", 11)
        for header, width in zip(headers, col_widths):
            self.cell(width, 8, header, border=1, align="C")
        self.ln()
        # Rows
        self.set_font("DejaVu", "", 10)
        for r in data:
            self.cell(col_widths[0], 8, r["feature"], border=1)
            self.cell(col_widths[1], 8, f"{r['mean_v1']:.3f}", border=1, align="R")
            self.cell(col_widths[2], 8, f"{r['mean_v2']:.3f}", border=1, align="R")
            self.cell(col_widths[3], 8, f"{r['delta']:.3f}", border=1, align="R")
            self.ln()
        self.ln(6)

def generate_pdf_report(
    summary: str,
    drift_data: List[Dict[str, Union[str, float]]],
    shap_data: List[Dict] = None,
    filename: str = "model_monitoring_report.pdf",
    chatgpt_summary: str = None
) -> str:
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Executive summary
    pdf.write_summary(summary, title="Executive Summary")

    # Drift table
    pdf.write_drift_table(drift_data)

    # SHAP table
    if shap_data:
        pdf.write_shap_table(shap_data)

    # AI‑generated summary
    if chatgpt_summary:
        pdf.write_summary(chatgpt_summary, title="AI-Generated Summary")

    # Save
    output_dir = "artifacts"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    pdf.output(output_path)
    return output_path
