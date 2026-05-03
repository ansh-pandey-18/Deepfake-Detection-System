"""
PDF Evidence Report Generator for Deepfake Detection System.

Generates professional forensic-style PDF reports summarizing
detection results, confidence scores, and recommended actions.
"""

import datetime
from io import BytesIO
from fpdf import FPDF


def _risk_level(confidence, label):
    """Determine risk level from prediction label and confidence."""
    if label == "REAL":
        if confidence >= 0.85:
            return "Low Risk", "Content appears authentic with high confidence."
        elif confidence >= 0.6:
            return "Moderate Risk", "Content likely authentic but confidence is moderate."
        else:
            return "Uncertain", "Prediction leans authentic but confidence is low."
    else:
        if confidence <= 0.15:
            return "Critical Risk", "Content is very likely AI-generated."
        elif confidence <= 0.35:
            return "High Risk", "Content shows strong indicators of manipulation."
        else:
            return "Elevated Risk", "Content may be AI-generated. Verify further."


def _recommended_action(label, risk):
    """Return recommended actions based on prediction outcome."""
    if label == "REAL":
        return "No immediate action required. Content appears authentic."
    if "Critical" in risk or "High" in risk:
        return ("Do NOT forward this content. Report to the relevant platform "
                "and consider filing a complaint at https://cybercrime.gov.in/")
    return ("Exercise caution before sharing. Verify the original source "
            "and cross-check with trusted outlets.")


def generate_pdf_report(filename, model_name, label, confidence,
                        media_type="Image", category=None,
                        gradcam_available=False,
                        video_summary=None):
    """
    Generate a PDF evidence report.

    Args:
        filename: Name of the uploaded file.
        model_name: Model used for prediction.
        label: Prediction label ('REAL' or 'AI Generated').
        confidence: Float confidence score.
        media_type: 'Image' or 'Video'.
        category: Selected input category string.
        gradcam_available: Whether Grad-CAM was generated.
        video_summary: Dict with video-specific stats or None.

    Returns:
        BytesIO buffer containing the PDF bytes.
    """
    risk, risk_desc = _risk_level(confidence, label)
    action = _recommended_action(label, risk)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # --- Header ---
    pdf.set_fill_color(10, 14, 39)
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(90, 249, 251)
    pdf.set_y(10)
    pdf.cell(0, 10, "Deepfake Detection System", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(180, 190, 210)
    pdf.cell(0, 6, "Evidence Report  |  Media Authenticity Analysis", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)

    # --- Report metadata ---
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Report Details", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(90, 249, 251)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    details = [
        ("Report Generated", now),
        ("File Name", filename or "N/A"),
        ("Media Type", media_type),
        ("Input Category", category or "N/A"),
        ("Model Used", model_name or "N/A"),
    ]
    for lbl, val in details:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(55, 7, f"{lbl}:", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, str(val), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # --- Prediction result ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Prediction Result", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Verdict
    if label == "REAL":
        pdf.set_fill_color(0, 180, 90)
    else:
        pdf.set_fill_color(220, 50, 50)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(80, 10, f"  Verdict:  {label}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(40, 40, 40)
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    pred_details = [
        ("Confidence Score", f"{confidence:.2%}"),
        ("Risk Level", risk),
        ("Assessment", risk_desc),
    ]
    for lbl, val in pred_details:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(55, 7, f"{lbl}:", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, str(val), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)

    # --- Grad-CAM ---
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(55, 7, "Grad-CAM Analysis:", new_x="RIGHT")
    pdf.set_font("Helvetica", "", 10)
    gcam_text = "Generated and available" if gradcam_available else "Not generated for this scan"
    pdf.cell(0, 7, gcam_text, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # --- Video summary ---
    if video_summary:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Video Analysis Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 10)
        vid_items = [
            ("Total Frames Analyzed", str(video_summary.get('total_frames', 'N/A'))),
            ("Fake Frames", str(video_summary.get('fake_count', 'N/A'))),
            ("Real Frames", str(video_summary.get('real_count', 'N/A'))),
            ("Fake Percentage", f"{video_summary.get('fake_percentage', 0):.1f}%"),
            ("Average Confidence", f"{video_summary.get('weighted_confidence', 0):.2%}"),
        ]
        for lbl, val in vid_items:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(55, 7, f"{lbl}:", new_x="RIGHT")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 7, val, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

    # --- Recommended action ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Recommended Action", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, action)
    pdf.ln(6)

    # --- Footer ---
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, "This report was auto-generated by the Deepfake Detection System for Media Authenticity.",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Developed for academic research by Ansh, Aryan & Kanishk.",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # Output to buffer
    buf = BytesIO()
    buf.write(pdf.output())
    buf.seek(0)
    return buf
