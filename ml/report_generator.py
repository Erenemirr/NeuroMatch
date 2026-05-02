"""
NeuroMatch — Report Generator
Generates a professional PDF report from the full pipeline output.
Uses ReportLab to create a patient-friendly or clinical report.
"""

import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ── COLORS ──
DARK_BLUE = colors.HexColor("#1e3a5f")
ACCENT_BLUE = colors.HexColor("#3b82f6")
LIGHT_GRAY = colors.HexColor("#f1f5f9")
GREEN = colors.HexColor("#10b981")
ORANGE = colors.HexColor("#f59e0b")
RED = colors.HexColor("#ef4444")
TEXT_DARK = colors.HexColor("#1e293b")
TEXT_GRAY = colors.HexColor("#64748b")


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        fontSize=22,
        fontName="Helvetica-Bold",
        textColor=DARK_BLUE,
        spaceAfter=4,
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        fontSize=11,
        fontName="Helvetica",
        textColor=TEXT_GRAY,
        spaceAfter=2,
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontSize=13,
        fontName="Helvetica-Bold",
        textColor=DARK_BLUE,
        spaceBefore=14,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="BodyText2",
        fontSize=10,
        fontName="Helvetica",
        textColor=TEXT_DARK,
        spaceAfter=4,
        leading=14
    ))
    styles.add(ParagraphStyle(
        name="SmallGray",
        fontSize=8,
        fontName="Helvetica",
        textColor=TEXT_GRAY,
        spaceAfter=2
    ))
    styles.add(ParagraphStyle(
        name="BoldBody",
        fontSize=10,
        fontName="Helvetica-Bold",
        textColor=TEXT_DARK,
        spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name="Disclaimer",
        fontSize=8,
        fontName="Helvetica-Oblique",
        textColor=TEXT_GRAY,
        alignment=TA_CENTER,
        spaceAfter=4
    ))

    return styles


def hex_color(color) -> str:
    """Converts ReportLab color to plain hex string for inline use."""
    return color.hexval().replace("0x", "").replace("0X", "").upper()


def status_color(status: str):
    return {
        "likely_eligible": GREEN,
        "needs_more_info": ORANGE,
        "likely_ineligible": RED
    }.get(status, TEXT_GRAY)


def status_label(status: str):
    return {
        "likely_eligible": "✓ Likely Eligible",
        "needs_more_info": "? Needs More Info",
        "likely_ineligible": "✗ Likely Ineligible"
    }.get(status, status)


def generate_neuromatch_report(pipeline_result: dict, output_path: str,
                                audience: str = "patient") -> str:
    """
    Generates a PDF report from the full pipeline output.

    Args:
        pipeline_result: Output from run_neuromatch()
        output_path: File path for the PDF
        audience: "patient" | "neurologist" | "gp"

    Returns:
        output_path on success
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    styles = build_styles()
    story = []

    # ── HEADER ──
    story.append(Paragraph("🧠 NeuroMatch", styles["ReportTitle"]))
    story.append(Paragraph("AI-Powered Neurological Clinical Trial Matching", styles["ReportSubtitle"]))
    story.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        styles["SmallGray"]
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=DARK_BLUE, spaceAfter=12))

    # ── SECTION 1: DIAGNOSIS ──
    story.append(Paragraph("Preliminary Diagnosis", styles["SectionHeader"]))

    diagnosis = pipeline_result.get("diagnosis", {})
    diagnoses = diagnosis.get("preliminary_diagnoses", [])
    urgency = diagnosis.get("urgency_level", "unknown")
    tests = diagnosis.get("recommended_tests", [])
    summary = diagnosis.get("summary", "")

    if summary:
        story.append(Paragraph(summary, styles["BodyText2"]))
        story.append(Spacer(1, 6))

    # Diagnosis table
    if diagnoses:
        diag_data = [["Condition", "Confidence", "Matching Symptoms"]]
        for d in diagnoses[:3]:
            pct = f"{int(d['confidence'] * 100)}%"
            matching = ", ".join(d.get("matching_symptoms", [])[:3]) or "—"
            diag_data.append([d["disease"], pct, matching])

        diag_table = Table(diag_data, colWidths=[2.2 * inch, 1.0 * inch, 3.5 * inch])
        diag_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 8))

    # Urgency + tests
    urgency_colors = {"high": RED, "medium": ORANGE, "low": GREEN}
    urg_color = urgency_colors.get(urgency, TEXT_GRAY)
    story.append(Paragraph(
        f"<b>Urgency Level:</b> <font color='#{hex_color(urg_color)}'>{urgency.upper()}</font>",
        styles["BodyText2"]
    ))
    if tests:
        story.append(Paragraph(
            f"<b>Recommended Tests:</b> {', '.join(tests)}",
            styles["BodyText2"]
        ))

    # ── SECTION 2: OASIS-2 REFERENCE DATA ──
    oasis = pipeline_result.get("oasis_estimates", {})
    if oasis.get("available"):
        story.append(Spacer(1, 8))
        story.append(Paragraph("Clinical Reference Data (OASIS-2)", styles["SectionHeader"]))
        story.append(Paragraph(
            f"Based on <b>{oasis['sample_size']}</b> similar patients from the OASIS-2 longitudinal MRI study:",
            styles["BodyText2"]
        ))

        mmse = oasis["mmse"]
        cdr = oasis["cdr"]
        brain = oasis.get("brain_volume", {})

        oasis_data = [
            ["Metric", "Value", "Estimated Range"],
            ["MMSE Score", str(mmse["mean"]), f"{mmse['estimated_range'][0]} – {mmse['estimated_range'][1]}"],
            ["CDR Score", str(cdr["most_common"]), "Most common in similar patients"],
        ]
        if brain:
            oasis_data.append(["Brain Volume (nWBV)", str(brain["mean"]), f"{brain['range'][0]} – {brain['range'][1]}"])

        oasis_table = Table(oasis_data, colWidths=[2.2 * inch, 1.5 * inch, 3.0 * inch])
        oasis_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(oasis_table)
        story.append(Spacer(1, 4))
        story.append(Paragraph(oasis.get("interpretation", ""), styles["SmallGray"]))

    # ── SECTION 3: CLINICAL TRIAL MATCHES ──
    story.append(Spacer(1, 8))
    story.append(Paragraph("Clinical Trial Matches", styles["SectionHeader"]))

    gap_analysis = pipeline_result.get("gap_analysis", [])
    summary_data = pipeline_result.get("summary", {})

    summary_row = [
        [Paragraph(f"<b>{summary_data.get('likely_eligible', 0)}</b>", styles["BoldBody"]),
         Paragraph("Likely Eligible", styles["SmallGray"])],
        [Paragraph(f"<b>{summary_data.get('needs_more_info', 0)}</b>", styles["BoldBody"]),
         Paragraph("Needs More Info", styles["SmallGray"])],
        [Paragraph(f"<b>{summary_data.get('likely_ineligible', 0)}</b>", styles["BoldBody"]),
         Paragraph("Likely Ineligible", styles["SmallGray"])],
        [Paragraph(f"<b>{len(gap_analysis)}</b>", styles["BoldBody"]),
         Paragraph("Total Analyzed", styles["SmallGray"])],
    ]
    summary_table = Table(summary_row, colWidths=[0.6*inch, 1.5*inch, 0.6*inch, 1.5*inch,
                                                   0.6*inch, 1.5*inch, 0.6*inch, 1.5*inch])
    story.append(Spacer(1, 6))

    # Individual trial cards
    for i, gap in enumerate(gap_analysis[:5], 1):
        llm = gap.get("llm_analysis", {})
        trial_status = gap.get("overall_status", "needs_more_info")
        confidence = int(gap.get("confidence", 0) * 100)

        trial_elements = []

        # Trial header
        header_data = [[
            Paragraph(f"<b>{i}. {gap['trial_title'][:70]}</b>", styles["BoldBody"]),
            Paragraph(
                f"<font color='#{hex_color(status_color(trial_status))}'><b>{status_label(trial_status)}</b></font>",
                styles["BoldBody"]
            )
        ]]
        header_table = Table(header_data, colWidths=[4.5 * inch, 2.2 * inch])
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ]))
        trial_elements.append(header_table)

        # NCT ID + condition
        trial_elements.append(Paragraph(
            f"<b>NCT ID:</b> {gap['trial_id']}  |  <b>Condition:</b> {gap['matched_condition']}  |  <b>Confidence:</b> {confidence}%",
            styles["SmallGray"]
        ))

        # Summary
        patient_summary = llm.get("summary_patient", "")
        if audience == "neurologist":
            patient_summary = llm.get("summary_clinician", patient_summary)
        if patient_summary:
            trial_elements.append(Paragraph(patient_summary, styles["BodyText2"]))

        # Strengths
        strengths = llm.get("strengths", [])
        if strengths:
            trial_elements.append(Paragraph("<b>✓ Strengths:</b>", styles["SmallGray"]))
            for s in strengths[:3]:
                trial_elements.append(Paragraph(f"  • {s}", styles["SmallGray"]))

        # Gaps
        gaps = llm.get("gaps", [])
        if gaps:
            trial_elements.append(Paragraph("<b>⚠ Information Needed:</b>", styles["SmallGray"]))
            for g in gaps[:3]:
                trial_elements.append(Paragraph(
                    f"  • {g.get('criterion', '')} → {g.get('action', '')}",
                    styles["SmallGray"]
                ))

        trial_elements.append(Spacer(1, 8))
        story.append(KeepTogether(trial_elements))

    # ── FOOTER ──
    story.append(HRFlowable(width="100%", thickness=1, color=TEXT_GRAY, spaceAfter=8))
    story.append(Paragraph(
        "⚠️ This report is generated by NeuroMatch AI for informational purposes only. "
        "It does not constitute a medical diagnosis or recommendation. "
        "Please consult a qualified neurologist before making any medical decisions.",
        styles["Disclaimer"]
    ))
    story.append(Paragraph(
        "OASIS-2 data: Marcus et al., Journal of Cognitive Neuroscience, 2010. doi:10.1162/jocn.2009.21407",
        styles["Disclaimer"]
    ))

    doc.build(story)
    print(f"[REPORT] PDF saved to: {output_path}")
    return output_path


# ── TEST ──

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # Load a sample pipeline result if available
    try:
        with open("ml/pipeline_test_output.json", "r") as f:
            result = json.load(f)
        print("[TEST] Loaded pipeline_test_output.json")
    except FileNotFoundError:
        # Use minimal mock data
        result = {
            "patient_id": "test_001",
            "audience": "patient",
            "diagnosis": {
                "preliminary_diagnoses": [
                    {"disease": "Alzheimer's Disease", "confidence": 0.80,
                     "matching_symptoms": ["memory loss", "confusion"],
                     "missing_indicators": ["amyloid PET scan"]},
                ],
                "recommended_tests": ["MRI", "MMSE", "Amyloid PET"],
                "urgency_level": "medium",
                "summary": "70-year-old female presenting with memory loss and confusion.",
                "disclaimer": "Not a medical diagnosis."
            },
            "oasis_estimates": {
                "available": True,
                "sample_size": 40,
                "mmse": {"mean": 24.8, "std": 4.2, "estimated_range": [20.6, 29.0]},
                "cdr": {"most_common": 0.5},
                "brain_volume": {"mean": 0.7277, "range": [0.6569, 0.7773]},
                "interpretation": "Among 40 similar OASIS-2 patients, MMSE ranges 20-29."
            },
            "gap_analysis": [
                {
                    "trial_id": "NCT06875986",
                    "trial_title": "REXULTI Drug General Use-results Survey",
                    "matched_condition": "Alzheimer's Disease",
                    "overall_status": "likely_eligible",
                    "confidence": 0.9,
                    "rule_checks": [],
                    "llm_analysis": {
                        "strengths": ["Age within range", "Diagnosis matches"],
                        "gaps": [{"criterion": "exclusion criteria",
                                  "type": "missing_info",
                                  "explanation": "Exclusion criteria not specified",
                                  "action": "Consult trial coordinator"}],
                        "summary_patient": "You appear to be a good candidate for this trial.",
                        "summary_clinician": "Patient meets inclusion criteria."
                    }
                }
            ],
            "summary": {"likely_eligible": 1, "needs_more_info": 0, "likely_ineligible": 0}
        }
        print("[TEST] Using mock data")

    output = "neuromatch_report.pdf"
    generate_neuromatch_report(result, output, audience="patient")
    print(f"[TEST] Report generated: {output}")