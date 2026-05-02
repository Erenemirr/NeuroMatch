from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import os

def generate_pdf_report(analysis_data: dict, output_path: str):
    """
    Generates a professional PDF report for the patient.
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("NeuroMatch Lite - Clinical Trial Eligibility Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Patient Summary
    story.append(Paragraph("<b>Patient Summary:</b>", styles['Heading2']))
    story.append(Paragraph(analysis_data.get('patient_summary', 'N/A'), styles['Normal']))
    story.append(Spacer(1, 12))

    # Trial Matches
    story.append(Paragraph("<b>Recommended Clinical Trials:</b>", styles['Heading2']))
    
    for match in analysis_data.get('matches', []):
        story.append(Paragraph(f"<b>{match.title} (Match Score: {int(match.match_score * 100)}%)</b>", styles['Heading3']))
        story.append(Paragraph(match.summary, styles['Normal']))
        
        # Criteria Table
        data = [["Criterion", "Status", "Details"]]
        for c in match.criteria_status:
            status = "✅ Met" if c.is_met else "❌ Not Met"
            data.append([c.name, status, c.details])
            
        t = Table(data, colWidths=[120, 80, 250])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        # Next Steps
        story.append(Paragraph("<b>Recommended Next Steps:</b>", styles['Normal']))
        for step in match.next_steps:
            story.append(Paragraph(f"• {step}", styles['Normal']))
        story.append(Spacer(1, 12))

    doc.build(story)
    return output_path
