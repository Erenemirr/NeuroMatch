from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import os

def generate_pdf_report(analysis_data: dict, output):
    """
    Generates a professional PDF report for the patient.
    'output' can be a file path string or a file-like object (e.g. BytesIO).
    """
    doc = SimpleDocTemplate(output, pagesize=letter)
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
        title = match.get('title', 'Unknown Trial')
        score = int(match.get('match_score', 0) * 100)
        summary = match.get('summary', '')
        
        story.append(Paragraph(f"<b>{title} (Match Score: {score}%)</b>", styles['Heading3']))
        story.append(Paragraph(summary, styles['Normal']))
        
        # Criteria Table
        data = [["Criterion", "Status", "Details"]]
        for c in match.get('criteria_status', []):
            status = "✅ Met" if c.get('is_met') else "❌ Not Met"
            data.append([c.get('name', 'N/A'), status, c.get('details', '')])
            
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
        for step in match.get('next_steps', []):
            story.append(Paragraph(f"• {step}", styles['Normal']))
        story.append(Spacer(1, 12))

    doc.build(story)
    return output
