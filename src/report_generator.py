from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import TableStyle
from reportlab.lib.units import inch
from datetime import datetime


def generate_report(output_path,
                    image_name,
                    cluster_count,
                    suspicious_count,
                    confidence,
                    execution_time,
                    method):

    confidence = int(float(confidence))

    doc = SimpleDocTemplate(output_path)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>COPY-MOVE FORGERY DETECTION REPORT</b>", styles['Title']))
    elements.append(Spacer(1, 0.4 * inch))

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    data = [
        ["Image Name", image_name],
        ["Detection Date", now],
        ["Clusters Detected", str(cluster_count)],
        ["Suspicious Points", str(suspicious_count)],
        ["Confidence (%)", str(confidence)],
        ["Execution Time (sec)", str(execution_time)],
        ["Method Used", method],
    ]

    table = Table(data, colWidths=[200, 250])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph("<b>Conclusion:</b>", styles['Heading2']))

    if confidence > 60:
        conclusion = "High Probability of Copy-Move Forgery Detected."
    elif confidence > 30:
        conclusion = "Moderate Probability of Forgery."
    else:
        conclusion = "Low Probability of Forgery."

    elements.append(Paragraph(conclusion, styles['Normal']))

    doc.build(elements)