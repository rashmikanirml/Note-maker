from __future__ import annotations

from html import escape
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def export_report(output_path: str | Path, title: str, summary: str, qa_pairs: list[tuple[str, str]], study_notes: list[str]) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(destination),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    body_style = ParagraphStyle("BodyTextWrapped", parent=styles["BodyText"], leading=14, spaceAfter=8)

    story = [Paragraph(escape(title), title_style), Spacer(1, 0.2 * inch)]
    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(Paragraph(escape(summary).replace("\n", "<br/>") , body_style))

    story.append(Paragraph("Question Answers", styles["Heading2"]))
    for question, answer in qa_pairs:
        story.append(Paragraph(escape(question), styles["Heading3"]))
        story.append(Paragraph(escape(answer).replace("\n", "<br/>") , body_style))

    story.append(Paragraph("Study Notes", styles["Heading2"]))
    for note in study_notes:
        story.append(Paragraph(escape(note), body_style))

    document.build(story)
    return destination
