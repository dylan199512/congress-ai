# -*- coding: utf-8 -*-
"""
pdf_export.py
-------------
Generates a formatted PDF research report from a Q&A result.

Public API
----------
    pdf_bytes = generate_qa_pdf(query, result)
    # then in Streamlit:
    st.download_button("Download PDF", pdf_bytes, "report.pdf", "application/pdf")
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

DARK       = colors.HexColor("#0d0f12")
GOLD       = colors.HexColor("#c8a96e")
BLUE       = colors.HexColor("#4a9eff")
GREEN      = colors.HexColor("#5fb88a")
LIGHT_GREY = colors.HexColor("#e8e4dc")
MID_GREY   = colors.HexColor("#6b7585")
SURFACE    = colors.HexColor("#14181e")
BORDER     = colors.HexColor("#252b33")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "Title",
            fontSize=22,
            textColor=GOLD,
            fontName="Helvetica-Bold",
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontSize=9,
            textColor=MID_GREY,
            fontName="Helvetica",
            spaceAfter=16,
            alignment=TA_LEFT,
        ),
        "section": ParagraphStyle(
            "Section",
            fontSize=11,
            textColor=GOLD,
            fontName="Helvetica-Bold",
            spaceBefore=14,
            spaceAfter=6,
        ),
        "question": ParagraphStyle(
            "Question",
            fontSize=13,
            textColor=LIGHT_GREY,
            fontName="Helvetica-Bold",
            spaceBefore=8,
            spaceAfter=8,
        ),
        "answer": ParagraphStyle(
            "Answer",
            fontSize=10,
            textColor=LIGHT_GREY,
            fontName="Helvetica",
            leading=16,
            spaceAfter=12,
        ),
        "body": ParagraphStyle(
            "Body",
            fontSize=9,
            textColor=LIGHT_GREY,
            fontName="Helvetica",
            leading=14,
            spaceAfter=4,
        ),
        "mono": ParagraphStyle(
            "Mono",
            fontSize=8,
            textColor=MID_GREY,
            fontName="Courier",
            leading=12,
            spaceAfter=2,
        ),
        "label": ParagraphStyle(
            "Label",
            fontSize=8,
            textColor=MID_GREY,
            fontName="Helvetica",
            spaceAfter=2,
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _divider():
    return HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8, spaceBefore=4)


def _topic_label(topic: str) -> str:
    topic = topic.upper()
    if topic == "HEALTHCARE":
        return f'<font color="#5fb88a">[{topic}]</font>'
    elif topic == "AI":
        return f'<font color="#4a9eff">[{topic}]</font>'
    return f"[{topic}]"


def _safe(val, fallback="") -> str:
    if val is None:
        return fallback
    s = str(val).strip()
    if s.lower() in ("nan", "none", ""):
        return fallback
    # Escape XML special chars for ReportLab
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_qa_pdf(query: str, result) -> bytes:
    """
    Generate a PDF research report for a single Q&A result.

    Args:
        query:  The user's question string.
        result: A QAResult dataclass from qa.py.

    Returns:
        PDF file contents as bytes, ready for st.download_button.
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
    )

    styles = _build_styles()
    story  = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Congress AI", styles["title"]))
    story.append(Paragraph(
        f"Healthcare &amp; AI Legislation Research Report &nbsp;·&nbsp; "
        f"{datetime.today().strftime('%B %d, %Y')}",
        styles["subtitle"],
    ))
    story.append(_divider())

    # ── Question ────────────────────────────────────────────────────────────
    story.append(Paragraph("Question", styles["section"]))
    story.append(Paragraph(_safe(query, "No question provided"), styles["question"]))

    # ── Answer ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Answer", styles["section"]))
    answer_text = _safe(result.answer, "No answer generated.")
    # Convert markdown-style bold (**text**) to ReportLab bold tags
    import re
    answer_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", answer_text)
    # Split into paragraphs on newlines
    for para in answer_text.split("\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, styles["answer"]))

    story.append(_divider())

    # ── Supporting Bills ────────────────────────────────────────────────────
    story.append(Paragraph("Supporting Bills", styles["section"]))

    if result.bill_hits is not None and not result.bill_hits.empty:
        for _, row in result.bill_hits.iterrows():
            bill_id = _safe(row.get("bill_id", ""))
            title   = _safe(row.get("title",   ""), "No title")
            summary = _safe(row.get("summary", ""), title)
            topics  = _safe(row.get("topics",  ""))
            sponsor = _safe(row.get("sponsor_bioguide_id", ""))

            topic_labels = " ".join(_topic_label(t.strip()) for t in topics.split(",") if t.strip())
            sponsor_line = f" &nbsp;·&nbsp; Sponsor: {sponsor}" if sponsor else ""

            block = [
                Paragraph(f"<b>{bill_id}</b> &nbsp; {topic_labels}{sponsor_line}", styles["mono"]),
                Paragraph(f"<b>{title}</b>", styles["body"]),
            ]
            if summary and summary != title:
                block.append(Paragraph(summary[:400] + ("…" if len(summary) > 400 else ""), styles["label"]))
            block.append(Spacer(1, 6))
            story.append(KeepTogether(block))
    else:
        story.append(Paragraph("No bills retrieved.", styles["label"]))

    story.append(_divider())

    # ── Member Stances ──────────────────────────────────────────────────────
    story.append(Paragraph("Member Stances", styles["section"]))

    if result.stance_hits is not None and not result.stance_hits.empty:
        for _, row in result.stance_hits.iterrows():
            bio   = _safe(row.get("bioguide_id", ""))
            date  = _safe(row.get("date",        ""))
            topic = _safe(row.get("topic",       ""))
            text  = _safe(row.get("text",        ""))
            url   = _safe(row.get("source_url",  ""))

            topic_label = _topic_label(topic) if topic else ""
            meta = f"<b>{bio}</b> &nbsp;·&nbsp; {date} &nbsp; {topic_label}"

            block = [
                Paragraph(meta, styles["mono"]),
                Paragraph(text[:350] + ("…" if len(text) > 350 else ""), styles["body"]),
            ]
            if url.startswith("http"):
                block.append(Paragraph(f'<link href="{url}"><font color="#4a9eff">{url[:80]}</font></link>', styles["label"]))
            block.append(Spacer(1, 6))
            story.append(KeepTogether(block))
    else:
        story.append(Paragraph("No stances retrieved.", styles["label"]))

    story.append(_divider())

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "Generated by Congress AI · Data: Congress.gov · GovInfo · legislators-current.csv",
        styles["label"],
    ))

    # ── Build ────────────────────────────────────────────────────────────────
    doc.build(story)
    return buffer.getvalue()
