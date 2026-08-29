"""ATS-safe output: plain text, DOCX, PDF.

Everything the renderer emits is deliberately boring — single column, no tables,
no text boxes, no images, standard headings, standard fonts, hyphen bullets.
That is what survives a parser.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, RGBColor

from .schema import ResumeDoc

SECTION_ORDER_HEADINGS = {
    "summary": "Professional Summary",
    "skills": "Technical Skills",
    "experience": "Professional Experience",
    "projects": "Projects",
    "education": "Education",
    "certifications": "Certifications",
}


def to_plain_text(doc: ResumeDoc) -> str:
    """Plain-text rendering with standard headings — also what we re-score against."""
    c = doc.contact
    lines: list[str] = [c.name or "", c.headline or ""]
    contact_bits = [c.location, c.phone, c.email, c.linkedin, c.github, c.portfolio]
    lines.append(" | ".join(b for b in contact_bits if b))
    lines.append("")

    if doc.summary:
        lines += [SECTION_ORDER_HEADINGS["summary"], doc.summary, ""]

    if doc.skills:
        lines.append(SECTION_ORDER_HEADINGS["skills"])
        for cat, vals in doc.skills.items():
            if vals:
                lines.append(f"{cat}: {', '.join(vals)}")
        lines.append("")

    if doc.experience:
        lines.append(SECTION_ORDER_HEADINGS["experience"])
        for exp in doc.experience:
            head = " | ".join(x for x in [exp.title, exp.company, exp.location] if x)
            lines.append(f"{head}{('  ' + exp.dates) if exp.dates else ''}")
            lines += [f"- {b.text}" for b in exp.bullets]
            lines.append("")

    if doc.projects:
        lines.append(SECTION_ORDER_HEADINGS["projects"])
        for proj in doc.projects:
            head = proj.name
            if proj.subtitle:
                head += f" | {proj.subtitle}"
            lines.append(head)
            if proj.tech:
                lines.append(f"Technologies: {', '.join(proj.tech)}")
            lines += [f"- {b.text}" for b in proj.bullets]
            lines.append("")

    if doc.education:
        lines.append(SECTION_ORDER_HEADINGS["education"])
        for ed in doc.education:
            lines.append(
                " | ".join(x for x in [ed.degree, ed.institution, ed.dates] if x)
            )
            if ed.details:
                lines.append(ed.details)
        lines.append("")

    if doc.certifications:
        lines.append(SECTION_ORDER_HEADINGS["certifications"])
        lines += [f"- {x}" for x in doc.certifications]
        lines.append("")

    for name, vals in doc.extras.items():
        if vals:
            lines.append(name)
            lines += [f"- {v}" for v in vals]
            lines.append("")

    return "\n".join(lines).strip()


# ------------------------------------------------------------------- DOCX
def _style(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    pf = normal.paragraph_format
    pf.space_after = Pt(2)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.05
    for section in document.sections:
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)
        section.left_margin = Inches(0.6)
        section.right_margin = Inches(0.6)


def _heading(document: Document, text: str) -> None:
    p = document.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text.upper())
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)
    # a single bottom border, the one flourish an ATS ignores harmlessly
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    pPr = p._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:color"), "999999")
    borders.append(bottom)
    pPr.append(borders)


def _bullet(document: Document, text: str) -> None:
    p = document.add_paragraph(text, style="List Bullet")
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.left_indent = Inches(0.22)


def write_docx(doc: ResumeDoc, path: str) -> str:
    document = Document()
    _style(document)
    c = doc.contact

    name_p = document.add_paragraph()
    name_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    name_run = name_p.add_run(c.name or "")
    name_run.bold = True
    name_run.font.size = Pt(18)

    if c.headline:
        h = document.add_paragraph()
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        hr = h.add_run(c.headline)
        hr.font.size = Pt(11.5)

    bits = [c.location, c.phone, c.email, c.linkedin, c.github, c.portfolio]
    contact_line = " | ".join(b for b in bits if b)
    if contact_line:
        cp = document.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.add_run(contact_line).font.size = Pt(9.5)

    if doc.summary:
        _heading(document, SECTION_ORDER_HEADINGS["summary"])
        document.add_paragraph(doc.summary)

    if doc.skills:
        _heading(document, SECTION_ORDER_HEADINGS["skills"])
        for cat, vals in doc.skills.items():
            if not vals:
                continue
            p = document.add_paragraph()
            p.add_run(f"{cat}: ").bold = True
            p.add_run(", ".join(vals))

    if doc.experience:
        _heading(document, SECTION_ORDER_HEADINGS["experience"])
        for exp in doc.experience:
            p = document.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.add_run(exp.title or "").bold = True
            tail = " | ".join(x for x in [exp.company, exp.location] if x)
            if tail:
                p.add_run(f" | {tail}")
            if exp.dates:
                p.add_run(f"  ({exp.dates})").italic = True
            for b in exp.bullets:
                _bullet(document, b.text)

    if doc.projects:
        _heading(document, SECTION_ORDER_HEADINGS["projects"])
        for proj in doc.projects:
            p = document.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.add_run(proj.name or "").bold = True
            if proj.subtitle:
                p.add_run(f" | {proj.subtitle}")
            if proj.link:
                p.add_run(f" | {proj.link}").font.size = Pt(9)
            if proj.tech:
                tp = document.add_paragraph()
                tp.add_run("Technologies: ").italic = True
                tp.add_run(", ".join(proj.tech))
            for b in proj.bullets:
                _bullet(document, b.text)

    if doc.education:
        _heading(document, SECTION_ORDER_HEADINGS["education"])
        for ed in doc.education:
            p = document.add_paragraph()
            p.add_run(ed.degree or "").bold = True
            if ed.institution:
                p.add_run(f" | {ed.institution}")
            if ed.dates:
                p.add_run(f"  ({ed.dates})").italic = True
            if ed.details:
                document.add_paragraph(ed.details)

    if doc.certifications:
        _heading(document, SECTION_ORDER_HEADINGS["certifications"])
        for cert in doc.certifications:
            _bullet(document, cert)

    for name, vals in doc.extras.items():
        if vals:
            _heading(document, name)
            for v in vals:
                _bullet(document, v)

    document.save(path)
    return path


# -------------------------------------------------------------------- PDF
def _write_pdf_native(doc: ResumeDoc, docx_path: str, out_dir: str | None) -> str | None:
    """Draw the resume straight to PDF with reportlab — no LibreOffice needed.

    Kept deliberately plain for ATS parsers: one column, real selectable text,
    no tables, no text boxes, no header/footer frames.
    """
    try:
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            HRFlowable, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer,
        )
    except ImportError:
        return None

    out_dir = out_dir or os.path.dirname(os.path.abspath(docx_path))
    path = os.path.join(
        out_dir, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf"
    )

    def esc(text: str) -> str:
        return (
            str(text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    body = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=9.5, leading=12.5, alignment=TA_LEFT,
        spaceAfter=2,
    )
    name = ParagraphStyle("name", parent=body, fontName="Helvetica-Bold", fontSize=17,
                          leading=20, spaceAfter=1)
    headline = ParagraphStyle("headline", parent=body, fontSize=10.5, leading=13,
                              spaceAfter=1)
    heading = ParagraphStyle("heading", parent=body, fontName="Helvetica-Bold",
                             fontSize=10.5, leading=13, spaceBefore=9, spaceAfter=3)
    item = ParagraphStyle("item", parent=body, fontName="Helvetica-Bold", spaceBefore=4)

    flow: list = []

    def section(title: str) -> None:
        flow.append(Paragraph(esc(title).upper(), heading))
        flow.append(HRFlowable(width="100%", thickness=0.6, spaceBefore=1, spaceAfter=4,
                               color="#666666"))

    def bullets(items: list[str]) -> None:
        if not items:
            return
        # Plain hyphen on purpose. The ZapfDingbats bullet extracts as the letter
        # "l" and a U+2022 in Helvetica extracts as (cid:127) — either way an ATS
        # reads garbage as the first word. A hyphen round-trips exactly.
        flow.append(ListFlowable(
            [ListItem(Paragraph(esc(t), body), leftIndent=12) for t in items],
            bulletType="bullet", bulletFontName="Helvetica", bulletFontSize=9.5,
            start="-", leftIndent=12, spaceBefore=1,
        ))

    c = doc.contact
    if c.name:
        flow.append(Paragraph(esc(c.name), name))
    if c.headline:
        flow.append(Paragraph(esc(c.headline), headline))
    contact_bits = [c.location, c.phone, c.email, c.linkedin, c.github, c.portfolio]
    if any(contact_bits):
        flow.append(Paragraph(esc(" | ".join(b for b in contact_bits if b)), body))

    if doc.summary:
        section(SECTION_ORDER_HEADINGS["summary"])
        flow.append(Paragraph(esc(doc.summary), body))

    if doc.skills:
        section(SECTION_ORDER_HEADINGS["skills"])
        for cat, vals in doc.skills.items():
            if vals:
                flow.append(Paragraph(
                    f"<b>{esc(cat)}:</b> {esc(', '.join(vals))}", body))

    if doc.experience:
        section(SECTION_ORDER_HEADINGS["experience"])
        for exp in doc.experience:
            head = " | ".join(x for x in [exp.title, exp.company, exp.location] if x)
            if exp.dates:
                head += f"  ({exp.dates})"
            flow.append(Paragraph(esc(head), item))
            bullets([b.text for b in exp.bullets])

    if doc.projects:
        section(SECTION_ORDER_HEADINGS["projects"])
        for proj in doc.projects:
            head = proj.name + (f" | {proj.subtitle}" if proj.subtitle else "")
            flow.append(Paragraph(esc(head), item))
            if proj.tech:
                flow.append(Paragraph(
                    f"<b>Technologies:</b> {esc(', '.join(proj.tech))}", body))
            bullets([b.text for b in proj.bullets])

    if doc.education:
        section(SECTION_ORDER_HEADINGS["education"])
        for ed in doc.education:
            flow.append(Paragraph(
                esc(" | ".join(x for x in [ed.degree, ed.institution, ed.dates] if x)),
                item))
            if ed.details:
                flow.append(Paragraph(esc(ed.details), body))

    if doc.certifications:
        section(SECTION_ORDER_HEADINGS["certifications"])
        bullets(list(doc.certifications))

    for extra_name, vals in doc.extras.items():
        if vals:
            section(extra_name)
            bullets(list(vals))

    flow.append(Spacer(1, 2))
    SimpleDocTemplate(
        path, pagesize=LETTER,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        title=c.name or "Resume", author=c.name or "",
    ).build(flow)
    return path if os.path.exists(path) else None


def write_pdf(
    docx_path: str, out_dir: str | None = None, doc: ResumeDoc | None = None
) -> str | None:
    """Best available PDF.

    LibreOffice first — it renders the real DOCX, so the PDF matches it exactly.
    Without it, fall back to drawing the resume directly with reportlab, which
    needs no external binary. Returns None only if neither route is available.
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return _write_pdf_native(doc, docx_path, out_dir) if doc is not None else None
    out_dir = out_dir or os.path.dirname(os.path.abspath(docx_path))
    with tempfile.TemporaryDirectory() as profile:
        try:
            subprocess.run(
                [soffice, "--headless", f"-env:UserInstallation=file://{profile}",
                 "--convert-to", "pdf", "--outdir", out_dir, docx_path],
                check=True, capture_output=True, timeout=180,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
    pdf_path = os.path.join(
        out_dir, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf"
    )
    return pdf_path if os.path.exists(pdf_path) else None
