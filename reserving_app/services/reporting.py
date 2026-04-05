from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


def build_pdf_report(
    portfolio: str,
    selected_factors: pd.Series,
    ibnr: pd.Series,
    bootstrap_summary: pd.Series | None,
    comments: str,
) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    w, h = LETTER

    y = h - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Actuarial Reserving Report")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Portfolio/Segment: {portfolio or 'All'}")
    y -= 15
    c.drawString(40, y, f"Generated: {datetime.utcnow().isoformat()} UTC")

    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Management Summary")
    y -= 15
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Total IBNR: {ibnr.sum():,.2f}")

    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Technical Appendix: Selected Factors")
    y -= 15
    c.setFont("Helvetica", 9)
    for idx, val in selected_factors.items():
        c.drawString(50, y, f"{idx}: {val:.4f}")
        y -= 12
        if y < 60:
            c.showPage()
            y = h - 50

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Technical Appendix: IBNR by Origin")
    y -= 15
    c.setFont("Helvetica", 9)
    for idx, val in ibnr.items():
        c.drawString(50, y, f"{idx}: {val:,.2f}")
        y -= 12
        if y < 60:
            c.showPage()
            y = h - 50

    if bootstrap_summary is not None:
        y -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Bootstrap Summary")
        y -= 15
        c.setFont("Helvetica", 9)
        for idx, val in bootstrap_summary.items():
            c.drawString(50, y, f"{idx}: {val:,.2f}")
            y -= 12
            if y < 60:
                c.showPage()
                y = h - 50

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Actuary Notes")
    y -= 15
    c.setFont("Helvetica", 9)
    for line in (comments or "No comments provided.").split("\n"):
        c.drawString(50, y, line[:100])
        y -= 12
        if y < 60:
            c.showPage()
            y = h - 50

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def export_tables_to_excel(tables: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31])
    output.seek(0)
    return output.read()
