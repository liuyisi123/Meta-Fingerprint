"""PDF report generator using ReportLab."""
from __future__ import annotations
import io, time
from pathlib import Path
from typing import Any
import numpy as np

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image
    )
    from reportlab.graphics.shapes import Drawing, Line, PolyLine, String, Rect
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.lib.colors import HexColor
    _RL_OK = True
except ImportError:
    _RL_OK = False


# ── Colours ────────────────────────────────────────────────────
C_BG      = HexColor("#060E1A") if _RL_OK else None
C_ACCENT  = HexColor("#00C8F0") if _RL_OK else None
C_TEXT    = HexColor("#E8F4FD") if _RL_OK else None
C_SEC     = HexColor("#7AAFCF") if _RL_OK else None
C_GREEN   = HexColor("#00E676") if _RL_OK else None
C_AMBER   = HexColor("#FFB300") if _RL_OK else None
C_RED     = HexColor("#FF4444") if _RL_OK else None
C_CARD    = HexColor("#0D2137") if _RL_OK else None
C_BORDER  = HexColor("#1A3A5C") if _RL_OK else None


def _waveform_drawing(data: np.ndarray, color_hex: str, width: float, height: float) -> "Drawing":
    d = Drawing(width, height)
    if len(data) == 0:
        return d
    xs = np.linspace(0, width, len(data))
    mn, mx = data.min(), data.max()
    rng = max(mx - mn, 1e-6)
    ys = (data - mn) / rng * (height - 10) + 5
    pts = []
    for x, y in zip(xs, ys):
        pts += [float(x), float(y)]
    pl = PolyLine(pts, strokeColor=HexColor(color_hex), strokeWidth=1.2)
    d.add(pl)
    return d


def generate_report(output_path: str, patient: dict[str, Any], session: dict[str, Any],
                    waveform_ecg: np.ndarray | None = None,
                    waveform_ppg: np.ndarray | None = None,
                    waveform_abp: np.ndarray | None = None,
                    waveform_abp_pred: np.ndarray | None = None) -> bool:
    if not _RL_OK:
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )

    W = A4[0] - 3 * cm
    styles = getSampleStyleSheet()

    def S(name, **kw):
        base = styles["Normal"]
        return ParagraphStyle(name, parent=base, **kw)

    TITLE   = S("title",  fontSize=22, textColor=C_ACCENT,  fontName="Helvetica-Bold", spaceAfter=4)
    SUBHEAD = S("sh",     fontSize=13, textColor=C_ACCENT,  fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
    BODY    = S("body",   fontSize=10, textColor=C_TEXT,    fontName="Helvetica",      spaceAfter=3)
    SMALL   = S("small",  fontSize=8,  textColor=C_SEC,     fontName="Helvetica")
    LABEL   = S("lbl",    fontSize=9,  textColor=C_SEC,     fontName="Helvetica-Bold")

    def metric_table(rows: list[tuple[str, str, str]]) -> Table:
        data = [["Metric", "Value", "Status"]] + [[r[0], r[1], r[2]] for r in rows]
        t = Table(data, colWidths=[W * 0.45, W * 0.30, W * 0.25])
        style = TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  C_CARD),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  C_ACCENT),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0),  9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_CARD, HexColor("#0A1628")]),
            ("TEXTCOLOR",    (0, 1), (-1, -1), C_TEXT),
            ("FONTSIZE",     (0, 1), (-1, -1), 9),
            ("GRID",         (0, 0), (-1, -1), 0.5, C_BORDER),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ])
        t.setStyle(style)
        return t

    def hr():
        return HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=6, spaceBefore=6)

    # ── Build content ───────────────────────────────────────────
    story = []

    # Header block (coloured rect illusion via table)
    hdr_data = [[
        Paragraph("Meta-Fingerprint", TITLE),
        Paragraph(f"<b>Hemodynamic Analysis Report</b><br/>"
                  f"<font size='9' color='#7AAFCF'>"
                  f"Generated: {time.strftime('%Y-%m-%d %H:%M')}</font>", BODY),
    ]]
    hdr_t = Table(hdr_data, colWidths=[W * 0.6, W * 0.4])
    hdr_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_CARD),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("LINEBELOW",     (0, 0), (-1, -1), 2, C_ACCENT),
    ]))
    story.append(hdr_t)
    story.append(Spacer(1, 0.4 * cm))

    # Patient info
    story.append(Paragraph("Patient Information", SUBHEAD))
    story.append(hr())
    pid = patient.get("id", "N/A")
    pname = patient.get("name", "Anonymous")
    pdob  = patient.get("dob", "—")
    psex  = patient.get("sex", "—")
    pht   = patient.get("height_cm", "—")
    pwt   = patient.get("weight_kg", "—")
    pbmi  = patient.get("bmi", "—")
    p_rows = [
        [Paragraph(f"<b>Patient ID:</b>  {pid}", BODY),   Paragraph(f"<b>Name:</b>  {pname}", BODY)],
        [Paragraph(f"<b>Date of Birth:</b>  {pdob}", BODY), Paragraph(f"<b>Sex:</b>  {psex}", BODY)],
        [Paragraph(f"<b>Height:</b>  {pht} cm", BODY),    Paragraph(f"<b>Weight:</b>  {pwt} kg  (BMI {pbmi})", BODY)],
    ]
    pt = Table(p_rows, colWidths=[W / 2, W / 2])
    pt.setStyle(TableStyle([("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3)]))
    story.append(pt)
    story.append(Spacer(1, 0.3 * cm))

    # Session info
    story.append(Paragraph("Session Details", SUBHEAD))
    story.append(hr())
    story.append(Paragraph(f"<b>Setting:</b>  {session.get('setting','custom')}    "
                            f"<b>Timestamp:</b>  {session.get('timestamp','—')}    "
                            f"<b>Notes:</b>  {session.get('notes','—')}", BODY))
    story.append(Spacer(1, 0.3 * cm))

    # Waveforms
    if any(w is not None for w in [waveform_ecg, waveform_ppg, waveform_abp_pred]):
        story.append(Paragraph("Signal Overview", SUBHEAD))
        story.append(hr())
        wave_rows = []
        for lbl, data, col in [
            ("ECG",              waveform_ecg,      "#00FF88"),
            ("PPG",              waveform_ppg,      "#00C8F0"),
            ("ABP (Reference)",  waveform_abp,      "#7AAFCF"),
            ("ABP (Predicted)",  waveform_abp_pred, "#FF6B35"),
        ]:
            if data is not None and len(data) > 0:
                wave_rows.append([
                    Paragraph(lbl, LABEL),
                    _waveform_drawing(data[:500], col, W * 0.80, 35),
                ])
        if wave_rows:
            wt = Table(wave_rows, colWidths=[W * 0.16, W * 0.84])
            wt.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), C_CARD),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ("LEFTPADDING",   (0,0), (-1,-1), 6),
                ("LINEBELOW",     (0,0), (-1,-1), 0.3, C_BORDER),
            ]))
            story.append(wt)
            story.append(Spacer(1, 0.3 * cm))

    # Clinical metrics
    story.append(Paragraph("Clinical Metrics", SUBHEAD))
    story.append(hr())
    sbp = session.get("rmse_sbp", "—")
    dbp = session.get("rmse_dbp", "—")
    f1  = session.get("macro_f1", "—")
    dsr = session.get("domain_shift_ratio", "—")
    aami_sbp = session.get("aami_sbp", False)
    aami_dbp = session.get("aami_dbp", False)
    pheno = session.get("phenotype", "—")

    def aami_str(v):
        return "✓ PASS" if v else "✗ FAIL"

    rows = [
        ("SBP RMSE (mmHg)",         f"{sbp:.2f}" if isinstance(sbp, float) else str(sbp), aami_str(aami_sbp)),
        ("DBP RMSE (mmHg)",         f"{dbp:.2f}" if isinstance(dbp, float) else str(dbp), aami_str(aami_dbp)),
        ("AAMI SP10 (SBP, σd ≤ 8)", "—",                                                  aami_str(aami_sbp)),
        ("AAMI SP10 (DBP, σd ≤ 8)", "—",                                                  aami_str(aami_dbp)),
        ("Macro-F1 (phenotype)",    f"{f1:.3f}" if isinstance(f1, float) else str(f1),    ""),
        ("Domain-Shift Ratio ρ",    f"{dsr:.2f}" if isinstance(dsr, float) else str(dsr), "Target ≤ 2.0"),
        ("Risk Phenotype",           str(pheno),                                            ""),
    ]
    story.append(metric_table(rows))
    story.append(Spacer(1, 0.3 * cm))

    # Disclaimer
    story.append(hr())
    story.append(Paragraph(
        "DISCLAIMER — This report is generated by Meta-Fingerprint, a research tool for cross-domain "
        "hemodynamic monitoring. It is <b>not a validated clinical diagnostic device</b>. Results are "
        "for research and evaluation purposes only. AAMI SP10 flags apply only to invasive-ABP-equipped "
        "settings; Setting-C (RWW, CNAP reference) results should be interpreted as CNAP-referenced "
        "transfer, not AAMI compliance.",
        SMALL))

    try:
        doc.build(story, onFirstPage=_page_template, onLaterPages=_page_template)
        return True
    except Exception:
        return False


def _page_template(canvas, doc):
    canvas.saveState()
    w, h = A4
    # footer
    canvas.setFillColor(HexColor("#1A3A5C"))
    canvas.rect(0, 0, w, 1.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(HexColor("#7AAFCF"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1.5 * cm, 0.4 * cm, "Meta-Fingerprint | Physics-Grounded Hemodynamic Monitoring | Research Use Only")
    canvas.drawRightString(w - 1.5 * cm, 0.4 * cm, f"Page {doc.page}")
    canvas.restoreState()
