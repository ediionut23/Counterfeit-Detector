"""
Build "Statistici dataset" — how many molecules were generated, per category
=============================================================================

Reads final_dataset/molecules.csv at build time, so the tables never drift
from the delivered dataset. Produces:
  - one combined table (all categories, M + E)
  - one table per generator (Version M, Version E)
  - full subcategory breakdown, grouped by category

Usage
-----
    venv/bin/python build_pdf_statistici_dataset.py
"""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

SRC = "final_dataset/molecules.csv"
AUTH = "final_dataset/authentic.csv"
OUT = "Statistici_Dataset_Categorii.pdf"

_TTF = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DJV", _TTF / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DJV-B", _TTF / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily("DJV", normal="DJV", bold="DJV-B")

INK = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#1f4e79")
MCOL = colors.HexColor("#2e6da4")
ECOL = colors.HexColor("#8a5a2b")
SOFT = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#c8d3dd")
BOXBG = colors.HexColor("#f4f7fa")

_ss = getSampleStyleSheet()
title = ParagraphStyle("t", parent=_ss["Title"], fontName="DJV-B", fontSize=18,
                       leading=23, textColor=ACCENT, spaceAfter=4)
subtitle = ParagraphStyle("st", parent=_ss["Normal"], fontName="DJV", fontSize=9.3,
                          leading=13.5, textColor=SOFT, alignment=TA_CENTER, spaceAfter=15)
h1 = ParagraphStyle("h1", parent=_ss["Heading1"], fontName="DJV-B", fontSize=13.5,
                    leading=17, textColor=ACCENT, spaceBefore=14, spaceAfter=7)
h2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontName="DJV-B", fontSize=10.5,
                    leading=13.5, textColor=INK, spaceBefore=10, spaceAfter=5)
body = ParagraphStyle("b", parent=_ss["Normal"], fontName="DJV", fontSize=9.3,
                      leading=14, textColor=INK, spaceAfter=6, alignment=TA_LEFT)
bullet = ParagraphStyle("bu", parent=body, leftIndent=11, spaceAfter=5)
boxed = ParagraphStyle("bx", parent=body, backColor=BOXBG, borderColor=RULE,
                       borderWidth=0.6, borderPadding=(8, 8, 8, 8),
                       spaceBefore=5, spaceAfter=9)
caption = ParagraphStyle("cap", parent=body, fontSize=8.1, textColor=SOFT,
                         leading=11.4, spaceBefore=2, spaceAfter=11)
th = ParagraphStyle("th", parent=body, fontName="DJV-B", fontSize=8.1, leading=10.4,
                    textColor=colors.white, spaceAfter=0)
thr = ParagraphStyle("thr", parent=th, alignment=TA_CENTER)
td = ParagraphStyle("td", parent=body, fontSize=8.1, leading=10.6, spaceAfter=0)
tdr = ParagraphStyle("tdr", parent=td, alignment=TA_CENTER)
tdb = ParagraphStyle("tdb", parent=td, fontName="DJV-B")
tdbr = ParagraphStyle("tdbr", parent=tdb, alignment=TA_CENTER)

CATS = ["homologation", "bioisostere", "scaffold-hop", "halogen-walk", "other"]
RO = {
    "homologation": "Homologare",
    "bioisostere": "Bioizosteri",
    "scaffold-hop": "Scaffold-hop",
    "halogen-walk": "Halogen-walk",
    "other": "Altele",
    "unconditioned": "Necondiționat",
}


def nr(n: int) -> str:
    """Romanian thousands separator."""
    return f"{n:,}".replace(",", ".")


def load():
    with open(SRC, newline="") as f:
        rows = list(csv.DictReader(f))
    by_cat = defaultdict(lambda: defaultdict(int))
    by_sub = defaultdict(lambda: defaultdict(int))
    for x in rows:
        c, s, m = x["primary_category"], x["primary_subcategory"], x["primary_method"]
        by_cat[c][m] += 1
        by_sub[(c, s)][m] += 1
    n_auth = 0
    if os.path.exists(AUTH):
        with open(AUTH, newline="") as f:
            n_auth = sum(1 for _ in f) - 1
    n_sub = len({s for _, s in by_sub})
    return rows, by_cat, by_sub, n_auth, n_sub


def mk_table(rows, widths, total_row=True):
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2 if total_row else -1),
         [colors.white, colors.HexColor("#f7f9fb")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if total_row:
        style += [
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8eef4")),
            ("LINEABOVE", (0, -1), (-1, -1), 0.9, ACCENT),
        ]
    t.setStyle(TableStyle(style))
    return t


def hdr(*labels):
    return [Paragraph(labels[0], th)] + [Paragraph(x, thr) for x in labels[1:]]


def datarow(label, *vals, bold=False):
    st = tdb if bold else td
    sr = tdbr if bold else tdr
    return [Paragraph(label, st)] + [Paragraph(v, sr) for v in vals]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DJV", 7.3)
    canvas.setFillColor(SOFT)
    canvas.drawString(20 * mm, 12 * mm, "Statistici dataset — molecule generate per categorie")
    canvas.drawRightString(190 * mm, 12 * mm, f"Pagina {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 15.5 * mm, 190 * mm, 15.5 * mm)
    canvas.restoreState()


def build():
    rows, by_cat, by_sub, n_auth, n_sub = load()
    tot_m = sum(v["M"] for v in by_cat.values())
    tot_e = sum(v["E"] for v in by_cat.values())
    tot = tot_m + tot_e

    doc = SimpleDocTemplate(
        OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="Statistici dataset — molecule generate per categorie",
        author="Ionut Eduard Tanasa")
    s = []
    W = doc.width

    # ------------------------------------------------------ PAGE 1
    s.append(Paragraph("Statistici dataset<br/>Molecule generate per categorie", title))
    s.append(Paragraph(
        f"Datasetul final conține <b>{nr(tot)}</b> molecule contrafăcute și "
        f"<b>{nr(n_auth)}</b> molecule autentice. Cifrele sunt citite direct din "
        f"<i>{SRC}</i> la generarea acestui document.", subtitle))

    s.append(Paragraph("Categorii sau subcategorii?", h1))
    s.append(Paragraph(
        f"Există <b>două niveluri</b> de clasificare, și probabil pe al doilea ți-l aminteai:",
        body))
    s.append(Paragraph(
        f"• <b>5 categorii</b> principale — tipul general al modificării chimice "
        f"(bioizosteri, scaffold-hop, halogen-walk, homologare, altele);<br/>"
        f"• <b>{n_sub} subcategorii</b> — semnătura exactă a modificării, de tipul "
        f"<i>Cl→F</i>, <i>add-O</i>, <i>aromatize</i>, <i>grow-single</i>.", bullet))
    s.append(Paragraph(
        "Subcategoriile nu sunt distribuite uniform: bioizosterii se împart în 52 de "
        "subcategorii (toate permutările de schimb între atomi), pe când „altele” are una "
        "singură. Detaliul complet e în Tabelul 4, la finalul documentului.", body))

    s.append(Paragraph("Tabelul 1 &nbsp;·&nbsp; Toate moleculele", h1))
    trows = [hdr("Categorie", "Versiunea M", "Versiunea E", "Total", "% din total")]
    for c in CATS:
        m, e = by_cat[c]["M"], by_cat[c]["E"]
        trows.append(datarow(RO[c], nr(m), nr(e), nr(m + e), f"{100*(m+e)/tot:.1f}%"))
    um, ue = by_cat["unconditioned"]["M"], by_cat["unconditioned"]["E"]
    trows.append(datarow(f"<i>{RO['unconditioned']}</i>", "—" if not um else nr(um),
                         nr(ue), nr(um + ue), f"{100*(um+ue)/tot:.1f}%"))
    trows.append(datarow("TOTAL", nr(tot_m), nr(tot_e), nr(tot), "100%", bold=True))
    s.append(mk_table(trows, [0.30 * W, 0.18 * W, 0.17 * W, 0.17 * W, 0.18 * W]))
    s.append(Paragraph(
        f"Versiunea M acoperă {100*tot_m/tot:.0f}% din dataset, Versiunea E {100*tot_e/tot:.0f}%. "
        "Categoriile sunt dezechilibrate: homologarea are de aproape patru ori mai multe "
        "molecule decât „altele”, diferență care trebuie menționată când se raportează "
        "scoruri per categorie.", caption))

    s.append(Paragraph("De reținut", h2))
    ue_share = 100 * ue / tot_e if tot_e else 0
    s.append(Paragraph(
        f"<b>{ue_share:.0f}% din moleculele Versiunii E nu au categorie.</b> Cele "
        f"{nr(ue)} de molecule marcate „necondiționat” provin din rulările în care NSGA-II a "
        f"căutat liber, fără filtrare pe tip de modificare. Sunt utilizabile la antrenare, dar "
        f"nu pot intra în evaluarea per categorie — adică "
        f"{100*(um+ue)/tot:.0f}% din dataset rămâne în afara benchmark-ului pe categorii.", boxed))

    s.append(PageBreak())

    # ------------------------------------------------------ PAGE 2: M
    s.append(Paragraph("Tabelul 2 &nbsp;·&nbsp; Versiunea M (generator pe reguli)", h1))
    s.append(Paragraph(
        "Moleculele produse prin aplicarea directă a regulilor MMP extrase din date reale. "
        "Fiecare moleculă poartă regula exactă folosită, deci are subcategorie precisă.", body))
    sub_per_cat = Counter()
    for (c, sname), v in by_sub.items():
        if v["M"] > 0:
            sub_per_cat[c] += 1
    trows = [hdr("Categorie", "Molecule", "Subcategorii", "% din M")]
    for c in CATS:
        m = by_cat[c]["M"]
        trows.append(datarow(RO[c], nr(m), str(sub_per_cat[c]), f"{100*m/tot_m:.1f}%"))
    trows.append(datarow("TOTAL", nr(tot_m), str(sum(sub_per_cat.values())), "100%", bold=True))
    s.append(mk_table(trows, [0.34 * W, 0.22 * W, 0.22 * W, 0.22 * W]))
    s.append(Paragraph(
        "Versiunea M nu produce molecule „necondiționate”: fiecare rând provine dintr-o regulă "
        "cunoscută, dintr-o categorie cunoscută.", caption))

    # ------------------------------------------------------ PAGE 2: E
    s.append(Paragraph("Tabelul 3 &nbsp;·&nbsp; Versiunea E (căutare evolutivă, NSGA-II)", h1))
    s.append(Paragraph(
        "Moleculele produse prin optimizare multi-obiectiv. Se împart în două regimuri de rulare:",
        body))
    trows = [hdr("Regim de rulare", "Molecule", "% din E")]
    for c in CATS:
        e = by_cat[c]["E"]
        trows.append(datarow(f"Condiționat pe „{RO[c]}”", nr(e), f"{100*e/tot_e:.1f}%"))
    trows.append(datarow("<i>Necondiționat (fără categorie)</i>", nr(ue), f"{100*ue/tot_e:.1f}%"))
    trows.append(datarow("TOTAL", nr(tot_e), "100%", bold=True))
    s.append(mk_table(trows, [0.46 * W, 0.27 * W, 0.27 * W]))
    s.append(Paragraph(
        "Spre deosebire de M, Versiunea E <b>nu are subcategorii</b>. Căutarea evolutivă "
        "combină mai multe modificări suprapuse, deci rezultatul nu corespunde unei singure "
        "semnături de transformare — poate fi etichetat cel mult la nivel de categorie.", caption))

    s.append(PageBreak())

    # ------------------------------------------------------ SUBCATEGORIES
    s.append(Paragraph("Tabelul 4 &nbsp;·&nbsp; Subcategorii, grupate pe categorie", h1))
    s.append(Paragraph(
        "Semnătura exactă a fiecărei modificări. Rândurile marcate "
        "<i>category_conditioned</i> reprezintă moleculele Versiunii E din acea categorie, "
        "care nu au granularitate mai fină.", body))

    for c in CATS + ["unconditioned"]:
        items = sorted([(sname, v["M"], v["E"]) for (cc, sname), v in by_sub.items() if cc == c],
                       key=lambda x: -(x[1] + x[2]))
        if not items:
            continue
        s.append(Paragraph(f"{RO[c]} &nbsp;—&nbsp; {len(items)} subcategorii, "
                           f"{nr(sum(a + b for _, a, b in items))} molecule", h2))
        if len(items) > 18:
            # two side-by-side blocks to keep long lists on one page
            half = (len(items) + 1) // 2
            left, right = items[:half], items[half:]
            while len(right) < len(left):
                right.append(("", 0, 0))
            trows = [hdr("Subcategorie", "Total", "Subcategorie", "Total")]
            for (ln, lm, le), (rn, rm, re) in zip(left, right):
                trows.append([
                    Paragraph(ln, td), Paragraph(nr(lm + le) if ln else "", tdr),
                    Paragraph(rn, td), Paragraph(nr(rm + re) if rn else "", tdr)])
            s.append(mk_table(trows, [0.31 * W, 0.19 * W, 0.31 * W, 0.19 * W],
                              total_row=False))
        else:
            trows = [hdr("Subcategorie", "Versiunea M", "Versiunea E", "Total")]
            for sname, m, e in items:
                trows.append(datarow(sname, nr(m) if m else "—", nr(e) if e else "—", nr(m + e)))
            s.append(mk_table(trows, [0.37 * W, 0.21 * W, 0.21 * W, 0.21 * W],
                              total_row=False))
        s.append(Spacer(1, 4))

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"scris {OUT} ({os.path.getsize(OUT)/1024:.0f} KB) — "
          f"{nr(tot)} contrafăcute, {n_sub} subcategorii")


if __name__ == "__main__":
    build()
