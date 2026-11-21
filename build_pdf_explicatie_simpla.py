"""
Build "Explicatie simpla" — a plain-language companion to the technical doc
===========================================================================

Three questions, answered as simply as possible:
  1. How does NSGA-II actually work?
  2. How does the Version M generator work (the simple one)?
  3. Can molecules from both generators be mixed in training?

Part 3 reports the measured numbers from research_runs/GENERATOR_TRANSFER_REPORT.md
and keeps that report's epistemic caveats intact.

Usage
-----
    venv/bin/python build_pdf_explicatie_simpla.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

OUT = "Explicatie_NSGA2_si_Generare.pdf"

# --- Romanian needs s-comma / t-comma (U+0219/U+021B), absent from WinAnsi ---
_TTF = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DJV", _TTF / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DJV-B", _TTF / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DJV-M", _TTF / "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DJV", normal="DJV", bold="DJV-B")

INK = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#1f4e79")
SOFT = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#c8d3dd")
BOXBG = colors.HexColor("#f4f7fa")
CODEBG = colors.HexColor("#f0f0ef")
GOODBG = colors.HexColor("#eef6ee")

_ss = getSampleStyleSheet()

title = ParagraphStyle("title", parent=_ss["Title"], fontName="DJV-B", fontSize=19,
                       leading=24, textColor=ACCENT, spaceAfter=4)
subtitle = ParagraphStyle("subtitle", parent=_ss["Normal"], fontName="DJV", fontSize=9.5,
                          leading=14, textColor=SOFT, alignment=TA_CENTER, spaceAfter=16)
h1 = ParagraphStyle("h1", parent=_ss["Heading1"], fontName="DJV-B", fontSize=14,
                    leading=18, textColor=ACCENT, spaceBefore=16, spaceAfter=7)
h2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontName="DJV-B", fontSize=10.8,
                    leading=14, textColor=INK, spaceBefore=11, spaceAfter=5)
body = ParagraphStyle("body", parent=_ss["Normal"], fontName="DJV", fontSize=9.4,
                      leading=14.4, textColor=INK, spaceAfter=6, alignment=TA_LEFT)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=11, spaceAfter=5, leading=14)
code = ParagraphStyle("code", parent=_ss["Normal"], fontName="DJV-M", fontSize=8.1,
                      leading=12.4, textColor=INK, backColor=CODEBG,
                      borderPadding=(7, 7, 7, 7), spaceBefore=4, spaceAfter=8)
boxed = ParagraphStyle("boxed", parent=body, backColor=BOXBG, borderColor=RULE,
                       borderWidth=0.6, borderPadding=(8, 8, 8, 8),
                       spaceBefore=5, spaceAfter=9, leading=14.2)
answer = ParagraphStyle("answer", parent=boxed, backColor=GOODBG,
                        borderColor=colors.HexColor("#a9c9a9"))
caption = ParagraphStyle("caption", parent=body, fontSize=8.2, textColor=SOFT,
                         leading=11.5, spaceBefore=2, spaceAfter=10)
th = ParagraphStyle("th", parent=body, fontName="DJV-B", fontSize=8.2, leading=10.6,
                    textColor=colors.white, spaceAfter=0)
td = ParagraphStyle("td", parent=body, fontSize=8.2, leading=10.8, spaceAfter=0)
tdc = ParagraphStyle("tdc", parent=td, alignment=TA_CENTER)
tdb = ParagraphStyle("tdb", parent=td, fontName="DJV-B")


def table(rows, widths, align_center_from=1):
    """Header row + body rows, all cells already Paragraphs."""
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def hdr(*labels):
    return [Paragraph(x, th) for x in labels]


def row(*cells, bold_first=False):
    out = []
    for i, c in enumerate(cells):
        st = tdb if (i == 0 and bold_first) else (td if i == 0 else tdc)
        out.append(Paragraph(c, st))
    return out


def steps(items):
    """Numbered steps as a borderless two-column table (number | text)."""
    rows = []
    for i, txt in enumerate(items, 1):
        rows.append([Paragraph(f"<b>{i}</b>", ParagraphStyle(
            "n", parent=body, fontName="DJV-B", fontSize=10, textColor=ACCENT,
            alignment=TA_CENTER, spaceAfter=0)),
            Paragraph(txt, ParagraphStyle("s", parent=body, spaceAfter=0, leading=13.6))])
    t = Table(rows, colWidths=[9 * mm, 155 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#e6ecf2")),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DJV", 7.4)
    canvas.setFillColor(SOFT)
    canvas.drawString(20 * mm, 12 * mm,
                      "Detectarea medicamentelor contrafăcute cu GNN — explicație simplificată")
    canvas.drawRightString(190 * mm, 12 * mm, f"Pagina {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 15.5 * mm, 190 * mm, 15.5 * mm)
    canvas.restoreState()


def build():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="Cum funcționează NSGA-II și generarea de contrafăcute",
        author="Ionut Eduard Tanasa",
    )
    s = []
    W = doc.width

    # ============================================================ TITLE
    s.append(Paragraph("Cum funcționează NSGA-II<br/>și generarea de contrafăcute", title))
    s.append(Paragraph(
        "Explicație simplificată, în trei părți: (1) algoritmul de optimizare NSGA-II, "
        "(2) generatorul pe reguli — Versiunea M, (3) dacă moleculele celor două "
        "generatoare pot fi folosite împreună la antrenare.", subtitle))

    # ============================================================ PART 1
    s.append(Paragraph("Partea 1 &nbsp;·&nbsp; NSGA-II pe înțelesul tuturor", h1))

    s.append(Paragraph("1.1 &nbsp;Problema: criterii care se bat între ele", h2))
    s.append(Paragraph(
        "Un contrafăcut util trebuie să îndeplinească simultan trei condiții, iar acestea "
        "se contrazic:", body))
    s.append(Paragraph(
        "• <b>să semene</b> cu medicamentul original — altfel nu păcălește pe nimeni;<br/>"
        "• <b>să fie o moleculă reală</b>, fabricabilă — altfel nu e o amenințare practică;<br/>"
        "• <b>să fie greu de detectat</b> — altfel nu are valoare ca test.", bullet))
    s.append(Paragraph(
        "Problema e că dacă forțezi una, strici alta. O moleculă foarte asemănătoare "
        "originalului e ușor de recunoscut ca analog. Una foarte greu de detectat ajunge, "
        "de obicei, să nu mai semene cu originalul.", body))
    s.append(Paragraph(
        "<b>Analogie:</b> cauți o mașină ieftină, rapidă și economică. Nu există una care să "
        "fie cea mai bună la toate trei. Există doar un set de variante bune, fiecare cu alt "
        "compromis — iar alegerea dintre ele depinde de ce îți trebuie.", boxed))

    s.append(Paragraph("1.2 &nbsp;Soluția tentantă, dar greșită: media ponderată", h2))
    s.append(Paragraph(
        "Prima idee ar fi să combini totul într-un singur scor:", body))
    s.append(Paragraph("scor = 0.5 × similaritate + 0.3 × realism + 0.2 × dificultate", code))
    s.append(Paragraph(
        "Problema: <b>de unde știi ponderile?</b> Scriind 0.5, 0.3 și 0.2 ai decis tu, "
        "arbitrar, care compromis e cel bun — și algoritmul îți va returna o singură moleculă, "
        "cea care se potrivește alegerii tale. Dacă ponderile erau greșite, tot rezultatul e "
        "greșit, și nu ai cum să afli.", body))

    s.append(Paragraph("1.3 &nbsp;Ideea NSGA-II: nu alegi, păstrezi tot", h2))
    s.append(Paragraph(
        "NSGA-II refuză să combine criteriile. În schimb, compară moleculele între ele "
        "folosind o singură regulă, numită <b>dominanță</b>:", body))
    s.append(Paragraph(
        "<b>A domină pe B</b> dacă A este cel puțin la fel de bună ca B la <b>toate</b> "
        "criteriile, și strict mai bună la <b>cel puțin unul</b>.", boxed))
    s.append(Paragraph(
        "Dacă A domină pe B, atunci B nu are nicio justificare să existe — A e mai bună "
        "oricum te uiți. B se elimină. Exemplu cu trei molecule (valori mai mici = mai bune):",
        body))

    s.append(table([
        hdr("Moleculă", "Similaritate", "Realism", "Dificultate", "Concluzie"),
        row("A", "0.05", "0.40", "0.60", "rămâne", bold_first=True),
        row("B", "0.30", "0.10", "0.20", "rămâne", bold_first=True),
        row("C", "0.35", "0.45", "0.65", "eliminată", bold_first=True),
    ], [0.14 * W, 0.19 * W, 0.15 * W, 0.17 * W, 0.35 * W]))
    s.append(Paragraph(
        "<b>C este eliminată</b> pentru că A e mai bună la toate trei criteriile "
        "(0.05&lt;0.35, 0.40&lt;0.45, 0.60&lt;0.65). <b>A și B rămân amândouă</b>: A e mai bună "
        "la similaritate, B e mai bună la celelalte două — niciuna nu o domină pe cealaltă. "
        "Sunt compromisuri diferite, ambele legitime.", caption))

    s.append(Paragraph("1.4 &nbsp;Frontul Pareto = răspunsul, la plural", h2))
    s.append(Paragraph(
        "Moleculele pe care nu le domină nimeni formează <b>frontul Pareto</b>. Acesta este "
        "rezultatul algoritmului: nu o moleculă câștigătoare, ci o <b>colecție</b> de molecule, "
        "fiecare reprezentând alt echilibru între cele trei criterii. Tu alegi apoi din front, "
        "știind ce compromisuri existau de fapt.", body))

    s.append(Paragraph("1.5 &nbsp;Crowding distance: de ce nu vrei clone", h2))
    s.append(Paragraph(
        "Dacă pe front încap doar 20 de molecule, dar sunt 35 nedominate, trebuie să alegi. "
        "Aici NSGA-II preferă moleculele <b>izolate</b> în loc de cele îngrămădite una peste alta.",
        body))
    s.append(Paragraph(
        "Motivul e practic: fără această regulă ai primi 20 de variante aproape identice ale "
        "aceleiași modificări. Cu ea, primești modificări variate — substituții de atomi, "
        "schimbări de schelet, deplasări de halogen — adică exact diversitatea de care are "
        "nevoie un benchmark.", body))

    s.append(Paragraph("1.6 &nbsp;Bucla completă", h2))
    s.append(steps([
        "Pornești cu 20 de molecule, obținute aplicând 1–3 modificări la medicamentul original.",
        "Fiecare moleculă primește trei note (cele trei criterii). Nimic nu e aruncat în acest pas.",
        "Moleculele se recombină două câte două și suferă mutații, producând ~20 de copii noi.",
        "Copiii primesc și ei trei note fiecare.",
        "Toate cele 40 (părinți + copii) se compară prin dominanță; se păstrează cele mai bune 20, "
        "cu preferință pentru cele izolate.",
        "Se repetă de la pasul 3. După ultima rundă, frontul Pareto este rezultatul.",
    ]))

    s.append(Paragraph("1.7 &nbsp;Trei lucruri pe care NSGA-II NU le face", h2))
    s.append(Paragraph(
        "• <b>Nu respinge molecule pe baza unui prag.</b> O moleculă „în afara parametrilor” "
        "nu e ștearsă — primește doar o notă proastă, deci e mai probabil să piardă la comparație.<br/>"
        "• <b>Nu adună niciodată criteriile.</b> Similaritatea se compară doar cu similaritatea, "
        "realismul doar cu realismul.<br/>"
        "• <b>Niciun criteriu nu are drept de veto.</b> O moleculă slabă la un criteriu poate "
        "supraviețui, dacă e excepțională la celelalte — exact cazul moleculei B din tabelul de mai sus.",
        bullet))

    s.append(PageBreak())

    # ============================================================ PART 2
    s.append(Paragraph("Partea 2 &nbsp;·&nbsp; Versiunea M — generatorul simplu", h1))

    s.append(Paragraph("2.1 &nbsp;Ideea în două propoziții", h2))
    s.append(Paragraph(
        "Dintr-o bază de medicamente reale au fost extrase automat <b>peste 3.200 de reguli "
        "de modificare</b> — tipare de genul „înlocuiește un clor cu un fluor” sau „lungește "
        "catena cu un carbon”, fiecare observată de multe ori în practică.", body))
    s.append(Paragraph(
        "Versiunea M ia un medicament, îi aplică o regulă din catalog, apoi verifică dacă "
        "rezultatul trece câteva filtre. Dacă trece, îl păstrează. Dacă nu, încearcă altă regulă. "
        "<b>Atât.</b> Nu există populație, nu există generații, nu există optimizare.", body))

    s.append(Paragraph("2.2 &nbsp;Pașii, exact ca în cod", h2))
    s.append(steps([
        "Ia un medicament din listă (ordinea e amestecată aleatoriu).",
        "Alege o regulă din categoria dorită. Regulile observate mai des în datele reale au "
        "șansă mai mare să fie alese.",
        "Aplică regula și verifică dacă molecula rezultată este chimic validă.",
        "<b>Respinge</b> dacă rezultatul e identic cu medicamentul de start, sau dacă molecula "
        "a mai fost generată înainte.",
        "<b>Respinge</b> dacă nu arată ca un medicament: sub 10 sau peste 70 de atomi grei, "
        "masă în afara intervalului 150–800 Da, atomi exotici, sau zero inele.",
        "<b>Respinge</b> dacă similaritatea cu părintele iese din intervalul 0.40–0.98.",
        "Dacă a trecut tot: salvează molecula, împreună cu regula folosită, categoria ei și "
        "<b>exact ce atomi s-au schimbat</b> — informație care devine „răspunsul corect” pentru "
        "evaluarea explicabilității.",
    ]))
    s.append(Paragraph(
        "Se încearcă maximum 6 reguli per medicament, apoi se trece la următorul.", caption))

    s.append(Paragraph("2.3 &nbsp;Diferența esențială față de NSGA-II", h2))
    s.append(Paragraph(
        "Cele două generatoare rezolvă aceeași problemă în moduri opuse. Un singur cuvânt le separă: "
        "<b>M filtrează, E optimizează</b>.", body))

    s.append(table([
        hdr("", "Versiunea M (reguli)", "Versiunea E (NSGA-II)"),
        row("Cum decide", "praguri fixe: trece / nu trece", "note comparative între molecule", bold_first=True),
        row("Încercări", "o regulă, o singură șansă", "20 de molecule × 6 generații", bold_first=True),
        row("Îmbunătățire", "nu există — ce iese, iese", "fiecare generație e mai bună", bold_first=True),
        row("Combină modificări", "nu, una singură", "da, mai multe suprapuse", bold_first=True),
        row("Viteză", "foarte mare", "de ordine de mărime mai lentă", bold_first=True),
        row("Rezultat", "molecule corecte, previzibile", "compromisuri rafinate, variate", bold_first=True),
    ], [0.26 * W, 0.37 * W, 0.37 * W]))

    s.append(Paragraph(
        "<b>Analogie:</b> Versiunea M e ca o rețetă — urmezi pașii și obții un rezultat corect, "
        "de fiecare dată același tip. Versiunea E e ca un bucătar care gustă, ajustează, "
        "gustă iar — mai lent, dar ajunge unde rețeta nu poate.", boxed))

    s.append(Paragraph("2.4 &nbsp;Cifre reale din cele două rulări", h2))
    s.append(table([
        hdr("", "Versiunea M", "Versiunea E"),
        row("Molecule generate", "50.000", "13.683", bold_first=True),
        row("Medicamente de start", "58.616", "8.000", bold_first=True),
        row("Similaritate medie", "0.59 – 0.64", "0.51", bold_first=True),
        row("Organizare", "10.000 per categorie × 5", "un singur set", bold_first=True),
        row("Timp de rulare", "minute", "39 minute pe 8 nuclee", bold_first=True),
    ], [0.34 * W, 0.33 * W, 0.33 * W]))
    s.append(Paragraph(
        "Cele cinci categorii ale Versiunii M: bioizosteri, scaffold-hop, halogen-walk, "
        "homologare și altele. Împărțirea permite evaluarea detectorului separat pe fiecare "
        "tip de modificare, în loc de un singur scor global.", caption))

    s.append(PageBreak())

    # ============================================================ PART 3
    s.append(Paragraph("Partea 3 &nbsp;·&nbsp; Se pot combina la antrenare?", h1))

    s.append(Paragraph("3.1 &nbsp;Întrebarea, pusă corect", h2))
    s.append(Paragraph(
        "Riscul nu e că moleculele ar fi „greșite”. Riscul este că fiecare generator ar putea "
        "lăsa o <b>amprentă proprie</b> — un tipar subtil care spune nu „aceasta este o "
        "contrafacere”, ci „aceasta a fost produsă de generatorul M”.", body))
    s.append(Paragraph(
        "Dacă acea amprentă există, un model antrenat pe amestec ar putea învăța să recunoască "
        "<b>generatorul</b> în loc de contrafacere. Ar da rezultate excelente pe datele tale și ar "
        "eșua pe orice altceva.", body))

    s.append(Paragraph("3.2 &nbsp;Cum a fost testat", h2))
    s.append(Paragraph(
        "Experimentul din proiect testează exact acest lucru. Au fost alese opt modificări "
        "elementare (Cl→F, C→N, C→O etc.) pe care <b>ambele</b> generatoare le pot produce. "
        "Apoi:", body))
    s.append(Paragraph(
        "• modelul a fost antrenat <b>numai</b> pe molecule din M;<br/>"
        "• a fost testat pe 320 de molecule M <b>și</b> 320 din E, potrivite între ele "
        "după proprietăți și similaritate;<br/>"
        "• spliturile nu împart molecule, părinți sau schelete, iar pragul de decizie s-a ales "
        "doar pe datele de validare.", bullet))
    s.append(Paragraph(
        "Dacă amprenta de generator ar fi puternică, performanța pe E ar trebui să se prăbușească.",
        body))

    s.append(Paragraph("3.3 &nbsp;Rezultatul", h2))
    s.append(table([
        hdr("Model", "AUC pe M", "AUC pe E", "Diferență"),
        row("Morgan + Random Forest", "0.851", "0.786", "−0.065", bold_first=True),
        row("Morgan + XGBoost", "0.840", "0.769", "−0.071", bold_first=True),
        row("Doar 3 proprietăți simple", "0.517", "0.511", "−0.006", bold_first=True),
    ], [0.40 * W, 0.20 * W, 0.20 * W, 0.20 * W]))
    s.append(Paragraph(
        "Performanța scade pe E, dar <b>moderat</b> — nu se prăbușește. Modelul antrenat doar pe "
        "M recunoaște în bună măsură și moleculele din E. Linia a treia e un martor: un model "
        "care folosește doar trei proprietăți globale stă la nivelul hazardului (0.50) pe ambele, "
        "deci diferența observată nu vine dintr-o simplă deplasare de masă sau lipofilie.", caption))

    s.append(Paragraph("3.4 &nbsp;Partea cea mai interesantă", h2))
    s.append(Paragraph(
        "Diferența nu este distribuită uniform. Când perechile M/E au fost separate după "
        "<b>vecinătatea chimică a atomului modificat</b> — adică ce se află imediat în jurul "
        "schimbării — imaginea s-a schimbat radical:", body))
    s.append(table([
        hdr("Perechile comparate", "Diferența de AUC"),
        row("vecinătate locală <b>identică</b> (≈212 din 320)", "−0.017", bold_first=True),
        row("vecinătate locală <b>diferită</b>", "−0.160", bold_first=True),
    ], [0.58 * W, 0.42 * W]))
    s.append(Paragraph(
        "Când contextul local al modificării este același, diferența aproape dispare. Toată "
        "scăderea agregată provine din perechile unde contextul diferă.", caption))
    s.append(Paragraph(
        "Interpretarea: <b>nu pare să existe o „semnătură de generator”</b> pe care modelul să o "
        "detecteze. Ce diferă este că E explorează <b>contexte chimice</b> pe care M nu le "
        "acoperă. Nu sunt date de alt tip — sunt date din zone pe care M nu le atinge.", boxed))

    s.append(Paragraph("3.5 &nbsp;Răspunsul", h2))
    s.append(Paragraph(
        "<b>Da, se pot combina — și combinarea este chiar de dorit.</b> Tocmai pentru că E acoperă "
        "contexte de editare pe care M le ratează, amestecul acoperă mai mult decât oricare "
        "singur. Dar numai cu patru condiții.", answer))
    s.append(steps([
        "<b>Marchează proveniența fiecărei molecule.</b> Fiecare rând trebuie să știe dacă vine "
        "din M sau din E — fără asta nu poți verifica nimic ulterior.",
        "<b>Ține un generator complet în afara antrenării, ca test.</b> Antrenezi pe amestec, dar "
        "evaluezi și pe un set provenit exclusiv din generatorul nefolosit. Dacă scorul e bun "
        "acolo, modelul a învățat chimie, nu proveniență.",
        "<b>Elimină suprapunerile.</b> Aceeași moleculă, sau molecule care împart părintele ori "
        "scheletul, nu trebuie să apară simultan în antrenare și în test — altfel scorurile sunt "
        "fals optimiste.",
        "<b>Controlează proporția.</b> M are 50.000 de molecule, E are 13.683. Lăsate nebalansate, "
        "contextele aduse de E se pierd în majoritatea M. Fie reponderezi, fie extinzi E.",
    ]))
    s.append(Paragraph(
        "Proiectul are deja scripturile necesare pentru condițiile 1–3; verificarea se reduce la "
        "a le rula și pe setul combinat.", caption))

    s.append(Paragraph("3.6 &nbsp;Ce nu demonstrează încă acest test", h2))
    s.append(Paragraph(
        "Limitele sunt importante și trebuie spuse înainte să întrebe altcineva:", body))
    s.append(Paragraph(
        "• Diferența de performanță <b>nu poate fi atribuită cauzal</b> algoritmului de generare — "
        "perechile potrivite nu egalizează întregul context chimic, doar o parte din el.<br/>"
        "• Toate datele provin din <b>aceeași bibliotecă sursă</b>. Nu există încă o verificare pe "
        "molecule din altă sursă, independentă.<br/>"
        "• Moleculele negative sunt compuși catalogați, <b>nu medicamente confirmate clinic</b>. "
        "Testul măsoară discriminarea între proveniențe de date, nu identificarea falsurilor reale.<br/>"
        "• Transferul a fost testat doar în direcția <b>M → E</b>. Direcția inversă rămâne de verificat.<br/>"
        "• La pragul operațional, rata de alarme false ajunge la 1–3%, peste bugetul țintă de 1%; "
        "AUC nu rezolvă această limită.", bullet))

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"scris {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
