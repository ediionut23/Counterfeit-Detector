"""Build a compact Romanian PDF from the audited final molecule bank."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "final_dataset"
OUT = ROOT / "output/pdf/Raport_dataset_final_2026-09-26.pdf"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
pdfmetrics.registerFont(TTFont("Arial", str(FONT_DIR / "Arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONT_DIR / "Arial Bold.ttf")))

NAVY = colors.HexColor("#142B42")
BLUE = colors.HexColor("#17669A")
PALE = colors.HexColor("#F3F7FA")
LINE = colors.HexColor("#D9E3EA")
GRAY = colors.HexColor("#526579")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleFinal", fontName="Arial-Bold", fontSize=20,
                          leading=25, textColor=NAVY, spaceAfter=9))
styles.add(ParagraphStyle(name="DeckFinal", fontName="Arial", fontSize=10,
                          leading=14, textColor=GRAY, spaceAfter=14))
styles.add(ParagraphStyle(name="HFinal", fontName="Arial-Bold", fontSize=12.5,
                          leading=17, textColor=NAVY, spaceBefore=13,
                          spaceAfter=6, keepWithNext=True))
styles.add(ParagraphStyle(name="BodyFinal", fontName="Arial", fontSize=9,
                          leading=13.4, textColor=NAVY, spaceAfter=7))
styles.add(ParagraphStyle(name="SmallFinal", fontName="Arial", fontSize=7.7,
                          leading=10.5, textColor=GRAY, spaceAfter=5))
styles.add(ParagraphStyle(name="CellFinal", fontName="Arial", fontSize=8,
                          leading=10.6, textColor=NAVY))
styles.add(ParagraphStyle(name="HeadCellFinal", fontName="Arial-Bold",
                          fontSize=8, leading=10, textColor=colors.white))


def j(name: str) -> dict:
    return json.loads((BANK / name).read_text())


def n(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def par(value: str, style: str = "BodyFinal") -> Paragraph:
    return Paragraph(value, styles[style])


def heading(value: str) -> Paragraph:
    return par(escape(value), "HFinal")


def tab(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    data = [[par(escape(str(x)), "HeadCellFinal") for x in headers]]
    data += [[par(escape(str(x)), "CellFinal") for x in row] for row in rows]
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, LINE),
    ]))
    result.spaceAfter = 5
    return result


class Document(BaseDocTemplate):
    def __init__(self, filename: Path):
        super().__init__(str(filename), pagesize=A4, leftMargin=19 * mm,
                         rightMargin=19 * mm, topMargin=20 * mm,
                         bottomMargin=18 * mm, title="Dataset final - Counterfeit Detector",
                         author="Counterfeit Detector project")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="main", frames=frame,
                                           onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        width, height = A4
        canvas.setStrokeColor(LINE)
        canvas.line(19 * mm, height - 13 * mm, width - 19 * mm, height - 13 * mm)
        canvas.line(19 * mm, 13 * mm, width - 19 * mm, 13 * mm)
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(19 * mm, 9 * mm, "Counterfeit Detector | Raport dataset final")
        canvas.drawRightString(width - 19 * mm, 9 * mm, f"Pagina {doc.page}")
        canvas.restoreState()


def main() -> None:
    summary = j("summary.json")
    audit = j("audit.json")
    conformers = j("conformers_report.json")
    conformer_audit = j("conformers_audit.json")
    training = j("examples/mixed_20k/summary.json")
    training_audit = j("examples/mixed_20k/audit.json")
    with (BANK / "subcategories/quality.csv").open(newline="") as stream:
        modules = list(csv.DictReader(stream))
    if not audit["passed"] or not conformer_audit["passed"] or not training_audit["passed"]:
        raise ValueError("One or more final audits failed")
    if conformers["missing"]:
        raise ValueError("3D coverage is incomplete")

    story = [
        par("Datasetul final pentru detectarea analogilor structurali", "TitleFinal"),
        par("Rezultate auditate | 26 septembrie 2026 | Banca modulară M + E", "DeckFinal"),
        par("<b>Ce am construit.</b> O bancă de analogi moleculari generați față de molecule de referință, cu identitate globală unică, relații părinte–analog, proveniență, subcategorii selectabile și conformeri 3D. M înseamnă reguli MMP minate; E înseamnă căutare evolutivă. Aceeași structură poate fi obținută prin ambele metode."),
        heading("1. Dimensiune și proveniență"),
        tab(["Indicator", "Rezultat"], [
            ["Analogi unici după filtre", n(summary["n_unique_counterfeits"])],
            ["Relații părinte–analog", n(summary["n_distinct_relations"])],
            ["Molecule de referință / eligibile pentru matching",
             f"{n(summary['n_authentic_reference'])} / {n(summary['n_authentic_eligible_for_training'])}"],
            ["Subcategorii selectabile", n(summary["n_subcategories"])],
            ["Schelete Bemis–Murcko ale analogilor", n(summary["n_unique_child_scaffolds"])],
            ["Audit structural independent", "TRECUT"],
        ], [314, 167]),
        par("Numărul de molecule unice nu se obține însumând subcategoriile: o moleculă poate avea mai multe relații valide. Fiecare sursă, scriptul de construire, catalogul de reguli și mediul software sunt înregistrate în summary.json.", "SmallFinal"),
        heading("2. Contribuția metodelor"),
        tab(["Metode prezente pentru moleculă", "Molecule unice"], [
            ["Doar M", n(summary["method_membership"].get("M", 0))],
            ["Doar E", n(summary["method_membership"].get("E", 0))],
            ["M și E", n(summary["method_membership"].get("EM", 0))],
        ], [314, 167]),
        heading("3. Calitate chimică"),
        tab(["Măsură", "Mediana", "P05", "P95"], [
            [label, str(summary[key]["median"]), str(summary[key]["p05"]),
             str(summary[key]["p95"])]
            for label, key in (
                ("QED", "molecule_qed_distribution"),
                ("SA score", "molecule_sa_distribution"),
                ("Tanimoto față de părinte", "relation_similarity_distribution"),
            )
        ], [230, 83, 83, 85]),
        par("Filtre recalculate din structuri: un singur component, 10-70 atomi grei, 150-800 Da, QED >= 0,35, SA <= 3,8, fără alerte PAINS/Brenk și similaritate Tanimoto Morgan radius 3 între 0,4 și 0,98. Au fost excluse moleculele prezente în biblioteca de referință.", "SmallFinal"),
        heading("4. Măștile atomilor modificați"),
        tab(["Stare", "Relații"], [
            [status.replace("_", " "), n(count)]
            for status, count in summary["edit_status_relation_counts"].items()
        ], [314, 167]),
        par("Numai măștile exact single atom au editarea locală verificată printr-o operație inversă unică. Măștile obținute prin substructură comună sau moștenite din sursă sunt estimări și se evaluează separat în studiile de explicabilitate.", "SmallFinal"),
        heading("5. Module gata de selectat"),
    ]
    ranked = sorted(modules, key=lambda r: int(r["unique_molecules"]), reverse=True)[:12]
    story.append(tab(["Modul (primele 12 după mărime)", "Molecule", "QED median", "Similaritate mediană"], [
        [row["module"], n(int(row["unique_molecules"])), row["median_qed"],
         row["median_similarity"]] for row in ranked
    ], [241, 76, 78, 86]))
    story += [
        par("Toate modulele și scorurile lor sunt în final_dataset/subcategories/quality.csv; index.json indică fișierul CSV aferent fiecăruia. Modulele se pot suprapune între ele.", "SmallFinal"),
        PageBreak(),
        heading("6. Geometrii 3D și exemplu de antrenare"),
        tab(["Verificare", "Rezultat"], [
            ["Structuri cu conformer 3D",
             f"{n(conformers['covered_total'])} / {n(conformers['wanted_unique_smiles'])}"],
            ["Acoperire / lipsă", f"{100 * conformers['coverage']:.2f}% / {n(conformers['missing'])}"],
            ["Auditul SDF", "TRECUT"],
            ["Exemplu modular M + E", f"{n(training['rows'])} rânduri echilibrate"],
            ["Test fără suprapunere de scaffold propriu", "0 suprapuneri; audit TRECUT"],
        ], [314, 167]),
        par("Exemplul a fost derivat din M/halogen-walk și E/unconditioned. Selecția elimină duplicatele, potrivește proprietățile moleculelor de referință, separă scaffoldurile părinților și filtrează testul pentru scaffolduri ale analogilor nevăzute la antrenare sau validare. Este un test funcțional al selecției, nu un rezultat de performanță pentru un model nou."),
        tab(["Split", "Referințe (0)", "Analogi (1)", "Total"], [
            [split, n(int(training["split_label_counts"][split]["0"])),
             n(int(training["split_label_counts"][split]["1"])),
             n(sum(int(x) for x in training["split_label_counts"][split].values()))]
            for split in ("train", "val", "test")
        ], [145, 112, 112, 112]),
        heading("7. Interpretare și cercetare viitoare"),
        par("Eticheta «analog generat» este relativă la un părinte din biblioteca publică. Nu dovedește existența unui produs contrafăcut în comerț, sinteza moleculei, activitatea ei biologică sau statutul de medicament aprobat al părintelui. Pentru o întrebare nouă de cercetare, perechile foarte apropiate chimic pot fi testate în coculturi umane pentru a afla dacă o schimbare moleculară mică produce o schimbare mare în comunicarea dintre tipuri de celule. Aceasta este o ipoteză experimentală, nu o etichetă prezentă în dataset."),
        heading("8. Studiu pilot propus pentru comunicarea celulară"),
        tab(["Etapă", "Măsură și control"], [
            ["Selecție", "12-24 perechi apropiate chimic, din clase M și E; confirmare identitate, disponibilitate, solubilitate și activitate documentată a părintelui."],
            ["Model biologic", "Coculturi umane cu două tipuri de celule la început; apoi organoizi tumorali, fibroblaste și celule imune, cu donori independenți."],
            ["Rezultat", "Viabilitate, compoziție celulară, proteine secretate și rețele spațiale ligand-receptor; validarea interacțiunilor prin blocare sau perturbare."],
            ["Analiză", "Diferență reproductibilă a rețelei între părinte și analog, ajustată pentru doză și viabilitate; donorul sau cultura este replica biologică."],
        ], [120, 361]),
        par("Literatură de plecare: COMMOT (Nature Methods 2023, DOI 10.1038/s41592-022-01728-4) pentru inferența spațială; MIX-Seq (Nature Communications 2020, DOI 10.1038/s41467-020-17440-w) pentru răspunsul unicelular la medicamente; MIRO (Nature Communications 2025, DOI 10.1038/s41467-025-56275-1) pentru modelul uman tumoră–stromă–imunitate. Ideea nu are încă date biologice proprii și necesită o revizuire sistematică înainte de o afirmație de prioritate.", "SmallFinal"),
        par("Fișiere de referință: final_dataset/README.md, REPORT.md, summary.json, audit.json, subcategories/quality.csv, conformers_report.json și RESEARCH_IDEA_CELL_COMMUNICATION.md.", "SmallFinal"),
        Spacer(1, 5),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Document(OUT).build(story)
    print(OUT)


if __name__ == "__main__":
    main()
