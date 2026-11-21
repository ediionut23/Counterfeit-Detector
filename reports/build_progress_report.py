"""Build the Romanian project progress PDF from the saved benchmark artifacts."""

from __future__ import annotations

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
    BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph, Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/Raport_progres_detector_contrafaceri_2026-09-26.pdf"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
pdfmetrics.registerFont(TTFont("Arial", str(FONT_DIR / "Arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONT_DIR / "Arial Bold.ttf")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")

NAVY = colors.HexColor("#142B42")
BLUE = colors.HexColor("#17669A")
LIGHT = colors.HexColor("#EAF2F7")
PALE = colors.HexColor("#F6F9FB")
GRAY = colors.HexColor("#536575")
LINE = colors.HexColor("#D9E3EA")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleRO", fontName="Arial-Bold", fontSize=21,
                          leading=26, textColor=NAVY, spaceAfter=10))
styles.add(ParagraphStyle(name="DeckRO", fontName="Arial", fontSize=10.4,
                          leading=15, textColor=GRAY, spaceAfter=15))
styles.add(ParagraphStyle(name="H1RO", fontName="Arial-Bold", fontSize=13,
                          leading=18, textColor=NAVY, spaceBefore=15, spaceAfter=7,
                          keepWithNext=True))
styles.add(ParagraphStyle(name="H2RO", fontName="Arial-Bold", fontSize=10.4,
                          leading=14, textColor=BLUE, spaceBefore=10, spaceAfter=5,
                          keepWithNext=True))
styles.add(ParagraphStyle(name="BodyRO", fontName="Arial", fontSize=9.1,
                          leading=13.5, textColor=NAVY, spaceAfter=7,
                          allowWidows=0, allowOrphans=0))
styles.add(ParagraphStyle(name="SmallRO", fontName="Arial", fontSize=7.7,
                          leading=10.5, textColor=GRAY, spaceAfter=5))
styles.add(ParagraphStyle(name="BulletRO", parent=styles["BodyRO"],
                          leftIndent=12, firstLineIndent=-9, spaceAfter=5))
styles.add(ParagraphStyle(name="CellRO", fontName="Arial", fontSize=7.5,
                          leading=10, textColor=NAVY))
styles.add(ParagraphStyle(name="CellHeadRO", fontName="Arial-Bold", fontSize=7.4,
                          leading=9.5, textColor=colors.white))


def j(path: str):
    return json.loads((ROOT / path).read_text())


def p(text: str, style="BodyRO"):
    return Paragraph(text, styles[style])


def heading(text: str, level=1):
    return p(escape(text), "H1RO" if level == 1 else "H2RO")


def body(text: str):
    return p(text)


def bullet(text: str):
    return p("•  " + text, "BulletRO")


def table(headers: list[str], rows: list[list[str]], widths: list[float] | None = None):
    data = [[p(escape(str(x)), "CellHeadRO") for x in headers]]
    data += [[p(escape(str(x)), "CellRO") for x in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, LINE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE),
    ]))
    t.spaceAfter = 2
    return t


def fmt(x: float) -> str:
    return f"{x:.3f}"


def mean_sd(obj: dict, digits=3) -> str:
    return f"{obj['mean']:.{digits}f} ± {obj['std']:.{digits}f}"


def pct_sd(obj: dict) -> str:
    return f"{100 * obj['mean']:.1f}% ± {100 * obj['std']:.1f} pp"


story = []
add = story.append

add(p("Raport de progres: benchmark pentru detectarea contrafacerilor moleculare", "TitleRO"))
add(p("Starea proiectului la 26 septembrie 2026  |  Date, evaluare, explicații și limite", "DeckRO"))
add(body("<b>Ideea proiectului.</b> Un exemplu contrafăcut este o moleculă derivată dintr-un medicament autentic printr-o modificare chimică. Eticheta descrie această relație cu molecula părinte; nu este o proprietate fizică măsurată a moleculei. Scopul benchmarkului este să testeze dacă un model recunoaște modificările, inclusiv pe familii moleculare și tipuri de editare noi."))
add(body("<b>Rezumatul progresului.</b> Am verificat și refăcut cele 90 de seturi de date, am restaurat biblioteca 3D, am reparat adnotări atomice verificabile, am rulat modele de referință și GNN-uri, apoi am măsurat robustețea la dezechilibru, mărime, diversitatea scaffoldurilor și transferul M către E. Am auditat separat explicațiile atomice și limitele splitului."))

add(heading("1. Ce există acum în repository"))
add(table(["Componentă", "Stare și dimensiune"], [
    ["Date tabelare", "90 fișiere CSV echilibrate; 781.692 rânduri în total; train/val/test fixate."],
    ["Sursă chimică", "58.616 molecule autentice pentru minerit; 3.257 reguli MMP clasificate."],
    ["Versiuni", "R: 21 reguli istorice; M: reguli minate și categorii; E: căutare evolutivă."],
    ["Geometrii 3D", "249.803 din 249.975 molecule unice au conformer; 172 lipsesc; acoperire 99,9312%."],
    ["Mediu și cod", "Mediu Python 3.12, dependențe fixate, scripturi de construire, audit, antrenare și predicție."],
], [120, 360]))
add(p("Surse: benchmark/index.json, benchmark/conformers_report.json, mined_transformations.json, benchmark/environment.json.", "SmallRO"))

add(heading("2. Cum sunt construite datele"))
for text in [
    "<b>A. Reguli MMP.</b> Molecule autentice sunt fragmentate în perechi care diferă într-o regiune mică. Transformările astfel observate primesc suport statistic și sunt grupate în bioizostere, schimbări de halogeni, omologare și modificări de scaffold.",
    "<b>B. Exemple M și E.</b> Versiunea M aplică regulile minate moleculelor părinte. Versiunea E folosește mutație, recombinare și selecție multiobiectiv NSGA-II pentru a căuta compuși similari, plauzibili și uneori dificili pentru detector.",
    "<b>C. Asamblare.</b> Exemplele autentice sunt potrivite aproximativ după masă, număr de atomi grei și lungimea SMILES. Fiecare tabel are clase 1:1 și splituri train/val/test echilibrate. Coloana scaffold a falsurilor reprezintă scaffoldul părintelui.",
    "<b>D. Conformeri.</b> Biblioteca 3D a fost restaurată din arhive și împărțită în cinci fișiere gzip. Geometriile sunt obținute cu ETKDGv3 și, când se poate, relaxare MMFF94. Cele 172 molecule fără conformer trebuie excluse sau raportate în studiile geometrice.",
]: add(bullet(text))

add(heading("3. Reparații și reproductibilitate"))
add(body("Am corectat poolul de molecule autentice folosit la potrivire, am refăcut toate tabelele și am verificat absența moleculelor și scaffoldurilor părinte comune între splituri. Am fixat corecțiile de mască într-un manifest versionat, ca un rebuild independent să poată reproduce exact rezultatul."))
add(table(["Audit", "Rezultat"], [
    ["Integritatea seturilor", "90/90 tabele trec auditul; 781.692 rânduri; clase echilibrate în fiecare split."],
    ["Rebuild independent", "90/90 CSV identice; rezumatele și indexul identice."],
    ["Editări verificabile", "23.826 perechi părinte-copil cu substituție/adăugare unică reversibilă; 49.741 rânduri."],
    ["Măști reparate", "40.298 rânduri corectate; 49.741/49.741 măști verificabile concordă acum."],
    ["Halogeni", "6.160 rânduri cu inversare exactă auditate, fără nepotriviri."],
], [156, 324]))
add(body("Verificarea de editare exactă acoperă doar o submulțime. În E, 3.578/46.251 falsuri (7,7%) au editare reversibilă cu un singur atom; în scaffold-hop, 654/32.166 (2,0%). Restul măștilor sunt aproximative și nu pot servi drept adevăr de referință pentru localizare atomică exactă."))
add(p("Surse: benchmark/reproduction_audit.json, benchmark/verified_edits/summary.json, benchmark/halogen_mask_audit.json.", "SmallRO"))

add(heading("4. Rezultatele principale pe setul evolutiv E"))
add(body("Comparația directă folosește aceleași 74.002 rânduri de train și 9.250 de test, semințele 42-44. GIN își selectează epoca pe validare; RF și XGBoost au parametri fixați și nu învață pe validare. AUC măsoară ordonarea scorurilor; FNR este fracția de falsuri ratate la pragul 0,5."))
e = j("benchmark/gnn_comparison/Evolved_summary.json")["models"]
add(table(["Model", "AUC ROC", "FNR", "Precizie", "F1"], [
    [label, mean_sd(e[key]["metrics"]["roc_auc"]), mean_sd(e[key]["metrics"]["fnr"]),
     mean_sd(e[key]["metrics"]["precision"]), mean_sd(e[key]["metrics"]["f1"])]
    for key, label in [("gin", "GIN"), ("rf", "Morgan + RF"), ("xgboost", "Morgan + XGBoost")]
], [116, 91, 91, 91, 91]))
add(body("<b>Interpretare.</b> GIN și RF au AUC apropiat (~0,884-0,885), dar RF ratează mai puține falsuri la pragul implicit. XGBoost are AUC mai mic în această configurație. Acestea sunt setări fixe, fără buget egal de optimizare a hiperparametrilor; concluzia este despre aceste implementări, nu despre superioritatea generală a unei arhitecturi."))
add(body("Un RF anterior, antrenat pe train+val, are AUC 0,902 pe E; nu este direct comparabil cu tabelul train-only. Bazele pe proprietăți fizicochimice sunt aproape de hazard pe seturile mari, dar unele subcategorii mici păstrează scurtături măsurabile."))
add(p("Sursa: benchmark/gnn_comparison/Evolved_summary.json și baseline_results.json.", "SmallRO"))

add(heading("5. Transfer între versiuni și auditul scaffoldului"))
add(body("Am construit un set M din patru categorii, cu deduplicare, eliminarea etichetelor contradictorii și split global. Modelele antrenate pe M au fost evaluate pe un subset E de 572 molecule, câte 286 pe clasă, fără SMILES sau scaffold părinte văzut în sursa M. Rezultatele de mai jos sunt mediile a trei semințe."))
cv = j("benchmark/cross_version/baseline_summary.json")["models"]
add(table(["M → E (572 test)", "AUC ROC", "FNR", "F1"], [
    ["Morgan + RF", mean_sd(cv["fingerprint_rf"]["roc_auc"]), mean_sd(cv["fingerprint_rf"]["false_negative_rate"]), mean_sd(cv["fingerprint_rf"]["f1"])],
    ["Morgan + XGBoost", mean_sd(cv["fingerprint_xgboost"]["roc_auc"]), mean_sd(cv["fingerprint_xgboost"]["false_negative_rate"]), mean_sd(cv["fingerprint_xgboost"]["f1"])],
], [170, 104, 103, 103]))
add(body("<b>Limită descoperită.</b> Splitul inițial este disjunct pe scaffoldul părintelui, însă scaffoldul moleculei copil poate apărea în train/val. Fenomenul afectează 55/90 de seturi. În E, 1.623/9.250 molecule test (17,55%) au propriul scaffold prezent în train/val; 1.333 dintre acestea sunt falsuri. Am creat teste suplimentare cu scaffoldul efectiv nevăzut."))
strict = j("benchmark/strict_child_scaffold/model_summary.json")["models"]
add(table(["Model E", "Test inițial AUC / FNR (9.250)", "Scaffold copil nou AUC / FNR (6.584)"], [
    [label,
     f"{fmt(strict[key]['original']['roc_auc']['mean'])} / {fmt(strict[key]['original']['fnr']['mean'])}",
     f"{fmt(strict[key]['strict']['roc_auc']['mean'])} / {fmt(strict[key]['strict']['fnr']['mean'])}"]
    for key, label in [("gin", "GIN"), ("rf", "Morgan + RF"), ("xgboost", "Morgan + XGBoost")]
], [116, 182, 182]))
add(body("AUC nu se prăbușește pe subsetul mai strict. Filtrarea schimbă însă și compoziția chimică, deci diferența nu estimează izolat efectul suprapunerii de scaffold. Transferul M → E are și un subset mai strict, 448 molecule; RF obține AUC 0,866 și FNR 0,292, iar XGBoost 0,811 și 0,409."))
add(p("Surse: benchmark/cross_version/, benchmark/child_scaffold_overlap_audit.json, benchmark/strict_child_scaffold/model_summary.json.", "SmallRO"))

add(heading("6. Reprezentarea atomilor: ablație pentru inele"))
add(body("Am adăugat șase indicatoare atomice pentru apartenența la inele de mărime 3-8, păstrând GIN, splitul și semințele. Efectul depinde de sarcina chimică."))
add(table(["Sarcină", "GIN AUC / FNR", "GIN + inele AUC / FNR", "Test"], [
    ["Cl → F", "0,976 / 0,057", "0,967 / 0,105", "542"],
    ["Redimensionare inel", "0,968 / 0,117", "0,981 / 0,075", "160"],
    ["Schimbare heteroatom inel", "0,979 / 0,019", "0,979 / 0,023", "556"],
], [147, 132, 143, 58]))
add(body("Pe redimensionarea inelului, adăugarea informației despre mărimea inelului îmbunătățește AUC și FNR în fiecare dintre cele trei semințe. Pe Cl → F le înrăutățește, iar pe schimbarea heteroatomului efectul este mic. Testul de redimensionare are numai 160 molecule, deci rezultatul este preliminar."))
add(p("Sursa: benchmark/gnn_comparison/*summary.json; medii pe semințele 42-44.", "SmallRO"))

add(heading("7. Dezechilibru și condiții de screening"))
add(body("În E am păstrat 37.001 exemple autentice de train și am redus falsurile de train la 3.700, 370 sau 37. Validarea și testul au rămas echilibrate și neschimbate. Tabelul arată o singură sămânță, la pragul 0,5; un prag recalibrat pe validare poate reduce FNR, dar poate crește alarmele false."))
add(table(["Falsuri : autentice în train", "RF AUC", "RF FNR", "XGB AUC", "XGB FNR"], [
    ["1:1", "0,884", "0,142", "0,834", "0,175"],
    ["1:10", "0,804", "0,995", "0,809", "0,974"],
    ["1:100", "0,660", "1,000", "0,706", "1,000"],
    ["1:1000", "0,547", "1,000", "0,585", "1,000"],
], [160, 80, 80, 80, 80]))
add(body("La un prag ales pentru cel mult 1% alarme false pe validare, FNR pe test este 0,981 pentru RF și 0,974 pentru XGBoost la 1:1000. Chiar și cu limita de alarme, aproape toate falsurile sunt ratate. Separat, o proiecție Bayes pe prevalență arată că precizia RF din testul echilibrat nu se păstrează când falsurile sunt rare; aceasta este o proiecție, nu un test de teren."))
add(p("Surse: benchmark/imbalance/baseline_summary.json, benchmark/prevalence_analysis.json.", "SmallRO"))

add(heading("8. Mărimea și diversitatea datelor de antrenare"))
add(body("Selectarea a 3.700 falsuri mici sau mari, în locul unui eșantion aleatoriu 1:10, reduce AUC pe același test: RF 0,804 → 0,704 / 0,676; XGBoost 0,809 → 0,615 / 0,663. Compoziția chimică se schimbă odată cu mărimea, deci testul indică sensibilitate la selecție, nu un efect cauzal pur al mărimii."))
sd = j("benchmark/scaffold_diversity/baseline_summary.json")["models"]
add(table(["Model", "10 scaffolduri AUC", "3.700 scaffolduri AUC", "FNR la 0,5: mică / mare"], [
    ["Morgan + RF", mean_sd(sd["fingerprint_rf"]["low"]["auc"]), mean_sd(sd["fingerprint_rf"]["high"]["auc"]), "0,926 / 0,181"],
    ["Morgan + XGBoost", mean_sd(sd["fingerprint_xgboost"]["low"]["auc"]), mean_sd(sd["fingerprint_xgboost"]["high"]["auc"]), "0,916 / 0,174"],
], [127, 115, 121, 117]))
add(KeepTogether([
    body("În testul de diversitate, cele două condiții au aceleași 3.700 exemple autentice și același număr de falsuri, cu aceleași trei benzi grosiere de mărime; validarea și testul sunt identice. AUC crește cu aproximativ 0,10 când falsurile provin din 3.700 scaffolduri în loc de 10. Chimia diferă totuși între subseturi, iar un buget de 1% alarme false lasă FNR de circa 0,95-0,97 în ambele condiții."),
    p("Surse: benchmark/graph_size/baseline_summary.json și benchmark/scaffold_diversity/baseline_summary.json.", "SmallRO"),
]))

add(heading("9. Explicații atomice și verificări pe perechi"))
add(body("Am salvat checkpointuri GIN și predicții pentru toate cele trei semințe E. Pentru GIN cu sămânța 42, doar 292/4.625 falsuri test au un loc de editare cu un singur atom verificat fără ambiguitate. Am comparat saliența gradient × intrare cu atomul editat și regiunea acestuia."))
add(table(["Măsură pe cele 292 perechi", "Observat", "Așteptare aleatoare"], [
    ["Atomul exact pe locul 1", "2,4%", "4,4%"],
    ["Regiunea editării pe locul 1", "25,0%", "9,5%"],
    ["Regiunea în primele 2 locuri", "33,9%", "18,3%"],
], [240, 120, 120]))
add(body("Saliența găsește regiunea mai des decât alegerea întâmplătoare, dar găsește slab atomul exact. Mascarea vectorului de trăsături al atomului cel mai salienț schimbă probabilitatea cu 0,575 în medie, față de 0,459 pentru un atom aleatoriu; vectorul nul este însă în afara distribuției chimice, deci această probă de fidelitate este limitată."))
pair = j("benchmark/pair_ordering/summary.json")["models"]
add(table(["Model, E test", "Copil scor > părinte", "Ambele clase corecte la 0,5", "Perechi"], [
    [label, pct_sd(pair[key]['all_test_pairs']['child_scores_higher_fraction']),
     pct_sd(pair[key]['all_test_pairs']['pair_correct_at_half_fraction']), "4.625"]
    for key, label in [("gin", "GIN"), ("rf", "Morgan + RF"), ("xgboost", "Morgan + XGBoost")]
], [113, 137, 174, 56]))
add(body("Această comparație folosește perechi chimic valide, identice în cele trei semințe 42-44; variația arată efectul seminței, nu al unui test nou. Modelul vede separat fiecare moleculă, fără identitatea părintelui: ordonarea perechii măsoară sensibilitatea la transformare, nu cauza internă a deciziei. Pe subsetul de 292 editări exacte, copilul are scor mai mare în medie în 86,5% dintre perechi pentru GIN, 87,1% pentru RF și 83,5% pentru XGBoost."))
add(p("Surse: benchmark/explanations/evolved_gin_seed42/, benchmark/pair_ordering/summary.json.", "SmallRO"))

add(heading("10. Ce este confirmat și ce urmează"))
add(table(["Confirmat de artefacte", "Încă deschis"], [
    ["90 seturi reproductibile; splituri și clase auditate; 99,9312% acoperire 3D.",
     "Conformeri pentru 172 molecule și validarea geometrică a unor experimente 3D."],
    ["RF concurează cu GIN pe E la setările fixe și ratează mai puține falsuri la pragul 0,5.",
     "Comparație cu buget egal de optimizare, GNN-uri eterogene și transformere preantrenate."],
    ["Scăderea performanței la puține falsuri și rolul diversității train sunt măsurate.",
     "Teste pe alte generații și protocoale de prevalență/alarme reprezentative pentru utilizare reală."],
    ["Editările exacte sunt delimitate; explicațiile sunt evaluate numai pe submulțimea sigură.",
     "Metode XAI suplimentare, fidelitate cu perturbări chimic valide și acoperire mai mare a adnotărilor."],
], [240, 240]))
add(body("<b>Concluzie de etapă.</b> Proiectul are acum o bază de date amplă, auditată și reproductibilă, plus rezultate C1-C4 inițiale. Principalul risc metodologic cunoscut este diferența dintre scaffoldul părintelui folosit la split și scaffoldul copilului; testele stricte au fost create și raportate separat. Rezultatele sunt experimente de cercetare pe molecule generate, nu o evaluare clinică sau un detector gata de utilizare."))
add(p("Fișiere de pornire pentru reproducere: WIKI.md, benchmark/README.md, benchmark/environment.json și scripturile audit_*.py / train_*.py din rădăcina repository-ului.", "SmallRO"))


class ProgressDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(filename, pagesize=A4, leftMargin=18 * mm,
                         rightMargin=18 * mm, topMargin=20 * mm,
                         bottomMargin=18 * mm, title="Raport de progres - Counterfeit Detector",
                         author="Counterfeit Detector project")
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="main", frames=frame,
                                           onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, width, height, fill=1, stroke=0)
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.6)
        canvas.line(18 * mm, height - 13 * mm, width - 18 * mm, height - 13 * mm)
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(18 * mm, height - 10.5 * mm, "COUNTERFEIT DETECTOR  /  RAPORT DE PROGRES")
        canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
        canvas.drawString(18 * mm, 9 * mm, "26.09.2026  |  Rezultate din repository")
        canvas.drawRightString(width - 18 * mm, 9 * mm, f"Pagina {doc.page}")
        canvas.restoreState()


OUT.parent.mkdir(parents=True, exist_ok=True)
ProgressDoc(str(OUT)).build(story)
print(OUT)
