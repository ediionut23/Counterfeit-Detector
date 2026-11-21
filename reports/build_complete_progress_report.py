"""Create the complete Romanian project-status report from audited repository results."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/Raport_complet_proiect_2026-09-29.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
pdfmetrics.registerFont(TTFont("ArialRO", str(FONT_DIR / "Arial.ttf")))
pdfmetrics.registerFont(TTFont("ArialRO-Bold", str(FONT_DIR / "Arial Bold.ttf")))
pdfmetrics.registerFontFamily("ArialRO", normal="ArialRO", bold="ArialRO-Bold")

NAVY = colors.HexColor("#142B42")
BLUE = colors.HexColor("#17669A")
TEAL = colors.HexColor("#087A80")
PALE = colors.HexColor("#F3F7FA")
LIGHT = colors.HexColor("#E8F2F5")
MUTED = colors.HexColor("#526575")
LINE = colors.HexColor("#CFDCE4")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleRO", fontName="ArialRO-Bold", fontSize=21, leading=25,
                          textColor=NAVY, spaceAfter=8))
styles.add(ParagraphStyle(name="DeckRO", fontName="ArialRO", fontSize=10, leading=14,
                          textColor=MUTED, spaceAfter=11))
styles.add(ParagraphStyle(name="LeadRO", fontName="ArialRO", fontSize=10.2, leading=15,
                          textColor=NAVY, spaceAfter=9))
styles.add(ParagraphStyle(name="H1RO", fontName="ArialRO-Bold", fontSize=13, leading=17,
                          textColor=NAVY, spaceBefore=13, spaceAfter=6, keepWithNext=True))
styles.add(ParagraphStyle(name="H2RO", fontName="ArialRO-Bold", fontSize=10.2, leading=14,
                          textColor=BLUE, spaceBefore=9, spaceAfter=4, keepWithNext=True))
styles.add(ParagraphStyle(name="BodyRO", fontName="ArialRO", fontSize=8.9, leading=12.8,
                          textColor=NAVY, spaceAfter=7, allowWidows=0, allowOrphans=0))
styles.add(ParagraphStyle(name="SmallRO", fontName="ArialRO", fontSize=7.5, leading=10.1,
                          textColor=MUTED, spaceAfter=5))
styles.add(ParagraphStyle(name="CellRO", fontName="ArialRO", fontSize=7.5, leading=10.3,
                          textColor=NAVY))
styles.add(ParagraphStyle(name="CellHeadRO", fontName="ArialRO-Bold", fontSize=7.5,
                          leading=9.7, textColor=colors.white))
styles.add(ParagraphStyle(name="KpiRO", fontName="ArialRO-Bold", fontSize=13, leading=16,
                          textColor=TEAL, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="KpiLabelRO", fontName="ArialRO", fontSize=7.5, leading=10,
                          textColor=MUTED, alignment=TA_CENTER))


def para(text: str, style: str = "BodyRO") -> Paragraph:
    return Paragraph(text, styles[style])


def h1(text: str) -> Paragraph:
    return para(escape(text), "H1RO")


def h2(text: str) -> Paragraph:
    return para(escape(text), "H2RO")


def source(text: str) -> Paragraph:
    return para("Surse în repository: " + escape(text), "SmallRO")


def tbl(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    cells = [[para(escape(str(value)), "CellHeadRO") for value in headers]]
    cells.extend([[para(escape(str(value)), "CellRO") for value in row] for row in rows])
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, LINE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE),
    ]))
    table.spaceAfter = 5
    return table


def note(text: str) -> Table:
    box = Table([[para(text, "BodyRO")]], colWidths=[174 * mm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("LINEBEFORE", (0, 0), (0, -1), 2, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    box.spaceAfter = 9
    return box


story = []
add = story.append

add(para("Raport complet de progres", "TitleRO"))
add(para("Benchmark molecular, dataset final, experimente și direcția disertației<br/>"
         "Situația documentată la 29 septembrie 2026", "DeckRO"))
add(HRFlowable(width="100%", thickness=1, color=LINE, spaceAfter=12))
add(para("Acest raport sintetizează munca documentată în repository de la pornirea proiectului. "
         "Am păstrat separat datele generate, rezultatele efectiv măsurate și experimentele planificate. "
         "Cifrele sunt cele din fișierele de audit și din rapoartele salvate, nu estimări noi.", "LeadRO"))

kpis = [
    [("216.165", "analogi sintetici unici"), ("222.772", "relații părinte-analog"),
     ("84", "module selectabile")],
    [("90", "seturi benchmark vechi"), ("3.257", "reguli MMP minate"),
     ("28.166", "relații cu edit exact")],
]
for row in kpis:
    blocks = [Table([[para(n, "KpiRO")], [para(label, "KpiLabelRO")]], colWidths=[58 * mm])
              for n, label in row]
    t = Table([blocks], colWidths=[58 * mm] * 3)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE),
                           ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    add(t)
    add(Spacer(1, 3 * mm))

add(h1("1. Evoluția lucrării"))
add(tbl(["Etapă", "Ce s-a realizat", "Rezultat verificabil"], [
    ["Înțelegerea repo-ului și a datelor", "Au fost urmărite WIKI.md, fluxul de generare, schemele CSV, arhivele 3D și construcția spliturilor.", "Harta clară a procesului: surse publice -> reguli MMP / căutare evolutivă -> benchmark -> audit -> modele."],
    ["Reparație și evaluare benchmark", "Rebuild, audit de măști și scaffold, comparații RF/XGBoost/GNN, stres pe dezechilibru și explicații.", "90/90 CSV reproduse identic; rezultate comparabile în snapshotul vechi."],
    ["Extinderea datasetului", "Au fost generate extensii M și E, filtrate și reunite într-o bancă modulară, auditată structural.", "216.165 structuri unice și 84 module; conformeri 3D pentru ținta finală."],
    ["Cercetarea pentru disertație", "Au fost construite controale MMP apropiate și splituri M -> E cu editări comparabile; s-a redactat protocolul HPC.", "Piloturi măsurate și întrebare de cercetare formulată; campania mare de tuning rămâne de executat."],
], [38 * mm, 69 * mm, 67 * mm]))
add(note("<b>Definiția etichetei:</b> `pozitiv` înseamnă analog virtual generat față de un părinte; "
         "`negativ` înseamnă moleculă din biblioteca publică de referință. Niciuna dintre etichete "
         "nu confirmă un produs farmaceutic fals sau eficacitatea biologică."))
add(source("WIKI.md; benchmark/README.md; final_dataset/REPORT.md; DISSERTATION_PUBLICATION_DIRECTION.md."))

add(PageBreak())
add(h1("2. Cum s-au creat și organizat moleculele"))
add(para("Am pornit de la molecule publice ChEMBL/PubChem. Din 58.616 molecule de sursă, "
         "procesul de matched molecular pairs (MMP) a extras 3.257 reguli clasificate. "
         "Metoda <b>M</b> aplică aceste reguli unor părinți; metoda <b>E</b> caută analogi prin "
         "operații evolutive și selecție multiobiectiv NSGA-II. Au fost apoi făcute extensii ale "
         "ambelor metode, reunite cu datele anterioare și deduplicate prin SMILES canonic."))
add(tbl(["Colecție", "Dimensiune", "Cum se interpretează"], [
    ["Benchmarkul anterior", "90 CSV; 781.692 rânduri", "Seturi echilibrate și subcategorii suprapuse; rândurile nu reprezintă analogi unici."],
    ["Banca finală", "216.165 analogi unici", "Câte un rând pe structură în molecules.csv; sursă pentru seturi viitoare de antrenare."],
    ["Relații", "222.772", "Un analog poate avea mai mulți părinți, reguli sau metode; relațiile stau separat."],
    ["Molecule de referință", "65.002; 27.261 eligibile", "A doua cifră trece filtrele de potrivire; nu ajunge pentru 216.165 negative unice într-un set 1:1."],
    ["Subcategorii", "84 module", "Selectabile după metodă / categorie / subcategorie; modulele se suprapun."],
], [49 * mm, 38 * mm, 87 * mm]))
add(h2("Compoziția băncii finale"))
add(tbl(["Proveniență per moleculă", "Număr unic", "Sens"], [
    ["M exclusiv", "138.941", "Obținută prin reguli MMP, nu prin E în banca deduplicată."],
    ["E exclusiv", "72.702", "Obținută prin căutare evolutivă, nu prin M în banca deduplicată."],
    ["M și E", "4.522", "Aceeași structură apare prin ambele metode; se exclude din contrastul strict M/E."],
], [64 * mm, 27 * mm, 83 * mm]))
add(para("Structurile acceptate sunt canonice, cu o componentă, absente din biblioteca de referință, "
         "cu părinte valid și cu filtre: 10-70 atomi grei, 150-800 Da, QED >= 0,35, SA <= 3,8, "
         "fără alerte PAINS/Brenk și similaritate Morgan față de părinte între 0,4 și 0,98. "
         "Media QED este 0,6847, media SA 2,7413, iar media similarității 0,6061. "
         "Aceste scoruri indică plauzibilitate computațională, nu posibilitatea demonstrată de sinteză."))
add(source("WIKI.md; final_dataset/README.md; final_dataset/REPORT.md; final_dataset/summary.json."))

add(PageBreak())
add(h1("3. Audit, adnotări și geometrii 3D"))
add(para("Am verificat integritatea snapshotului anterior și am reparat adnotările de editare "
         "acolo unde transformarea inversă este verificabilă. Spliturile benchmarkului sunt "
         "disjuncte pe scheletul părintelui în fiecare CSV, dar scheletul analogului poate "
         "reapărea între partiții. Acest detaliu a dus la teste mai stricte."))
add(tbl(["Verificare", "Rezultat", "Limită importantă"], [
    ["Reproducerea snapshotului", "90/90 CSV identice", "Valabil pentru snapshotul auditat, nu pentru un nou set construit din banca finală."],
    ["Edituri exacte în benchmark", "23.826 perechi; 49.741 rânduri", "Subsetul este alcătuit din substituții/adăugări reversibile cu un atom."],
    ["Corecții de mască", "40.298 rânduri corectate", "Măștile aproximative rămân distincte de adnotarea exactă."],
    ["Edituri exacte în banca finală", "28.166 relații", "191.794 relații au mască MCS aproximativă; nu se unesc în testele XAI exacte."],
    ["Suprapunere schelet copil în E", "1.623 / 9.250 rânduri test", "Splitul inițial separă scheletul părintelui; testul strict selectează 6.584 rânduri."],
], [52 * mm, 45 * mm, 77 * mm]))
add(h2("Conformeri 3D"))
add(tbl(["Colecție", "Acoperire", "Interpretare"], [
    ["Biblioteca benchmark arhivată", "249.803 / 249.975 (99,9312%)", "Cinci fișiere SDF.gz; 172 structuri fără conformer."],
    ["Ținta băncii finale", "243.426 / 243.426 (100%)", "Ținta definită în conformers_report.json, folosind arhiva existentă plus fișiere extra; nu înseamnă că toate moleculele din orice sursă externă au 3D."],
], [55 * mm, 49 * mm, 70 * mm]))
add(para("S-au folosit ETKDGv3 și încercări de relaxare MMFF94; pentru structurile noi apare și "
         "un fallback ETDG. O geometrie calculată nu este o structură experimentală. "
         "Banca finală păstrează hash-uri și rapoarte de audit pentru reproducere."))
add(source("benchmark/reproduction_audit.json; benchmark/verified_edits/summary.json; "
           "benchmark/child_scaffold_overlap_audit.json; benchmark/conformers_report.json; "
           "final_dataset/REPORT.md; final_dataset/conformers_report.json."))

add(PageBreak())
add(h1("4. Modele antrenate și rezultate"))
add(para("Au fost rulate modele simple pe proprietăți, Random Forest și XGBoost pe amprente "
         "Morgan, plus GIN, GCN și GAT pe grafuri moleculare. Comparația uniformă cea mai "
         "ușor de citit folosește subsetul `mixed_20k` derivat din banca finală: 31.746 rânduri "
         "train, 3.968 val și 2.948 test. Sunt trei semințe (42-44) și setări fixe, fără "
         "optimizare echitabilă a hiperparametrilor pentru fiecare familie."))
add(tbl(["Model", "AUC ROC", "FNR la 0,5", "F1"], [
    ["3 proprietăți", "0,528 ± 0,000", "0,453 ± 0,000", "0,535 ± 0,000"],
    ["Morgan + RF", "0,851 ± 0,001", "0,229 ± 0,010", "0,771 ± 0,004"],
    ["Morgan + XGBoost", "0,789 ± 0,001", "0,358 ± 0,003", "0,693 ± 0,003"],
    ["GIN", "0,815 ± 0,008", "0,290 ± 0,020", "0,732 ± 0,012"],
    ["GCN", "0,781 ± 0,010", "0,358 ± 0,038", "0,684 ± 0,013"],
    ["GAT", "0,791 ± 0,004", "0,403 ± 0,009", "0,674 ± 0,004"],
], [55 * mm, 37 * mm, 43 * mm, 39 * mm]))
add(para("În acest experiment cu setări fixe, RF are cel mai mare AUC. Nu rezultă că RF este "
         "superior oricărui GNN optimizat. AUC descrie ordonarea scorurilor pe toate pragurile; "
         "pentru screening contează câți analogi sunt ratați când alarmele false sunt puține."))
add(h2("Prag calibrat pe validare pentru cel mult 1% alarme false"))
add(tbl(["Model", "FPR pe test", "FNR pe test", "Analogi detectați, aprox."], [
    ["Morgan + RF", "0,008 ± 0,002", "0,860 ± 0,011", "14,0%"],
    ["Morgan + XGBoost", "0,010 ± 0,001", "0,850 ± 0,007", "15,0%"],
    ["GIN", "0,014 ± 0,002", "0,821 ± 0,028", "17,9%"],
    ["GCN", "0,007 ± 0,002", "0,838 ± 0,016", "16,2%"],
    ["GAT", "0,008 ± 0,002", "0,833 ± 0,023", "16,7%"],
], [53 * mm, 37 * mm, 39 * mm, 45 * mm]))
add(note("<b>Interpretare:</b> la bugetul strict, toate modelele ratează majoritatea analogilor "
         "din acest test. FPR de 1% a fost impus pe validare; GIN ajunge la 1,4% pe test. "
         "Precizia măsurată pe un test echilibrat nu se transferă automat la prevalența reală."))
add(source("training_runs/mixed_20k/README.md; architecture_summary_80.md; fpr_budget_summary_80.md."))

add(PageBreak())
add(h1("5. Analize de robustețe și explicații atomice"))
add(tbl(["Întrebare", "Rezultat observat", "Cum îl citim"], [
    ["Transfer M -> E inițial", "RF AUC 0,866 pe 572 rânduri; FNR 0,287", "Test intern mic; contrastul nu izolează generatorul ori contextul chimic."],
    ["Schelet copil nou în E", "GIN AUC 0,884 -> 0,887; RF 0,885 -> 0,893", "Subsetul mai strict schimbă și compoziția, deci nu dovedește efectul cauzal al splitului."],
    ["Diversitate în train", "RF AUC 0,645 -> 0,740 când cresc scheletele de la 10 la 3.700", "Număr de pozitive și clase de mărime potrivite; compoziția chimică diferă."],
    ["Dezechilibru 1:1.000 în train", "RF reglat pe F1 poate semnala toate rândurile test", "FNR devine 0, dar precizia este 0,500 pe testul echilibrat: pragul poate fi inutil."],
    ["Atribuție pe edituri E", "Top-1 în regiunea editului 0,250 vs 0,095 aleator; sit exact 0,024 vs 0,044 aleator", "Atingerea regiunii nu garantează identificarea atomului sau fidelitatea explicației."],
], [45 * mm, 63 * mm, 66 * mm]))
add(para("Pentru o subcategorie Cl -> F cu edit verificat, explicația GIN indică regiunea "
         "modificării ca atom de vârf în 62,3% dintre 220 analogi, dar nu identifică "
         "atomul exact de F în acel protocol. Măștile aproximative nu trebuie tratate drept "
         "adevăr atomic. Ablația trăsăturilor de inel a ajutat la redimensionarea inelelor "
         "(160 molecule test), însă a dăunat pe Cl -> F și a schimbat puțin altă sarcină. "
         "Acestea sunt observații pe subcategorii, nu un clasament universal al arhitecturilor."))
add(source("benchmark/README.md; benchmark/cross_version/; benchmark/strict_child_scaffold/; "
           "benchmark/scaffold_diversity/; benchmark/imbalance/; benchmark/explanations/; "
           "benchmark/gnn_comparison/."))

add(PageBreak())
add(h1("6. Experimentele de cercetare pentru disertație"))
add(h2("A. Martori MMP apropiați"))
add(para("Am construit un test intern cu 1.000 perechi părinte-potrivite: 500 analogi M și "
         "500 E, fiecare comparat cu o moleculă din biblioteca sursă apropiată de același "
         "părinte. Train are 19.824 rânduri, val 2.766, test 2.000. Pragul a fost ales pentru "
         "FPR de aproximativ 0,9% pe validare. Mediile sunt pe trei semințe ale modelului."))
add(tbl(["Model", "FPR pe martori test", "TPR analogi", "AUC"], [
    ["Morgan + RF", "0,046 ± 0,005", "0,135 ± 0,010", "0,676 ± 0,002"],
    ["Morgan + XGBoost", "0,023 ± 0,004", "0,110 ± 0,012", "0,663 ± 0,001"],
    ["RF + 130 martori grei", "0,044 ± 0,007", "0,142 ± 0,014", "0,677 ± 0,002"],
], [56 * mm, 51 * mm, 38 * mm, 29 * mm]))
add(para("Adăugarea celor 130 martori în train nu a produs un câștig robust. Martorii sunt "
         "din aceeași bibliotecă folosită la minarea regulilor M; nu constituie validare "
         "externă și nu certifică medicamente aprobate."))
add(h2("B. Transfer controlat M -> E pe aceeași schimbare elementară"))
add(para("În pilotul mai strict s-au folosit opt substituții atomice verificabile. Train "
         "conține doar M exclusiv; fiecare test are 320 M, 320 E și aceleași 1.000 negative. "
         "Perechile sunt potrivite după proprietăți; o variantă potrivește și părinții, iar "
         "alta cere identitatea primei vecinătăți a atomului editat. Structurile EM sunt excluse."))
add(tbl(["Potrivire", "Model", "AUC M", "AUC E", "Diferență E - M"], [
    ["Proprietăți", "RF", "0,817", "0,770", "-0,046"],
    ["Părinți", "RF", "0,851", "0,786", "-0,065"],
    ["Vecinătate locală", "RF", "0,829", "0,796", "-0,033"],
    ["Vecinătate locală", "XGBoost", "0,818", "0,792", "-0,026"],
], [42 * mm, 31 * mm, 29 * mm, 29 * mm, 43 * mm]))
add(para("În subsetul post hoc al splitului cu părinți potriviți, 212/320 perechi au aceeași "
         "primă vecinătate. Diferența RF scade la -0,017 acolo, față de -0,160 în perechile "
         "fără această potrivire. Acest lucru arată că mare parte din decalajul agregat este "
         "asociată cu contextul chimic local. Nu demonstrează o cauză unică. În testul cu "
         "vecinătate locală impusă, FPR RF este ~2,3%, peste ținta de validare de 1%."))
add(source("research_runs/MMP_PILOT_REPORT.md; research_runs/GENERATOR_TRANSFER_REPORT.md."))

add(PageBreak())
add(h1("7. Ce am formulat ca direcție de cercetare"))
add(para("Direcția principală pentru o disertație și o lucrare separată este un benchmark "
         "critic al generalizării detectorului de analogi sintetici: ce rămâne din performanță "
         "când schimbăm generatorul, familia editului, scheletul și martorii negativi, la "
         "un număr mic de alarme false. Contribuția posibilă este protocolul controlat și "
         "diagnosticul scurtăturilor de dataset; o metodă nouă ar trebui introdusă doar dacă "
         "depășește bazele simple pe același test și buget de tuning."))
add(tbl(["Idee discutată", "Stadiu actual", "Ce ar trebui adăugat"], [
    ["Comunicare între celule", "Ipoteză de cercetare: analogi apropiați ar putea produce salturi în interacțiuni celulare.", "Măsurători pe coculturi umane, donori independenți și validare experimentală a relațiilor ligand-receptor."],
    ["Nanosenzori", "Propunere de citire multiplexată a unor semnale locale, eventual mai timpurie decât viabilitatea.", "Sonde validate, controale de interferență și etichete biologice independente."],
    ["Explorator vascular", "Propunere de măsurare activă pe vas pe cip: controlul alege unde și când să observe leucocit-endoteliu.", "Platformă fizică, imagini și markeri, comparație cu traseu fix; nu există un nanobot uman realizat aici."],
], [48 * mm, 61 * mm, 65 * mm]))
add(para("Documentele de idei au inclus o căutare orientată a literaturii, însă noutatea nu "
         "este demonstrată prin acest raport și necesită revizuire sistematică înainte de "
         "o revendicare de prioritate. Datasetul actual nu conține răspuns biologic, spectre "
         "ale produselor ori măsurători ale comunicării dintre celule."))
add(source("DISSERTATION_PUBLICATION_DIRECTION.md; RESEARCH_IDEA_CELL_COMMUNICATION.md; "
           "RESEARCH_IDEA_NANOSENSORS.md; RESEARCH_IDEA_VASCULAR_MICROROBOT.md."))

add(PageBreak())
add(h1("8. Planul pentru calculatorul facultății"))
add(para("A fost redactat un plan amplu de antrenare și optimizare. Este un plan, nu un "
         "rezultat deja rulat pe calculatorul facultății. Separă două întrebări: transferul "
         "M -> E / E -> M pe edituri comparabile și efectul creșterii numărului de exemple "
         "și al capacității modelului. Datele pilot folosite deja devin date de dezvoltare; "
         "testul final trebuie construit și înghețat înainte de căutarea hiperparametrilor."))
add(tbl(["Pas", "Protocol planificat", "De ce contează"], [
    ["Date", "Train, validare HPO, calibrare prag, test final separate; audit pe molecule, componente părinte-copil și ambele schelete.", "Previne alegerea indirectă a modelului după test și scurgerile între partiții."],
    ["Modele", "Controale simple; Morgan + RF/XGBoost; GIN/GCN/GAT și GINE sau message passing cu legături.", "Modelul graf mare trebuie comparat corect cu alternative puternice."],
    ["Scară", "Aproximativ 5k, 10k, 20k pozitive; GNN mici, medii și mari; măsurare timp/memorie.", "Separă efectul datelor de efectul capacității."],
    ["Tuning", "10 probe scurte per familie, apoi ~25-40 trialuri pentru familiile importante; mai multe splituri și semințe.", "Bugete comparabile și selecție pe validare, nu pe test."],
    ["Evaluare", "AUC parțial la FPR 0-5%, FNR/TPR la prag calibrat pentru 1% FPR, intervale pe schelete, cost computațional.", "AUC global singur ascunde rata mare de omisiuni la prag strict."],
], [31 * mm, 82 * mm, 61 * mm]))
add(h2("Stadiu la data raportului"))
add(note("<b>Gata:</b> banca modulară și auditurile, subsetul mixed_20k, comparația mică de "
         "arhitecturi, pilotul MMP și pilotul M -> E, plus planul HPC. <b>Încă de făcut:</b> "
         "holdout final înghețat, refactorizarea trainerului GNN, joburile GPU, optimizarea "
         "echitabilă, teste repetate și validare pe o sursă independentă."))
add(source("HPC_TRAINING_PLAN.md; training_runs/mixed_20k/; research_runs/."))

add(h1("9. Concluzia onestă"))
add(para("Ținta numerică de 200-300 de mii de analogi a fost atinsă: 216.165 structuri "
         "unice într-o bancă auditată, modulară. Sunt disponibile transformări MMP și "
         "analogii evolutivi, metadate, relații, edituri exacte pentru o submulțime și "
         "conformeri calculați. Experimentele arată că modelele pot discrimina bine într-un "
         "test obișnuit, dar performanța la foarte puține alarme false și la martori "
         "apropiați este încă modestă. Acesta este un rezultat științific important pentru "
         "designul disertației: problema trebuie definită și controlată mai strict înainte "
         "de a trage concluzii despre detectarea falsurilor reale."))
add(para("Fișiere de pornire pentru reproducere: `WIKI.md`, `final_dataset/README.md`, "
         "`final_dataset/REPORT.md`, `training_runs/mixed_20k/README.md`, "
         "`research_runs/MMP_PILOT_REPORT.md`, `research_runs/GENERATOR_TRANSFER_REPORT.md`, "
         "`DISSERTATION_PUBLICATION_DIRECTION.md` și `HPC_TRAINING_PLAN.md`.", "SmallRO"))


def page_decor(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(18 * mm, height - 17 * mm, width - 18 * mm, height - 17 * mm)
    canvas.setFont("ArialRO", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, height - 14 * mm, "Counterfeit-Detector  |  Raport de progres")
    canvas.line(18 * mm, 18 * mm, width - 18 * mm, 18 * mm)
    canvas.drawString(18 * mm, 13 * mm, "29 septembrie 2026  |  Rezultate din repository")
    canvas.drawRightString(width - 18 * mm, 13 * mm, f"Pagina {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4,
                        leftMargin=18 * mm, rightMargin=18 * mm,
                        topMargin=23 * mm, bottomMargin=24 * mm,
                        title="Raport complet de progres - Counterfeit-Detector",
                        author="Counterfeit-Detector")
doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
print(OUTPUT)
