# Direcție recomandată pentru disertație și o lucrare nouă

## Concluzie

**Întrebare recomandată:** cât din performanța unui detector de analogi
moleculari sintetici supraviețuiește când se schimbă simultan familia de
transformări, algoritmul de generare, scheletul chimic și tipul de molecule
folosite ca martor negativ, la un buget mic de alarme false?

Contribuția propusă este un **benchmark de generalizare și validitate a
etichetelor pentru modificări moleculare locale**, urmat de o metodă simplă
care încearcă să reducă erorile demonstrate de benchmark. Este diferit de
extinderea numărului de molecule sau de o comparație între GNN-uri.

Titlu de lucru: **Do Molecular Edit Detectors Generalize Beyond Their
Generator? A Controlled Benchmark with Real-Source Near-Analogue Negatives**.
Termenul „counterfeit medicine detection” ar trebui rezervat pentru o
evaluare pe mostre contrafăcute reale sau pe măsurători fizice ale produsului.

## De ce aceasta este întrebarea corectă pentru datele existente

Banca finală are 216.165 analogi sintetici unici, 222.772 relații
părinte–analog, 84 de module selectabile și 28.166 relații cu un edit exact
verificat. Există două surse diferite de analogi: reguli MMP (`M`) și căutare
evolutivă (`E`). Auditul, proveniența și împărțirile pe schelete fac posibile
experimente care separă unele surse de variație, nu doar o medie de AUC.
Vezi [raportul datasetului](final_dataset/REPORT.md).

Există și martori apropiați cu molecule preluate din biblioteca sursă: baza
locală `mmp_mining_big/mined.mmpdb` conține 336.298 rânduri de perechi MMP.
O verificare locală la 26 septembrie 2026 a găsit 10.797 din 11.144 molecule
din această bază în `final_dataset/authentic.csv`, respectiv **311.926 rânduri
de perechi cu ambele molecule în biblioteca sursă** (109.598 perechi distincte).
Dintre cei 38.186 părinți distincți ai analogilor generați, 5.883 apar și
printre aceste molecule MMP. După excluderea moleculelor care apar și ca
analogi generați, rămân 76.532 perechi distincte candidate, acoperind 5.878
părinți și 8.667 molecule de control distincte. Numerele sunt reproductibile
cu [`audit_near_analogue_controls_feasibility.py`](audit_near_analogue_controls_feasibility.py)
și [rezultatul auditului](reports/near_analogue_controls_feasibility.json).
Acestea sunt controale de **proveniență din biblioteca publică**, nu
medicamente aprobate și nici exemple certificate clinic ca „autentice”.
Regulile `M` au fost minate din această sursă: controalele nu sunt externe
procesului de dezvoltare. Avem nevoie de audit de proveniență și de un set
dintr-o sursă nouă pentru o validare cu adevărat externă.

Pilotul actual, cu setări fixe, dă Random Forest AUC 0,851 pe subsetul
`mixed_20k`, dar la pragul ales pentru FPR de cel mult 1% pe validare ratează
în medie 86,0% dintre analogii din test. GIN ratează 82,1% și depășește
țintele de alarme false pe test. Rezultatul este preliminar și nu poate fi
extins automat la întreaga bancă. Vezi [comparația de arhitecturi](training_runs/mixed_20k/architecture_summary_80.md)
și [analiza la pragul strict](training_runs/mixed_20k/fpr_budget_summary_80.md).

## Problema conceptuală care poate deveni contribuția lucrării

În setul actual, `label=1` înseamnă că molecula a fost **generată** dintr-un
părinte, nu că un laborator a constatat un fals farmaceutic. „Contrafăcut” nu
este o proprietate intrinsecă a grafului molecular. Mai mult, dacă sunt deja
cunoscute exact structura suspectă și cea declarată pe produs, un algoritm
determinist le poate compara și găsi editarea; un model ML nu este necesar
pentru simpla constatare că grafurile diferă. Întrebarea ML interesantă este
ce generalizează un model când proveniența, editarea și contextul se schimbă
sau când măsurarea structurii este incompletă.

De aceea lucrarea ar măsura explicit **cât din scor este transferabil și cât
este o urmă a procesului de construire a datasetului**. Un rezultat negativ,
obținut prin controale riguroase, poate fi o contribuție utilă: împiedică
interpretarea unui AUC bun drept dovadă de detectare a falsurilor reale.

## Ce știm deja din literatură și unde ar putea fi golul

| Lucrare apropiată | Ce acoperă deja | Diferența propusă aici |
|---|---|---|
| [Hidden bias in DUD-E](https://pmc.ncbi.nlm.nih.gov/articles/PMC6701836/), [Beware of Simple Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10015451/) și [MUBDsyn](https://www.sciencedirect.com/science/article/pii/S001048252400249X) | Au demonstrat sau controlat biasul produs de construcția decoy-urilor în screening virtual. | Nu folosesc această bancă de **editări locale** cu două familii de generatoare, subcategorii și măști exacte pentru auditul analogilor. Nu revendicăm descoperirea generală a biasului. |
| [UB-GOLD, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e58fa42d4b7a798eef8d0d75098f87ad-Abstract-Conference.html) și [SDM, ICML 2025](https://proceedings.mlr.press/v267/cai25k.html) | Benchmark și metodă pentru anomalii la nivel de graf, inclusiv anomalii sintetice. | Testăm substanțe chimice foarte apropiate de un părinte, proveniența editării și martori MMP din biblioteca sursă. |
| [ACNet](https://arxiv.org/abs/2302.07541) și [ACANet](https://www.nature.com/articles/s41467-026-75713-2) | Modelează perechi cu diferențe **măsurate** de activitate biologică. | Noi nu prezicem bioactivitate; nu putem folosi analogii virtuali drept „activity cliffs”. Împrumutăm disciplina evaluării pe perechi, nu revendicăm noutatea modelelor pereche. |
| [VibraCLIP](https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00269a) și [Mol2Raman](https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00210a) | Leagă grafuri de spectre IR/Raman sau prezic spectre. | O direcție ulterioară ar testa diferențierea analogilor noi din spectre reale de instrument portabil; datele spectrale lipsesc acum. |

**Ipoteza de noutate, formulată prudent:** nu am identificat, în căutarea
orientată de mai sus, un benchmark care combină pentru această problemă
controlul pe generator și familie de editări, martori MMP catalogați în
biblioteca sursă, calibrarea la FPR mic și măști atomice exacte. Un set extern
cu proveniență separată ar întări semnificativ afirmația de generalizare.
Fiecare ingredient are precedenți; combinația și rezultatul empiric trebuie
verificate printr-o revizuire sistematică înainte de o revendicare de
prioritate.

## Experimentul minim publicabil

1. **Definirea sarcinii și auditul etichetei.** Separăm în articol
   „analog sintetic generat” de „medicament contrafăcut”. Înghețăm un manifest
   cu structuri, părinți, regulă, generator, sursă, audit și split. Raportăm
   orice suprapunere de molecule și relații între partiții.
2. **Martori dificili din sursa reală.** Pentru fiecare părinte eligibil,
   alegem analogi MMP din biblioteca sursă, absenți din banca sintetică și
   potriviți pe similitudine, mărime, proprietăți și tip de editare unde este
   posibil. Aceștia testează alarma falsă pe molecule catalogate, nu statutul
   legal sau clinic al unui produs. Audităm separat potrivirile imposibile și
   regulile ori moleculele care au intrat în minarea transformărilor `M`.
   Pentru validare externă, adăugăm ulterior o bibliotecă independentă.
3. **Trei axe de testare.** Ținem în afara antrenării (a) componentele
   părinte–copil și scheletele, (b) familii de editări, (c) generatorul M sau E.
   Le evaluăm întâi separat, apoi în combinațiile cu suficiente exemple;
   eliminăm moleculele `EM` din comparația strictă M→E sau E→M.
   Calibrăm pragul pe validare din domeniul de antrenare; nu recalibrăm pe
   generatorul sau familia ascunsă din test. Păstrăm mărimea și prevalența
   comparabile între comparațiile în care interpretăm diferențe de scor.
4. **Baselines obligatorii.** Similaritate și caracteristici chimice simple,
   Morgan + RF/XGBoost, GIN/GCN/GAT, plus o metodă de anomalie antrenată pe
   moleculele sursă. Egalizăm bugetul de selecție a hiperparametrilor și
   repetăm pe mai multe semințe. Modelul nou, dacă îl introducem, trebuie să
   bată aceste comparații la același buget, nu doar un GNN netunat.
5. **Metrici operaționale.** FNR și TPR la FPR de 1% ales pe validare, FPR
   observat pe fiecare test, AUPRC la prevalențe simulate, calibrare și
   intervale de incertitudine. Raportăm separat fiecare generator și clasă de
   editări. AUC singur nu este suficient.
6. **Diagnostic al mecanismului.** Un model secundar încearcă să ghicească
   generatorul din structura moleculei. Pe cele 28.166 relații cu măști exacte,
   comparăm explicația detectorului cu editul și testăm dacă modificarea
   fragmentului explicat schimbă scorul. Separăm localizarea de fidelitate.

Prima lucrare poate fi un **benchmark critic** chiar dacă nu găsim o metodă
care rezolvă problema. Pentru o contribuție metodologică suplimentară,
încercăm antrenare cu martori MMP dificili și echilibrare între familiile de
editări, apoi evaluăm câștigul pe generatorul ascuns. Aceasta devine o
contribuție numai dacă bate controalele simple și nu consumă un buget de
etichetare disproporționat.

## Ce ar susține sau ar infirma ipoteza

- **Susține:** performanța aparent bună pe testul obișnuit se schimbă
  substanțial pe analogi MMP din biblioteca sursă ori pe generator/familie
  ascunsă; analiza de atribuire arată că modelul urmărește regiuni
  irelevante; o metodă controlată reduce eroarea fără să crească FPR.
- **O infirmă sau o îngustează:** aceeași performanță rămâne stabilă pe toate
  aceste controale și martorii MMP nu sunt mai dificili. Rezultatul ar fi tot
  informativ, dar ideea de „bias de generator” nu ar fi concluzia lucrării.
- **Blochează interpretarea:** martorii nu pot fi potriviți corect, există
  suprapuneri ascunse între splituri, ori clasele de editări sunt prea mici.
  Atunci publicăm numai afirmațiile pe subgrupurile auditate și nu extrapolăm.

## Raportarea la disertație și la prima lucrare

Lucrarea deja depusă la DSAA și propunerea actuală descriu un prim set de
date și o extindere de benchmark. Pentru o a doua publicație, simpla mărire
de la 50.000 la 216.165 molecule și antrenarea unor GNN-uri mai mari riscă să
fie insuficiente ca diferență științifică. **Noua întrebare, martorii MMP,
spliturile conjuncte și evaluarea la FPR mic** trebuie să fie contribuțiile
separate, formulate și demonstrate clar.

Direcția cu nanoroboți și comunicare celulară poate rămâne motivație pe
termen lung. Pentru disertația bazată pe repo-ul actual ar necesita etichete
biologice și o platformă experimentală pe care banca moleculară nu le oferă.
Multimodalitatea structură–Raman este un pas aplicat promițător, dar datele
spectrale reale pentru perechile noastre lipsesc și există deja modele
generale de aliniere graf–spectru; nu este calea cea mai sigură pentru prima
lucrare nouă.

## Progres experimental: 27 septembrie 2026

Am construit un [pilot cu 1.000 de perechi părinte–analog generat / control MMP](research_runs/MMP_PILOT_REPORT.md).
Spliturile separă structurile, componentele părinte–copil și scheletele;
pragurile sunt alese pe validare. Pe trei semințe ale aceluiași split,
Random Forest are FPR pe controalele MMP de 4,6% ± 0,5% la un FPR de
validare de circa 0,9%, cu TPR de 13,5% ± 1,0% pe analogii generați.
Înlocuirea a 130 de negative de antrenare cu controale MMP nu a produs un
câștig stabil. Aceste rezultate sunt un diagnostic al schimbării de
distribuție pe date interne, nu validare pe falsuri farmaceutice reale și
nici un test care izolează deja efectul generatorului.

Un [audit al editărilor exacte comune M/E](research_runs/shared_exact_edit_signatures.json)
arată 13 semnături elementare cu cel puțin 100 de structuri exclusive pentru
fiecare metodă. Acestea permit construirea următorului test pe familii de
schimbări comparabile; semnătura elementară trebuie verificată și la nivelul
legăturilor și contextului chimic înainte de a echivala transformările.
Relațiile filtrate sunt în [manifestul M/E](research_runs/shared_exact_edit_relations.csv).

Am rulat și [pilotul de transfer M→E](research_runs/GENERATOR_TRANSFER_REPORT.md)
pe opt semnături de substituție exactă: antrenare numai pe structuri `M`,
test cu 320 de perechi `M`/`E` exclusive, 1.000 de negative comune din
biblioteca sursă și separarea componentelor și scheletelor între splituri.
Pe trei semințe, diferența medie AUC `E−M` este −0,046 pentru Morgan + RF
și −0,049 pentru Morgan + XGBoost în potrivirea după proprietăți. În varianta
cu părinți potriviți mai strâns, diferențele sunt −0,065 și −0,071.
Două alte semințe de selecție păstrează direcția diferenței. Dar auditul
vecinătății imediate a atomului editat schimbă interpretarea: pe perechile
cu aceeași primă vecinătate, diferența AUC la sămânța 42 scade la −0,017
pentru RF și −0,019 pentru XGBoost în varianta cu părinți potriviți;
perechile cu vecinătăți diferite concentrează aproape tot contrastul.
Un split nou, construit de la început cu vecinătate identică în toate cele
320 de perechi, păstrează totuși o diferență mai mică: aproximativ −0,033
pentru RF și −0,026 pentru XGBoost pe trei semințe ale modelului. Contextul
chimic mai îndepărtat și selecția moleculelor rămân explicații posibile.
Acestea sunt rezultate pe selecții interne din aceeași bancă, nu o
demonstrație cauzală a biasului de generator sau o validare clinică.
FPR pe test depășește deseori bugetul de 1% fixat pe validare. Urmează
potrivire extinsă la două legături, validare cu negative externe
și verificarea arhitecturilor pe același protocol.
