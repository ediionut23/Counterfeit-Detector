# Ghid C1 — Datasetul: tot ce trebuie să știi + întrebările coordonatorului

Document de pregătire pentru discuția cu coordonatorii (prof. Breabăn, asist. Cornei).
Termenii tehnici sunt în engleză (ca în propunere); explicațiile în română.

> **Actualizare 26.09.2026:** snapshotul curent are 90 de dataseturi și
> 781.692 de rânduri. Baseline-urile RF pentru `evolved`, `pooled`, cele
> patru categorii mari și experimentul M→E strict au fost rerulate.
> Rezultatele GIN pe categoriile mari rămân istorice; există o comparație nouă
> GIN/GCN/GAT pe subcategoria Cl→F. Arhiva 3D acoperă 99,9312% dintre molecule; vezi `WIKI.md` și
> `benchmark/README.md`.

**Banca modulară nouă:** `final_dataset/` conține câte un rând pentru fiecare
analog sintetic unic, relațiile cu părinții autentici și fișiere separate pe
metodă/categorie/subcategorie. `final_dataset/summary.json` și `audit.json`
dau numerele finale și verificările; `final_dataset/README.md` descrie formatul.
Cele 90 de CSV-uri echilibrate din `benchmark/` sunt instantanee mai vechi de
evaluare, iar suma rândurilor lor nu reprezintă molecule contrafăcute unice.

---

## PARTEA 1 — Ce este, de fapt, datasetul

**Sarcina:** detecția de medicamente contrafăcute din **structura moleculară**, cu
Graph Neural Networks. Fiecare moleculă e un graf (atomi = noduri, legături =
muchii). Clasificare binară: `0` = autentic, `1` = contrafăcut.

**Ce e o contrafacere aici:** un medicament real cu o **modificare mică,
chimic-conștientă** (o substituție bioisosterică, un halogen swap etc.). Semnalul
e **local** — clasele diferă doar într-o regiune mică.

**Ideea centrală (și cea mai importantă de înțeles):**
> "Counterfeit" **nu e o proprietate** a moleculei (ca toxicitatea, care e
> măsurabilă). E o moleculă obișnuită pe care o *interpretăm* drept contrafacere
> **relativ la medicamentul pe care îl imită**. Eticheta e **atribuită**, nu
> măsurată — deci **alegerea transformărilor e la fel de importantă ca modelul.**

De aici rezultă **de ce e nevoie de un benchmark** (nu doar un dataset): trebuie
o *familie* de dataset-uri, un protocol de evaluare fix, și baseline-uri.

---

## PARTEA 2 — Cum e construit (pipeline cap-coadă)

```
molecule reale (ChEMBL+PubChem)
   │  fetch_public_molecules.py
   ▼
[A] ÎNVAȚĂ regulile  →  mine_mmp_rules.py  →  catalog MMP (cu support + categorii)
   │
   ├──► [B1] APLICĂ regulile  → build_version_m.py → Version M (per categorie)
   └──► [B2] EVOLUEAZĂ         → evolutionary_generator.py (GA+NSGA-II) → Version E
   │
   ▼
[C] ASAMBLEAZĂ  → assemble_benchmark.py
     (pereche cu autentice + property-match + scaffold split)
   │
   ▼
[D] CONFORMERI 3D → generate_conformers.py
```

**[A] Mining (Matched Molecular Pairs):** tai fiecare moleculă la o legătură →
parte variabilă + parte constantă. Perechile care împart partea constantă dar
diferă în variabilă → o regulă `V1→V2`. **Support** = câte perechi reale o
susțin (măsura de realism). Algoritm: Hussain–Rea, via `mmpdb`.

**[B1] Version M (rule-based):** aplic o regulă pe un părinte autentic, filtrez
(druglike + Tanimoto 0.4–0.98). Emis **per categorie**.

**[B2] Version E (evolutiv):** algoritm genetic pe graf (crossover cut+molzip +
mutație cu reguli MMP) sub **NSGA-II** multi-obiectiv. **Combină** edituri,
**descoperă** modificări neenumerate.

**[C] Asamblare:** fiecare set de contrafaceri primește negative autentice
1:1, **property-matched** (DUD-E), split **scaffold** fix.

**[D] Conformeri 3D:** ETKDGv3+MMFF, unul per moleculă.

---

## PARTEA 3 — Deciziile de design (și DE CE)

| Decizie | De ce |
|---|---|
| **Mai multe versiuni** (R/M/E), nu una | fiecare răspunde la altă întrebare |
| **Per categorie/subcategorie** | un F1 mediu ascunde *ce* prinde modelul; per categorie spune "prinde halogen-walk, ratează scaffold-hop" |
| **Reguli minate** (nu 21 scrise de mână) | data-driven, cu dovadă de realism (support); catalogul crește automat cu date |
| **Version E (evolutiv)** | un catalog fix are plafon; un contrafăcător real nu e limitat la o listă |
| **Property matching (DUD-E)** | ca niciun clasificator să nu separe pe greutate/mărime → forțează semnal structural |
| **Split pe scaffoldul părintelui** (nu random) | ține familia părinte–analog în același split; scaffoldul structurii analogului poate apărea în alte splituri, deci raportez separat un test mai strict |
| **Conformeri 3D** | pentru experimentele geometrice (C1/C3) |
| **Atomii editați** | permit evaluarea explicațiilor; pentru substituțiile halogenice reversibile verificăm exact locul editării, iar măștile MCS ambigue se raportează separat |
| **E model-free la benchmark** | un dataset trebuie să fie independent de model, reproductibil |

---

## PARTEA 4 — Numerele actuale

- **Molecule reale de mining:** ~58.600 (ChEMBL + PubChem)
- **Catalog:** 1.732.359 reguli brute → **3.257 reguli** filtrate (support median ridicat)
- **Version M:** 50.000 contrafaceri (10k × 5 categorii)
- **Benchmark E:** 46.251 contrafaceri evoluate după asamblare și matching
- **Benchmark:** 90 dataseturi, 781.692 de rânduri în total
- **pooled:** 104.718 molecule (1:1 autentic:contrafăcut)
- **Conformeri 3D:** 249.803/249.975 molecule (99,9312%), arhivate în 5 fișiere gzip cu chei `smiles`

**Categoriile:** bioisostere, scaffold-hop, halogen-walk, homologation, other.

---

## PARTEA 5 — Validarea (dovezile, nu presupunerile)

**Realism** (`audit_realism.py`): în cele 90 de seturi, contrafacerile sunt
valide și unice în fiecare set. Noutatea față de biblioteca autentică completă
este 0,983 pentru E și 0,976 pentru pooled; într-o subcategorie cu 80 de
contrafaceri scade la 0,738. Similaritatea medie cu părintele este 0,628 pe E.
Matchingul reduce shortcut-urile de proprietate, dar unele subcategorii mici
au încă scoruri LR triviale ridicate; verifică `summary.json` și mărimea
eșantionului în `realism_audit.json` pentru fiecare.

**Învățabilitate** (`train_baseline.py`, `train_gnn.py`), pe split scaffold:
- **M, snapshotul curent:** fingerprint AUC 0,872–0,952 pe cele patru categorii mari.
- **E, snapshotul curent:** fingerprint AUC **0,902**, F1 **0,825**, FNR **0,150**.
- **property-only:** AUC 0,521 pe E, dar până la 0,592 pe halogen-walk.
- **GIN/GCN/GAT pe Cl→F:** AUC mediu pe 3 seeduri 0,973 / 0,973 / 0,979;
  RF și XGBoost obțin 0,984 / 0,983, tot pe 3 seeduri.
- **M→E strict:** sursă M comună fără conflicte, test E fără schele/SMILES
  văzute: RF AUC **0,866 ± 0,002**, F1 **0,756 ± 0,006**, FNR **0,287 ± 0,004**;
  XGBoost AUC **0,810 ± 0,005**, F1 **0,680 ± 0,008**, FNR **0,395 ± 0,010**
  pe 572 exemple, trei seeduri.

**Constatarea curentă:** la scară mai mare, E nu mai are AUC 0,67; Random
Forest ajunge la **0,902**. Gap-ul vechi M–E era specific versiunilor mici.

---

## PARTEA 6 — Limitările oneste (spune-le TU primul, nu aștepta să te prindă)

1. **Nu există date de contrafaceri reale.** Nu există public structuri de
   falsuri sechestrate. Deci nu "învățăm din falsuri reale", ci din **spațiul
   real de analogi medicamentoși** (cel mai apropiat proxy). *Metoda* de extragere
   există; limitarea e datele, nu metoda.
2. **Eticheta e atribuită, nu măsurată** (vezi Partea 1). O contrafacere e o
   moleculă plauzibilă; în alt context ar putea fi un analog legitim.
3. **E parțial mai ușor decât realitatea** — property matching-ul poate lăsa
   mici shortcut-uri reziduale pe unele subcategorii (le raportăm).
4. **E (evoluat) e costisitor** — generarea NSGA-II per părinte rămâne scumpă,
   chiar dacă versiunea curentă are 46.251 contrafaceri după matching.
5. **Un singur autor/paper până acum** (DSAA, sub review) — de aici nevoia de
   benchmark ca să devină comparabil.

---

## PARTEA 7 — Întrebări pe care le-ar putea pune coordonatorul (cu răspunsuri)

### A. Conceptuale / framing

**Î: Ce înseamnă că o moleculă e "contrafăcută"? Nu e o proprietate reală.**
R: Corect — nu e. E o etichetă *relațională*: molecula e un fals *al unui
anumit medicament* de la care a fost derivată printr-o transformare cunoscută.
De-aia eticheta e atribuită, nu măsurată, și de-aia alegerea transformărilor e
parte din contribuție.

**Î: De ce benchmark și nu doar un dataset?**
R: Un benchmark = task + date + split-uri fixe + protocol + baseline-uri. Fără
protocol fix și baseline-uri, rezultatele nu-s comparabile (vezi Errica 2020, Lv
2021 — câștiguri care dispar când tunezi corect). C1 e componenta de date.

**Î: De ce GNN și nu descriptori/fingerprints?**
R: E o întrebare deschisă — de-aia am și baseline fingerprint (C2). Pe M,
fingerprint-ul e puternic (0.8-0.95); pe E și pooled, GNN-ul ar trebui să ajute
(vede structura, nu doar biți). Van Tilborg 2022 avertizează că pe activity
cliffs descriptorii bat deep — de aici prudența.

### B. Metodologie

**Î: Cum extragi regulile? Cât de strict e?**
R: MMP (Hussain–Rea) via mmpdb. Filtre: support ≥ prag (câte perechi reale),
single-cut, fragment ≤8 atomi, fără izotopi/sarcini, compilabil RDKit. Butonul
de strictețe = `--min-support`. Support-ul e măsura de realism.

**Î: Ce e NSGA-II și de ce multi-obiectiv?**
R: O contrafacere e "bună" doar dacă e simultan similară-în-bandă, plauzibilă
(QED+alerte) și sintetizabilă (SA) — obiective care se contrazic. NSGA-II
păstrează **frontul Pareto** (soluțiile nedominate) + crowding distance pentru
diversitate. Nu comprim într-o sumă ponderată (ar fi arbitrar).

**Î: Cum eviți shortcut-urile? De unde știi că modelul nu trișează pe greutate?**
R: Property matching stil DUD-E: potrivesc autentice și contrafaceri pe MW/heavy/
lungime. Raportez Cohen's d (~0) și acuratețea unui LR trivial pe acele
proprietăți (~0.5 = nu se separă pe acele variabile). Un scor aproape de 0,5
susține interpretarea că semnalul util vine din structură.
În snapshotul curent verific fiecare subcategorie separat; unele păstrează
shortcut-uri reziduale, deci nu presupun că matchingul le-a eliminat complet.

**Î: De ce scaffold split și nu random?**
R: Un split random lasă modelul să memoreze scaffold-uri și să pară bun fals.
Splitul pe scaffoldul părintelui ține familiile de origine separate și reduce
memorarea lor; singur nu demonstrează că modelul a învățat
*editul*. În snapshotul curent, train/val/test
sunt disjuncte pe scaffoldul stocat și fiecare split păstrează balansul 1:1.
Auditul arată că 17,55% din rândurile E de test au scaffoldul structurii proprii
prezent în train/validare; `benchmark/strict_child_scaffold/` oferă un test E
mai strict. Vechiul
rezultat GNN 0,90 pe val și 0,67 pe test provenea dintr-un set mai mic.
Pentru M→E folosesc un set M derivat cu split global și un subset E filtrat;
spliturile originale ale categoriilor nu se pot concatena direct.

**Î: Cum faci balansul 1:1 dacă un părinte dă 10 contrafaceri?**
R: Raportul "1 părinte : N falsuri" e la generare, irelevant pentru balans.
Balansul 1:1 e la asamblare: iau **la fel de multe molecule autentice câte
contrafaceri**, molecule *separate* de părinți, property-matched. Părintele e
reținut doar ca origine (pentru split și identificarea editării). Pentru
evaluarea explicațiilor la nivel de atom folosesc subsetul din
`benchmark/verified_edits/`: 23.826 perechi cu un loc unic de editare
reversibilă. Celelalte măști pot fi aproximative și nu constituie automat
ground truth exact.

### C. Realism / validitate (cele mai probabile)

**Î: Sunt realiste contrafacerile astea? Sau inventezi molecule?**
R: Le-am *măsurat* (audit_realism.py): valide/unice în fiecare set, noutate
0,983 pe E și 0,976 pe pooled, similaritate medie cu părintele 0,628 pe E.
Unele subcategorii mici au noutate și proprietăți mai puțin uniforme.
Regulile vin din perechi de medicamente reale (support median mare). Deci
ancorate în chimie reală, nu inventate.

**Î: Nu e circular? Generezi și testezi cu aceleași reguli?**
R: Pentru M, regulile minate NU au fost folosite la antrenarea vreunui model —
sunt doar sursa datelor. Pentru E adversarial *ar fi* circular (de-aia la
benchmark E e **model-free**). Iar întrebarea de generalizare o testez R/M→E
(antrenezi pe unele, testezi pe altele).

**Î: De unde știi că nu e doar zgomot (moleculele indistinctibile)?**
R: Pe scaffoldurile părinților nevăzute, fingerprint-ul dă AUC 0,872–0,952 pe categoriile
M mari și 0,902 pe E. Property-only dă 0,521 pe E, dar ajunge la 0,592 pe
halogen-walk; semnalul structural este real, iar shortcut-urile reziduale se
raportează separat.

**Î: De ce este costisitoare generarea moleculelor evoluate (E)?**
R: Fiecare părinte cere o rulare NSGA-II, apoi păstrez doar frontul Pareto
filtrat (in-band + druglike + dedup). Versiunea curentă are 46.251
contrafaceri E după matching, dar generarea rămâne mai scumpă decât aplicarea
regulilor MMP.

### D. Comparație cu literatura

**Î: Cu ce diferă de DUD-E / decoys?**
R: DUD-E generează *decoys* (molecule diferite, matched pe proprietăți). Aici
contrafacerile sunt *derivate din* medicament printr-un edit local cunoscut —
avem și indicii atomilor modificați, cu locul verificat exact acolo unde o
substituție inversă unică reconstruiește părintele. Am împrumutat de la DUD-E doar
protocolul de property-matching anti-shortcut.

**Î: Legătura cu activity cliffs (MoleculeACE)?**
R: Foarte apropiată — activity cliffs = perechi aproape identice cu activitate
diferită. Contrafacerile sunt perechi aproape identice cu "autenticitate"
diferită. De-aia și avertismentul lor (descriptorii pot bate deep) e relevant și
îl testez cu baseline-ul fingerprint.

**Î: De ce nu un generator deep (diffusion/VAE)?**
R: Pentru sarcina de *modificare locală a unui medicament real*, MMP+GA e mai
potrivit și mai realist decât generarea de novo (care inventează de la zero). Un
generator deep ar fi peste scop; MMP din date reale ancorează editurile.

### E. Detalii tehnice (posibile "prinderi")

**Î: Câte reguli? Ce support?** → 3.257 la support≥8 (din 1.7M brute). Median
ridicat, max sute de perechi.

**Î: De ce max 8 atomi pe fragment?** → ca editurile să fie *mici/locale* (o
contrafacere e o modificare subtilă, nu o rescriere).

**Î: Ce faci cu regulile care nu se aplică?** → Filtrate; doar cele compilabile
RDKit + care produc molecule druglike ajung în dataset.

**Î: Cum garantezi reproductibilitatea?** → Datele se reconstruiesc exact cu
sursele arhivate, manifestul măștilor și split-urile fixe; experimentele salvează
semințele, numărul de fire CPU și versiunile pachetelor. Un seed singur nu
garantează rezultate identice pe orice hardware.

**Î: Ce metrici raportezi?** → FNR (un fals ratat e eroarea costisitoare) întâi,
apoi Precision/Recall/F1, plus AUC — și **per categorie**, nu doar media.

---

## Cheat-sheet: 5 lucruri de spus sigur

1. **"Eticheta e atribuită, nu măsurată"** — arată că înțelegi natura problemei.
2. **"Property matching + split pe scaffoldul părintelui + audit al
   scaffoldului analogului"** — arată ce controlezi și ce suprapunere rămâne;
   testul E strict elimină acea suprapunere.
3. **"Support-ul din mining = realism data-driven"** — regulile-s reale, nu
   inventate.
4. **"E ajunge la AUC 0,902 la scara actuală"** — rezultatul vechi de 0,67
   nu se aplică snapshotului curent.
5. **Limitarea onestă:** "nu există date de falsuri reale; folosim proxy-ul de
   analogi reali + validăm realismul." — onestitatea impresionează.
