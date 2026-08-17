# Ghid C1 — Datasetul: tot ce trebuie să știi + întrebările coordonatorului

Document de pregătire pentru discuția cu coordonatorii (prof. Breabăn, asist. Cornei).
Termenii tehnici sunt în engleză (ca în propunere); explicațiile în română.

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
| **Scaffold split** (nu random) | testezi pe schelete nevăzute → măsori învățarea *editului*, nu memorarea scaffold-urilor |
| **Conformeri 3D** | pentru experimentele geometrice (C1/C3) |
| **Atomii editați (ground-truth)** | permit scorarea explicabilității (explanation accuracy, Fidelity, Sparsity) "gratis" |
| **E model-free la benchmark** | un dataset trebuie să fie independent de model, reproductibil |

---

## PARTEA 4 — Numerele actuale

- **Molecule reale de mining:** ~58.600 (ChEMBL + PubChem)
- **Catalog:** 1.732.359 reguli brute → **3.257 reguli** filtrate (support median ridicat)
- **Version M:** 50.000 contrafaceri (10k × 5 categorii)
- **Version E:** ~1.900 contrafaceri evoluate (mic, e nivelul greu)
- **Benchmark:** 44 dataset-uri (5 categorii + subcategorii + evolved + pooled)
- **pooled:** ~97.000 molecule (1:1 autentic:contrafăcut)
- **Conformeri 3D:** ~105.000 molecule, ~100% coverage

**Categoriile:** bioisostere, scaffold-hop, halogen-walk, homologation, other.

---

## PARTEA 5 — Validarea (dovezile, nu presupunerile)

**Realism** (`audit_realism.py`): contrafacerile sunt valide/unice/noi ~100%,
similaritate cu părintele ~0.6-0.7 (bandă realistă), QED egal cu al medicamentelor
reale, **fără shortcut de proprietate** (LR trivial ~0.5).

**Învățabilitate** (`train_baseline.py`, `train_gnn.py`), pe split scaffold:
- **M**: fingerprint AUC 0.80-0.95, GNN 0.97 → **învățabil, semnal structural**.
- **E**: 0.53 pe puține date → **0.67** pe 2× date → **greu dar învățabil**.
- property-only ~0.5 → **nu e shortcut**.

**Constatarea cheie:** gap-ul **M ≈ 0.95 vs E ≈ 0.67** = *detectorii prind
modificările enumerate, dar se chinuie cu cele descoperite evolutiv.* Asta e
întrebarea centrală a benchmark-ului, cu răspuns măsurat.

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
4. **E (evoluat) e mic** — generarea evolutivă e scumpă (NSGA-II per părinte).
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
proprietăți (~0.5 = nu se separă). Deci semnalul trebuie să vină din structură.

**Î: De ce scaffold split și nu random?**
R: Un split random lasă modelul să memoreze scaffold-uri și să pară bun fals.
Scaffold split (familii disjuncte, MoleculeNet/OGB) testează pe schelete
nevăzute → măsoară dacă a învățat *editul*. Dovada: pe E, GNN-ul dă 0.90 pe val
(same-scaffold) dar 0.67 pe test (scaffold nevăzut) — random ar fi mințit.

**Î: Cum faci balansul 1:1 dacă un părinte dă 10 contrafaceri?**
R: Raportul "1 părinte : N falsuri" e la generare, irelevant pentru balans.
Balansul 1:1 e la asamblare: iau **la fel de multe molecule autentice câte
contrafaceri**, molecule *separate* de părinți, property-matched. Părintele e
reținut doar ca origine (pentru split + ground-truth).

### C. Realism / validitate (cele mai probabile)

**Î: Sunt realiste contrafacerile astea? Sau inventezi molecule?**
R: Le-am *măsurat* (audit_realism.py): valide/unice/noi ~100%, similaritate cu
părintele ~0.65 (bandă realistă, nu clone, nu random), QED egal cu autenticele.
Regulile vin din perechi de medicamente reale (support median mare). Deci
ancorate în chimie reală, nu inventate.

**Î: Nu e circular? Generezi și testezi cu aceleași reguli?**
R: Pentru M, regulile minate NU au fost folosite la antrenarea vreunui model —
sunt doar sursa datelor. Pentru E adversarial *ar fi* circular (de-aia la
benchmark E e **model-free**). Iar întrebarea de generalizare o testez R/M→E
(antrenezi pe unele, testezi pe altele).

**Î: De unde știi că nu e doar zgomot (moleculele indistinctibile)?**
R: Testul de învățabilitate: pe scaffold-uri nevăzute, fingerprint-ul dă AUC
0.8-0.95 pe M → există semnal real. Iar property-only ~0.5 → nu e shortcut de
proprietate.

**Î: De ce ai puține molecule evoluate (E)?**
R: E e scump — fiecare părinte cere o rulare NSGA-II completă, apoi păstrez doar
frontul Pareto filtrat (in-band + druglike + dedup) → ~2-3/părinte. E nivelul
*greu, descoperit*, gândit să fie mic și dificil, nu produs în masă. Scalabil cu
mai mulți părinți dacă e nevoie.

### D. Comparație cu literatura

**Î: Cu ce diferă de DUD-E / decoys?**
R: DUD-E generează *decoys* (molecule diferite, matched pe proprietăți). Aici
contrafacerile sunt *derivate din* medicament printr-un edit local cunoscut —
avem și ground-truth-ul atomilor modificați. Am împrumutat de la DUD-E doar
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

**Î: Cum garantezi reproductibilitatea?** → Totul determinist cu `--seed` (42);
cod, split-uri fixe, catalog, mediu versionat — se re-derivă orice număr.

**Î: Ce metrici raportezi?** → FNR (un fals ratat e eroarea costisitoare) întâi,
apoi Precision/Recall/F1, plus AUC — și **per categorie**, nu doar media.

---

## Cheat-sheet: 5 lucruri de spus sigur

1. **"Eticheta e atribuită, nu măsurată"** — arată că înțelegi natura problemei.
2. **"Property matching + scaffold split"** — arată că eviți shortcut-urile și
   testezi onest.
3. **"Support-ul din mining = realism data-driven"** — regulile-s reale, nu
   inventate.
4. **"Gap M vs E măsurat (0.95 vs 0.67)"** — ai un rezultat, nu doar date.
5. **Limitarea onestă:** "nu există date de falsuri reale; folosim proxy-ul de
   analogi reali + validăm realismul." — onestitatea impresionează.
