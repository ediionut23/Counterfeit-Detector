# Registru de cazuri și pregătirea validării cu date reale

Versiune de lucru: 31 octombrie 2026. Obiectivul acestei etape este să facă
distincția verificabilă între molecule generate, compoziție măsurată și
falsificarea confirmată într-un raport oficial. Un detector general de
medicamente falsificate nu este încă validat prin aceste rezultate.

**Ce am construit.** Registrul `real_world_falsification/v1/cases.csv` are
17 înregistrări de produs/prezentare/lot din
11 rapoarte oficiale. Sursele sunt WHO,
NSW Health și US DOJ, cu rezultate de laborator FDA. Am arhivat
18 documente/pagini oficiale, cu URL și SHA-256. Cele
19 substanțe au identități PubChem
descărcate și păstrate separat, fără SMILES ghicite.

Exemple de dovezi sunt [ponatinib declarat, paracetamol identificat](https://www.who.int/news/item/01-02-2019-medical-product-alert-n-2-2019-%28english-version%29)
și [hidroclorotiazidă declarată, glibenclamidă identificată](https://www.who.int/news/item/17-04-2019-medical-product-alert-n-6-2019-%28english-version%29).
Compoziția identificată este specifică probelor/loturilor din fiecare raport.
Nu o extindem tuturor produselor cu aceeași denumire sau același număr de lot.

**Ce acoperă datele.** Sunt 9 înregistrări
de substituție cu un compus declarat și un substitut identificat, reprezentând
7 relații chimice distincte.
Celelalte înregistrări descriu 4 cazuri cu API declarat nedetectat și substitut
necunoscut, un amestec de substituție, un contaminant și 2 cazuri fără rezultat
de compoziție publicat. Loturile repetate au grupuri comune de sursă/relație.
Numărul surselor nu demonstrează independență statistică.

`components.csv` și `composition_sets.jsonl` păstrează separat declarația,
detecția, nedetectarea, cantitățile raportate și necunoscutele. Lipsa API-ului nu
devine o moleculă goală. Un contaminant identificat nu devine întreaga compoziție
a produsului. Substanțele sunt reprezentate prin forma activă normalizată;
sarea și formularea declarată rămân în câmpuri distincte. Structurile PubChem
nu sunt structuri 3D experimentale ale produselor analizate.

**Intrări pentru modele.** `load_claim_observation_graphs.py` produce pentru
fiecare caz lista grafurilor declarate, lista componentelor identificate și
stările de absență/analiză. Amestecurile păstrează mai multe grafuri distincte.
Substanțele individuale nu primesc automat eticheta de produs autentic sau
falsificat. Toate cazurile oficiale rămân `external_case_challenge_only`.

**Expunerea în datele vechi.** Auditul a scanat setul mare existent: train
171,616, validation
21,452, test
20,092 rânduri. În intrările de train
apar 5 identități din registru
și 12 compuși au scheletul prezent
în train. Au existat 0 normalizări nereușite,
listate separat. Aceasta este expunere chimică, fără a afirma că etichetele
cazurilor reale au fost folosite la antrenare.

Am legat 8 relații
M/E la identități de referință din registru. Acestea rămân simulări de substituție
structurală, cu `confirmed_real_case=0`, și sunt marcate pentru carantină în
viitoare antrenări care folosesc acest set de cazuri ca holdout.
`external_holdout.json` listează identitățile și scheletele de exclus din ambele
laturi ale viitoarelor perechi de antrenare. Modelul deja antrenat nu devine
retroactiv lipsit de suprapuneri.

**Diagnosticul GATv2 existent.** Am încărcat checkpointul salvat, fără o nouă
antrenare. Coloanele de mai jos reprezintă probabilitatea clasei vechi
„analog generat”, pe structura normalizată. Nu reprezintă probabilitatea ca
produsul să fie falsificat.

| Compus declarat | Compus identificat | p(generat), declarat | p(generat), identificat |
|---|---|---:|---:|
| ponatinib | paracetamol | 0.004 | 0.381 |
| phenoxymethylpenicillin | paracetamol | 0.005 | 0.381 |
| hydrochlorothiazide | glibenclamide | 0.624 | 0.573 |
| diazepam | haloperidol | 0.261 | 0.317 |
| oxymorphone | metonitazene | 0.124 | 0.283 |
| orlistat | sibutramine | 0.807 | 0.546 |
| diazepam | bromazolam | 0.261 | 0.375 |

Aceeași moleculă, de exemplu paracetamolul, primește exact același scor indiferent
de substanța declarată. Modelul nu primește acel context. Datele de normalizare
pot diferi de formele farmaceutice din antrenarea veche; tabelul este un diagnostic.
Nu raportăm ROC-AUC, F1 sau specificitate medicală pe acest registru de cazuri
pozitive fără controale reale autentificate. Cu identitățile chimice deja
stabilite în laborator, comparația deterministă a structurilor este reperul de
bază și identifică diferențele în toate perechile selectate prin construcție.

**Date măsurate complementare.** În
`real_world_falsification/field_spectra_doxy/` am importat date publice NIR/HPLC
din [Awotunde și colaboratorii, 2024](https://doi.org/10.1039/D4AY00001C), cu
[depozitul autorilor](https://github.com/aawotund/Mitigating-gelatin-capsule-impact-on-detection-of-substandard-and-falsified-pharmaceuticals-with-NIR)
fixat la un commit și licența originală păstrată. Probele comerciale și
calibrările simulate sunt evidențiate distinct. Exportul are
26 înregistrări de metadate, grupate conservator
în 23 grupuri de probă și
7 grupuri lot/producător. Sunt
90 scanări asociate fără
ambiguitate cu metadatele HPLC, pe 228 canale spectrale.
Aceste scanări nu reprezintă tot atâtea probe independente.

Am păstrat discrepanța față de numărătorile din articol și o valoare HPLC de
121,6 mg care depășește limita de 120 mg discutată în studiu. Coloana MG=100
dintr-un alt fișier reprezintă doza nominală; nu a fost folosită drept rezultat
HPLC. Cele 208 scanări fără asociere sigură cu rezultatele de laborator rămân
arhivate separat. Rezultatul HPLC în afara limitelor
indică neconformitate cantitativă; nu demonstrează singur intenția de fraudă.
Cu un singur API declarat, un encoder molecular primește aceeași substanță la
toate aceste probe; reperul adecvat este mai întâi un model spectral. O viitoare
fuziune graf–spectru cere date de la mai multe API-uri și loturi.
Pentru un predictor spectral, intrările sunt coloanele `nir_*`; rezultatele HPLC,
stările de conformitate și identificatorii de proveniență sunt ținte/metadate și
nu trebuie introduși ca predictori.

**Verificări efectuate.** Am verificat identitățile și parsarea RDKit, unicitatea
ID-urilor de caz, sursele obligatorii, păstrarea amestecurilor, absențele distincte
de necunoscute, lipsa etichetelor de autenticitate pe grafurile componentelor,
partiția externă a cazurilor și limitele de citare ale fragmentelor. Descărcările
conțin date/documente și licențe; codul autorilor nu este executat. Hashurile,
versiunile bibliotecilor și scripturile sunt în manifestul proiectului pilot.

**Ce urmează pentru validarea detectorului.** Avem nevoie de observații măsurate
legate de eticheta de pe produs și de un verdict documentat, împreună cu probe
conforme autentificate. Validarea trebuie grupată după probă, lot/producător și
sursă, astfel încât replicile spectrale să nu ajungă în train și test.
Pentru substituții putem folosi referința moleculară plus un encoder al
observației; pentru doză sunt necesare măsurători cantitative, iar pentru
falsificarea provenienței sunt necesare informații de trasabilitate/ambalaj.
Mărirea băncii M/E oferă preantrenare sintetică, fără să creeze cazuri reale.

Reproducerea completă și comenzile pentru fiecare etapă sunt în
[README-ul pilotului](real_world_falsification/README.md). Nicio nouă antrenare
lungă nu a fost pornită în această etapă.
