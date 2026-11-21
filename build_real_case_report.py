#!/usr/bin/env python3
"""Write the project report from the case, overlap and measured-data outputs."""

import csv
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


root = Path("real_world_falsification/v1")
case_summary = read_json(root / "summary.json")
overlap = read_json(root / "prior_model_overlap/summary.json")
diagnostic = read_json(root / "gnn_diagnostic/summary.json")
field = read_json("real_world_falsification/field_spectra_doxy/summary.json")
with (root / "gnn_diagnostic/case_diagnostics.csv").open(newline="") as stream:
    scored = list(csv.DictReader(stream))
unique = {r["relation_id"]: r for r in scored}
source_manifest = read_json("real_world_falsification/source_cache/manifest.json")
cached_count = sum(r["status"] in {"downloaded", "cached"} for r in source_manifest["records"])
table = "\n".join(f"| {r['declared_name']} | {r['observed_name']} | {float(r['declared_p_generated']):.3f} | {float(r['observed_p_generated']):.3f} |"
                  for r in unique.values())

report = f"""# Registru de cazuri și pregătirea validării cu date reale

Versiune de lucru: 31 octombrie 2026. Obiectivul acestei etape este să facă
distincția verificabilă între molecule generate, compoziție măsurată și
falsificarea confirmată într-un raport oficial. Un detector general de
medicamente falsificate nu este încă validat prin aceste rezultate.

**Ce am construit.** Registrul `real_world_falsification/v1/cases.csv` are
{case_summary['reported_case_records']} înregistrări de produs/prezentare/lot din
{case_summary['source_report_clusters']} rapoarte oficiale. Sursele sunt WHO,
NSW Health și US DOJ, cu rezultate de laborator FDA. Am arhivat
{cached_count} documente/pagini oficiale, cu URL și SHA-256. Cele
{case_summary['unique_compound_records']} substanțe au identități PubChem
descărcate și păstrate separat, fără SMILES ghicite.

Exemple de dovezi sunt [ponatinib declarat, paracetamol identificat](https://www.who.int/news/item/01-02-2019-medical-product-alert-n-2-2019-%28english-version%29)
și [hidroclorotiazidă declarată, glibenclamidă identificată](https://www.who.int/news/item/17-04-2019-medical-product-alert-n-6-2019-%28english-version%29).
Compoziția identificată este specifică probelor/loturilor din fiecare raport.
Nu o extindem tuturor produselor cu aceeași denumire sau același număr de lot.

**Ce acoperă datele.** Sunt {case_summary['single_graph_pair_records']} înregistrări
de substituție cu un compus declarat și un substitut identificat, reprezentând
{case_summary['unique_single_substitution_relations']} relații chimice distincte.
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
{overlap['source_rows_by_split'].get('train', 0):,}, validation
{overlap['source_rows_by_split'].get('val', 0):,}, test
{overlap['source_rows_by_split'].get('test', 0):,} rânduri. În intrările de train
apar {len(overlap['training_input_exact_identity_exposure'])} identități din registru
și {len(overlap['training_input_scaffold_exposure'])} compuși au scheletul prezent
în train. Au existat {overlap['normalization_failures']} normalizări nereușite,
listate separat. Aceasta este expunere chimică, fără a afirma că etichetele
cazurilor reale au fost folosite la antrenare.

Am legat {overlap['synthetic_parent_relations_linked_to_case_compounds']} relații
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
{table}

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
{field['field_source_sample_records']} înregistrări de metadate, grupate conservator
în {field['conservative_source_specimen_groups']} grupuri de probă și
{field['conservative_source_lot_groups']} grupuri lot/producător. Sunt
{field['field_scans_with_unambiguous_assay_metadata']} scanări asociate fără
ambiguitate cu metadatele HPLC, pe {field['spectral_channels']} canale spectrale.
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
"""
Path("REAL_WORLD_FALSIFICATION_REPORT.md").write_text(report)
print("Wrote REAL_WORLD_FALSIFICATION_REPORT.md")
