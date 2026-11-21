# Raport: banca extinsă de molecule și seturile de antrenare

**Data raportului:** 27 octombrie 2026  
**Banca sursă:** `molecule_bank_500k/`  
**Setul combinat:** `balanced_datasets/generator_versions_200k/combined/`

## 1. Banca extinsă

`molecule_bank_500k/molecules.csv` conține **657.106 structuri unice**:

| Grup | Molecule | Proveniență |
|---|---:|---|
| Analogi generați | 512.104 | 435.254 din banca agregată, 45.836 analogi GA adăugați, 31.014 molecule M brute adăugate |
| Referințe publice | 145.002 | 65.002 din biblioteca publică a proiectului și 80.000 de compuși ChEMBL |
| **Total** | **657.106** | Structuri unice după eliminarea suprapunerilor exacte |

Adăugarea ChEMBL a parcurs 216 pagini de catalog. Din 87.556 candidați care
treceau filtrele proiectului, au fost selectați 80.000. Înainte de selecție au
fost eliminate duplicatele și suprapunerile exacte cu molecule existente.
Filtrele au inclus SMILES valid și canonic, o singură componentă, masă
moleculară 150–800 Da, 10–70 atomi grei, QED ≥ 0,35, SA ≤ 3,8 și absența
alertelor structurale configurate. Proveniența, versiunea sursei și licența
sunt păstrate în `molecule_bank_500k/chembl_controls.summary.json`.

**Interpretarea etichetelor este limitată:** label 1 înseamnă analog generat,
nu contrafacere confirmată; label 0 înseamnă compus public de referință,
nu medicament autentic sau aprobat verificat. La redistribuirea compușilor
ChEMBL trebuie atribuită sursa și respectată licența CC BY-SA 3.0.

## 2. Cum a fost construit setul mare de antrenare

Scriptul principal este `build_generator_balanced_datasets.py`. Cu sursele
curente, ieșirile sunt în `balanced_datasets/generator_versions_200k/`:

| Versiune | Total | Referințe, label 0 | Generați, label 1 |
|---|---:|---:|---:|
| M | 213.422 | 106.711 | 106.711 |
| E, inclusiv familia GA | 205.396 | 102.698 | 102.698 |
| Combinat M+E | 213.160 | 106.580 | 106.580 |

Pentru versiunea combinată, distribuția pe split-uri este:

| Split | Label 0 | Label 1 | Din label 1: M | Din label 1: E |
|---|---:|---:|---:|---:|
| Train | 85.808 | 85.808 | 42.904 | 42.904 |
| Val | 10.726 | 10.726 | 5.363 | 5.363 |
| Test | 10.046 | 10.046 | 5.023 | 5.023 |

Ordinea logică din script este:

1. Canonicalizează SMILES și colectează candidații cu proveniență M sau E.
   Moleculele atribuite simultan ambelor metode sunt excluse din versiunile
   individuale. Analogii GA unici sunt adăugați familiei E.
2. Grupează moleculele în train/val/test după scaffold-ul Bemis–Murcko al
   moleculei parentale, pentru ca aceeași familie de surse să nu traverseze
   split-urile.
3. Elimină din test moleculele al căror scaffold propriu apare în train sau
   val.
4. Echilibrează **în fiecare split** numărul de referințe și analogi generați.
   În setul combinat, alege și același număr de molecule M și E în clasa
   generată. Selecția din fiecare grup folosește seed-ul din build.

### Răspunsul despre subcategorii

Setul mare **nu este echilibrat pe subcategorii**. El este echilibrat între
clasele binare 0/1 în fiecare split și, pentru versiunea combinată, între
sursele M/E din clasa generată. Scriptul nu impune același număr de exemple
pentru `bioisostere`, `halogen-walk`, `grow-single`, `unconditioned` sau alte
categorii și subcategorii. Acestea rămân în coloanele `category` și
`subcategory`, cu frecvențe determinate de câte molecule există în sursă.

Pentru comparație, există setul separat
`balanced_datasets/subcategory_balanced/dataset.csv`: acesta include 85
subcategorii și **71 de analogi generați per subcategorie**, apoi tot atâtea
referințe publice, în total 12.070 rânduri. Trei subcategorii cu suport sub
50 au fost excluse. Echilibrul pe subcategorie este la nivelul întregului
fișier; split-urile individuale nu păstrează exact 71 per subcategorie,
deoarece moleculele sunt separate după scaffold.

### Varianta recomandată când vrem și scară, și echilibru pe subcategorii

Am adăugat `build_subcategory_weighted_training.py`, care creează
`balanced_datasets/training_200k/subcategory_weighted/dataset.csv`. Acesta
păstrează toate cele 213.160 de rânduri și split-urile existente; nu duplică
molecule și păstrează coloanele originale de proveniență. Pentru echilibrare,
folosește 79 de grupuri M și **6 grupe E armonizate**. Banca avea 12 etichete
E brute: cele cinci etichete E category-conditioned și cele cinci etichete
category-conditioned GA descriu aceleași cinci categorii chimice. Acestea sunt
grupate pe bioisostere, halogen-walk, homologation, other și scaffold-hop;
`unconditioned` și `size_stratified` formează grupa unconditioned. Câmpurile
`category`, `subcategory` și `generator_variant` originale rămân în fișier,
iar `balancing_category`/`balancing_subcategory` arată grupa folosită la
ponderare.

După armonizarea taxonomiei, banca are deja cel puțin 9.509 molecule unice în
fiecare grup E chimic (în afară de unconditioned, care are 69.370). Prin
urmare, pentru pragul de 5.000 molecule unice per grup E nu a fost nevoie să
generăm molecule suplimentare; deficitul aparent provenea din separarea
variantelor de generator ca subcategorii diferite.

| Grup E armonizat | Molecule unice în bancă |
|---|---:|
| Bioisostere | 10.223 |
| Halogen-walk | 9.651 |
| Homologation | 10.178 |
| Other | 9.509 |
| Scaffold-hop | 9.607 |
| Unconditioned | 69.370 |

În train, suportul observat pentru cele șase grupe E este:

| Grup E folosit la echilibrare | Exemple train |
|---|---:|
| Bioisostere | 3.608 |
| Halogen-walk | 3.579 |
| Homologation | 3.633 |
| Other | 3.381 |
| Scaffold-hop | 3.556 |
| Unconditioned | 25.147 |

Coloana `sample_weight` echilibrează contribuția la funcția de pierdere:

- label 0 are pondere 1 per referință;
- jumătate din ponderea totală a clasei generate este alocată familiei M și
  jumătate familiei E;
- în fiecare familie, fiecare grup folosit la echilibrare primește aceeași
  pondere totală, împărțită egal între moleculele din acel grup.

Astfel, suma ponderilor este egală între clase, M/E rămân egale ca familie,
iar subcategoriile au contribuții egale în interiorul familiei după
armonizarea de mai sus. În val și test
`sample_weight=1`; evaluarea rămâne neponderată. Nu se schimbă numărul de
molecule, ci importanța lor în loss. Unele grupuri au foarte puține exemple
(minimum 6 în train), deci greutatea maximă per moleculă ajunge la aproximativ
90,5. Ponderarea nu creează exemple noi; pentru concluzii solide trebuie
raportate și metrici separate pe subcategorie și interpretate prudent clasele
rare.

`train_gnn.py` citește `sample_weight` și îl folosește în loss. Pentru setul
ponderat, lansarea explicită este:

```bash
venv/bin/python train_gnn.py \
  --root balanced_datasets/training_200k \
  --dataset subcategory_weighted --model gin \
  --subcategory-sampling none
```

`load_3d_graph_dataset.iter_graphs()` returnează aceeași coloană de pondere
pentru grafurile din train; val/test au pondere 1. Modelul 3D trebuie să
înmulțească loss-ul fiecărui graf cu `sample_weight` și apoi să normalizeze
după suma ponderilor.

## 3. Exportul gata de antrenare

`export_training_splits.py` separă versiunea combinată fără să schimbe rânduri
sau etichete:

- `balanced_datasets/training_200k/train.csv`: 171.616 molecule, 85.808 per etichetă;
- `val.csv`: 21.452 molecule, 10.726 per etichetă;
- `test.csv`: 20.092 molecule, 10.046 per etichetă.

Acesta este setul 2D/SMILES de peste 100k molecule pentru train. Păstrează
metoda generatorului, categoria, subcategoria, scaffold-urile și metadatele
chimice, dar **nu reechilibrează subcategoriile**.

## 4. Versiunea 3D pentru GNN

`prepare_balanced_training_3d.py` a construit grafuri packed în shard-uri din
setul combinat. Pentru fiecare moleculă, procesul încearcă conformeri ETKDGv3,
minimizează cu MMFF94s (UFF fallback) și păstrează hidrogenii. Grafurile
conțin caracteristici atomice și de legătură, coordonate în Å, muchii covalente,
vecinătăți spațiale la raza de 5 Å, distanțe și unghiuri în radiani.

- Au fost create grafuri pentru **213.150 din 213.160** molecule; 10 eșecuri
  sunt în `balanced_datasets/training_200k/3d/failed_embeddings.csv`.
- `build_balanced_graph_index.py` a creat `balanced_graph_index.csv`, care
  selectează **213.146 grafuri** cu label 0/1 echilibrat exact în fiecare split.
  Patru grafuri din clasa majoritară au fost excluse pentru echilibru exact după
  eșecurile de geometrie.
- Train 3D: **171.614 grafuri**, 85.807 per etichetă. Val: 21.448; test: 20.084.
- `load_3d_graph_dataset.py` oferă `iter_graphs(root, split)` pentru citirea
  grafurilor cu indici locali per moleculă.

Coordonatele sunt predicții de conformer minimizat, nu date experimentale.
Versiunea 3D păstrează aceleași subcategorii, dar **nu le echilibrează**.

## 5. Comenzi de reproducere

```bash
venv/bin/python build_large_molecule_bank.py \
  --bank final_dataset_400k --ga-root version_ga --m-root version_m \
  --controls molecule_bank_500k/chembl_controls.csv \
  --out molecule_bank_500k --minimum 500000

venv/bin/python build_generator_balanced_datasets.py \
  --bank final_dataset_400k --ga-root version_ga \
  --authentic-extra molecule_bank_500k/chembl_controls.csv \
  --out balanced_datasets/generator_versions_200k \
  --seed 20261031 --min-per-class 100000

venv/bin/python export_training_splits.py \
  --source balanced_datasets/generator_versions_200k/combined/dataset.csv \
  --out balanced_datasets/training_200k

venv/bin/python prepare_balanced_training_3d.py \
  --input balanced_datasets/generator_versions_200k/combined/dataset.csv \
  --out balanced_datasets/training_200k/3d \
  --workers 6 --conformers 2 --shard-size 15000 --seed 20261026

venv/bin/python build_balanced_graph_index.py \
  --root balanced_datasets/training_200k/3d --seed 20261026
```

Sursele numerice detaliate sunt `molecule_bank_500k/summary.json`,
`balanced_datasets/generator_versions_200k/summary_all.json` și rezumatele
locale din directoarele de antrenare.
