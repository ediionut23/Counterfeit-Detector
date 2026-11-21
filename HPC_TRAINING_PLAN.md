# Plan de antrenare și optimizare pentru disertație

## Decizia principală

Da, merită antrenate modele graf mai capabile și optimizați hiperparametrii pe
calculatorul facultății. **Mărirea modelului este un experiment, nu obiectivul
științific în sine.** Întrebarea lucrării este cât de bine se transferă
detecția analogilor generați între metode, editări și contexte chimice, la
puține alarme false. Rezultatul se compară cu metode simple optimizate cu
aceeași grijă.

În [pilotul M→E](research_runs/GENERATOR_TRANSFER_REPORT.md), potrivirea
vecinătății imediate a editului reduce mare parte din diferența observată
între `M` și `E`. Un GNN mai mare care obține AUC ridicat pe splitul simplu
nu rezolvă automat această problemă. De aceea ordinea corectă este:
**protocol de evaluare → date și controale → implementare GPU → tuning →
test final**.

## 1. Ce există și ce limitează antrenarea astăzi

| Resursă | Situația curentă | Consecință pentru plan |
|---|---|---|
| Banca finală | 216.165 structuri generate unice, 222.772 relații, 84 de module; `M`, `E` și `EM` | Nu există un singur split universal; selectăm sarcina și păstrăm proveniența. |
| Negative eligibile | 27.261 molecule din biblioteca sursă | Un set echilibrat 1:1 cu toate cele 216.165 pozitive unice este imposibil cu negative unice din această bancă. |
| Edituri exacte | 28.166 relații `exact_single_atom` | Doar această parte susține evaluarea precisă a localizării editului. |
| Comparația existentă | 20k, GIN/GCN/GAT mici, 3 semințe, setări fixe | RF a avut AUC 0,851; GIN 0,815. Nu s-a făcut tuning cu buget comparabil. |
| `train_gnn.py` | 3 straturi, batch 64, opțiuni limitate, selecție pe AUC de validare; scor de test la fiecare rulare | Trebuie refăcut înainte de HPO: setări configurabile, scoruri salvate și test blocat în timpul căutării. |
| Constructorul general de date | `make_training_dataset_from_bank.py` separă scheletul părintelui; `--strict-child-test` filtrează o parte din suprapunerile copilului | Nu garantează singur separarea tuturor componentelor părinte–copil; este necesar un nou audit pentru benchmarkul final. |

„Pozitiv” înseamnă **analog generat**; „negativ” înseamnă **moleculă din
biblioteca sursă**. Niciuna dintre etichete nu certifică statutul clinic al
unui medicament. Această formulare trebuie păstrată în numele taskului,
figuri și lucrare.

## 2. Două trasee experimentale, cu rezultate raportate separat

### Traseul A: întrebarea de cercetare principală

Antrenare pe `M` exclusiv și evaluare pe `M` și `E` exclusive, cu același
set negativ. Folosim opt substituții elementare prezente în ambele surse.
Construim perechi M/E potrivite după semnătură, proprietăți, similitudine
față de părinte și prima vecinătate a atomului modificat. Adăugăm variante
cu potrivire la două legături și control al contextului mai îndepărtat.
Ținem separat componentele părinte–copil și ambele schelete ale fiecărei
molecule. Excludem structurile `EM` din comparația strictă a surselor.

Seturile deja văzute în `research_runs/generator_transfer_*` sunt **date de
dezvoltare**. Pentru rezultatul final se construiește un holdout nou și se
îngheață manifestul înainte de tuning. Testăm și direcția inversă `E→M`;
dacă numărul de exemple `E` este prea mic pentru o familie, reducem familia
sau raportăm lipsa de putere, fără a completa artificial cu `EM`.

### Traseul B: scalarea pe banca mare

Testăm efectul numărului de exemple și al capacității modelului. Începem cu
seturi de antrenare de aproximativ **5k, 10k și 20k pozitive**, cu negative
unice potrivite unde sunt suficiente. Numerele finale se stabilesc după
separarea strictă a datelor, fiindcă 27.261 este totalul eligibil înainte
de holdout și de restricțiile pe schelete. Un experiment ulterior poate
folosi o parte mai mare din cele 216k pozitive cu ponderare ori eșantionare
echilibrată în minibatch; evaluarea rămâne pe distribuții fixe și raportează
explicit prevalența. Nu repetăm aceleași negative în validare sau test
pentru a simula exemple noi.

Traseul B răspunde la întrebarea „ajută mai multe date și un model mai mare?”;
Traseul A răspunde la întrebarea publicabilă „ce transferă modelul?”.
Rezultatele nu se combină într-un singur scor mediu.

## 3. Protocolul de date înainte de primul job GPU

1. **Înghețăm o versiune a băncii**: hash pentru `molecules.csv`, `relations.csv`,
   `authentic_eligible.csv`, codul constructorului și lista modulelor.
2. **Alegem unitatea de independență**: moleculă, părinte, relație și componentă
   părinte–copil. Audităm și scheletul copilului, nu numai pe cel al părintelui.
   Dacă separarea simultană este prea restrictivă pe banca mare, măsurăm
   exact câte exemple pierdem și raportăm o variantă strictă și una mai largă.
3. **Fixăm patru roluri**: train; validare pentru checkpoint și HPO; set
   separat de calibrare a pragului; test final. Testul final nu intră în
   jurnalele de HPO și nu decide alegerea arhitecturii.
4. **Păstrăm controalele**: negative din biblioteca sursă, martori MMP
   apropiați de părinte, generator ascuns, familii de edituri și contexte
   locale. Controalele din aceeași bibliotecă rămân teste interne; o sursă
   independentă este necesară pentru validare externă.
5. **Verificăm distribuțiile** pe fiecare rol: `M/E/EM`, categorie,
   semnătură, masă, logP, număr de atomi, similitudine, schelete și părinți.
   Fiecare split primește CSV, manifest JSON și SHA-256.

Pentru o estimare credibilă la FPR de 1% sunt necesare mai multe negative
decât cele 1.000 folosite în pilot: acolo un singur fals pozitiv schimbă
FPR cu 0,1 puncte procentuale. Ținta practică este **cel puțin 5.000 de
negative în calibrare și 5.000 în fiecare test important**, dacă rămân
suficiente după separarea datelor. Dacă nu, publicăm intervale de încredere
largi și nu pretindem că am demonstrat un prag strict de 1%.

## 4. Familiile de modele

| Grup | Modele | De ce intră în comparație |
|---|---|---|
| Controale simple | 3 proprietăți, regresie logistică pe Morgan, vecin apropiat pe Morgan | Detectează scurtături chimice și oferă o referință reproductibilă. |
| Modele tabulare puternice | Random Forest și XGBoost pe aceleași amprente Morgan | RF este deja cel mai bun AUC pe subsetul 20k cu setări fixe; trebuie optimizat serios. |
| GNN de bază | GIN, GCN, GAT/GATv2 | Reproducem arhitecturile existente pe același protocol nou. |
| GNN cu legături | GINE ori un model de tip directed message passing | Testăm dacă tipul legăturii și reprezentarea editului explică un câștig real. |
| Opțional, separat | Model părinte–candidat cu două grafuri; model 3D | Altă informație la intrare și altă întrebare. Nu le comparăm direct cu detectorul care primește doar candidatul. |

Comparația GNN folosește inițial trei trepte de capacitate: **mic** (3
straturi, 64/128 dimensiuni), **mediu** (4–5 straturi, 128/256) și **mare**
(6–8 straturi, 256/512, cu conexiuni reziduale și normalizare). Acestea
sunt intervale de căutare, nu promisiuni că cel mai mare va câștiga. În
Traseul A sunt aproximativ 4.000 de pozitive `M` de antrenare în spliturile
actuale; acolo un model foarte mare poate învăța particularități ale
sursei, nu o regulă generală.

Toate modelele din sarcina **structură doar** primesc structura candidatului
și caracteristici atom/legătură derivate din ea. `method`, `category`,
`parent_smiles`, `edited_atoms`, `split` și identificatorii sunt metadate
pentru construcție/evaluare, nu predictori. Modelul părinte–candidat se
raportează ca sarcină separată și se compară cu o regulă deterministă de
diferență între grafuri.

## 5. Tuningul: ce căutăm și pe ce metrică

Facem mai întâi **10 încercări scurte per familie** pe un subset de
dezvoltare pentru a verifica fluxul, memoria și timpul per epocă. Urmează
aproximativ **25–40 încercări per familie importantă** pe datele de
dezvoltare complete, cu aceeași politică de selecție și jurnal al
costului. Acesta este un buget inițial, care se ajustează după timpul real
al unui trial; raportăm atât numărul de trialuri, cât și orele GPU/CPU.
Nu alegem un GNN după zeci de încercări și lăsăm RF/XGBoost pe setările
implicite.

| Parametru | Domeniu inițial rezonabil |
|---|---|
| GNN: straturi | 2–6 în HPO; 8 numai în testul explicit de scalare |
| GNN: dimensiune ascunsă | 64, 128, 256; 512 doar dacă profilarea justifică |
| GNN: batch | 64, 128, 256, eventual 512 dacă încape și este mai rapid |
| GNN: rată de învățare | 1e-4–3e-3, eșantionare logaritmică |
| GNN: weight decay | 1e-6–1e-3, eșantionare logaritmică |
| GNN: dropout | 0–0,4; normalizare și rezidual ca opțiuni declarate |
| RF | 300–1.000 arbori, frunză minimă 1–10, subeșantionarea caracteristicilor |
| XGBoost | adâncime 3–8, rată 0,01–0,2, `min_child_weight`, subsample, `colsample`, regularizare și oprire timpurie |

La început nu căutăm simultan zeci de parametri. Pentru epoci folosim o
limită de circa 100–150 și oprire timpurie pe validare; trialurile slab
promițătoare pot fi întrerupte după o perioadă minimă. [Optuna](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
oferă eșantionare și prunere; [Ray Tune](https://docs.ray.io/en/latest/tune/getting-started.html)
oferă ASHA și alocare de resurse pentru trialuri. Pentru o singură mașină,
începem cu Optuna și joburi independente. Nu adăugăm Ray decât dacă
planificarea multor GPU-uri sau noduri o cere.

**Selecția modelului** se face pe validare, cu AUC parțial în zona FPR
0–5% ca obiectiv principal de HPO și AUC global ca diagnostic. AUC parțial
este disponibil prin `roc_auc_score(max_fpr=0.05)` în
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
După alegerea configurației, pragul pentru FPR ≤1% se fixează pe setul
de calibrare, fără recalibrare pe test. Calibrarea și reglarea pragului pe
datele de antrenare ar produce optimism; vezi
[documentația scikit-learn despre praguri](https://scikit-learn.org/1.5/modules/classification_threshold.html).

## 6. Cum folosim calculatorul facultății

### Inventar și probă de funcționare

Pe calculator, într-un cont obișnuit, notăm modelul GPU, memoria GPU,
numărul de GPU-uri, RAM, CPU, spațiul liber și dacă există Slurm:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
lscpu
free -h
df -h
command -v sbatch || true
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Nu copiem mediul `venv` de pe macOS. Creăm un mediu Linux nou și instalăm
PyTorch pentru driverul și acceleratorul mașinii după
[selectorul oficial](https://pytorch.org/get-started/locally/), apoi PyG
conform [compatibilității PyTorch/CUDA](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
Pinăm versiunile verificate într-un fișier separat și facem un test de
import plus o epocă de antrenare. Biblioteca PyG folosește
[`DataLoader` pentru loturi de grafuri independente](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html);
`NeighborLoader` este destinat în principal eșantionării din grafuri mari,
nu necesar aici pentru mii de molecule mici.

Precalculăm grafurile RDKit și amprentele **o singură dată pentru fiecare
versiune de date**, păstrând hashul SMILES și schema de caracteristici.
Profilăm separat conversia SMILES, încărcarea datelor, pasul GPU și
evaluarea. Testăm `num_workers`, `pin_memory` și `persistent_workers`
doar după măsurare; [ghidul PyTorch pentru încărcarea datelor](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)
arată de ce acestea pot ajuta. Testăm mixed precision pe GPU și verificăm
că scorurile nu se schimbă material; [PyTorch AMP](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
documentează `autocast` și scalarea gradientului.

Cu mai multe GPU-uri, **rulăm mai întâi trialuri HPO independente pe GPU-uri
diferite**. Trecem la `DistributedDataParallel` numai dacă un singur trial
mare este prea lent și o probă arată accelerare netă; DDP replică modelul
pe GPU-uri și sincronizează gradientele, conform
[tutorialului oficial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html).
Dacă există Slurm, un job cere explicit GPU, CPU, memorie și timp prin
`sbatch`; opțiunile exacte depind de configurația facultății
([documentația Slurm](https://slurm.schedmd.com/sbatch.html)).

### Buget calculat dintr-o măsurătoare, nu dintr-o presupunere

După 3–5 trialuri pilot notăm timpul unei epoci, epocile până la oprire,
memoria maximă GPU și utilizarea GPU. Estimăm:

`ore GPU ≈ număr de trialuri × durata medie a trialului × GPU-uri per trial`

Rezervăm separat timp pentru refacerea celor mai bune configurații pe
**3 semințe de split × 3 semințe de model** și pentru toate testele finale.
Pe un singur GPU se execută secvențial; pe mai multe GPU-uri se distribuie
trialurile fără suprasubscrierea memoriei. Salvăm checkpoint la cel mai bun
scor de validare și posibilitatea de reluare după expirarea unui job.

## 7. Evaluarea finală și tabelele lucrării

După HPO înghețăm pentru fiecare model configurația, epoca de selecție și
politica de calibrare. Reantrenăm modelele finaliste pe semințele stabilite
și aplicăm **o singură dată** pe testele finale:

| Test | Metrici minime |
|---|---|
| În domeniu, componente/schelete neîntâlnite | AUC, AUC parțial, TPR la pragul de 1%, FPR efectiv, AUPRC și timp de inferență |
| Transfer M→E și E→M | Aceleași metrici, separat pe `M` și `E`; ΔAUC și ΔTPR pe perechi |
| Vecinătate edit identică/diferită | Rezultate per subgrup și per semnătură; câte exemple are fiecare |
| Negative MMP apropiate | FPR pe martori apropiați, nu doar pe negative aleatoare |
| Calibrare și prevalență | Diagramă de calibrare, Brier/ECE, precizie proiectată la prevalențe declarate |
| Stabilitate | Medie și dispersie între semințe de split/model; intervale bootstrap grupate pe schelet/componente |

Pentru FPR raportăm și numărul de alarme false, numărul de negative și
intervalul de încredere, nu doar un procent rotunjit. Pentru AUC parțial,
declarăm exact intervalul FPR și varianta standardizată. Nu agregăm
rezultatele din subcategorii foarte mici într-o afirmație generală.
Testul final nu se folosește pentru a mai alege hiperparametri după primul
rezultat.

## 8. Ce trebuie implementat în repo înaintea rulării mari

1. Constructor de splituri stricte și manifest înghețat pentru traseele A/B,
   cu audit independent al componentelor, părinților și ambelor schelete.
2. Refactorizare `train_gnn.py` într-un trainer configurabil: straturi,
   dimensiune, batch, dropout, edge features, normalizare, residual, AMP,
   checkpoint, jurnal per epocă, predicții și scoruri pe validare. Modul HPO
   **nu citește testul**. Actualul `train_gnn.py` trebuie păstrat ca reper.
3. Runner HPO care înregistrează toate trialurile, timpul, memoria, config,
   splitul și cauza opririi; aceeași disciplină pentru RF/XGBoost.
4. Evaluator unic pentru toate arhitecturile: calibrare pe set separat,
   teste de proveniență/hard-negative, metrici și intervale, fișiere de
   predicții cu ID stabil.
5. Scripturi pentru joburi GPU/Slurm și manifest de mediu, fără căi locale
   macOS. Test de reluare din checkpoint înainte de campania completă.

Structura recomandată pentru fiecare rulare:

```text
runs/<experiment>/<split_seed>/<model>/<trial_or_seed>/
  config.json
  data_manifest.json
  environment.json
  history.csv
  checkpoint.pt
  validation_predictions.csv
  calibration_predictions.csv
  test_predictions.csv          # creat numai în etapa finală
  metrics.json
```

## 9. Calendar de execuție și criterii de trecere

Ordinea de mai jos este mai importantă decât durata; săptămânile sunt o
estimare de organizare și se ajustează după timpul de acces la calculator.

| Interval orientativ | Activitate principală |
|---|---|
| Săptămâna 1 | Inventar hardware, splituri noi, audit, mediu Linux și trainer GPU fără acces la test. |
| Săptămâna 2 | Baselines puternice, probă de throughput, trialuri scurte și fixarea bugetului HPO. |
| Săptămânile 3–4 | HPO pe Traseul A, apoi pe Traseul B dacă A funcționează; arhivarea fiecărui trial. |
| Săptămâna 5 | Reantrenări pe semințe, calibrare separată, evaluarea finală și intervale de incertitudine. |
| Săptămâna 6 | Ablări, figuri, verificare reproductibilitate și redactarea secțiunilor de metode/rezultate. |

| Etapă | Livrabil | Trecem mai departe când... |
|---|---|---|
| Pregătire date | splituri și audit semnate prin hash | nu există suprapuneri nedeclarate și fiecare grup de test are număr suficient |
| Adaptare GPU | o epocă + checkpoint + reluare | modelul produce aceleași tipuri de scoruri ca evaluatorul și folosește GPU-ul corect |
| Pilot hardware | tabel timp/epocă, memorie, throughput | putem estima bugetul real și nu saturăm RAM/VRAM |
| HPO | istoric complet pentru baselines și GNN | selecția s-a făcut fără acces la test, cu buget comparabil documentat |
| Confirmare | 3×3 semințe pentru finaliști, teste înghețate | efectul depășește variația între splituri și este relevant la FPR mic |
| Redactare | tabele, ablații, cod, manifest, limite | concluziile se referă la analogi virtuali și la sursele efectiv testate |

Dacă RF/XGBoost rămân mai bune după tuning echitabil, acesta este un rezultat
științific util. Dacă GNN câștigă numai AUC global, dar pierde la FPR mic
sau pe editurile locale bine potrivite, nu îl prezentăm drept progres
operațional. Prioritatea pentru conferință este un benchmark riguros și o
explicație verificabilă a erorilor, nu dimensiunea rețelei.
