# Extinderea băncii și îmbunătățirea generării

31 octombrie 2026. Plan pentru o bancă sintetică mare, cu identitate declarată
și scenarii explicite, legată de registrul de dovezi reale.

Rezumatul actual din `molecule_bank_500k/summary.json` raportează 657.106 molecule
unice: 512.104 analogi generați și 145.002 referințe publice. Acestea sunt
numărătorile băncii consolidate, nu o nouă numărătoare a tuturor batch-urilor
brute existente. Referința publică nu dovedește statut de medicament aprobat,
autenticitatea unui produs sau utilizarea într-un caz real.

**Utilizarea băncii mari.** Recomandarea este să folosim catalogul chimic pentru
preantrenare: modelul învață să reconstruiască atomi/legături mascate sau să
asocieze reprezentarea 2D cu geometria 3D. Acest pas poate utiliza multe molecule
fără etichete de falsificare. Modelul este apoi adaptat la observația condiționată
de substanța declarată, cu etichete ale sarcinii respective și evaluare externă.
Strategia este susținută metodologic de lucrări despre
[preantrenarea GNN-urilor](https://arxiv.org/abs/1905.12265) și
[GraphMVP pentru 2D/3D](https://openreview.net/forum?id=xQUe1pOKPam).
Beneficiul pentru acest proiect trebuie măsurat; transferul nu este garantat.
Preantrenarea poate utiliza aproape întregul catalog disponibil, după
excluderile de evaluare, fără plafonul impus de balansarea claselor din setul
supervizat de 213.160 de rânduri. La această scară, grafurile trebuie citite în
shard-uri de pe disc și încărcate pe batch-uri; loaderul actual care construiește
liste cu toate grafurile în RAM trebuie extins înaintea unei rulări cu un milion.

**Unitatea de antrenare.** Catalogul chimic trebuie să stocheze fiecare structură
o singură dată, iar tabelul de observații trebuie să stocheze separat identitatea
declarată, componentele identificate/simulate, scenariul și sursa etichetei.
Paracetamolul poate fi compatibil cu un produs care declară paracetamol și
incompatibil cu un produs care declară ponatinib. Eticheta aparține observației
și declarației; identitatea moleculei singure nu decide falsificarea produsului.

Propunem trei tabele distincte:

| Tabel | Unitatea numărată | Câmpuri esențiale |
|---|---|---|
| `molecules` | structură chimică unică | ID, SMILES original și izomeric standardizat, InChIKey, proveniență, forme/săruri, calitate chimică |
| `relations` | relație declarată–observată | referință declarată, molecule observate, M/E sau compus cunoscut, operații, categorie verificată, scenariu, evidence tier |
| `samples` | observație de produs/compoziție | componente și cantități, metodă de măsurare/simulare, lot/grup, semnale disponibile, partiție |

Un milion de perechi, mai multe amestecuri sau mai mulți conformeri nu înseamnă
automat un milion de molecule distincte. Toate numărătorile trebuie raportate.

**Familii de generare.**

1. **Analogi structurali sintetici.** M produce modificări locale, iar E explorează
   combinații de modificări și alte schelete. Fiecare rezultat păstrează referința
   inițială și operațiile. Aceste date pot susține preantrenarea și evaluarea
   controlată a modificărilor structurale.
2. **Substituții cu compuși cunoscuți.** Un sampler produce perechi de identități
   medicamentoase/compuși identificați din surse curate. Scenariile simulate sunt
   distincte de substituțiile documentate în cazurile externe. Niciun compus nu
   devine global „contrafăcut” pentru că apare într-o astfel de pereche.
3. **Compoziții, absențe și cantități.** Amestecurile sunt seturi de componente;
   lipsa API-ului și doza sunt observații/stări. Aceste scenarii pot avea date
   simulate de compoziție, dar nu produc molecule noi și nu creează spectre reale.
   Un model de doză sau contaminare necesită intrări analitice relevante.

**Referințe cu identitate păstrată.** Pregătirea următoarei bănci începe cu un
registru de API-uri și compuși de referință cu nume, identificatori și statutul
sursei (de exemplu ChEMBL plus etichete de produs verificate). Menținem distincte
substanța activă, sarea și stereochimia. Părinții generici existenți rămân
marcați ca referințe publice, fără statut de API confirmat inventat ulterior.
`evolve_category_pilot.py` păstrează acum doar primul token SMILES din fișierul
de părinți; extinderea va transmite identificatorul referinței prin întregul flux.

**Fitness adaptat scenariului.** Obiectivul actual `anti-shortcut` preferă
similaritate în jur de 0,70 și penalizează pierderea scheletului. Îl păstrăm ca
reper pentru analogii apropiați, dar viitoarea selecție folosește benzi de
similaritate și obiective configurabile pentru fiecare scenariu. Categoria
scaffold-hop nu trebuie să primească automat aceeași penalizare pentru schimbarea
scheletului ca o substituție periferică. Diferența numărului de atomi nu este
echivalentă cu numărul atomilor modificați; istoricul operațiilor și verificarea
finală a transformării sunt mai potrivite pentru această mărime.

QED și SA sunt indicatori calculați, nu validări de eficacitate, siguranță sau
sinteză. Un filtru universal de drug-likeness ar elimina inclusiv substanțe mici
și contaminanți relevanți. Filtrele de analogi medicinali, compuși cunoscuți și
contaminanți trebuie definite separat.

**Acoperire și diversitate.** În locul unei singure cote identice pentru cele
cinci categorii E, următorul scheduler va aloca bugetul după deficitul de
acoperire în celule: referință/familie de schelet × categorie verificată ×
bandă de similaritate × mărimea modificării. Părinții se aleg și după existența
unor situsuri unde regula se poate aplica. Istoricul încrucișării și mutației
este păstrat, iar eticheta chimică se verifică pe perechea finală.

Faptul că o căutare a folosit mutații din categoria bioisostere nu dovedește că
toate efectele încrucișării aparțin acelei categorii. `generation_method=E` și
categoria transformării chimice rămân câmpuri diferite.

Selecția globală trebuie să țină cont de noutatea față de întreaga bancă,
diversitatea scheletelor și a referințelor, pe lângă diversitatea copiilor unui
singur părinte. Bugetul nu trebuie consumat de repetarea unor familii deja
suprareprezentate. Celulele fără suport chimic suficient sunt raportate, fără
copierea acelorași molecule pentru a simula exemple independente.

**Scalare operațională.** Generarea rulează în batch-uri care pot fi reluate,
cu indici persistenți pentru deduplicare și cu relații păstrate separat.
Descriptorii și amprentele se pot memora în cache, iar calculele MCS și 3D se
aplică după selecția candidaților relevanți. Outputurile trebuie să fie
reproductibile când ordinea de finalizare a proceselor paralele diferă.

Auditul codului a găsit două probleme concrete de randament: NSGA calcula
adnotările MCS pentru întreaga arhivă înaintea unei noi adnotări finale în
runner; selecția pe schelete se putea opri după candidați blocați de cote, chiar
dacă existau candidați eligibili mai jos în aceleași cozi. Corecțiile sunt
verificate separat cu exemple mici, fără o lansare de producție de mare volum.
Un factor numeric de accelerare trebuie măsurat pe același buget de generare,
nu presupus din schimbarea codului.

Corecții aplicate în această etapă:

- `evolutionary_generator.py`: NSGA nu mai calculează MCS/edit metrics pentru
  întreaga arhivă când runnerul va adnota selecția finală. Flagul `archive_history`
  este definit și transmis corect și pentru backendul probabilistic-crowding.
- `evolve_category_pilot.py`: selecția caută dincolo de capetele blocate ale
  cozilor pe schelete; cozile folosesc `deque`. La scoruri egale, părintele
  reprezentativ al unui duplicat este ales prin criterii deterministe.
- Verificare cu populație 4 și 2 generații: 8 candidați NSGA de arhivă au aceleași
  SMILES, fitness și descriptori ca referința cu adnotare imediată, cu 0 în loc
  de 8 apeluri MCS și 0 în loc de 8 calcule edit-metrics. Modul normal continuă
  să adnoteze rezultatele. Verificarea pentru crowding a trecut de asemenea.
- Exemplele de selecție au trecut pentru recuperarea candidaților de după
  un cap blocat, 20 de permutări ale ordinii de intrare, respectarea cotelor și
  egalitățile între părinți. Nu am măsurat încă randamentul de producție.

**Etape și criterii de decizie.**

- Pilot pe 100–200 de identități curate, excluzând identitățile rezervate pentru
  evaluarea externă. Comparație M/E și vechi/nou pe același buget de evaluări,
  cu țintă orientativă de 10.000–20.000 de structuri acceptate și deduplicate.
- Raportăm validitatea, rata de acceptare, motivele respingerii, noutatea,
  randamentul/oră, numărul referințelor și scheletelor, diversitatea și suportul
  pe celule. Numărul final poate fi sub țintă dacă restricțiile chimice cer asta.
- După pilot, obiectivul de producție poate fi un catalog de un milion de
  structuri unice, plus observațiile condiționate asociate. De exemplu,
  2.000 de referințe × 500 de propuneri înseamnă un milion de propuneri;
  numărul acceptat este mai mic după filtrare și deduplicare și trebuie măsurat.
- Pregătim întâi grafurile 2D și doar ulterior conformerii 3D pentru selecțiile
  utilizate efectiv. Conformerii calculați rămân predicții, cu statut de
  convergență și proveniență, nu geometrii experimentale validate.

**Evaluare.** Pentru generalizarea la API-uri noi, împărțirea după referință și
familia chimică se face înainte de generare. Pentru un API cunoscut și loturi
noi, gruparea trebuie să țină laolaltă loturile și replicile măsurate. Cazurile
oficiale rezervate și observațiile de test nu intră în fitness-ul generatorului.
Un scor adversarial pe un detector curent este doar un criteriu de dificultate
sintetică; nu dovedește falsificarea și poate reproduce erorile acelui model.

Cele 17 cazuri oficiale și spectrele NIR/HPLC importate sunt pentru verificarea
realismului, a acoperirii și pentru evaluări externe atent definite. Ele nu sunt
multiplicate pentru a pretinde o bancă mare de cazuri reale.

Referințe metodologice: [MOSES](https://arxiv.org/abs/1811.12823) și
[GuacaMol](https://doi.org/10.1021/acs.jcim.8b00839) descriu evaluarea generării
prin validitate, unicitate/noutate, diversitate și distribuția proprietăților.
Standardizarea chimică trebuie explicitată, folosind, de exemplu,
[instrumentele RDKit](https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html),
fără a șterge distincții relevante de sare sau stereochimie.
