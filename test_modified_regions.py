#!/usr/local/bin/python3.12
"""
test_modified_regions.py

Validează calitatea explicabilității GNNExplainer din app.py.

Logică corectă:
  Explicabilitatea NU îți spune ce ai schimbat —
  îți spune CE CONTEAZĂ pentru predicție.

  Testul real:
    1. Analizează molecula originală → obține ranking atomi
    2. Perturbă atomii cu scor MARE (top-3)
       → predicția trebuie să se schimbe mult (ΔConf mare)
    3. Perturbă atomii cu scor MIC (bottom-3)
       → predicția trebuie să rămână stabilă (ΔConf mic)

  PASS dacă: ΔConf(important) > ΔConf(pasiv)
  Asta demonstrează că explicabilitatea e consistentă cu modelul.

Utilizare:
  python3.12 test_modified_regions.py
  python3.12 test_modified_regions.py --url http://localhost:8001
  python3.12 test_modified_regions.py --only 0
"""

import argparse
import json
import re
import sys
import time

import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

BASE_URL = "http://localhost:8001"

# Molecule de test — preferabil cu structuri bine cunoscute
TEST_MOLECULES = [
    {
        "name": "Aspirin",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    },
    {
        "name": "Paracetamol",
        "smiles": "CC(=O)NC1=CC=C(O)C=C1",
    },
    {
        "name": "Ibuprofen",
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    },
    {
        "name": "Ciprofloxacin",
        "smiles": "OC(=O)c1cn(C2CC2)c3cc(N4CCNCC4)c(F)cc3c1=O",
    },
    {
        "name": "Diazepam",
        "smiles": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",
    },
]

# Substituenți neutrali pentru perturbație (mici, izolați)
PERTURBATION_SUBS = {
    "C": "[CH3]",   # înlocuire carbon cu metil explicit — aproape neutru
    "N": "[NH2]",
    "O": "[OH]",
    "F": "Cl",
    "Cl": "F",
    "Br": "Cl",
    "S": "[SH]",
}


# ──────────────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────────────

def check_server(url: str) -> bool:
    try:
        return requests.get(url, timeout=5).status_code == 200
    except Exception:
        return False


def analyze(url: str, smiles: str) -> dict | None:
    try:
        r = requests.post(
            f"{url}/analyze",
            json={"smiles": smiles},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


def extract_atom_scores(html: str) -> dict[int, float]:
    """Extrage {atom_index: normalized_score} din HTML-ul 3D viewer."""
    m = re.search(r"const AD = (\{.*?\});", html, re.DOTALL)
    if not m:
        return {}
    try:
        raw = json.loads(m.group(1))
        return {v["index"]: v["score"] for v in raw.values()}
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Perturbație moleculară
# ──────────────────────────────────────────────────────────────────────────────

_MAX_VALENCE = {'C': 4, 'N': 3, 'O': 2, 'S': 2, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1}
_CANDIDATES  = ['N', 'O', 'S', 'C', 'F', 'Cl', 'Br']


def _perturb_aromatic(smiles: str, atom_idx: int) -> str | None:
    """
    Perturbație pentru atomi aromatici: adaugă H explicit, înlocuiește
    un H vecin cu F sau Cl, apoi scoate H-urile înapoi.
    Asta evită orice conflict cu valența aromatică.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)
    target = mol_h.GetAtomWithIdx(atom_idx)
    h_neighbors = [n.GetIdx() for n in target.GetNeighbors() if n.GetAtomicNum() == 1]
    if not h_neighbors:
        return None

    for atomic_num in (9, 17):  # F, Cl
        rw = Chem.RWMol(mol_h)
        rw.GetAtomWithIdx(h_neighbors[0]).SetAtomicNum(atomic_num)
        try:
            Chem.SanitizeMol(rw)
            mol_clean = Chem.RemoveHs(rw.GetMol())
            new_smiles = Chem.MolToSmiles(mol_clean)
            if new_smiles and Chem.MolFromSmiles(new_smiles) is not None and new_smiles != smiles:
                return new_smiles
        except Exception:
            continue
    return None


def _perturb_nonaromatic(smiles: str, atom_idx: int) -> str | None:
    """
    Perturbație prin înlocuire pentru atomi non-aromatici.
    Compatibilitatea se verifică pe suma ordinelor de legătură.
    """
    mol = Chem.MolFromSmiles(smiles)
    atom = mol.GetAtomWithIdx(atom_idx)
    symbol = atom.GetSymbol()
    bond_order_sum = int(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()))

    pt = Chem.GetPeriodicTable()
    for new_symbol in _CANDIDATES:
        if new_symbol == symbol:
            continue
        if _MAX_VALENCE.get(new_symbol, 4) < bond_order_sum:
            continue
        try:
            rw = Chem.RWMol(Chem.MolFromSmiles(smiles))
            a = rw.GetAtomWithIdx(atom_idx)
            a.SetAtomicNum(pt.GetAtomicNumber(new_symbol))
            a.SetNoImplicit(False)
            a.SetNumExplicitHs(0)
            a.SetFormalCharge(0)
            Chem.SanitizeMol(rw)
            new_smiles = Chem.MolToSmiles(rw)
            if new_smiles and Chem.MolFromSmiles(new_smiles) is not None and new_smiles != smiles:
                return new_smiles
        except Exception:
            continue
    return None


def perturb_atom(smiles: str, atom_idx: int) -> str | None:
    """
    Alege strategia de perturbație în funcție de tipul atomului:
    - aromatic  → adaugă substituent (F/Cl) la poziția liberă
    - non-arom. → înlocuiește atomul cu unul de valență compatibilă
    """
    base_mol = Chem.MolFromSmiles(smiles)
    if base_mol is None or atom_idx >= base_mol.GetNumAtoms():
        return None

    atom = base_mol.GetAtomWithIdx(atom_idx)
    if atom.GetIsAromatic():
        return _perturb_aromatic(smiles, atom_idx)
    return _perturb_nonaromatic(smiles, atom_idx)


def get_perturbable_atoms(smiles: str, atom_indices: list[int]) -> list[int]:
    """Returnează doar indicii pentru care perturb_atom produce un SMILES valid."""
    return [idx for idx in atom_indices if perturb_atom(smiles, idx) is not None]


# ──────────────────────────────────────────────────────────────────────────────
# Test individual
# ──────────────────────────────────────────────────────────────────────────────

def run_test(mol_info: dict, url: str) -> dict:
    smiles = mol_info["smiles"]
    name   = mol_info["name"]

    print(f"\n{'─'*64}")
    print(f"  MOLECULĂ: {name}")
    print(f"  SMILES:   {smiles}")

    # ── 1. Analiză inițială ────────────────────────────────────
    resp = analyze(url, smiles)
    if resp is None:
        print("  [EROARE] API a eșuat pentru molecula originală")
        return {"name": name, "passed": False, "error": "API fail"}

    conf_orig   = resp["confidence"]
    pred_orig   = resp["prediction"]
    atom_scores = extract_atom_scores(resp["html"])

    if not atom_scores:
        print("  [EROARE] Nu s-au putut extrage scorurile atomilor")
        return {"name": name, "passed": False, "error": "no atom scores"}

    sorted_atoms = sorted(atom_scores.items(), key=lambda x: x[1], reverse=True)
    total = len(sorted_atoms)

    print(f"\n  Predicție originală: {pred_orig} ({conf_orig:.1%})")
    print(f"  Total atomi: {total}")

    # ── 2. Selectare atomi importanți vs. pasivi ───────────────
    top_candidates    = [idx for idx, _ in sorted_atoms[:5]]
    bottom_candidates = [idx for idx, _ in sorted_atoms[-5:]]

    top_perturbable    = get_perturbable_atoms(smiles, top_candidates)
    bottom_perturbable = get_perturbable_atoms(smiles, bottom_candidates)

    print(f"\n  Top-5 atomi (importanți): {top_candidates}")
    print(f"  Bottom-5 atomi (pasivi) : {bottom_candidates}")
    print(f"  Perturbabili importanți : {top_perturbable}")
    print(f"  Perturbabili pasivi     : {bottom_perturbable}")

    if not top_perturbable or not bottom_perturbable:
        print("  [SKIP] Nu s-au găsit suficienți atomi perturbabili "
              "(probabil moleculă complet aromatică)")
        return {"name": name, "passed": None, "error": "no perturbable atoms"}

    # ── 3. Perturbație atomi importanți ────────────────────────
    print(f"\n  --- Perturbând atomi IMPORTANȚI ---")
    delta_important = []
    for atom_idx in top_perturbable[:3]:
        new_smiles = perturb_atom(smiles, atom_idx)
        if new_smiles is None or new_smiles == smiles:
            print(f"  Atom {atom_idx:>3} (scor {atom_scores[atom_idx]:.3f}): "
                  f"perturbație imposibilă")
            continue
        r = analyze(url, new_smiles)
        if r is None:
            print(f"  Atom {atom_idx:>3} (scor {atom_scores[atom_idx]:.3f}): "
                  f"API invalid ({new_smiles[:40]})")
            continue
        delta = abs(r["confidence"] - conf_orig)
        pred_new = r["prediction"]
        changed = "→ " + pred_new if pred_new != pred_orig else "  același"
        print(f"  Atom {atom_idx:>3} (scor {atom_scores[atom_idx]:.3f}): "
              f"ΔConf = {delta:.3f}  {changed}")
        delta_important.append(delta)
        time.sleep(0.3)

    # ── 4. Perturbație atomi pasivi ────────────────────────────
    print(f"\n  --- Perturbând atomi PASIVI ---")
    delta_passive = []
    for atom_idx in bottom_perturbable[:3]:
        new_smiles = perturb_atom(smiles, atom_idx)
        if new_smiles is None or new_smiles == smiles:
            print(f"  Atom {atom_idx:>3} (scor {atom_scores[atom_idx]:.3f}): "
                  f"perturbație imposibilă")
            continue
        r = analyze(url, new_smiles)
        if r is None:
            print(f"  Atom {atom_idx:>3} (scor {atom_scores[atom_idx]:.3f}): "
                  f"API invalid ({new_smiles[:40]})")
            continue
        delta = abs(r["confidence"] - conf_orig)
        pred_new = r["prediction"]
        changed = "→ " + pred_new if pred_new != pred_orig else "  același"
        print(f"  Atom {atom_idx:>3} (scor {atom_scores[atom_idx]:.3f}): "
              f"ΔConf = {delta:.3f}  {changed}")
        delta_passive.append(delta)
        time.sleep(0.3)

    # ── 5. Concluzie ───────────────────────────────────────────
    if not delta_important and not delta_passive:
        print("\n  [SKIP] Nicio perturbație nu a reușit")
        return {"name": name, "passed": None, "error": "no successful perturbations"}
    if not delta_important:
        print("\n  [SKIP] Niciun atom important nu a putut fi perturbat")
        return {"name": name, "passed": None, "error": "no important perturbations"}
    if not delta_passive:
        print("\n  [SKIP] Niciun atom pasiv nu a putut fi perturbat")
        return {"name": name, "passed": None, "error": "no passive perturbations"}

    avg_imp = sum(delta_important) / len(delta_important)
    avg_pas = sum(delta_passive)   / len(delta_passive)

    print(f"\n  ΔConf mediu — atomi IMPORTANȚI : {avg_imp:.4f}")
    print(f"  ΔConf mediu — atomi PASIVI     : {avg_pas:.4f}")

    passed = avg_imp > avg_pas
    status = "PASS ✓" if passed else "FAIL ✗"
    reason = ("Atomii importanți perturbați produc impact mai mare"
              if passed else
              "Atomii pasivi au produs impact similar sau mai mare — "
              "explicabilitatea nu e consistentă cu modelul")
    print(f"  Rezultat: {status} — {reason}")

    return {
        "name": name,
        "passed": passed,
        "prediction": pred_orig,
        "confidence": conf_orig,
        "avg_delta_important": avg_imp,
        "avg_delta_passive": avg_pas,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validare explicabilitate GNNExplainer"
    )
    parser.add_argument("--url",  default=BASE_URL)
    parser.add_argument("--only", type=int,
                        help="Rulează doar molecula cu indexul dat (0-based)")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    print("=" * 64)
    print("  PharmaAI — Validare consistență explicabilitate GNNExplainer")
    print(f"  Server: {url}")
    print("=" * 64)
    print()
    print("  Principiu: dacă explicabilitatea e corectă,")
    print("  perturbarea atomilor CU SCOR MARE trebuie să schimbe")
    print("  predicția mai mult decât perturbarea atomilor CU SCOR MIC.")

    if not check_server(url):
        print(f"\n[EROARE] Serverul nu răspunde la {url}")
        print("Pornește-l cu: python3.12 app.py")
        sys.exit(1)
    print("\nServer activ.")

    molecules = TEST_MOLECULES if args.only is None else [TEST_MOLECULES[args.only]]

    results = []
    for mol in molecules:
        result = run_test(mol, url)
        results.append(result)
        time.sleep(0.5)

    # ── Sumar ─────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  SUMAR FINAL")
    print(f"{'='*64}")

    passed  = sum(1 for r in results if r.get("passed") is True)
    failed  = sum(1 for r in results if r.get("passed") is False)
    skipped = sum(1 for r in results if r.get("passed") is None)

    for r in results:
        s   = r.get("passed")
        tag = "PASS ✓" if s is True else ("SKIP ~" if s is None else "FAIL ✗")
        detail = ""
        if "avg_delta_important" in r:
            detail = (f"  ΔImp={r['avg_delta_important']:.4f} "
                      f"ΔPas={r['avg_delta_passive']:.4f}")
        print(f"  [{tag}]  {r['name']}{detail}")

    print()
    print(f"  {passed} passed  |  {failed} failed  |  {skipped} skipped  "
          f"(din {len(results)} total)")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
