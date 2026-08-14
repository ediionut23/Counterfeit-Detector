"""
Data-driven MMP rule mining (Phase A of the benchmark plan)
===========================================================

Instead of hand-coding a fixed catalogue of ~21 transformation SMARTS
(see `mmp_transformations.py`), this script *learns* transformation rules
from a real set of drug molecules using the Matched Molecular Pair (MMP)
algorithm of Hussain & Rea (2010), as implemented by `mmpdb` (A. Dalke).

This directly answers the research question:
    "Can the molecular modifications be extracted/learned from data,
     rather than enumerated by hand?"

The mined rules come with a *support count* (how many matched pairs in the
real data exhibit that transformation), which is a first, data-grounded
signal of how *frequent / realistic* a modification is — useful later both
as a catalogue for the rule-based dataset version and as the mutation-operator
library for the evolutionary (NSGA-II) generator.

Pipeline
--------
    authentic SMILES  ->  mmpdb fragment  ->  mmpdb index  ->  SQLite mmpdb
                       ->  extract rules ranked by support
                       ->  keep single-cut, RDKit-compilable rules
                       ->  export:  mined_transformations.json   (full, with stats)
                                    mined_transformations.py     (import-ready 6-tuples)

The exported `mined_transformations.py` matches the exact 6-tuple format of
`MMP_TRANSFORMATIONS` in `mmp_transformations.py`:
    (name, difficulty, category, reactant_smarts, product_smarts, citation)
so the existing generator can consume it with no code change.

Usage
-----
    # Extract authentic SMILES from an existing .pt dataset and mine:
    python mine_mmp_rules.py --from-pt intelligent_pharma_50k_v2.pt

    # Or mine from a plain SMILES file (one SMILES per line, optional <TAB> id):
    python mine_mmp_rules.py --smiles-file authentic.smi

Requirements: mmpdb (`pip install mmpdb`), rdkit, torch (only for --from-pt).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
#  Step 1 — obtain the input SMILES set
# ------------------------------------------------------------------
def extract_smiles_from_pt(pt_path: str, only_authentic: bool = True,
                           max_mols: Optional[int] = None) -> List[Tuple[str, str]]:
    """Pull (smiles, id) pairs out of a saved (graphs, labels, meta) .pt dataset.

    Only label == 0 (authentic) graphs are used by default: we want to mine
    modifications between *real* drugs, not between already-counterfeit ones.
    """
    import torch  # imported lazily so plain --smiles-file runs need no torch

    logger.info(f"Loading dataset {pt_path} ...")
    graphs, labels, _meta = torch.load(pt_path, weights_only=False)
    out: List[Tuple[str, str]] = []
    seen = set()
    for i, (g, lab) in enumerate(zip(graphs, labels)):
        if only_authentic and int(lab) != 0:
            continue
        smi = getattr(g, "smiles", None)
        if not smi or smi in seen:
            continue
        seen.add(smi)
        out.append((smi, f"mol{i}"))
        if max_mols and len(out) >= max_mols:
            break
    logger.info(f"Extracted {len(out)} unique "
                f"{'authentic ' if only_authentic else ''}SMILES")
    return out


def read_smiles_file(path: str, max_mols: Optional[int] = None) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen = set()
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smi = parts[0]
            mol_id = parts[1] if len(parts) > 1 else f"mol{i}"
            if smi in seen:
                continue
            seen.add(smi)
            out.append((smi, mol_id))
            if max_mols and len(out) >= max_mols:
                break
    logger.info(f"Read {len(out)} unique SMILES from {path}")
    return out


def canonicalize(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Canonicalize + drop unparseable. mmpdb wants clean input."""
    out = []
    for smi, mid in pairs:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out.append((Chem.MolToSmiles(mol), mid))
    logger.info(f"Canonicalized: {len(out)}/{len(pairs)} parseable")
    return out


def write_smi(pairs: List[Tuple[str, str]], path: Path) -> None:
    with open(path, "w") as fh:
        for smi, mid in pairs:
            fh.write(f"{smi}\t{mid}\n")
    logger.info(f"Wrote {len(pairs)} molecules to {path}")


# ------------------------------------------------------------------
#  Step 2/3 — run mmpdb fragment + index (CLI is the stable interface)
# ------------------------------------------------------------------
def run_mmpdb(smi_path: Path, workdir: Path, max_heavies_transf: int = 12,
              max_variable_heavies: int = 10, max_radius: int = 0) -> Path:
    if shutil.which("mmpdb") is None and _mmpdb_module() is None:
        logger.error("mmpdb not found. Install with: pip install mmpdb")
        sys.exit(1)

    frag_path = workdir / "fragments.fragdb"
    mmpdb_path = workdir / "mined.mmpdb"

    base = _mmpdb_invocation()

    logger.info("mmpdb fragment ... (this scans every molecule for cut bonds)")
    subprocess.run(
        base + ["fragment", str(smi_path),
                "-o", str(frag_path),
                "--max-heavies", str(max_variable_heavies + 20)],
        check=True,
    )

    # --max-radius 0 keeps only the context-free rule (constant environment
    # C(0) is empty), which still yields every transformation rule + its total
    # matched-pair count. We default to 0 because computing radius>0 circular
    # environments triggers an RDKit 2026.03 canonicalization bug
    # ("neither end atom traversed" in MolFragmentToSmiles) on some fragments.
    logger.info(f"mmpdb index (max-radius={max_radius}) ... (builds matched pairs + rules)")
    subprocess.run(
        base + ["index", str(frag_path),
                "-o", str(mmpdb_path),
                "--max-variable-heavies", str(max_variable_heavies),
                "--max-radius", str(max_radius)],
        check=True,
    )
    logger.info(f"Built MMP database: {mmpdb_path}")
    return mmpdb_path


def _mmpdb_module():
    try:
        import mmpdblib  # noqa: F401
        return mmpdblib
    except ImportError:
        return None


def _mmpdb_invocation() -> List[str]:
    """Prefer the console script; fall back to `python -m mmpdblib.cli`."""
    if shutil.which("mmpdb"):
        return ["mmpdb"]
    return [sys.executable, "-m", "mmpdblib.cli"]


# ------------------------------------------------------------------
#  Step 4 — extract + rank + filter rules from the SQLite mmpdb
# ------------------------------------------------------------------
_HALOGENS = {"F", "Cl", "Br", "I"}


def _frag_profile(frag_smiles: str) -> Optional[Dict]:
    """Structural profile of a fragment (dummy [*:1] ignored)."""
    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is None:
        return None
    ri = mol.GetRingInfo()
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 0]
    # ring fingerprint: multiset of ring sizes + heteroatoms that sit in rings
    ring_sizes = tuple(sorted(len(r) for r in ri.AtomRings()))
    ring_hetero = frozenset(
        mol.GetAtomWithIdx(idx).GetSymbol()
        for ring in ri.AtomRings() for idx in ring
        if mol.GetAtomWithIdx(idx).GetSymbol() != "C")
    # count non-aromatic multiple bonds (unsaturation introduced outside rings)
    n_multi = sum(1 for b in mol.GetBonds()
                  if not b.GetIsAromatic()
                  and b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE))
    return {
        "n_rings": rdMolDescriptors.CalcNumRings(mol),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "ring_sizes": ring_sizes,
        "ring_hetero": ring_hetero,
        "halogens": sum(1 for a in heavy if a.GetSymbol() in _HALOGENS),
        "heteroatoms": frozenset(a.GetSymbol() for a in heavy
                                 if a.GetSymbol() not in ("C",)),
        "n_multi_bonds": n_multi,
        "n_carbons": sum(1 for a in heavy if a.GetSymbol() == "C"),
        "n_heavy": len(heavy),
        "aromatic_atoms": sum(1 for a in heavy if a.GetIsAromatic()),
    }


def _transition(a: frozenset, b: frozenset) -> str:
    """Compact label for an element-set change, e.g. {F}->{Cl} => 'F->Cl'."""
    added, removed = sorted(b - a), sorted(a - b)
    if removed and added:
        return f"{'/'.join(removed)}->{'/'.join(added)}"
    if added:
        return f"add-{'/'.join(added)}"
    if removed:
        return f"remove-{'/'.join(removed)}"
    return "count-change"


def _subcategorize(category: str, pf: Dict, pt: Dict, delta_heavies: int) -> str:
    """Finer chemical subcategory within a top-level category (proposal C1:
    'per subcategory wherever the chemistry supports the distinction')."""
    if category == "halogen-walk":
        hf = frozenset(x for x in pf["heteroatoms"] if x in _HALOGENS)
        ht = frozenset(x for x in pt["heteroatoms"] if x in _HALOGENS)
        return _transition(hf, ht)  # e.g. H->F (add-F), F->Cl, remove-Cl
    if category == "homologation":
        d = abs(delta_heavies)
        size = "single" if d <= 1 else ("short" if d <= 3 else "long")
        return f"{'grow' if delta_heavies > 0 else 'shrink'}-{size}"
    if category == "scaffold-hop":
        if (pf["aromatic_atoms"] > 0) != (pt["aromatic_atoms"] > 0):
            return "aromatize" if pt["aromatic_atoms"] > 0 else "saturate"
        if pf["n_rings"] != pt["n_rings"]:
            return "add-ring" if pt["n_rings"] > pf["n_rings"] else "remove-ring"
        if pf["ring_hetero"] != pt["ring_hetero"]:
            return "ring-heteroatom-swap"
        if pf["ring_sizes"] != pt["ring_sizes"]:
            return "ring-resize"
        return "ring-other"
    if category == "bioisostere":
        if pf["heteroatoms"] == pt["heteroatoms"] and pf["n_multi_bonds"] != pt["n_multi_bonds"]:
            return "unsaturation-change"
        return _transition(pf["heteroatoms"], pt["heteroatoms"])
    return "other"


def categorize_rule(from_smiles: str, to_smiles: str,
                    delta_heavies: int) -> Tuple[str, str]:
    """Heuristic (category, subcategory) for a mined transformation, aligned with
    the benchmark's per-category / per-subcategory dataset design (proposal C1).

    Top-level categories:
      scaffold-hop  : ring system changes (count / aromaticity / size /
                      in-ring heteroatom composition, e.g. benzene->thiophene)
      halogen-walk  : only halogen identity/placement changes (low-effort)
      homologation  : pure saturated carbon-chain grow/shrink, no new
                      heteroatom, no new unsaturation (methyl/ethyl walk,
                      N-demethylation)
      bioisostere   : heteroatom composition or functional group changes
                      (acylation, phosphorylation, ester/amide, C=O added, ...)
      other         : anything else
    """
    pf, pt = _frag_profile(from_smiles), _frag_profile(to_smiles)
    if pf is None or pt is None:
        return "other", "other"

    # (1) any change to the ring system -> scaffold-hop. Now also catches
    #     in-ring heteroatom swaps (benzene<->thiophene) and ring-size changes.
    ring_change = (pf["n_rings"] != pt["n_rings"]
                   or pf["n_aromatic_rings"] != pt["n_aromatic_rings"]
                   or pf["ring_sizes"] != pt["ring_sizes"]
                   or pf["ring_hetero"] != pt["ring_hetero"]
                   or (pf["aromatic_atoms"] > 0) != (pt["aromatic_atoms"] > 0))
    if ring_change:
        cat = "scaffold-hop"
    else:
        only_halogen_hetero = (pf["heteroatoms"] <= _HALOGENS
                               and pt["heteroatoms"] <= _HALOGENS)
        halogen_change = (pf["halogens"] != pt["halogens"]
                          or pf["heteroatoms"] != pt["heteroatoms"])
        if only_halogen_hetero and halogen_change:
            cat = "halogen-walk"
        elif (pf["heteroatoms"] == pt["heteroatoms"]
                and pf["halogens"] == pt["halogens"] == 0
                and pf["n_multi_bonds"] == pt["n_multi_bonds"]
                and abs(delta_heavies) >= 1):
            cat = "homologation"
        elif (pf["heteroatoms"] != pt["heteroatoms"]
                or pf["n_multi_bonds"] != pt["n_multi_bonds"]):
            cat = "bioisostere"
        else:
            cat = "other"
    return cat, _subcategorize(cat, pf, pt, delta_heavies)


def is_clean_fragment(frag_smiles: str) -> bool:
    """Reject fragments that make poor counterfeit-modification operators:
    isotopic labels ([18F], [2H]), non-zero formal charges (protonation-state
    'transformations' like COOH->COO-), and radicals. These are mmpdb artefacts
    of how the source data was recorded, not structural edits a counterfeiter
    would make."""
    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetIsotope() != 0:
            return False
        if atom.GetFormalCharge() != 0:
            return False
        if atom.GetNumRadicalElectrons() != 0:
            return False
    return True


def extract_rules(mmpdb_path: Path, min_support: int,
                  min_heavies: int, max_heavies: int,
                  drop_noise: bool = True) -> List[Dict]:
    """Read transformation rules ranked by number of supporting matched pairs.

    mmpdb schema (v3): a `rule` is (from_smiles_id -> to_smiles_id) over
    fragment SMILES that carry [*:1] attachment points; each rule fans out
    into `rule_environment` rows (one per radius) that store `num_pairs`.
    Total support for a rule = SUM(num_pairs) across its environments.
    """
    con = sqlite3.connect(str(mmpdb_path))
    con.row_factory = sqlite3.Row

    query = """
        SELECT r.id                AS rule_id,
               fs.smiles           AS from_smiles,
               ts.smiles           AS to_smiles,
               fs.num_heavies      AS from_heavies,
               ts.num_heavies      AS to_heavies,
               SUM(re.num_pairs)   AS n_pairs,
               COUNT(DISTINCT re.id) AS n_envs
        FROM rule r
        JOIN rule_smiles fs ON r.from_smiles_id = fs.id
        JOIN rule_smiles ts ON r.to_smiles_id   = ts.id
        JOIN rule_environment re ON re.rule_id = r.id
        GROUP BY r.id
        ORDER BY n_pairs DESC
    """
    rows = con.execute(query).fetchall()
    con.close()
    logger.info(f"mmpdb produced {len(rows)} raw rules")

    rules: List[Dict] = []
    for row in rows:
        frm, to = row["from_smiles"], row["to_smiles"]
        n_pairs = int(row["n_pairs"] or 0)
        if n_pairs < min_support:
            continue
        # Single-cut only: exactly one attachment point on each side.
        if frm.count("*") != 1 or to.count("*") != 1:
            continue
        if frm == to:
            continue
        fh_, th_ = int(row["from_heavies"]), int(row["to_heavies"])
        if not (min_heavies <= fh_ <= max_heavies and min_heavies <= th_ <= max_heavies):
            continue
        if drop_noise and not (is_clean_fragment(frm) and is_clean_fragment(to)):
            continue
        rules.append({
            "rule_id": int(row["rule_id"]),
            "from_smiles": frm,
            "to_smiles": to,
            "from_heavies": fh_,
            "to_heavies": th_,
            "delta_heavies": th_ - fh_,
            "support_pairs": n_pairs,
            "n_environments": int(row["n_envs"]),
        })
    logger.info(f"After filtering (support>={min_support}, single-cut, "
                f"heavies in [{min_heavies},{max_heavies}]): {len(rules)} rules")
    return rules


# ------------------------------------------------------------------
#  Step 5 — turn fragment rules into RDKit-compilable reaction SMARTS
# ------------------------------------------------------------------
def to_reaction(frm: str, to: str) -> Optional[Tuple[str, str]]:
    """mmpdb fragments already carry [*:1]; `from>>to` is a reaction SMILES.

    Returns (reactant, product) only if RDKit can compile the reaction, so
    the exported catalogue is guaranteed usable by the existing generator's
    `_build_transformations` (which calls ReactionFromSmarts).
    """
    reactant, product = frm, to
    try:
        rxn = AllChem.ReactionFromSmarts(f"{reactant}>>{product}")
        if rxn is None:
            return None
        rxn.Initialize()
        if rxn.GetNumReactantTemplates() != 1 or rxn.GetNumProductTemplates() != 1:
            return None
        return reactant, product
    except Exception:
        return None


def difficulty_of(delta_heavies: int) -> str:
    d = abs(delta_heavies)
    if d <= 1:
        return "subtle"
    if d <= 3:
        return "moderate"
    return "obvious"


def make_name(idx: int, frm: str, to: str) -> str:
    def tag(s: str) -> str:
        return (s.replace("[*:1]", "R").replace("(", "").replace(")", "")
                 .replace("[", "").replace("]", "").replace("@", "")
                 .replace("=", "eq").replace("#", "tri").replace(":", "")
                 .replace("/", "").replace("\\", "")[:14])
    return f"mmp{idx:04d}_{tag(frm)}_to_{tag(to)}"


# ------------------------------------------------------------------
#  Step 6 — export
# ------------------------------------------------------------------
def export(rules: List[Dict], out_stem: Path, source_label: str) -> None:
    compiled: List[Dict] = []
    for i, rule in enumerate(rules):
        rxn = to_reaction(rule["from_smiles"], rule["to_smiles"])
        if rxn is None:
            continue
        reactant, product = rxn
        rule = dict(rule)
        rule["name"] = make_name(i, rule["from_smiles"], rule["to_smiles"])
        rule["difficulty"] = difficulty_of(rule["delta_heavies"])
        cat, subcat = categorize_rule(rule["from_smiles"], rule["to_smiles"],
                                      rule["delta_heavies"])
        rule["category"] = cat
        rule["subcategory"] = subcat
        rule["reactant_smarts"] = reactant
        rule["product_smarts"] = product
        rule["citation"] = f"mmpdb-mined({source_label})"
        compiled.append(rule)

    logger.info(f"{len(compiled)}/{len(rules)} rules compile under RDKit")

    # --- JSON (full, with support stats — for the thesis tables) ---
    json_path = out_stem.with_suffix(".json")
    with open(json_path, "w") as fh:
        json.dump({
            "source": source_label,
            "n_rules": len(compiled),
            "difficulty_distribution": _dist(compiled, "difficulty"),
            "category_distribution": _dist(compiled, "category"),
            "subcategory_distribution": _dist(compiled, "subcategory"),
            "rules": compiled,
        }, fh, indent=2)
    logger.info(f"Wrote {json_path}")

    # --- Python module (import-ready, matches MMP_TRANSFORMATIONS format) ---
    py_path = out_stem.with_suffix(".py")
    with open(py_path, "w") as fh:
        fh.write('"""AUTO-GENERATED by mine_mmp_rules.py — do not edit by hand.\n\n')
        fh.write(f'Mined {len(compiled)} single-cut MMP transformation rules from '
                 f'{source_label},\nranked by matched-pair support. Format matches '
                 'MMP_TRANSFORMATIONS in\nmmp_transformations.py:\n'
                 '    (name, difficulty, category, reactant, product, citation)\n"""\n\n')
        fh.write("from typing import List, Tuple\n\n")
        fh.write("MMP_TRANSFORMATIONS: List[Tuple[str, str, str, str, str, str]] = [\n")
        for r in compiled:
            fh.write(f"    ({r['name']!r}, {r['difficulty']!r}, {r['category']!r},\n")
            fh.write(f"     {r['reactant_smarts']!r},\n")
            fh.write(f"     {r['product_smarts']!r},\n")
            fh.write(f"     {r['citation']!r}),  # support={r['support_pairs']} pairs, "
                     f"Δheavy={r['delta_heavies']}\n")
        fh.write("]\n\n")
        fh.write("# Support counts (matched pairs) per rule, for weighting:\n")
        fh.write("SUPPORT = {\n")
        for r in compiled:
            fh.write(f"    {r['name']!r}: {r['support_pairs']},\n")
        fh.write("}\n\n")
        fh.write("# Finer chemical subcategory per rule (proposal C1):\n")
        fh.write("SUBCATEGORY = {\n")
        for r in compiled:
            fh.write(f"    {r['name']!r}: {r['subcategory']!r},\n")
        fh.write("}\n")
    logger.info(f"Wrote {py_path}")

    _print_top(compiled, n=20)


def _dist(rules: List[Dict], key: str) -> Dict[str, int]:
    d: Dict[str, int] = defaultdict(int)
    for r in rules:
        d[r[key]] += 1
    return dict(d)


def _print_top(rules: List[Dict], n: int = 20) -> None:
    logger.info(f"\nCategory distribution: {_dist(rules, 'category')}")
    logger.info(f"Subcategory distribution: {_dist(rules, 'subcategory')}")
    logger.info(f"\nTop {n} mined transformations by support:")
    logger.info(f"  {'support':>8}  {'Δheavy':>6}  {'category':>13}  from  ->  to")
    for r in rules[:n]:
        logger.info(f"  {r['support_pairs']:>8}  {r['delta_heavies']:>+6}  "
                    f"{r['category']:>13}  {r['from_smiles']}  ->  {r['to_smiles']}")


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-pt", type=str,
                     help="Extract authentic (label==0) SMILES from a .pt dataset")
    src.add_argument("--smiles-file", type=str,
                     help="Plain SMILES file (one per line, optional <TAB> id)")
    p.add_argument("--max-mols", type=int, default=None,
                   help="Cap number of input molecules (speed up a first run)")
    p.add_argument("--min-support", type=int, default=5,
                   help="Keep rules seen in at least this many matched pairs")
    p.add_argument("--min-heavies", type=int, default=1,
                   help="Min heavy atoms on each fragment side")
    p.add_argument("--max-heavies", type=int, default=8,
                   help="Max heavy atoms on each fragment side (small edits)")
    p.add_argument("--max-radius", type=int, default=0,
                   help="mmpdb environment radius (0 = context-free rules; "
                        ">0 may hit an RDKit 2026.03 canonicalization bug)")
    p.add_argument("--reuse-db", action="store_true",
                   help="Skip fragment+index if the mmpdb already exists in "
                        "--workdir (re-extract rules with new filters only)")
    p.add_argument("--keep-noise", action="store_true",
                   help="Keep isotope/charged/radical fragments (default drops them)")
    p.add_argument("--workdir", type=str, default="mmp_mining",
                   help="Scratch dir for fragdb/mmpdb files")
    p.add_argument("--out", type=str, default="mined_transformations",
                   help="Output stem (writes .json and .py)")
    args = p.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(exist_ok=True)
    mmpdb_path = workdir / "mined.mmpdb"

    if args.reuse_db and mmpdb_path.exists():
        logger.info(f"--reuse-db: reusing existing {mmpdb_path}, "
                    "re-extracting rules only")
        source_label = (f"authentic@{Path(args.from_pt).name}" if args.from_pt
                        else Path(args.smiles_file).name)
    else:
        # 1. input
        if args.from_pt:
            pairs = extract_smiles_from_pt(args.from_pt, only_authentic=True,
                                           max_mols=args.max_mols)
            source_label = f"authentic@{Path(args.from_pt).name}"
        else:
            pairs = read_smiles_file(args.smiles_file, max_mols=args.max_mols)
            source_label = Path(args.smiles_file).name
        pairs = canonicalize(pairs)
        if len(pairs) < 50:
            logger.error("Too few molecules to mine meaningful MMPs (need >= ~50).")
            sys.exit(1)

        smi_path = workdir / "input.smi"
        write_smi(pairs, smi_path)

        # 2/3. fragment + index
        mmpdb_path = run_mmpdb(smi_path, workdir, max_radius=args.max_radius)

    # 4. extract + filter
    rules = extract_rules(mmpdb_path, min_support=args.min_support,
                          min_heavies=args.min_heavies, max_heavies=args.max_heavies,
                          drop_noise=not args.keep_noise)
    if not rules:
        logger.error("No rules survived filtering. Try lowering --min-support "
                     "or widening --max-heavies.")
        sys.exit(1)

    # 5/6. compile + export
    export(rules, Path(args.out), source_label)
    logger.info("\nDONE. Import the catalogue with:  "
                f"from {Path(args.out).stem} import MMP_TRANSFORMATIONS, SUPPORT")


if __name__ == "__main__":
    main()
