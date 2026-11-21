"""Build an audited, modular counterfeit-only molecule bank from source CSVs.

The molecule catalogue is globally unique. The relation table preserves every
distinct accepted parent/child transformation and its source provenance.
Subcategory CSVs are selections of relations; they can overlap by molecule ID.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from build_verified_single_atom_edits import verify as verify_single_atom_edit
from generate_100k_counterfeits import has_structural_alerts, sascorer
from mined_transformations import SUBCATEGORY
from repair_edited_atoms import edited_child_atoms


RDLogger.DisableLog("rdApp.*")
if sascorer is None:
    raise RuntimeError("RDKit SA_Score is required; refusing to impute accessibility")

ROOT = Path(__file__).resolve().parent
CATEGORIES = {"bioisostere", "halogen-walk", "homologation", "scaffold-hop", "other"}
MOLECULE_FIELDS = (
    "molecule_id", "smiles", "label", "primary_method", "primary_category",
    "primary_subcategory", "methods", "n_relations", "n_sources", "mw",
    "logp", "qed", "sa_score", "heavy_atoms", "smiles_length",
    "child_scaffold", "has_verified_edit",
)
RELATION_FIELDS = (
    "relation_id", "molecule_id", "smiles", "parent_smiles", "method",
    "category", "subcategory", "rule_name", "sources", "similarity_to_parent",
    "edited_atoms", "edit_status", "parent_scaffold", "child_scaffold",
)


def source_list(extension: Path | None, e_extension: Path | None) -> list[tuple[str, Path, str]]:
    base = [
        ("catalog_100k", ROOT / "counterfeits_100k/counterfeits_100k.csv", "mixed"),
        ("evolved_100k", ROOT / "version_e_100k/counterfeits_evolved.csv", "E"),
        ("evolved_tiered", ROOT / "version_e_tiered/counterfeits_size_stratified.csv", "E"),
        ("mined_50k", ROOT / "version_m/counterfeits_all.csv", "M"),
        ("boosted", ROOT / "counterfeits_boosted.csv", "M"),
    ]
    base += [("evolved_conditioned_" + path.stem.removeprefix("counterfeits_evolved_"), path, "E")
             for path in sorted((ROOT / "version_e_categories").glob("counterfeits_evolved_*.csv"))]
    if extension is not None:
        base.append(("extension_m", extension, "M"))
    if e_extension is not None:
        base.append(("extension_e", e_extension, "E"))
    return base


def molecule_id(smiles: str) -> str:
    return "mol_" + hashlib.sha256(smiles.encode()).hexdigest()[:20]


def relation_id(key: tuple[str, ...]) -> str:
    return "rel_" + hashlib.sha256("\x1f".join(key).encode()).hexdigest()[:20]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def murcko(mol: Chem.Mol) -> str:
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core) if core.GetNumAtoms() else "<acyclic>"


@lru_cache(maxsize=10000)
def child_quality(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if Chem.MolToSmiles(mol) != smiles:
        return "noncanonical"
    if len(Chem.GetMolFrags(mol)) != 1:
        return "multicomponent_child"
    heavy = mol.GetNumHeavyAtoms()
    mw = float(Descriptors.MolWt(mol))
    if heavy < 10 or heavy > 70 or mw < 150 or mw > 800:
        return "size_or_mass"
    qed = float(Descriptors.qed(mol))
    if qed < 0.35:
        return "low_qed"
    sa = float(sascorer.calculateScore(mol))
    if sa > 3.8:
        return "high_sa"
    if has_structural_alerts(mol):
        return "structural_alert"
    return {
        "mw": round(mw, 2), "logp": round(float(Descriptors.MolLogP(mol)), 3),
        "qed": round(qed, 4), "sa_score": round(sa, 3),
        "heavy_atoms": heavy, "n_atoms": mol.GetNumAtoms(), "smiles_length": len(smiles),
        "child_scaffold": murcko(mol),
        "fingerprint": rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, 2048),
    }


@lru_cache(maxsize=5000)
def parent_quality(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or Chem.MolToSmiles(mol) != smiles or len(Chem.GetMolFrags(mol)) != 1:
        return None
    return (rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, 2048), murcko(mol))


def classify(source_method: str, row: dict[str, str]) -> tuple[str, str, str]:
    raw_category = row.get("category", "")
    method = "E" if source_method == "E" or raw_category.startswith("evolved") else "M"
    if method == "E":
        category = raw_category.removeprefix("evolved_") if raw_category.startswith("evolved_") else "unconditioned"
        subcategory = "category_conditioned" if category != "unconditioned" else (
            "size_stratified" if raw_category.startswith("evolved-") else "unconditioned")
        return method, category, subcategory
    category = raw_category
    rule = row.get("rule_name", "")
    subcategory = row.get("subcategory") or SUBCATEGORY.get(rule, "other" if category == "other" else "unknown")
    return method, category, subcategory


def mask_value(row: dict[str, str], n_atoms: int, verified: dict[tuple[str, str], str]) -> tuple[str, str]:
    pair = (row.get("parent_smiles", ""), row.get("counterfeit_smiles", ""))
    if pair in verified:
        return verified[pair], "exact_single_atom"
    raw = row.get("edited_atoms", "")
    try:
        indices = sorted({int(part) for part in re.findall(r"\d+", raw)})
    except ValueError:
        return "", "missing"
    if indices and indices[-1] < n_atoms:
        return " ".join(map(str, indices)), "approximate"
    return "", "missing"


def write_csv(path: Path, fields: tuple[str, ...], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def safe_name(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:55] or "unnamed"
    return label + "_" + hashlib.sha1(text.encode()).hexdigest()[:8]


def distribution(values) -> dict[str, float | int]:
    data = sorted(float(value) for value in values)
    if not data:
        return {"n": 0}
    def at(frac: float) -> float:
        return round(data[round(frac * (len(data) - 1))], 4)
    return {
        "n": len(data), "min": round(data[0], 4), "p05": at(0.05),
        "median": at(0.5), "p95": at(0.95), "max": round(data[-1], 4),
        "mean": round(statistics.fmean(data), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "final_dataset")
    parser.add_argument("--extension", type=Path,
                        default=ROOT / "counterfeits_extension_m/counterfeits_100k.csv")
    parser.add_argument("--without-extension", action="store_true")
    parser.add_argument("--e-extension", type=Path, default=None,
                        help="Additional evolved CSV after generation completes")
    parser.add_argument("--limit", type=int, default=None,
                        help="Source rows per file for a smoke test")
    parser.add_argument("--skip-new-edit-verification", action="store_true",
                        help="Skip exact reverse-edit search beyond the saved manifest")
    parser.add_argument("--skip-mcs-mask-reconstruction", action="store_true",
                        help="Use source masks only for non-exact edits (smoke tests)")
    args = parser.parse_args()
    extension = None if args.without_extension else args.extension
    if extension is not None and not extension.exists():
        raise FileNotFoundError(extension)
    if args.e_extension is not None and not args.e_extension.exists():
        raise FileNotFoundError(args.e_extension)

    authentic_files = [ROOT / "public_molecules_100k.smi", ROOT / "public_molecules.smi"]
    auth = set()
    authentic_source_counts = {}
    for path in authentic_files:
        with path.open() as stream:
            source_smiles = {line.split()[0] for line in stream if line.strip()}
        authentic_source_counts[path.name] = len(source_smiles)
        auth.update(source_smiles)
    verified = {}
    with (ROOT / "benchmark/verified_edits/manifest.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            verified[(row["parent_smiles"], row["smiles"])] = row["verified_mask"]

    reject = Counter()
    source_counts = Counter()
    relations: dict[tuple[str, ...], dict] = {}
    molecules: dict[str, dict] = {}
    source_files = source_list(extension, args.e_extension)
    for source, path, source_method in source_files:
        print(f"Scanning {source}: {path}", flush=True)
        with path.open(newline="") as stream:
            for i, row in enumerate(csv.DictReader(stream)):
                if args.limit is not None and i >= args.limit:
                    break
                source_counts[source] += 1
                child = row.get("counterfeit_smiles", "")
                parent = row.get("parent_smiles", "")
                if not child or not parent:
                    reject["missing_smiles"] += 1
                    continue
                if child in auth or child == parent:
                    reject["authentic_overlap_or_identity"] += 1
                    continue
                if parent not in auth:
                    reject["parent_not_in_authentic_library"] += 1
                    continue
                if "." in parent:
                    reject["multicomponent_parent"] += 1
                    continue
                method, category, subcat = classify(source_method, row)
                if category not in CATEGORIES and category != "unconditioned":
                    reject["unknown_category"] += 1
                    continue
                quality = child_quality(child)
                if not isinstance(quality, dict):
                    reject[str(quality or "invalid_child")] += 1
                    continue
                pinfo = parent_quality(parent)
                if pinfo is None:
                    reject["invalid_parent"] += 1
                    continue
                similarity = DataStructs.TanimotoSimilarity(pinfo[0], quality["fingerprint"])
                if not 0.4 <= similarity <= 0.98:
                    reject["similarity_outside_band"] += 1
                    continue

                rule = row.get("rule_name", "") if method == "M" else ""
                key = (child, parent, method, category, subcat, rule)
                edited, edit_status = mask_value(row, quality["n_atoms"], verified)
                if key in relations:
                    old = relations[key]
                    old["sources"].add(source)
                    if edit_status == "exact_single_atom" and old["edit_status"] != edit_status:
                        old["edited_atoms"], old["edit_status"] = edited, edit_status
                    continue
                if edit_status != "exact_single_atom" and not args.skip_new_edit_verification:
                    exact = verify_single_atom_edit(parent, child)
                    if exact is not None:
                        edited, edit_status = exact[2], "exact_single_atom"
                if edit_status != "exact_single_atom" and not args.skip_mcs_mask_reconstruction:
                    mcs_indices = edited_child_atoms(parent, child, max_timeout=1)
                    if mcs_indices:
                        edited = " ".join(map(str, mcs_indices))
                        edit_status = "mcs_approximate"
                    elif edit_status == "approximate":
                        edit_status = "source_approximate"
                rel = {
                    "relation_id": relation_id(key), "molecule_id": molecule_id(child),
                    "smiles": child, "parent_smiles": parent, "method": method,
                    "category": category, "subcategory": subcat, "rule_name": rule,
                    "sources": {source}, "similarity_to_parent": round(similarity, 4),
                    "edited_atoms": edited, "edit_status": edit_status,
                    "parent_scaffold": pinfo[1], "child_scaffold": quality["child_scaffold"],
                }
                relations[key] = rel
                if child not in molecules:
                    molecules[child] = {k: v for k, v in quality.items()
                                        if k not in ("fingerprint", "n_atoms")}
                if (i + 1) % 20000 == 0:
                    print(f"  rows={i+1:,}, accepted molecules={len(molecules):,}", flush=True)

    by_child = defaultdict(list)
    for rel in relations.values():
        by_child[rel["smiles"]].append(rel)
    method_counts = Counter()
    primary_counts = Counter()
    molecule_rows = []
    source_rank = {name: i for i, (name, _, _) in enumerate(source_files)}
    for smiles in sorted(molecules):
        group = by_child[smiles]
        primary = min(group, key=lambda r: (min(source_rank[s] for s in r["sources"]),
                                            r["method"], r["category"], r["subcategory"],
                                            r["parent_smiles"]))
        methods = "".join(sorted({r["method"] for r in group}))
        method_counts[methods] += 1
        primary_counts[f"{primary['method']}/{primary['category']}/{primary['subcategory']}"] += 1
        molecule_rows.append({
            "molecule_id": molecule_id(smiles), "smiles": smiles, "label": 1,
            "primary_method": primary["method"], "primary_category": primary["category"],
            "primary_subcategory": primary["subcategory"], "methods": methods,
            "n_relations": len(group),
            "n_sources": len({source for r in group for source in r["sources"]}),
            "has_verified_edit": int(any(r["edit_status"] == "exact_single_atom" for r in group)),
            **molecules[smiles],
        })
    relation_rows = []
    for rel in sorted(relations.values(), key=lambda r: (r["method"], r["category"], r["subcategory"], r["smiles"], r["parent_smiles"])):
        relation_rows.append({**rel, "sources": ";".join(sorted(rel["sources"]))})

    out = args.out
    write_csv(out / "molecules.csv", MOLECULE_FIELDS, molecule_rows)
    write_csv(out / "relations.csv", RELATION_FIELDS, relation_rows)
    write_csv(out / "authentic.csv", ("molecule_id", "smiles", "label"),
              ({"molecule_id": molecule_id(s), "smiles": s, "label": 0} for s in sorted(auth)))
    eligible_auth = []
    auth_reject = Counter()
    for smiles in sorted(auth):
        quality = child_quality(smiles)
        if not isinstance(quality, dict):
            auth_reject[str(quality or "invalid_molecule")] += 1
            continue
        eligible_auth.append({
            "molecule_id": molecule_id(smiles), "smiles": smiles, "label": 0,
            **{k: v for k, v in quality.items() if k not in ("fingerprint", "n_atoms")},
        })
    write_csv(out / "authentic_eligible.csv",
              ("molecule_id", "smiles", "label", "mw", "logp", "qed",
               "sa_score", "heavy_atoms", "smiles_length", "child_scaffold"),
              eligible_auth)

    slices = defaultdict(dict)
    for rel in relation_rows:
        key = (rel["method"], rel["category"], rel["subcategory"])
        child = rel["smiles"]
        slices[key].setdefault(child, rel)
    slice_index = {}
    for (method, category, subcat), entries in sorted(slices.items()):
        label = f"{method}/{category}/{subcat}"
        relpath = Path("subcategories") / method / category / f"{safe_name(subcat)}.csv"
        write_csv(out / relpath, RELATION_FIELDS,
                  (entries[s] for s in sorted(entries)))
        slice_index[label] = {"path": str(relpath), "unique_molecules": len(entries)}

    summary = {
        "schema_version": 1,
        "n_unique_counterfeits": len(molecule_rows),
        "n_distinct_relations": len(relation_rows),
        "new_single_atom_verification_enabled": not args.skip_new_edit_verification,
        "mcs_mask_reconstruction_enabled": not args.skip_mcs_mask_reconstruction,
        "n_authentic_reference": len(auth),
        "authentic_source_unique_counts": authentic_source_counts,
        "n_authentic_eligible_for_training": len(eligible_auth),
        "authentic_excluded_by_quality_reason": dict(sorted(auth_reject.items())),
        "method_membership": dict(sorted(method_counts.items())),
        "n_subcategories": len(slice_index),
        "n_unique_child_scaffolds": len({r["child_scaffold"] for r in molecule_rows}),
        "n_unique_parent_scaffolds": len({r["parent_scaffold"] for r in relation_rows}),
        "edit_status_relation_counts": dict(sorted(Counter(
            r["edit_status"] for r in relation_rows).items())),
        "nonmissing_edit_mask_atom_count_distribution": distribution(
            len(r["edited_atoms"].split()) for r in relation_rows if r["edited_atoms"]),
        "molecule_qed_distribution": distribution(r["qed"] for r in molecule_rows),
        "molecule_sa_distribution": distribution(r["sa_score"] for r in molecule_rows),
        "relation_similarity_distribution": distribution(
            r["similarity_to_parent"] for r in relation_rows),
        "source_rows": dict(source_counts),
        "source_sha256": {name: sha256_file(path) for name, path, _ in source_files},
        "authentic_source_sha256": {path.name: sha256_file(path) for path in authentic_files},
        "rules_sha256": sha256_file(ROOT / "mined_transformations.py"),
        "verified_manifest_sha256": sha256_file(ROOT / "benchmark/verified_edits/manifest.csv"),
        "builder_sha256": sha256_file(Path(__file__)),
        "environment": json.loads((ROOT / "benchmark/environment.json").read_text()),
        "rejected_source_rows_by_reason": dict(sorted(reject.items())),
        "primary_category_counts": dict(sorted(primary_counts.items())),
        "quality_filters": {
            "authentic_overlap": "excluded", "parent_must_be_in_reference": True,
            "single_component_parent_and_child": True,
            "canonical_smiles": True, "heavy_atoms": [10, 70], "mw": [150, 800],
            "qed_min": 0.35, "sa_score_max": 3.8,
            "PAINS_Brenk_alerts": "excluded", "Morgan_radius3_Tanimoto": [0.4, 0.98],
        },
        "files": {"molecules": "molecules.csv", "relations": "relations.csv",
                  "authentic": "authentic.csv",
                  "authentic_eligible": "authentic_eligible.csv",
                  "subcategories": "subcategories/index.json"},
    }
    (out / "subcategories/index.json").write_text(json.dumps(slice_index, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_unique_counterfeits", "n_distinct_relations", "method_membership", "n_subcategories")}, indent=2))


if __name__ == "__main__":
    main()
