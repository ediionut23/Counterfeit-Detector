"""Find verified elemental edit families shared by exclusive M and E analogs.

An elemental signature is a one-element multiset change, not a complete
reaction rule. Use it only to identify candidate strata for a later graph-edit
and source-provenance audit.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def atom_counts(smiles: str) -> Counter:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


def signature(parent: str, child: str) -> str | None:
    before, after = atom_counts(parent), atom_counts(child)
    removed, added = before - after, after - before
    if sum(removed.values()) > 1 or sum(added.values()) > 1 or not (removed or added):
        return None
    old = next(iter(removed)) if removed else "∅"
    new = next(iter(added)) if added else "∅"
    return f"{old}→{new}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("final_dataset"))
    parser.add_argument("--out", type=Path,
                        default=Path("research_runs/shared_exact_edit_signatures.json"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("research_runs/shared_exact_edit_relations.csv"))
    args = parser.parse_args()
    with (args.bank / "molecules.csv").open(newline="") as stream:
        molecule_methods = {r["smiles"]: r["methods"] for r in csv.DictReader(stream)}
    counts = Counter()
    children = defaultdict(set)
    parents = defaultdict(set)
    provenance = Counter()
    manifest_rows = []
    with (args.bank / "relations.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["edit_status"] != "exact_single_atom":
                continue
            method = row["method"]
            if molecule_methods[row["smiles"]] != method:
                provenance["dual_method_child_excluded"] += 1
                continue
            sig = signature(row["parent_smiles"], row["smiles"])
            if sig is None:
                provenance["not_single_element_change"] += 1
                continue
            key = (sig, method)
            counts[key] += 1
            children[key].add(row["smiles"])
            parents[key].add(row["parent_smiles"])
            provenance["accepted_relations"] += 1
            manifest_rows.append({
                "relation_id": row["relation_id"], "smiles": row["smiles"],
                "parent_smiles": row["parent_smiles"], "method": method,
                "signature": sig, "category": row["category"],
                "subcategory": row["subcategory"],
                "similarity_to_parent": row["similarity_to_parent"],
                "edited_atoms": row["edited_atoms"],
                "parent_scaffold": row["parent_scaffold"],
                "child_scaffold": row["child_scaffold"],
            })
    signatures = []
    for sig in {key[0] for key in counts}:
        signatures.append({
            "signature": sig,
            "M_relations": counts[sig, "M"], "E_relations": counts[sig, "E"],
            "M_children": len(children[sig, "M"]), "E_children": len(children[sig, "E"]),
            "M_parents": len(parents[sig, "M"]), "E_parents": len(parents[sig, "E"]),
        })
    signatures.sort(key=lambda r: (-min(r["M_children"], r["E_children"]),
                                   -r["E_children"], r["signature"]))
    output = {
        "bank": str(args.bank),
        "definition": "Element-count delta for exact_single_atom relations with exclusive M or E child provenance.",
        "limitation": "Elemental signature does not verify bond topology, edit site, chemical context, or independent generator provenance.",
        "counts": dict(provenance),
        "shared_signatures_with_100_children_per_method": sum(
            row["M_children"] >= 100 and row["E_children"] >= 100 for row in signatures),
        "signatures": signatures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(json.dumps({"counts": output["counts"],
                      "shared_signatures_with_100_children_per_method":
                      output["shared_signatures_with_100_children_per_method"],
                      "top_signatures": signatures[:12]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
