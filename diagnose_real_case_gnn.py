#!/usr/bin/env python3
"""Describe a saved molecule-only GNN's outputs on documented structure pairs.

This is a diagnostic of the existing generated-versus-public-reference model.
It is not clinical validation or a falsification classifier: the registry has
no independently adjudicated negative controls, and the model never receives
the claimed ingredient, product composition, dose, or source evidence.

Required inputs under --benchmark are molecules.csv and structure_pairs.csv.
Outputs preserve case/evidence provenance without treating repeated reports as
independent evaluations. No F1, accuracy, or ROC-AUC is computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader

from train_gnn import GIN, MessagePassingClassifier, smiles_to_data


MOLECULE_FIELDS = {"compound_id", "name", "normalized_smiles", "pubchem_cid", "inchikey"}
PAIR_FIELDS = {"case_id", "evidence_group_id", "relation_id",
               "declared_compound_id", "observed_compound_id"}
SUPPORTED_MODELS = {"gin", "gcn", "gat", "gat2"}


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a train_gnn.py checkpoint dictionary")
    architecture = checkpoint.get("model")
    if architecture not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported checkpoint model {architecture!r}; expected one of {sorted(SUPPORTED_MODELS)}")
    required = {"in_dim", "hidden", "layers", "state_dict", "ring_size_features"}
    if required - checkpoint.keys():
        raise ValueError(f"Checkpoint lacks reconstruction fields: {sorted(required - checkpoint.keys())}")
    in_dim, hidden, layers = (int(checkpoint[k]) for k in ("in_dim", "hidden", "layers"))
    if min(in_dim, hidden, layers) < 1:
        raise ValueError("Checkpoint dimensions and layer count must be positive")
    if architecture == "gin":
        model = GIN(in_dim, hidden=hidden, layers=layers)
    else:
        model = MessagePassingClassifier(architecture, in_dim, hidden=hidden, layers=layers)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_compounds(compounds: list[dict[str, str]], model, checkpoint: dict,
                      batch_size: int) -> tuple[dict[str, float], dict[str, str]]:
    canonical_by_id = {}
    graphs_by_smiles = {}
    for compound in compounds:
        compound_id = compound["compound_id"].strip()
        if not compound_id or compound_id in canonical_by_id:
            raise ValueError(f"Missing or duplicate compound_id: {compound_id!r}")
        mol = Chem.MolFromSmiles(compound["normalized_smiles"])
        if mol is None or not mol.GetNumAtoms():
            raise ValueError(f"Invalid normalized_smiles for {compound_id}")
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        canonical_by_id[compound_id] = canonical
        if canonical in graphs_by_smiles:
            continue
        # y=0 is a required batching placeholder, not a ground-truth judgment.
        graph = smiles_to_data(canonical, label=0,
                               ring_size_features=bool(checkpoint["ring_size_features"]))
        if graph is None or graph.x.shape[1] != int(checkpoint["in_dim"]):
            raise ValueError(f"Cannot reconstruct checkpoint features for {compound_id}")
        graphs_by_smiles[canonical] = graph
    structures = sorted(graphs_by_smiles)
    probabilities = []
    for batch in DataLoader([graphs_by_smiles[s] for s in structures], batch_size=batch_size,
                            shuffle=False):
        logits = model(batch.x, batch.edge_index, batch.batch)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"Expected two model output classes, got {tuple(logits.shape)}")
        score = torch.softmax(logits, dim=-1)[:, 1]
        if not torch.isfinite(score).all():
            raise ValueError("Model produced a non-finite probability")
        probabilities.extend(score.tolist())
    probability_by_structure = dict(zip(structures, probabilities, strict=True))
    return {key: probability_by_structure[smi] for key, smi in canonical_by_id.items()}, canonical_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("real_world_falsification/v1"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("training_runs/gat2_subcategory_20261031/large_weighted.pt"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1:
        parser.error("--batch-size and --threads must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    molecules_path = args.benchmark / "molecules.csv"
    pairs_path = args.benchmark / "structure_pairs.csv"
    compounds = read_csv(molecules_path, MOLECULE_FIELDS)
    pairs = read_csv(pairs_path, PAIR_FIELDS)
    if not compounds or not pairs:
        raise ValueError("Need nonempty molecules.csv and structure_pairs.csv")
    model, checkpoint = restore_model(args.checkpoint)
    probabilities, canonical = predict_compounds(compounds, model, checkpoint, args.batch_size)
    compounds_by_id = {c["compound_id"]: c for c in compounds}
    roles = defaultdict(set)
    relation_pairs = {}
    for pair in pairs:
        if any(not pair[k].strip() for k in PAIR_FIELDS):
            raise ValueError("Structure pair has a missing required identity/provenance field")
        declared, observed = pair["declared_compound_id"], pair["observed_compound_id"]
        if declared not in probabilities or observed not in probabilities:
            raise ValueError(f"Unknown compound ID in relation {pair['relation_id']}")
        identity = (declared, observed)
        prior = relation_pairs.setdefault(pair["relation_id"], identity)
        if prior != identity:
            raise ValueError(f"relation_id {pair['relation_id']} maps to conflicting compound pairs")
        roles[declared].add("declared")
        roles[observed].add("observed")

    molecule_predictions = []
    for compound in sorted(compounds, key=lambda row: row["compound_id"]):
        compound_id = compound["compound_id"]
        molecule_predictions.append({
            **{key: compound[key] for key in sorted(MOLECULE_FIELDS)},
            "model_input_canonical_smiles": canonical[compound_id],
            "roles_in_structure_pairs": ";".join(sorted(roles[compound_id])),
            "p_generated": probabilities[compound_id],
            # argmax in the original two-class model selects class 0 at a tie.
            "model_class_at_0_5": int(probabilities[compound_id] > 0.5),
        })

    case_diagnostics = []
    for pair in pairs:
        declared, observed = pair["declared_compound_id"], pair["observed_compound_id"]
        case_diagnostics.append({
            **{key: pair[key] for key in sorted(PAIR_FIELDS)},
            "declared_name": compounds_by_id[declared]["name"],
            "observed_name": compounds_by_id[observed]["name"],
            "same_model_input_structure": int(canonical[declared] == canonical[observed]),
            "declared_p_generated": probabilities[declared],
            "observed_p_generated": probabilities[observed],
            "observed_minus_declared_p_generated": probabilities[observed] - probabilities[declared],
            "declared_model_class_at_0_5": int(probabilities[declared] > 0.5),
            "observed_model_class_at_0_5": int(probabilities[observed] > 0.5),
            "model_receives_declared_context": 0,
        })

    observed_claims = defaultdict(set)
    for declared, observed in relation_pairs.values():
        observed_claims[observed].add(declared)
    groups = defaultdict(set)
    for pair in pairs:
        groups[pair["evidence_group_id"]].add(pair["relation_id"])
    unique_relations = []
    seen_relations = set()
    for row in case_diagnostics:
        if row["relation_id"] not in seen_relations:
            unique_relations.append(row)
            seen_relations.add(row["relation_id"])

    out = args.out or args.benchmark / "gnn_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "molecule_predictions.csv", molecule_predictions, list(molecule_predictions[0]))
    write_csv(out / "case_diagnostics.csv", case_diagnostics, list(case_diagnostics[0]))
    summary = {
        "diagnostic_type": "unary generated-versus-public-reference GNN applied to case component structures",
        "clinical_or_falsification_validation": False,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_model": checkpoint["model"],
        "checkpoint_training_dataset": checkpoint.get("dataset"),
        "inputs": {str(p): sha256(p) for p in (molecules_path, pairs_path)},
        "original_model_threshold": 0.5, "tie_policy": "class 0, matching two-class argmax",
        "model_label_semantics": {"0": "public-reference-like model output", "1": "generated-analogue-like model output"},
        "registry_ground_truth_negative_labels_assigned": False,
        "unique_compounds": len(compounds), "unique_input_structures": len(set(canonical.values())),
        "structure_pair_rows": len(pairs), "unique_cases": len({r["case_id"] for r in pairs}),
        "unique_relations": len(relation_pairs), "unique_evidence_groups": len(groups),
        "relation_counts_by_evidence_group": {k: len(v) for k, v in sorted(groups.items())},
        "compound_model_output_counts": dict(sorted(Counter(str(r["model_class_at_0_5"]) for r in molecule_predictions).items())),
        "unique_relation_observed_model_output_counts": dict(sorted(Counter(str(r["observed_model_class_at_0_5"]) for r in unique_relations).items())),
        "unique_relations_with_identical_declared_observed_structure": sum(r["same_model_input_structure"] for r in unique_relations),
        "compounds_in_both_declared_and_observed_roles": sorted(k for k, v in roles.items() if v == {"declared", "observed"}),
        "observed_compounds_with_multiple_declared_identities": {k: sorted(v) for k, v in sorted(observed_claims.items()) if len(v) > 1},
        "caveats": [
            "These are descriptive model outputs, not sensitivity, specificity, F1, accuracy, or ROC-AUC on a validated population.",
            "The model only receives an individual molecular graph and was trained on generated analogues versus public references.",
            "Registry active-moiety normalization may differ from pharmaceutical forms in the original training CSV; these are diagnostic outputs on the normalized structures.",
            "Claimed components are not automatically approved medicines, authentic products, or negative labels.",
            "The same observed structure has exactly the same cached probability regardless of the claimed ingredient or case context.",
            "Repeated cases, relations, and evidence groups are counted separately in the report and are not independent validation samples.",
            "Unknown ingredients, mixtures, absent APIs, doses, spectra, packaging and product provenance are outside this molecular input.",
            "Known claimed and observed structures also permit a deterministic identity comparison; GNN output does not replace it.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "unique_compounds": len(compounds),
                      "unique_relations": len(relation_pairs), "unique_evidence_groups": len(groups),
                      "clinical_or_falsification_validation": False}, indent=2))


if __name__ == "__main__":
    main()
