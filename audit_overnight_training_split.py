#!/usr/bin/env python3
"""Predeclare an immutable common molecular-origin cohort for 2D and 3D runs.

No model results are read. Original training membership is retained where
chemistry, external exclusion, unique connectivity and graph availability allow.
Held-out rows are filtered against all earlier declared/observed families.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scalable_bank_store import chemistry_init, normalize_identity


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def rank(source_id, seed, purpose):
    return hashlib.sha256(f"{seed}|{purpose}|{source_id}".encode()).hexdigest()


def normalized_chunk(rows):
    output = []
    for row in rows:
        try:
            _, smiles, connectivity, scaffold = normalize_identity(row["smiles"])
            parent = row.get("parent_smiles") or ""
            if parent:
                _, psmiles, pconn, pscaffold = normalize_identity(parent)
            else:
                psmiles = pconn = pscaffold = ""
            output.append({"source_id": row["molecule_id"], "normalized_smiles": smiles,
                "normalized_connectivity": connectivity, "normalized_own_scaffold": scaffold,
                "normalized_parent_smiles": psmiles, "normalized_parent_connectivity": pconn,
                "normalized_parent_scaffold": pscaffold})
        except Exception as error:
            output.append({"source_id": row["molecule_id"], "normalization_error": str(error)})
    return output


def read_csv(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def families(row):
    return ({row[key] for key in ("normalized_connectivity", "normalized_parent_connectivity") if row.get(key)},
            {row[key] for key in ("normalized_own_scaffold", "normalized_parent_scaffold") if row.get(key)})


def overlap_report(rows):
    report = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        a = [row for row in rows if row["split"] == left]
        b = [row for row in rows if row["split"] == right]
        report[left + "_vs_" + right] = {
            "observed_connectivity": len({r.get("normalized_connectivity") for r in a} & {r.get("normalized_connectivity") for r in b}),
            "observed_own_scaffold": len({r.get("normalized_own_scaffold") for r in a} & {r.get("normalized_own_scaffold") for r in b}),
            "declared_or_observed_connectivity": len(set().union(*(families(r)[0] for r in a)) & set().union(*(families(r)[0] for r in b))),
            "declared_or_observed_scaffold": len(set().union(*(families(r)[1] for r in a)) & set().union(*(families(r)[1] for r in b))),
        }
    return report


def select_rows(rows, graphs, holdout, seed):
    """Pure deterministic selection. Reasons may overlap for the same source."""
    reasons = defaultdict(set)
    blocked_conn = set(holdout.get("blocked_connectivity_smiles", []))
    blocked_scaf = set(holdout.get("blocked_scaffold_keys", []))
    labels = defaultdict(set)
    for row in rows:
        if "normalization_error" not in row:
            labels[row["normalized_connectivity"]].add(row["label"])
    ambiguous = {key for key, values in labels.items() if len(values) > 1}
    for row in rows:
        identity = row["molecule_id"]
        if "normalization_error" in row:
            reasons[identity].add("normalization_error")
            continue
        if identity not in graphs:
            reasons[identity].add("no_available_graph_in_original_balanced_index")
        if row["normalized_connectivity"] in ambiguous:
            reasons[identity].add("global_opposite_labels_same_normalized_connectivity")
        conn, scaffold = families(row)
        if conn & blocked_conn:
            reasons[identity].add("external_holdout_declared_or_observed_connectivity")
        if scaffold & blocked_scaf:
            reasons[identity].add("external_holdout_declared_or_observed_scaffold")
    selected, earlier_conn, earlier_scaffold, seen_connectivity = [], set(), set(), set()
    for split in ("train", "val", "test"):
        candidates = sorted((r for r in rows if r["split"] == split and not reasons[r["molecule_id"]]),
                            key=lambda r: rank(r["molecule_id"], seed, "deduplicate"))
        accepted = []
        for row in candidates:
            identity = row["molecule_id"]
            conn, scaffold = families(row)
            if split != "train" and conn & earlier_conn:
                reasons[identity].add("declared_or_observed_connectivity_exposed_in_earlier_split")
            if split != "train" and scaffold & earlier_scaffold:
                reasons[identity].add("declared_or_observed_scaffold_exposed_in_earlier_split")
            if row["normalized_connectivity"] in seen_connectivity:
                reasons[identity].add("duplicate_observed_normalized_connectivity")
            if reasons[identity]:
                continue
            seen_connectivity.add(row["normalized_connectivity"])
            accepted.append(row)
        if split != "train":
            counts = Counter(r["label"] for r in accepted)
            per_label = min(counts.get("0", 0), counts.get("1", 0))
            balanced = []
            for label in ("0", "1"):
                ordered = sorted((r for r in accepted if r["label"] == label),
                                 key=lambda r: rank(r["molecule_id"], seed, "heldout_balance"))
                balanced.extend(ordered[:per_label])
                for row in ordered[per_label:]:
                    reasons[row["molecule_id"]].add("deterministic_heldout_label_balance")
            accepted = balanced
        selected.extend(accepted)
        for row in accepted:
            conn, scaffold = families(row)
            earlier_conn.update(conn)
            earlier_scaffold.update(scaffold)
    return selected, reasons, ambiguous


def set_training_weights(rows):
    train = [r for r in rows if r["split"] == "train"]
    reference_count = sum(r["label"] == "0" for r in train)
    groups = Counter((r["generation_method"], r.get("balancing_category", r["category"]),
                      r.get("balancing_subcategory", r["subcategory"])) for r in train if r["label"] == "1")
    method_groups = Counter(key[0] for key in groups)
    if not reference_count or not groups:
        raise ValueError("Training requires both molecular-origin labels")
    for row in rows:
        if row["split"] != "train":
            weight = 1.0
        elif row["label"] == "0":
            weight = len(train) / (2 * reference_count)
        else:
            group = (row["generation_method"], row.get("balancing_category", row["category"]),
                     row.get("balancing_subcategory", row["subcategory"]))
            weight = len(train) / (2 * len(method_groups) * method_groups[group[0]] * groups[group])
        row["sample_weight"] = format(weight, ".12g")
    return {"definition": "training weighted mass: public references 1/2; generated 1/2; generated methods equal, groups equal within each method; heldout weights 1",
            "train_reference_count": reference_count, "train_group_counts": {"/".join(k): v for k, v in sorted(groups.items())}}


def build(source, graph_index, graph_root, holdout_path, out, *, seed=42, workers=2, minimum_train=100001):
    source, graph_index, graph_root, holdout_path, out = map(Path, (source, graph_index, graph_root, holdout_path, out))
    if out.exists():
        raise FileExistsError(f"Immutable output already exists: {out}")
    if workers < 1 or minimum_train < 1:
        raise ValueError("Workers and minimum training count must be positive")
    inputs = {str(p.resolve()): sha256(p) for p in (source, graph_index, holdout_path, Path(__file__))}
    rows, fields = read_csv(source)
    if any(r["split"] not in {"train", "val", "test"} or r["label"] not in {"0", "1"} for r in rows):
        raise ValueError("Unexpected source split/label")
    by_id = {row["molecule_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Original source has duplicate source molecule IDs")
    graph_rows, graph_fields = read_csv(graph_index)
    graphs = {row["source_id"]: row for row in graph_rows}
    if len(graphs) != len(graph_rows):
        raise ValueError("Original graph index has duplicate source IDs")
    for identity, row in graphs.items():
        if identity not in by_id or any(row[k] != by_id[identity][k] for k in ("split", "label")):
            raise ValueError(f"Graph index source identity, label or split disagrees: {identity}")
    chunks = (rows[i:i + 1000] for i in range(0, len(rows), 1000))
    with ProcessPoolExecutor(max_workers=workers, initializer=chemistry_init) as pool:
        processed = 0
        for normalized in pool.map(normalized_chunk, chunks, chunksize=1):
            for result in normalized:
                by_id[result["source_id"]].update(result)
            processed += len(normalized)
            if processed % 10000 == 0 or processed == len(rows):
                print(f"Normalized {processed:,}/{len(rows):,} source rows", flush=True)
    valid = [r for r in rows if "normalization_error" not in r]
    original_overlaps = overlap_report(valid)
    holdout = json.loads(holdout_path.read_text())
    selected, reasons, ambiguous = select_rows(rows, graphs, holdout, seed)
    weight_policy = set_training_weights(selected)
    counts = {split: dict(Counter(r["label"] for r in selected if r["split"] == split)) for split in ("train", "val", "test")}
    train_count = sum(counts["train"].values())
    if train_count < minimum_train:
        raise ValueError(f"Only {train_count:,} strict training rows, require {minimum_train:,}; selection will not be relaxed based on results")
    if not all(counts[s].get("0", 0) == counts[s].get("1", 0) and counts[s].get("0", 0) > 0 for s in ("val", "test")):
        raise ValueError(f"No nonempty balanced strict held-out split remains: {counts}")
    strict_overlaps = overlap_report(selected)
    if any(value for overlap in strict_overlaps.values() for value in overlap.values()):
        raise AssertionError("Strict selection still leaks a normalized family across splits")
    selected.sort(key=lambda r: (("train", "val", "test").index(r["split"]), r["molecule_id"]))
    selected_graphs = []
    by_file = defaultdict(list)
    for row in selected:
        graph = dict(graphs[row["molecule_id"]])
        graph["sample_weight"] = row["sample_weight"]
        graph["graph_file"] = str((graph_root / graph["graph_file"]).resolve())
        selected_graphs.append(graph)
        by_file[graph["graph_file"]].append(graph)
    shard_manifest = {}
    for filename, items in sorted(by_file.items()):
        path = Path(filename)
        before = path.stat()
        digest = sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            ids, labels, splits = archive["source_id"], archive["label"], archive["split"]
            molecule_ids = archive["molecule_id"]
            for row in items:
                i = int(row["graph_index"])
                if i < 0 or i >= len(ids) or str(ids[i]) != row["source_id"] or str(int(labels[i])) != row["label"] or str(splits[i]) != row["split"] or str(molecule_ids[i]) != row["molecule_id"]:
                    raise ValueError(f"NPZ/index identity mismatch: {filename}:{i}")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Graph shard moved during audit: {path}")
        shard_manifest[filename] = {"sha256": digest, "bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "selected_graphs": len(items)}
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise RuntimeError(f"Input moved during immutable cohort preparation: {path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + out.name + "_", dir=out.parent))
    try:
        extra_fields = ["normalized_smiles", "normalized_connectivity", "normalized_own_scaffold", "normalized_parent_smiles", "normalized_parent_connectivity", "normalized_parent_scaffold"]
        fields = list(dict.fromkeys(fields + ["sample_weight"] + extra_fields))
        graph_fields = list(dict.fromkeys(graph_fields + ["sample_weight"]))
        write_csv(staging / "dataset.csv", selected, fields)
        write_csv(staging / "graph_index.csv", selected_graphs, graph_fields)
        excluded = [{"source_id": row["molecule_id"], "original_split": row["split"], "original_label": row["label"], "reasons_json": encoded(sorted(reasons[row["molecule_id"]]))} for row in rows if reasons[row["molecule_id"]]]
        write_csv(staging / "excluded_rows.csv", excluded, ["source_id", "original_split", "original_label", "reasons_json"])
        cohort_payload = [[r["molecule_id"], r["split"], r["label"]] for r in sorted(selected, key=lambda r: r["molecule_id"])]
        cohort_hash = hashlib.sha256(encoded(cohort_payload).encode()).hexdigest()
        manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
            "selection_declared_without_model_metrics": True, "source_files_sha256": inputs,
            "source_rows": len(rows), "original_available_graphs": len(graph_rows),
            "rows": len(selected), "train_rows": train_count, "split_label_counts": counts,
            "exclusion_reasons": dict(Counter(reason for values in reasons.values() for reason in values)),
            "excluded_unique_source_rows": len(excluded), "ambiguous_connectivity_count": len(ambiguous),
            "before_overlap": original_overlaps, "after_overlap": strict_overlaps,
            "weight_policy": weight_policy, "graph_shards": shard_manifest,
            "cohort_sha256": cohort_hash,
            "cohort_hash_definition": "SHA256(json.dumps(sorted([[source_id,split,label],...], key=source_id),sort_keys=True,separators=(',',':'),allow_nan=False))",
            "policy": "no split reassignment; global opposite-label normalized connectivity excluded; external holdout on declared and observed sides; one observed connectivity; train first, then val then test; heldout declared/observed connectivity and scaffolds disjoint from all earlier retained rows; val/test label balance by SHA256 only",
            "label_semantics": {"0": "public reference molecular origin; no measured authentic product established", "1": "generated analogue origin; no real counterfeit established"},
            "objective_limit": "Molecule-only models compare molecular origins. Claim-conditioned identity/composition detection requires declared reference and observed component inputs; this cohort does not establish real-world fraud detection.",
        }
        manifest["selected_files"] = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(staging.iterdir()) if p.is_file()}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "README.md").write_text("# Cohortă comună strictă pentru rulările de noapte\n\nSelecția este stabilită înaintea oricărui rezultat de model. `dataset.csv` și `graph_index.csv` conțin exact aceleași ID-uri, etichete și partiții. Fișierele NPZ originale sunt doar referite prin căi absolute; hashurile lor sunt în `manifest.json`.\n\nAntrenarea păstrează partiția originală după excluderea cazurilor externe, a ambiguităților de etichetă, a conectivităților duplicate și a structurilor fără graf. Validarea și testul sunt filtrate pentru separarea conectivităților și scheletelor de pe ambele laturi, declarată și observată, apoi balansate 0/1 prin hash determinist. Nicio moleculă nu este mutată între partiții. `excluded_rows.csv` păstrează toate motivele. Ponderile sunt calculate numai din antrenare; evaluarea folosește ponderi 1.\n\nEtichetele descriu originea moleculară publică/generată, fără dovadă clinică de autenticitate sau falsificare. GAT și modelul geometric trebuie comparate pe această cohortă comună. Alegerea hiperparametrilor și a pragului folosește exclusiv validarea; testul se accesează după înghețarea modelului.\n")
        staging.replace(out)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("balanced_datasets/training_200k/subcategory_weighted/dataset.csv"))
    parser.add_argument("--graph-index", type=Path, default=Path("balanced_datasets/training_200k/3d/balanced_graph_index.csv"))
    parser.add_argument("--graph-root", type=Path, default=Path("balanced_datasets/training_200k/3d"))
    parser.add_argument("--holdout", type=Path, default=Path("real_world_falsification/v1/external_holdout.json"))
    parser.add_argument("--out", type=Path, default=Path("balanced_datasets/training_200k/overnight_strict_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--minimum-train", type=int, default=100001)
    args = parser.parse_args()
    manifest = build(args.source, args.graph_index, args.graph_root, args.holdout, args.out,
                     seed=args.seed, workers=args.workers, minimum_train=args.minimum_train)
    print(json.dumps({key: manifest[key] for key in ("rows", "train_rows", "split_label_counts", "cohort_sha256", "exclusion_reasons")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
