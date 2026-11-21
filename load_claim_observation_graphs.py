"""Load component graphs with the declared identity and reported observations.

No graph-level authentic/counterfeit label is attached to individual chemicals.
A mixture is a list of component graphs, not a single covalently bonded graph.
An absent or unreported API is a status annotation, never an invented graph.
These inputs presume identified components; they do not infer them from spectra.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from train_gnn import smiles_to_data


def iter_case_graphs(root=Path("real_world_falsification/v1")):
    root = Path(root)
    with (root / "molecules.csv").open(newline="") as stream:
        compounds = {row["compound_id"]: row for row in csv.DictReader(stream)}
    with (root / "composition_sets.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)

            def graphs(ids):
                result = []
                for compound_id in ids:
                    graph = smiles_to_data(compounds[compound_id]["normalized_smiles"], label=0)
                    if graph is None:
                        raise ValueError(f"Invalid resolved compound: {compound_id}")
                    for field in ("y", "sample_weight", "group_id"):
                        del graph[field]
                    graph.compound_id = compound_id
                    result.append(graph)
                return result

            has_chemical_finding = record["scenario"] != "source_or_packaging_falsification"
            yield {
                "case_id": record["case_id"], "evidence_group_id": record["evidence_group_id"],
                "declared_compound_ids": record["declared_compound_ids"],
                "detected_compound_ids": record["detected_compound_ids"],
                "declared_graphs": graphs(record["declared_compound_ids"]),
                "detected_graphs": graphs(record["detected_compound_ids"]),
                "not_detected_declared_compound_ids": record["not_detected_declared_compound_ids"],
                "composition_mismatch": 1 if has_chemical_finding else None,
                "analysis_status": record["analysis_status"], "scenario": record["scenario"],
                "reported_detected_quantities": record["reported_detected_quantities"],
                "single_graph_pair_eligible": record["single_graph_pair_eligible"],
                "partition": record["partition"], "complete_sample_composition_known": False,
            }


if __name__ == "__main__":
    for sample in iter_case_graphs():
        print(sample["case_id"], "declared graphs:", len(sample["declared_graphs"]),
              "detected graphs:", len(sample["detected_graphs"]),
              "missing API evidence:", sample["not_detected_declared_compound_ids"])
