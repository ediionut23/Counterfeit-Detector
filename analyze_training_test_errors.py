"""Report FNR by generator and subcategory for a saved GNN test prediction."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def rows(path: Path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("final_dataset/examples/mixed_20k/dataset.csv"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    test = {row["id"]: row for row in rows(args.dataset) if row["split"] == "test"}
    pred = list(rows(args.predictions))
    if len(pred) != len(test) or {r["id"] for r in pred} != set(test):
        raise ValueError("Prediction IDs do not exactly match test rows")
    counts = defaultdict(lambda: [0, 0])
    for row in pred:
        source = test[row["id"]]
        if row["label"] != source["label"] or row["split"] != "test":
            raise ValueError(f"Wrong label or split for {row['id']}")
        score = float(row["counterfeit_score"])
        if not 0 <= score <= 1 or int(row["predicted_label"]) != int(score > 0.5):
            raise ValueError(f"Invalid score/prediction for {row['id']}")
        if source["label"] != "1":
            continue
        method = source["generation_method"]
        if method not in ("M", "E"):
            raise ValueError(f"Unexpected counterfeit method: {method}")
        category = source["category"]
        subcategory = source["subcategory"]
        for key in ("all", f"method/{method}",
                    f"category/{method}/{category}",
                    f"subcategory/{method}/{category}/{subcategory}"):
            counts[key][0] += 1
            counts[key][1] += score <= 0.5
    result = {name: {"positive_test_rows": n, "false_negatives": fn,
                     "fnr": round(fn / n, 4)}
              for name, (n, fn) in sorted(counts.items())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "dataset": str(args.dataset), "predictions": str(args.predictions),
        "test_rows": len(test), "groups": result,
        "note": "Subcategory FNR uses only positive test rows; small groups are unstable."
    }, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in result if key == "all" or key.startswith("method/")}, indent=2))


if __name__ == "__main__":
    main()
