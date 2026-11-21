"""Aggregate saved E parent-child ordering audits over the three model seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    "child_score_mean", "parent_score_mean", "mean_child_minus_parent",
    "child_scores_higher_fraction", "pair_correct_at_half_fraction",
    "parent_scores_authentic_fraction",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("benchmark/pair_ordering"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    reports = [json.loads((args.directory / f"summary_seed{seed}.json").read_text())
               for seed in args.seeds]
    for seed, report in zip(args.seeds, reports):
        if report["seed"] != seed or report["test_counterfeit_pairs"] != 4625:
            raise ValueError(f"Unexpected seed or pair count in seed {seed}")
    output = {"dataset": "evolved", "seeds": args.seeds,
              "test_counterfeit_pairs": 4625,
              "note": "Same fixed test pairs; sample SD describes model-seed variation, not independent test cohorts",
              "models": {}}
    for model in ("gin", "rf", "xgboost"):
        output["models"][model] = {}
        for subset in ("all_test_pairs", "verified_one_atom_pairs"):
            values = [report["models"][model][subset] for report in reports]
            n = values[0]["n"]
            if any(record["n"] != n for record in values):
                raise ValueError(f"Subset size changed for {model}/{subset}")
            output["models"][model][subset] = {"n": n}
            for metric in METRICS:
                numbers = [record[metric] for record in values]
                output["models"][model][subset][metric] = {
                    "mean": round(statistics.mean(numbers), 4),
                    "std": round(statistics.stdev(numbers), 4),
                    "values": numbers,
                }
    path = args.directory / "summary.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path)
    for model, group in output["models"].items():
        print(model, group["all_test_pairs"]["child_scores_higher_fraction"],
              group["all_test_pairs"]["pair_correct_at_half_fraction"])


if __name__ == "__main__":
    main()
