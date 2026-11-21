"""Record the software and compute environment used for benchmark baselines."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("benchmark/environment.json"))
    args = parser.parse_args()

    import rdkit
    import sklearn
    import torch
    import torch_geometric
    import xgboost

    report = {
        "python": sys.version.split()[0],
        "system": platform.system(),
        "system_release": platform.release(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "rdkit": rdkit.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
