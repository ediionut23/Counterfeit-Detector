#!/usr/bin/env python3
"""Record source, data, pipeline and environment fingerprints for this pilot."""

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


root = Path("real_world_falsification")
output = root / "manifest_sha256.json"
scripts = ["real_case_curation.py", "fetch_real_case_structures.py", "fetch_real_case_sources.py",
           "build_real_case_benchmark.py", "load_claim_observation_graphs.py",
           "diagnose_real_case_gnn.py", "audit_real_case_overlap.py", "fetch_field_spectra.py",
           "audit_real_case_artifacts.py", "build_real_case_report.py", "write_real_case_manifest.py", "REAL_WORLD_FALSIFICATION_REPORT.md"]
data = {str(p): digest(p) for p in sorted(root.rglob("*")) if p.is_file() and p != output}
pipeline = {name: digest(Path(name)) for name in scripts if Path(name).exists()}
output.write_text(json.dumps({"written_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "packages": {name: importlib.metadata.version(name) for name in ("rdkit", "torch", "torch-geometric")},
    "pipeline_sha256": pipeline, "data_sha256": data}, indent=2) + "\n")
print(f"Wrote {output}: {len(data)} data files, {len(pipeline)} pipeline/report files")
