"""Write a SHA-256 inventory for the generated GA source bank and 3D files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("version_ga"))
    args = parser.parse_args()
    manifest_path = args.root / "manifest_sha256.json"
    files = {}
    for path in sorted(p for p in args.root.rglob("*") if p.is_file() and p != manifest_path):
        files[str(path.relative_to(args.root))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest = {"schema_version": 1, "root": str(args.root), "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"n_files": len(files), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
