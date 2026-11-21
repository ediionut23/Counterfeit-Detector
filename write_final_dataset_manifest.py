"""Record sizes and SHA-256 hashes for every final-bank deliverable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    root = Path("final_dataset")
    manifest = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest_sha256.json" or path.name.endswith(".partial.sdf"):
            continue
        manifest[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    output = root / "manifest_sha256.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Hashed {len(manifest)} files in {output}")


if __name__ == "__main__":
    main()
