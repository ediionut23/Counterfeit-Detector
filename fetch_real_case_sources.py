#!/usr/bin/env python3
"""Archive the curated official source documents and their SHA-256 digests."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from real_case_curation import SOURCES


def main():
    out = Path("real_world_falsification/source_cache")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for source_id, source in SOURCES.items():
        for kind in ("url", "annex_url", "supporting_url"):
            url = source.get(kind)
            if not url:
                continue
            record = {"source_id": source_id, "kind": kind, "url": url}
            stem = source_id + "_" + kind
            try:
                existing = next(iter(out.glob(stem + ".*")), None)
                if existing:
                    payload = existing.read_bytes()
                    record.update({"file": existing.name, "status": "cached"})
                else:
                    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (scientific-source archival)"})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        content_type = response.headers.get("Content-Type", "")
                        payload = response.read(8_000_001)
                        if len(payload) > 8_000_000:
                            raise ValueError("Source exceeded 8 MB download limit")
                    suffix = ".pdf" if payload.startswith(b"%PDF") else ".html"
                    existing = out / (stem + suffix)
                    existing.write_bytes(payload)
                    record.update({"file": existing.name, "content_type": content_type, "status": "downloaded"})
                record["sha256"] = hashlib.sha256(payload).hexdigest()
                record["bytes"] = len(payload)
            except Exception as error:
                record.update({"status": "download_failed", "error": str(error)})
            records.append(record)
            print(source_id, kind, record["status"], flush=True)
    (out / "manifest.json").write_text(json.dumps({"archived_utc": datetime.now(timezone.utc).isoformat(),
        "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
