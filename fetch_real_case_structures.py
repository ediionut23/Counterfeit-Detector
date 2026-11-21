#!/usr/bin/env python3
"""Cache PubChem identities for the manually curated real-case compounds.

Only public chemical names are sent to the official NIH PubChem API. Existing
cache entries are reused. Raw responses and request URLs make the mapping
reviewable; biochemical or clinical validity is not inferred from the lookup.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from real_case_curation import COMPOUNDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("real_world_falsification/pubchem_cache"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for compound_id, spec in COMPOUNDS.items():
        output = args.out / f"{compound_id}.json"
        if output.exists():
            print(f"cached {compound_id}", flush=True)
            continue
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               + urllib.parse.quote(spec["query"], safe="")
               + "/property/SMILES,ConnectivitySMILES,InChIKey,Title/JSON")
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "Counterfeit-Detector research identity lookup"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    payload = json.load(response)
                entries = payload["PropertyTable"]["Properties"]
                if len(entries) != 1:
                    raise ValueError(f"Ambiguous PubChem name lookup for {compound_id}: {len(entries)} records")
                output.write_text(json.dumps({"compound_id": compound_id, "query": spec["query"],
                    "request_url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "response": payload}, indent=2) + "\n")
                print(f"resolved {compound_id}: CID {entries[0]['CID']}", flush=True)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
        time.sleep(0.4)  # stay below the documented PubChem request-rate limit


if __name__ == "__main__":
    main()
