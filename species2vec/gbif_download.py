"""Download GBIF occurrences for a single order via the search API.

Used here for the small-taxon retrain demo. The full corpus used by the
original paper requires a registered GBIF download (multi-GB tsv) — out of
scope for the demo.

We pull Squamata (lizards + snakes): rich enough for spatial signal
(~10k+ species globally) but bounded enough to fit in a search-API budget.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

PAGE_LIMIT = 300
GBIF = "https://api.gbif.org/v1"


def order_key(name: str, rank: str | None = "order") -> int:
    """Resolve a taxon name to its GBIF backbone usageKey.

    `rank` constrains the lookup (default ORDER for backward compat).
    Set rank=None to let GBIF pick the best match across all ranks.
    """
    params = {"name": name}
    if rank:
        params["rank"] = rank
    r = requests.get(f"{GBIF}/species/match", params=params, timeout=30)
    r.raise_for_status()
    res = r.json()
    if "usageKey" not in res:
        raise RuntimeError(f"GBIF backbone returned no usageKey for {name}: {res}")
    return res["usageKey"]


def fetch(order_name: str, out_path: Path, max_records: int = 60000) -> int:
    key = order_key(order_name)
    rows = []
    offset = 0
    pbar = tqdm(total=max_records, desc=order_name)
    flush_every = 1500
    last_flushed = 0
    while offset < max_records:
        try:
            r = requests.get(
                f"{GBIF}/occurrence/search",
                params={
                    "orderKey": key,
                    "hasCoordinate": "true",
                    "hasGeospatialIssue": "false",
                    "limit": PAGE_LIMIT,
                    "offset": offset,
                },
                timeout=60,
            )
            r.raise_for_status()
            resp = r.json()
        except Exception as e:
            print(f"\nGBIF error at offset {offset}: {e}", file=sys.stderr)
            time.sleep(5)
            continue
        results = resp.get("results", [])
        if not results:
            break
        for rec in results:
            sp = rec.get("species")
            lat = rec.get("decimalLatitude")
            lon = rec.get("decimalLongitude")
            if sp and lat is not None and lon is not None:
                rows.append((sp, lat, lon))
        offset += PAGE_LIMIT
        pbar.update(len(results))
        if len(rows) - last_flushed >= flush_every:
            pd.DataFrame(
                rows, columns=pd.Index(["species", "decimalLatitude", "decimalLongitude"])
            ).to_csv(out_path, index=False)
            last_flushed = len(rows)
        if resp.get("endOfRecords"):
            break
    pbar.close()
    pd.DataFrame(
        rows, columns=pd.Index(["species", "decimalLatitude", "decimalLongitude"])
    ).to_csv(out_path, index=False)
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--order", default="Squamata")
    p.add_argument("--out", default="data/squamata.csv")
    p.add_argument("--max", type=int, default=60000)
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = fetch(args.order, out, max_records=args.max)
    print(f"wrote {n} rows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
