"""Country-sliced parallel downloader to dodge GBIF's per-query rate limit.

Issues several concurrent search queries, each filtered to a different
country code. Saves to a single CSV with incremental flushes.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from species2vec.gbif_download import GBIF, PAGE_LIMIT, order_key

LOCK = threading.Lock()

DEFAULT_COUNTRIES = [
    "US", "AU", "BR", "MX", "ZA", "ID", "IN", "CN", "MY", "TH",
    "CO", "PE", "AR", "VE", "MG", "KE", "TZ", "ES", "FR", "IT",
]


def worker(
    key: int, country: str, max_records: int, all_rows: list, pbar: tqdm,
    year_range: str | None = None,
) -> None:
    offset = 0
    while offset < max_records:
        try:
            params = {
                "taxonKey": key,
                "country": country,
                "hasCoordinate": "true",
                "hasGeospatialIssue": "false",
                "limit": PAGE_LIMIT,
                "offset": offset,
            }
            if year_range:
                params["year"] = year_range
            r = requests.get(
                f"{GBIF}/occurrence/search",
                params=params,
                timeout=60,
            )
            r.raise_for_status()
            resp = r.json()
        except Exception as e:
            print(f"\n[{country}] error: {e}", file=sys.stderr)
            time.sleep(10)
            continue
        results = resp.get("results", [])
        if not results:
            break
        batch = []
        for rec in results:
            sp = rec.get("species")
            lat = rec.get("decimalLatitude")
            lon = rec.get("decimalLongitude")
            year = rec.get("year")
            if sp and lat is not None and lon is not None:
                batch.append((sp, lat, lon, year))
        with LOCK:
            all_rows.extend(batch)
            pbar.update(len(results))
        offset += PAGE_LIMIT
        if resp.get("endOfRecords"):
            break


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--order", default="Squamata")
    p.add_argument("--rank", default="order",
                   help="Backbone rank for name lookup (order, family, genus, ...). "
                        "Use 'any' to let GBIF pick the best match.")
    p.add_argument("--out", default="data/squamata.csv")
    p.add_argument("--per-country", type=int, default=10000)
    p.add_argument(
        "--countries",
        nargs="+",
        default=DEFAULT_COUNTRIES,
    )
    p.add_argument("--workers", type=int, default=6)
    p.add_argument(
        "--year",
        default=None,
        help='GBIF year filter, e.g. "1990,2009" for an inclusive range.',
    )
    args = p.parse_args()

    rank = None if args.rank == "any" else args.rank
    key = order_key(args.order, rank=rank)
    print(f"taxonKey({args.order}, rank={rank}) = {key}"
          + (f"  year={args.year}" if args.year else ""))
    all_rows: list = []
    pbar = tqdm(total=args.per_country * len(args.countries), desc=args.order)

    threads: list[threading.Thread] = []
    countries_left = list(args.countries)
    while countries_left or any(t.is_alive() for t in threads):
        threads = [t for t in threads if t.is_alive()]
        while len(threads) < args.workers and countries_left:
            c = countries_left.pop(0)
            t = threading.Thread(
                target=worker,
                args=(key, c, args.per_country, all_rows, pbar),
                kwargs={"year_range": args.year},
            )
            t.start()
            threads.append(t)
        # incremental flush
        with LOCK:
            if all_rows:
                pd.DataFrame(
                    all_rows,
                    columns=pd.Index(["species", "decimalLatitude", "decimalLongitude", "year"]),
                ).drop_duplicates().to_csv(args.out, index=False)
        time.sleep(3)

    pbar.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        all_rows, columns=pd.Index(["species", "decimalLatitude", "decimalLongitude", "year"])
    ).drop_duplicates()
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} unique rows ({df['species'].nunique():,} species) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
