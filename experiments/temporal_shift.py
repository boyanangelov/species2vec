"""Temporal range-shift experiment.

Trains two species2vec models on disjoint year ranges of Squamata GBIF
records (a "pre" slice, default 1990-2009, and a "post" slice, default
2017-2026) and tests whether species whose occurrence centroid has
shifted geographically between the two epochs also shift in embedding
space.

The two slices must be downloaded separately with explicit GBIF year
filters because GBIF's default ordering is overwhelmingly biased toward
recent uploads — a naive global download yields almost no pre-2010
records. See species2vec/gbif_download_parallel.py --year.

Output: runs/temporal/{pre,post}/embeddings.vec, shift_summary.csv,
         manuscript/figures/range_shift.pdf
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "temporal"
FIG = ROOT / "manuscript" / "figures"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_slice(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude", "species"])
    return df


def main():
    from species2vec.pipeline import run as train_run

    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-csv", default=str(ROOT / "data" / "squamata_pre.csv"))
    ap.add_argument("--post-csv", default=str(ROOT / "data" / "squamata_post.csv"))
    ap.add_argument("--min-records", type=int, default=10,
                    help="min records per species in BOTH slices")
    ap.add_argument("--geohash-precision", type=int, default=4)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    pre = load_slice(Path(args.pre_csv))
    post = load_slice(Path(args.post_csv))
    if "year" in pre.columns:
        pre_yr = pd.to_numeric(pre["year"], errors="coerce").dropna().astype(int)
        if len(pre_yr) > 0:
            print(f"pre : {len(pre):,} records, {pre['species'].nunique():,} sp, "
                  f"years {pre_yr.min()}-{pre_yr.max()}")
    if "year" in post.columns:
        post_yr = pd.to_numeric(post["year"], errors="coerce").dropna().astype(int)
        if len(post_yr) > 0:
            print(f"post: {len(post):,} records, {post['species'].nunique():,} sp, "
                  f"years {post_yr.min()}-{post_yr.max()}")

    print(f"training pre slice -> {OUT/'pre'}")
    train_run(pre, workdir=OUT / "pre",
              geohash_precision=args.geohash_precision, seed=42, min_count=3)
    print(f"training post slice -> {OUT/'post'}")
    train_run(post, workdir=OUT / "post",
              geohash_precision=args.geohash_precision, seed=42, min_count=3)

    pre_kv = KeyedVectors.load_word2vec_format(str(OUT / "pre" / "embeddings.vec"))
    post_kv = KeyedVectors.load_word2vec_format(str(OUT / "post" / "embeddings.vec"))
    common_vocab = sorted(
        w for w in (set(pre_kv.key_to_index) & set(post_kv.key_to_index))
        if "_" in w
    )
    print(f"common vocab: {len(common_vocab)}")

    pre["sp"] = pre["species"].str.replace(" ", "_")
    post["sp"] = post["species"].str.replace(" ", "_")
    pre_c = pre.groupby("sp")[["decimalLatitude", "decimalLongitude"]].mean()
    post_c = post.groupby("sp")[["decimalLatitude", "decimalLongitude"]].mean()
    pre_n = pre.groupby("sp").size()
    post_n = post.groupby("sp").size()

    rows = []
    for sp in common_vocab:
        if sp not in pre_c.index or sp not in post_c.index:
            continue
        if pre_n.get(sp, 0) < args.min_records or post_n.get(sp, 0) < args.min_records:
            continue
        v_pre, v_post = pre_kv[sp], post_kv[sp]
        cos = float(v_pre @ v_post /
                    (np.linalg.norm(v_pre) * np.linalg.norm(v_post) + 1e-9))
        dkm = float(haversine_km(
            pre_c.loc[sp, "decimalLatitude"], pre_c.loc[sp, "decimalLongitude"],
            post_c.loc[sp, "decimalLatitude"], post_c.loc[sp, "decimalLongitude"],
        ))
        rows.append((sp, int(pre_n[sp]), int(post_n[sp]), cos, dkm))

    summary = pd.DataFrame(
        rows, columns=["species", "n_pre", "n_post", "emb_cos", "centroid_km"]
    )
    summary.to_csv(OUT / "shift_summary.csv", index=False)
    print(f"species with >={args.min_records} records both slices: {len(summary)}")
    if len(summary) < 30:
        print("too few species for a stable correlation; aborting plot")
        return

    rho, p = spearmanr(1 - summary["emb_cos"], summary["centroid_km"])
    print(f"Spearman rho (1 - cos vs centroid km): {rho:.3f}   p = {p:.2e}")
    print(f"median centroid drift: {summary['centroid_km'].median():.0f} km")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax = axes[0]
    bins = np.linspace(0, summary["centroid_km"].quantile(0.97), 40)
    ax.hist(summary["centroid_km"], bins=bins, color="#3c6eb3", alpha=0.85)
    ax.set_xlabel("centroid drift (km)")
    ax.set_ylabel("species")
    ax.set_title(
        f"Spatial drift, pre -> post\n"
        f"median = {summary['centroid_km'].median():.0f} km, "
        f"n = {len(summary)}"
    )

    ax = axes[1]
    ax.scatter(summary["centroid_km"], 1 - summary["emb_cos"],
               s=10, alpha=0.45, color="#3c6eb3", edgecolor="none")
    ax.set_xlabel("centroid drift (km)")
    ax.set_ylabel(r"embedding distance $1 - \cos(v_{\mathrm{pre}}, v_{\mathrm{post}})$")
    ax.set_title(f"Embedding tracks spatial drift\nSpearman $\\rho$ = {rho:.3f}")

    fig.tight_layout()
    out = FIG / "range_shift.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
