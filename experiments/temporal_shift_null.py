"""Permutation null for the temporal range-shift correlation.

Pools all records from pre+post slices, then randomly reassigns
each record to a synthetic "pre" or "post" bucket (preserving the
original epoch sizes), retrains both embeddings, and recomputes
the Spearman correlation between centroid drift and embedding
distance. Repeats N times to build a null distribution.

If the observed correlation is well outside the null, the result
is real. If the null is also positive (~ observed), most of the
"signal" is sampling artifact.

Usage:
    PYTHONPATH=. uv run python experiments/temporal_shift_null.py \\
        --pre-csv data/bombus_pre.csv --post-csv data/bombus_post.csv \\
        --n-perms 10 --label Bombus
"""
from __future__ import annotations
from pathlib import Path
import argparse
import shutil
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_rho(pre: pd.DataFrame, post: pd.DataFrame, workdir: Path,
                geohash_precision: int, min_records: int, seed: int) -> tuple[float, int]:
    from species2vec.pipeline import run as train_run
    if workdir.exists():
        shutil.rmtree(workdir)
    train_run(pre, workdir=workdir / "pre",
              geohash_precision=geohash_precision, seed=seed, min_count=3)
    train_run(post, workdir=workdir / "post",
              geohash_precision=geohash_precision, seed=seed, min_count=3)
    pre_kv = KeyedVectors.load_word2vec_format(str(workdir / "pre" / "embeddings.vec"))
    post_kv = KeyedVectors.load_word2vec_format(str(workdir / "post" / "embeddings.vec"))
    common = sorted(
        w for w in (set(pre_kv.key_to_index) & set(post_kv.key_to_index))
        if "_" in w
    )
    pre = pre.copy()
    post = post.copy()
    pre["sp"] = pre["species"].str.replace(" ", "_")
    post["sp"] = post["species"].str.replace(" ", "_")
    pre_c = pre.groupby("sp")[["decimalLatitude", "decimalLongitude"]].mean()
    post_c = post.groupby("sp")[["decimalLatitude", "decimalLongitude"]].mean()
    pre_n = pre.groupby("sp").size()
    post_n = post.groupby("sp").size()
    rows = []
    for sp in common:
        if sp not in pre_c.index or sp not in post_c.index:
            continue
        if pre_n.get(sp, 0) < min_records or post_n.get(sp, 0) < min_records:
            continue
        v_pre, v_post = pre_kv[sp], post_kv[sp]
        cos = float(v_pre @ v_post /
                    (np.linalg.norm(v_pre) * np.linalg.norm(v_post) + 1e-9))
        dkm = float(haversine_km(
            pre_c.loc[sp, "decimalLatitude"], pre_c.loc[sp, "decimalLongitude"],
            post_c.loc[sp, "decimalLatitude"], post_c.loc[sp, "decimalLongitude"],
        ))
        rows.append((1 - cos, dkm))
    if len(rows) < 30:
        return float("nan"), len(rows)
    emb_d = np.array([r[0] for r in rows])
    geo_d = np.array([r[1] for r in rows])
    rho, _ = spearmanr(emb_d, geo_d)
    return float(rho), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-csv", required=True)
    ap.add_argument("--post-csv", required=True)
    ap.add_argument("--n-perms", type=int, default=10)
    ap.add_argument("--min-records", type=int, default=10)
    ap.add_argument("--geohash-precision", type=int, default=4)
    ap.add_argument("--label", default="taxon")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else \
        ROOT / "runs" / f"temporal_null_{args.label.lower()}"
    outdir.mkdir(parents=True, exist_ok=True)

    pre_df = pd.read_csv(args.pre_csv).dropna(
        subset=["decimalLatitude", "decimalLongitude", "species"])
    post_df = pd.read_csv(args.post_csv).dropna(
        subset=["decimalLatitude", "decimalLongitude", "species"])
    print(f"{args.label}: pre={len(pre_df):,} post={len(post_df):,}")

    print("\n=== observed (true split) ===")
    rho_obs, n_obs = compute_rho(pre_df, post_df, outdir / "obs",
                                  args.geohash_precision, args.min_records,
                                  seed=42)
    print(f"observed rho = {rho_obs:.3f}  (n = {n_obs})")

    pooled = pd.concat([pre_df, post_df], ignore_index=True)
    n_pre = len(pre_df)
    rng = np.random.default_rng(0)

    null_rhos = []
    for k in range(args.n_perms):
        print(f"\n=== perm {k + 1}/{args.n_perms} ===")
        idx = rng.permutation(len(pooled))
        fake_pre = pooled.iloc[idx[:n_pre]].reset_index(drop=True)
        fake_post = pooled.iloc[idx[n_pre:]].reset_index(drop=True)
        rho_k = float("nan")
        n_k = 0
        for retry in range(3):
            try:
                rho_k, n_k = compute_rho(
                    fake_pre, fake_post, outdir / f"perm{k}",
                    args.geohash_precision, args.min_records,
                    seed=42 + k * 31 + retry,
                )
                break
            except RuntimeError as e:
                print(f"  retry {retry + 1}: {e}")
                continue
        print(f"  perm rho = {rho_k:.3f}  (n = {n_k})")
        if not np.isnan(rho_k):
            null_rhos.append(rho_k)

    null = np.array(null_rhos)
    print(f"\n========== {args.label} null summary ==========")
    print(f"observed rho   = {rho_obs:.3f}  (n = {n_obs})")
    print(f"null mean      = {null.mean():.3f}")
    print(f"null sd        = {null.std():.3f}")
    print(f"null 95% range = [{np.quantile(null, 0.025):.3f}, "
          f"{np.quantile(null, 0.975):.3f}]")
    p_perm = float((null >= rho_obs).mean())
    print(f"permutation p  = {p_perm:.3f}  "
          f"({int((null >= rho_obs).sum())}/{len(null)} perms >= observed)")

    pd.DataFrame({"perm": list(range(len(null))) + ["observed"],
                  "rho": list(null) + [rho_obs]}).to_csv(
        outdir / "null_summary.csv", index=False)
    print(f"wrote {outdir / 'null_summary.csv'}")


if __name__ == "__main__":
    main()
