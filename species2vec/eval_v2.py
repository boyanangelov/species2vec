"""Continuous-distance evaluation of species2vec embeddings.

Three checks that do not depend on a single geohash precision (so they
cannot be gamed by picking the binning that matches training):

1. mantel_test
   Spearman rho between pairwise embedding cosine and great-circle distance
   between species range centroids (km). Significance via row/column
   permutation of the geographic distance matrix (Mantel).
   A useful embedding has rho clearly negative (closer in space -> higher
   cosine). Permutation null gives a p-value.

2. distance_decay
   Bin all species pairs by inter-centroid distance (log-spaced km), report
   median cosine per bin. Monotone decreasing curve = real spatial signal.
   The slope of cosine vs log-distance is a single-number summary.

3. precision_sweep
   Re-evaluate sympatry AUC at multiple geohash precisions (p3..p7).
   A robust spatial signal is roughly stable across scales; a precision
   that wildly outperforms the rest indicates the model only learned
   that one binning.

A Jaccard baseline is also provided: rank pairs by raw bin-Jaccard on the
training fold and report sympatry AUC. If the embedding does not beat
Jaccard, it is not adding information.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import pandas as pd
import pygeohash
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from species2vec.eval import _load_vec, sympatry_eval


EARTH_R_KM = 6371.0088


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def _normalize_species(s) -> pd.Series:
    return pd.Series(s).astype(str).str.strip().str.replace(" ", "_", regex=False)


def _centroids(
    df: pd.DataFrame,
    vocab: set[str],
    min_occ: int = 5,
) -> pd.DataFrame:
    """Spherical centroid (mean of unit vectors) per species. Avoids the
    lat/lon wraparound that a flat mean produces near the antimeridian.
    """
    d = cast(
        pd.DataFrame,
        df[["species", "decimalLatitude", "decimalLongitude"]].dropna().copy(),
    )
    d["species"] = _normalize_species(d["species"])
    d = cast(pd.DataFrame, d.loc[d["species"].isin(list(vocab))].copy())
    lat = np.radians(d["decimalLatitude"].to_numpy())
    lon = np.radians(d["decimalLongitude"].to_numpy())
    d["x"] = np.cos(lat) * np.cos(lon)
    d["y"] = np.cos(lat) * np.sin(lon)
    d["z"] = np.sin(lat)
    g = cast(
        pd.DataFrame,
        d.groupby("species").agg(
            n=("x", "size"), x=("x", "mean"), y=("y", "mean"), z=("z", "mean")
        ),
    )
    g = cast(pd.DataFrame, g.loc[g["n"] >= min_occ])
    norm = np.sqrt(g["x"] ** 2 + g["y"] ** 2 + g["z"] ** 2)
    g["lat"] = np.degrees(np.arcsin(g["z"] / norm))
    g["lon"] = np.degrees(np.arctan2(g["y"], g["x"]))
    return g[["lat", "lon", "n"]].reset_index()


def _pairwise_geo_km(cent: pd.DataFrame) -> np.ndarray:
    lat = cent["lat"].to_numpy()
    lon = cent["lon"].to_numpy()
    n = len(cent)
    lat1 = lat[:, None].repeat(n, axis=1)
    lat2 = lat[None, :].repeat(n, axis=0)
    lon1 = lon[:, None].repeat(n, axis=1)
    lon2 = lon[None, :].repeat(n, axis=0)
    return _haversine_km(lat1, lon1, lat2, lon2)


def _pairwise_cosine(kv: KeyedVectors, species: list[str]) -> np.ndarray:
    V = np.stack([kv[s] for s in species])
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V @ V.T


@dataclass
class MantelResult:
    rho: float
    p_value: float
    n_species: int
    n_pairs: int
    n_permutations: int


def mantel_test(
    kv: KeyedVectors,
    df: pd.DataFrame,
    *,
    min_occ: int = 5,
    max_species: int = 800,
    n_permutations: int = 999,
    seed: int = 0,
) -> MantelResult | None:
    """Spearman(cosine, geo_km) with Mantel permutation null.

    max_species caps memory at O(N^2). 800 -> ~640k pairs -> ~5 MB float32.
    """
    vocab = set(kv.key_to_index)
    cent = _centroids(df, vocab, min_occ=min_occ)
    if len(cent) < 20:
        return None
    if len(cent) > max_species:
        cent = cent.sample(max_species, random_state=seed).reset_index(drop=True)

    species = cent["species"].tolist()
    geo = _pairwise_geo_km(cent)
    cos = _pairwise_cosine(kv, species)

    iu, ju = np.triu_indices(len(species), k=1)
    geo_v = geo[iu, ju]
    cos_v = cos[iu, ju]

    rho_obs = float(spearmanr(cos_v, geo_v)[0])  # type: ignore[index]
    rng = np.random.default_rng(seed)
    n = len(species)
    null = np.empty(n_permutations)
    for k in range(n_permutations):
        perm = rng.permutation(n)
        geo_p = geo[np.ix_(perm, perm)]
        null[k] = float(spearmanr(cos_v, geo_p[iu, ju])[0])  # type: ignore[index]
    p = float((np.sum(np.abs(null) >= abs(rho_obs)) + 1) / (n_permutations + 1))
    return MantelResult(
        rho=float(rho_obs),
        p_value=p,
        n_species=n,
        n_pairs=int(len(geo_v)),
        n_permutations=n_permutations,
    )


@dataclass
class DistanceDecayResult:
    edges_km: list[float]
    median_cosine: list[float]
    n_pairs_per_bin: list[int]
    slope_per_log10_km: float


def distance_decay(
    kv: KeyedVectors,
    df: pd.DataFrame,
    *,
    min_occ: int = 5,
    max_species: int = 800,
    n_bins: int = 12,
    seed: int = 0,
) -> DistanceDecayResult | None:
    vocab = set(kv.key_to_index)
    cent = _centroids(df, vocab, min_occ=min_occ)
    if len(cent) < 20:
        return None
    if len(cent) > max_species:
        cent = cent.sample(max_species, random_state=seed).reset_index(drop=True)
    species = cent["species"].tolist()
    geo = _pairwise_geo_km(cent)
    cos = _pairwise_cosine(kv, species)
    iu, ju = np.triu_indices(len(species), k=1)
    geo_v = geo[iu, ju]
    cos_v = cos[iu, ju]
    mask = geo_v > 0
    geo_v = geo_v[mask]
    cos_v = cos_v[mask]

    edges = np.logspace(
        np.log10(max(geo_v.min(), 1.0)),
        np.log10(geo_v.max()),
        n_bins + 1,
    )
    med, counts = [], []
    log_centers, log_medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (geo_v >= lo) & (geo_v < hi)
        if sel.sum() < 5:
            med.append(float("nan"))
            counts.append(int(sel.sum()))
            continue
        m = float(np.median(cos_v[sel]))
        med.append(m)
        counts.append(int(sel.sum()))
        log_centers.append(0.5 * (np.log10(lo) + np.log10(hi)))
        log_medians.append(m)
    if len(log_centers) >= 3:
        slope = float(np.polyfit(log_centers, log_medians, 1)[0])
    else:
        slope = float("nan")
    return DistanceDecayResult(
        edges_km=[float(e) for e in edges],
        median_cosine=med,
        n_pairs_per_bin=counts,
        slope_per_log10_km=slope,
    )


@dataclass
class PrecisionRow:
    precision: int
    cell_km: float
    sympatry_auc: float | None
    sympatry_auc_nocong: float | None
    n_pairs: int


@dataclass
class PrecisionSweepResult:
    rows: list[PrecisionRow] = field(default_factory=list)

    def pretty(self) -> str:
        head = f"{'p':>3} {'~km':>7} {'AUC':>7} {'AUC_xgen':>9} {'n_pairs':>9}"
        body = "\n".join(
            f"{r.precision:>3} {r.cell_km:>7.2f} "
            f"{'n/a' if r.sympatry_auc is None else f'{r.sympatry_auc:.3f}':>7} "
            f"{'n/a' if r.sympatry_auc_nocong is None else f'{r.sympatry_auc_nocong:.3f}':>9} "
            f"{r.n_pairs:>9}"
            for r in self.rows
        )
        return head + "\n" + body


# Approximate cell width at the equator. Geohash cells are roughly square at
# the equator and shrink toward the poles. Numbers from the standard
# geohash spec.
_GEOHASH_CELL_KM = {
    3: 156.0,
    4: 39.1,
    5: 4.89,
    6: 1.22,
    7: 0.153,
    8: 0.0382,
}


def precision_sweep(
    kv: KeyedVectors,
    holdout: pd.DataFrame,
    *,
    precisions: Sequence[int] = (3, 4, 5, 6, 7),
    seed: int = 0,
) -> PrecisionSweepResult:
    out = PrecisionSweepResult()
    for p in precisions:
        auc, n = sympatry_eval(kv, holdout, geohash_precision=p, seed=seed)
        auc_nc, _ = sympatry_eval(
            kv, holdout, geohash_precision=p, seed=seed, exclude_congeners=True
        )
        out.rows.append(
            PrecisionRow(
                precision=p,
                cell_km=_GEOHASH_CELL_KM.get(p, float("nan")),
                sympatry_auc=auc,
                sympatry_auc_nocong=auc_nc,
                n_pairs=n,
            )
        )
    return out


def jaccard_baseline(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    *,
    geohash_precision: int = 5,
    vocab: set[str] | None = None,
    max_pairs: int = 20000,
    seed: int = 0,
    exclude_congeners: bool = False,
) -> tuple[float | None, int]:
    """Rank pairs by Jaccard(set of training bins per species). If the
    embedding does not beat this on held-out sympatry, it adds nothing.
    """
    rng = random.Random(seed)
    tr = cast(
        pd.DataFrame,
        train_df[["species", "decimalLatitude", "decimalLongitude"]].dropna().copy(),
    )
    tr["species"] = _normalize_species(tr["species"])
    if vocab is not None:
        tr = cast(pd.DataFrame, tr.loc[tr["species"].isin(list(vocab))].copy())
    tr["geohash"] = tr.apply(
        lambda r: pygeohash.encode(
            float(r["decimalLatitude"]),
            float(r["decimalLongitude"]),
            precision=geohash_precision,
        ),
        axis=1,
    )
    sp_bins: dict[str, set[str]] = defaultdict(set)
    for sp, gh in zip(tr["species"], tr["geohash"]):
        sp_bins[sp].add(gh)
    species = list(sp_bins)
    if len(species) < 2:
        return None, 0

    ho = cast(
        pd.DataFrame,
        holdout_df[["species", "decimalLatitude", "decimalLongitude"]].dropna().copy(),
    )
    ho["species"] = _normalize_species(ho["species"])
    if vocab is not None:
        ho = cast(pd.DataFrame, ho.loc[ho["species"].isin(list(vocab))].copy())
    ho = cast(pd.DataFrame, ho.loc[ho["species"].isin(list(sp_bins))].copy())
    ho["geohash"] = ho.apply(
        lambda r: pygeohash.encode(
            float(r["decimalLatitude"]),
            float(r["decimalLongitude"]),
            precision=geohash_precision,
        ),
        axis=1,
    )
    bins = cast(
        pd.Series,
        ho.groupby("geohash")["species"].apply(lambda s: sorted(set(s))),
    )
    bins = cast(pd.Series, bins[bins.map(len) >= 2])
    if bins.empty:
        return None, 0

    def _genus(name: str) -> str:
        return name.split("_", 1)[0]

    pos_pairs: set[tuple[str, str]] = set()
    for sp_list in bins:
        for i, a in enumerate(sp_list):
            for b in sp_list[i + 1 :]:
                if exclude_congeners and _genus(a) == _genus(b):
                    continue
                pos_pairs.add((a, b))
    pos_pairs = set(rng.sample(sorted(pos_pairs), min(len(pos_pairs), max_pairs // 2)))

    neg_pairs: set[tuple[str, str]] = set()
    attempts = 0
    target = len(pos_pairs)
    while len(neg_pairs) < target and attempts < target * 20:
        a, b = rng.sample(species, 2)
        if exclude_congeners and _genus(a) == _genus(b):
            attempts += 1
            continue
        pair = (a, b) if a < b else (b, a)
        if pair not in pos_pairs:
            neg_pairs.add(pair)
        attempts += 1
    if not neg_pairs:
        return None, 0

    def _jac(a: str, b: str) -> float:
        sa, sb = sp_bins[a], sp_bins[b]
        u = len(sa | sb)
        return 0.0 if u == 0 else len(sa & sb) / u

    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    scores = [_jac(a, b) for a, b in list(pos_pairs) + list(neg_pairs)]
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        return None, len(pos_pairs) + len(neg_pairs)
    return auc, len(pos_pairs) + len(neg_pairs)


@dataclass
class EvalV2Report:
    mantel: MantelResult | None
    decay: DistanceDecayResult | None
    sweep: PrecisionSweepResult
    jaccard_auc: float | None
    jaccard_auc_nocong: float | None
    jaccard_n_pairs: int

    def pretty(self) -> str:
        parts = []
        if self.mantel is not None:
            parts.append(
                "Mantel (cosine vs geo_km):\n"
                f"  rho                 = {self.mantel.rho:+.3f}\n"
                f"  p (perm, n={self.mantel.n_permutations}) = {self.mantel.p_value:.4f}\n"
                f"  species             = {self.mantel.n_species}\n"
                f"  pairs               = {self.mantel.n_pairs}"
            )
        else:
            parts.append("Mantel: insufficient data")

        if self.decay is not None:
            parts.append(
                f"Distance decay:\n"
                f"  slope d(cos)/d(log10 km) = {self.decay.slope_per_log10_km:+.3f}\n"
                f"  (negative = closer in space -> higher cosine)"
            )
        else:
            parts.append("Distance decay: insufficient data")

        parts.append("Precision sweep:\n" + self.sweep.pretty())

        j = "n/a" if self.jaccard_auc is None else f"{self.jaccard_auc:.3f}"
        j_nc = "n/a" if self.jaccard_auc_nocong is None else f"{self.jaccard_auc_nocong:.3f}"
        parts.append(
            f"Jaccard baseline (training-bin sets, holdout-bin pairs):\n"
            f"  AUC            = {j}\n"
            f"  AUC cross-gen  = {j_nc}\n"
            f"  n_pairs        = {self.jaccard_n_pairs}"
        )
        return "\n\n".join(parts)


def evaluate_v2(
    vec_path: str | Path,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    *,
    precisions: Sequence[int] = (3, 4, 5, 6, 7),
    mantel_max_species: int = 800,
    mantel_permutations: int = 999,
    min_occ: int = 5,
    seed: int = 0,
) -> EvalV2Report:
    kv = _load_vec(vec_path)
    full = pd.concat([train_df, holdout_df], ignore_index=True)
    mantel = mantel_test(
        kv,
        full,
        max_species=mantel_max_species,
        n_permutations=mantel_permutations,
        min_occ=min_occ,
        seed=seed,
    )
    decay = distance_decay(
        kv, full, max_species=mantel_max_species, min_occ=min_occ, seed=seed
    )
    sweep_holdout = holdout_df.copy()
    sweep_holdout["species"] = _normalize_species(sweep_holdout["species"])
    sweep = precision_sweep(kv, sweep_holdout, precisions=list(precisions), seed=seed)
    jac_auc, jac_n = jaccard_baseline(
        train_df, holdout_df, vocab=set(kv.key_to_index), seed=seed
    )
    jac_auc_nc, _ = jaccard_baseline(
        train_df,
        holdout_df,
        vocab=set(kv.key_to_index),
        seed=seed,
        exclude_congeners=True,
    )
    return EvalV2Report(
        mantel=mantel,
        decay=decay,
        sweep=sweep,
        jaccard_auc=jac_auc,
        jaccard_auc_nocong=jac_auc_nc,
        jaccard_n_pairs=jac_n,
    )


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vec", required=True)
    ap.add_argument("--csv", required=True, help="full GBIF csv (will be split)")
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument(
        "--precisions",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7],
    )
    ap.add_argument("--permutations", type=int, default=999)
    ap.add_argument("--mantel-max-species", type=int, default=800)
    ap.add_argument("--min-occ", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - args.holdout))
    train_df = df.iloc[idx[:cut]].reset_index(drop=True)
    holdout_df = df.iloc[idx[cut:]].reset_index(drop=True)

    rep = evaluate_v2(
        args.vec,
        train_df,
        holdout_df,
        precisions=args.precisions,
        mantel_max_species=args.mantel_max_species,
        mantel_permutations=args.permutations,
        min_occ=args.min_occ,
        seed=args.seed,
    )
    print(rep.pretty())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
