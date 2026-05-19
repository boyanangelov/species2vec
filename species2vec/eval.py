"""Quantitative evaluation of species2vec embeddings.

Three orthogonal checks:

1. congener_score
   For each species with at least one congener in the vocab, measure mean
   cosine to congeners vs mean cosine to a random sample of non-congeners.
   A useful embedding scores higher on congeners (correlated with shared
   range), but the gap should not be ~1.0 — that means it learned names,
   not geography.

2. sympatry_auc
   Hold out 10% of occurrence rows. For every (species_a, species_b) pair
   observed in the same held-out geohash bin, label = 1. Sample a matched
   number of pairs never co-occurring in the held-out fold, label = 0.
   Score = ROC-AUC of cosine similarity as a discriminator.
   This is the spatial-distributional signal the model is meant to capture.

3. neighborhood_self_rank
   For each held-out (species, bin) row, rank all vocabulary species by
   cosine to the centroid of OTHER species in that bin. Report the median
   rank of the held-out species (lower = better). Random baseline = |V|/2.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pygeohash
from gensim.models import KeyedVectors
from sklearn.metrics import roc_auc_score


@dataclass
class EvalReport:
    n_species: int
    congener_score: float | None
    congener_gap: float | None
    sympatry_auc: float | None
    sympatry_n_pairs: int
    sympatry_auc_nocong: float | None
    sympatry_n_pairs_nocong: int
    neighborhood_median_rank: float | None
    neighborhood_median_pct: float | None

    def pretty(self) -> str:
        def fmt(v, p=3):
            return "n/a" if v is None else f"{v:.{p}f}"

        return (
            f"species in vocab:              {self.n_species}\n"
            f"congener mean cosine:          {fmt(self.congener_score)}\n"
            f"  gap vs non-congener:         {fmt(self.congener_gap)}\n"
            f"sympatry pair AUC:             {fmt(self.sympatry_auc)}  "
            f"(n_pairs={self.sympatry_n_pairs})\n"
            f"sympatry AUC (cross-genus):    {fmt(self.sympatry_auc_nocong)}  "
            f"(n_pairs={self.sympatry_n_pairs_nocong})\n"
            f"held-out median self-rank:     {fmt(self.neighborhood_median_rank, 1)}  "
            f"({fmt(self.neighborhood_median_pct, 1)} %ile)"
        )


def _load_vec(path: str | Path) -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(str(path))


def _genus(name: str) -> str:
    return name.split("_", 1)[0]


def congener_eval(
    kv: KeyedVectors,
    n_samples: int = 5000,
    n_neg_per_sp: int = 20,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    rng = random.Random(seed)
    vocab = list(kv.key_to_index.keys())
    by_genus: dict[str, list[str]] = defaultdict(list)
    for w in vocab:
        by_genus[_genus(w)].append(w)
    eligible = [w for w in vocab if len(by_genus[_genus(w)]) >= 2]
    if not eligible:
        return None, None
    if len(eligible) > n_samples:
        eligible = rng.sample(eligible, n_samples)

    pos_means, neg_means = [], []
    for w in eligible:
        cong = [c for c in by_genus[_genus(w)] if c != w]
        pos = [float(kv.similarity(w, c)) for c in cong]
        non = [v for v in vocab if _genus(v) != _genus(w)]
        if len(non) > n_neg_per_sp:
            non = rng.sample(non, n_neg_per_sp)
        neg = [float(kv.similarity(w, c)) for c in non]
        if pos and neg:
            pos_means.append(np.mean(pos))
            neg_means.append(np.mean(neg))
    if not pos_means:
        return None, None
    pos_mean = float(np.mean(pos_means))
    neg_mean = float(np.mean(neg_means))
    return pos_mean, pos_mean - neg_mean


def sympatry_eval(
    kv: KeyedVectors,
    holdout: pd.DataFrame,
    *,
    geohash_precision: int,
    max_pairs: int = 20000,
    seed: int = 0,
    exclude_congeners: bool = False,
) -> tuple[float | None, int]:
    rng = random.Random(seed)
    df = holdout.copy()
    df["geohash"] = df.apply(
        lambda r: pygeohash.encode(
            float(r["decimalLatitude"]),
            float(r["decimalLongitude"]),
            precision=geohash_precision,
        ),
        axis=1,
    )
    df = df[df["species"].isin(kv.key_to_index)]
    bins = df.groupby("geohash")["species"].apply(lambda s: sorted(set(s)))
    bins = bins[bins.map(len) >= 2]
    if bins.empty:
        return None, 0

    pos_pairs: set[tuple[str, str]] = set()
    for sp_list in bins:
        for i, a in enumerate(sp_list):
            for b in sp_list[i + 1 :]:
                if exclude_congeners and _genus(a) == _genus(b):
                    continue
                pos_pairs.add((a, b))
    pos_pairs = set(rng.sample(sorted(pos_pairs), min(len(pos_pairs), max_pairs // 2)))

    vocab = list(kv.key_to_index.keys())
    neg_pairs: set[tuple[str, str]] = set()
    attempts = 0
    target = len(pos_pairs)
    while len(neg_pairs) < target and attempts < target * 20:
        a, b = rng.sample(vocab, 2)
        if exclude_congeners and _genus(a) == _genus(b):
            attempts += 1
            continue
        pair = (a, b) if a < b else (b, a)
        if pair not in pos_pairs:
            neg_pairs.add(pair)
        attempts += 1

    if not neg_pairs:
        return None, 0

    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    scores = [
        float(kv.similarity(a, b)) for a, b in list(pos_pairs) + list(neg_pairs)
    ]
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        return None, len(pos_pairs) + len(neg_pairs)
    return auc, len(pos_pairs) + len(neg_pairs)


def neighborhood_eval(
    kv: KeyedVectors,
    holdout: pd.DataFrame,
    *,
    geohash_precision: int,
    max_samples: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    rng = random.Random(seed)
    df = holdout.copy()
    df["geohash"] = df.apply(
        lambda r: pygeohash.encode(
            float(r["decimalLatitude"]),
            float(r["decimalLongitude"]),
            precision=geohash_precision,
        ),
        axis=1,
    )
    df = df[df["species"].isin(kv.key_to_index)]
    bins = df.groupby("geohash")["species"].apply(lambda s: sorted(set(s)))
    bins = bins[bins.map(len) >= 3]
    if bins.empty:
        return None, None

    vocab = list(kv.key_to_index.keys())
    V = kv.get_normed_vectors()
    idx = {w: i for i, w in enumerate(vocab)}

    samples: list[tuple[str, list[str]]] = []
    for sp_list in bins:
        for held in sp_list:
            others = [s for s in sp_list if s != held]
            samples.append((held, others))
    if len(samples) > max_samples:
        samples = rng.sample(samples, max_samples)

    ranks = []
    for held, others in samples:
        centroid = V[[idx[s] for s in others]].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        sims = V @ centroid
        order = np.argsort(-sims)
        rank = int(np.where(order == idx[held])[0][0])
        ranks.append(rank)
    if not ranks:
        return None, None
    med = float(np.median(ranks))
    pct = 100.0 * med / len(vocab)
    return med, pct


def evaluate(
    vec_path: str | Path,
    holdout_df: pd.DataFrame | None = None,
    *,
    geohash_precision: int = 5,
    seed: int = 0,
) -> EvalReport:
    kv = _load_vec(vec_path)
    cong, gap = congener_eval(kv, seed=seed)
    if holdout_df is not None and len(holdout_df):
        auc, n_pairs = sympatry_eval(
            kv, holdout_df, geohash_precision=geohash_precision, seed=seed
        )
        auc_nc, n_pairs_nc = sympatry_eval(
            kv,
            holdout_df,
            geohash_precision=geohash_precision,
            seed=seed,
            exclude_congeners=True,
        )
        med_rank, med_pct = neighborhood_eval(
            kv, holdout_df, geohash_precision=geohash_precision, seed=seed
        )
    else:
        auc, n_pairs, med_rank, med_pct = None, 0, None, None
        auc_nc, n_pairs_nc = None, 0
    return EvalReport(
        n_species=len(kv.key_to_index),
        congener_score=cong,
        congener_gap=gap,
        sympatry_auc=auc,
        sympatry_n_pairs=n_pairs,
        sympatry_auc_nocong=auc_nc,
        sympatry_n_pairs_nocong=n_pairs_nc,
        neighborhood_median_rank=med_rank,
        neighborhood_median_pct=med_pct,
    )
