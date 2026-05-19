"""species2vec corrected pipeline.

Fixes vs original repo:
1. Geohash precision is explicit (default 5 ≈ 5 km cell), not the default 12 (~mm).
2. One "sentence" per geohash bin written as its own line — no cross-bin
   context bleed during skip-gram window slides.
3. Species deduplicated within a bin so abundance reporting does not bias
   co-occurrence counts.
4. fastText trained with minn=0, maxn=0 to disable character n-grams.
   Otherwise the model learns taxonomic similarity from shared name
   substrings instead of spatial co-occurrence.
5. Randomness seeded; modern fastText API (train_unsupervised).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import fasttext
import numpy as np
import pandas as pd
import pygeohash
from tqdm import tqdm


def _normalize_species(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(" ", "_", regex=False)


def build_corpus(
    df: pd.DataFrame,
    out_path: str | os.PathLike,
    *,
    geohash_precision: int = 5,
    min_bin_size: int = 2,
    species_col: str = "species",
    lat_col: str = "decimalLatitude",
    lon_col: str = "decimalLongitude",
) -> dict:
    """Group occurrences into geohash bins and write one sentence per bin.

    Each line of `out_path` is the deduplicated set of species observed inside
    one geohash cell, in occurrence-frequency order. Lines are independent —
    fastText will not slide its context window across them.
    """
    needed = {species_col, lat_col, lon_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df = df[[species_col, lat_col, lon_col]].dropna().copy()
    df[species_col] = _normalize_species(df[species_col])
    df = df[df[species_col].str.len() > 0]

    tqdm.pandas(desc="geohash")
    df["geohash"] = df.progress_apply(
        lambda r: pygeohash.encode(
            float(r[lat_col]), float(r[lon_col]), precision=geohash_precision
        ),
        axis=1,
    )

    counts = df.groupby(["geohash", species_col]).size().rename("n").reset_index()
    counts = counts.sort_values(["geohash", "n"], ascending=[True, False])

    bin_sizes = counts.groupby("geohash")[species_col].nunique()
    keep = bin_sizes[bin_sizes >= min_bin_size].index
    counts = counts[counts["geohash"].isin(keep)]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    n_tokens = 0
    with out_path.open("w") as fh:
        for _, grp in counts.groupby("geohash", sort=True):
            sentence = " ".join(grp[species_col].tolist())
            fh.write(sentence + "\n")
            n_lines += 1
            n_tokens += len(grp)

    return {
        "corpus_path": str(out_path),
        "n_bins": n_lines,
        "n_tokens": n_tokens,
        "n_species": int(df[species_col].nunique()),
        "n_occurrences": int(len(df)),
        "geohash_precision": geohash_precision,
    }


def train_embeddings(
    corpus_path: str | os.PathLike,
    vec_path: str | os.PathLike,
    *,
    dim: int = 100,
    epoch: int = 25,
    window: int = 8,
    min_count: int = 3,
    lr: float = 0.025,
    seed: int = 42,
    threads: int = 4,
) -> fasttext.FastText._FastText:
    """Skip-gram, char n-grams disabled (minn=0, maxn=0)."""
    random.seed(seed)
    np.random.seed(seed)
    model = fasttext.train_unsupervised(
        str(corpus_path),
        model="skipgram",
        dim=dim,
        epoch=epoch,
        ws=window,
        minCount=min_count,
        minn=0,
        maxn=0,
        lr=lr,
        thread=threads,
        verbose=0,
    )

    vec_path = Path(vec_path)
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    words = model.get_words()
    with vec_path.open("w") as fh:
        fh.write(f"{len(words)} {model.get_dimension()}\n")
        for w in words:
            vec = model.get_word_vector(w)
            fh.write(w + " " + " ".join(f"{x:.6f}" for x in vec) + "\n")
    return model


def run(
    df: pd.DataFrame,
    workdir: str | os.PathLike,
    *,
    geohash_precision: int = 5,
    min_bin_size: int = 2,
    dim: int = 100,
    epoch: int = 25,
    window: int = 8,
    min_count: int = 3,
    lr: float = 0.025,
    seed: int = 42,
) -> dict:
    workdir = Path(workdir)
    corpus_path = workdir / "corpus.txt"
    vec_path = workdir / "embeddings.vec"
    stats = build_corpus(
        df,
        corpus_path,
        geohash_precision=geohash_precision,
        min_bin_size=min_bin_size,
    )
    train_embeddings(
        corpus_path,
        vec_path,
        dim=dim,
        epoch=epoch,
        window=window,
        min_count=min_count,
        lr=lr,
        seed=seed,
    )
    stats["vec_path"] = str(vec_path)
    return stats
