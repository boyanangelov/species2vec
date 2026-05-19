"""End-to-end comparison: broken pipeline vs corrected pipeline.

Trains two models on the same occurrence data:
  A) broken — replicates the original repo's method (one big sentence,
              default high-precision geohash sort, fastText with subword
              n-grams enabled).
  B) fixed  — geohash-binned sentences, no subwords, dedup, seeded.

Both are evaluated against a held-out 10% of occurrences using
species2vec.eval. Prints a side-by-side report.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import fasttext
import numpy as np
import pandas as pd
import pygeohash
from tqdm import tqdm

from species2vec.eval import evaluate
from species2vec.pipeline import build_corpus, train_embeddings


def _split(
    df: pd.DataFrame, holdout: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < holdout
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def broken_pipeline(df: pd.DataFrame, workdir: Path, seed: int) -> Path:
    """Replicates the original repo: precision-12 geohash, sort, concat
    everything into one space-separated line, fasttext skipgram with
    default subword n-grams enabled."""
    df = df.dropna().copy()
    df["species"] = df["species"].astype(str).str.replace(" ", "_")
    tqdm.pandas(desc="broken-geohash")
    df["location_index"] = df.progress_apply(
        lambda r: pygeohash.encode(
            float(r["decimalLatitude"]), float(r["decimalLongitude"])
        ),
        axis=1,
    )
    df = df.sort_values(by="location_index")
    text = " ".join(df["species"].tolist())
    corpus = workdir / "broken_corpus.txt"
    corpus.write_text(text)

    random.seed(seed)
    np.random.seed(seed)
    model = fasttext.train_unsupervised(
        str(corpus),
        model="skipgram",
        dim=100,
        epoch=25,
        ws=5,
        minCount=3,
        thread=4,
        verbose=0,
    )
    vec = workdir / "broken.vec"
    words = model.get_words()
    with vec.open("w") as fh:
        fh.write(f"{len(words)} {model.get_dimension()}\n")
        for w in words:
            v = model.get_word_vector(w)
            fh.write(w + " " + " ".join(f"{x:.6f}" for x in v) + "\n")
    return vec


def fixed_pipeline(df: pd.DataFrame, workdir: Path, seed: int, precision: int) -> Path:
    corpus = workdir / "fixed_corpus.txt"
    stats = build_corpus(df, corpus, geohash_precision=precision, min_bin_size=2)
    print(
        f"  fixed corpus: bins={stats['n_bins']} tokens={stats['n_tokens']} "
        f"species={stats['n_species']}"
    )
    vec = workdir / "fixed.vec"
    train_embeddings(
        corpus,
        vec,
        dim=100,
        epoch=25,
        window=8,
        min_count=2,
        lr=0.025,
        seed=seed,
    )
    return vec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--workdir", default="runs/compare")
    p.add_argument("--precision", type=int, default=5)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep", action="store_true")
    args = p.parse_args()

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["species", "decimalLatitude", "decimalLongitude"])
    print(f"loaded {len(df):,} occurrences, {df['species'].nunique():,} species")

    train_df, hold_df = _split(df, args.holdout, args.seed)
    print(f"  train: {len(train_df):,}  holdout: {len(hold_df):,}")

    print("\n== training BROKEN pipeline (original repo method) ==")
    broken_vec = broken_pipeline(train_df, workdir, args.seed)
    print("\n== training FIXED pipeline ==")
    fixed_vec = fixed_pipeline(train_df, workdir, args.seed, args.precision)

    hold_df_norm = hold_df.copy()
    hold_df_norm["species"] = hold_df_norm["species"].astype(str).str.replace(" ", "_")

    print("\n== EVAL: BROKEN ==")
    print(evaluate(broken_vec, hold_df_norm, geohash_precision=args.precision).pretty())
    print("\n== EVAL: FIXED ==")
    print(evaluate(fixed_vec, hold_df_norm, geohash_precision=args.precision).pretty())

    if not args.keep:
        # Keep .vec files but remove corpora.
        for f in workdir.glob("*_corpus.txt"):
            f.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
