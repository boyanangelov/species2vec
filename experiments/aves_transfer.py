"""Train Apodiformes (hummingbirds+swifts) embeddings + run eval harness.

Tests whether species2vec generalises beyond Squamata. Compares against
the Squamata baseline on the same metrics + adds class-level transfer.

Output: runs/aves/embeddings.vec, figures/aves_transfer.pdf
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import random

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "passeriformes.csv"
OUT = ROOT / "runs" / "aves"
FIG = ROOT / "manuscript" / "figures"
TAXON = "Passeriformes"


def _genus(name: str) -> str:
    return name.split("_", 1)[0] if "_" in name else name.split(" ", 1)[0]


def congener_eval(kv: KeyedVectors, seed: int = 0, n: int = 2000):
    rng = random.Random(seed)
    vocab = [w for w in kv.index_to_key if "_" in w]
    by_g: dict[str, list[str]] = {}
    for w in vocab:
        by_g.setdefault(_genus(w), []).append(w)
    groups = [v for v in by_g.values() if len(v) >= 2]
    cong, rand = [], []
    for _ in range(n):
        g = rng.choice(groups)
        a, b = rng.sample(g, 2)
        cong.append(float(kv.similarity(a, b)))
        x, y = rng.sample(vocab, 2)
        while _genus(x) == _genus(y):
            x, y = rng.sample(vocab, 2)
        rand.append(float(kv.similarity(x, y)))
    return np.array(cong), np.array(rand)


def sympatry_auc(kv: KeyedVectors, df: pd.DataFrame, geohash_precision: int = 4,
                 n_pairs: int = 5000, seed: int = 0):
    from sklearn.metrics import roc_auc_score
    import pygeohash
    df = df.copy()
    df["sp"] = df["species"].str.replace(" ", "_")
    df = df[df["sp"].isin(kv.key_to_index)]
    df["gh"] = [pygeohash.encode(float(la), float(lo), precision=geohash_precision)
                for la, lo in zip(df["decimalLatitude"], df["decimalLongitude"])]

    bin_species = df.groupby("gh")["sp"].apply(lambda s: list(set(s)))
    bin_species = bin_species[bin_species.map(len) >= 2]

    rng = random.Random(seed)
    pos, neg = [], []
    bins = bin_species.index.tolist()
    for _ in range(n_pairs):
        b = rng.choice(bins)
        sps = bin_species[b]
        a, c = rng.sample(sps, 2)
        if _genus(a) == _genus(c):
            continue
        pos.append((a, c))

    vocab = list(set(s for spl in bin_species for s in spl))
    # build a set of co-occurring pairs to exclude
    cooc = set()
    for spl in bin_species:
        for i in range(len(spl)):
            for j in range(i + 1, len(spl)):
                cooc.add(frozenset((spl[i], spl[j])))

    while len(neg) < len(pos):
        a, b = rng.sample(vocab, 2)
        if frozenset((a, b)) in cooc or _genus(a) == _genus(b):
            continue
        neg.append((a, b))

    y, scores = [], []
    for a, b in pos:
        y.append(1); scores.append(float(kv.similarity(a, b)))
    for a, b in neg:
        y.append(0); scores.append(float(kv.similarity(a, b)))
    return roc_auc_score(y, scores), len(pos)


def main():
    from species2vec.pipeline import run as train_run

    df = pd.read_csv(CSV)
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude", "species"])
    print(f"{TAXON}: {len(df):,} records, {df['species'].nunique()} species")

    train_run(df, workdir=OUT, geohash_precision=4, seed=42, min_count=3)
    kv = KeyedVectors.load_word2vec_format(str(OUT / "embeddings.vec"))
    print(f"vocab: {len(kv.key_to_index)}")

    cong, rand = congener_eval(kv)
    gap = float(cong.mean() - rand.mean())
    auc, npos = sympatry_auc(kv, df, geohash_precision=4)
    print(f"{TAXON} congener gap = {gap:.3f}")
    print(f"{TAXON} cross-genus sympatry AUC = {auc:.3f}  (n+={npos})")

    sq_kv = KeyedVectors.load_word2vec_format(str(ROOT / "runs" / "compare" / "fixed.vec"))
    sq_cong, sq_rand = congener_eval(sq_kv)
    sq_gap = float(sq_cong.mean() - sq_rand.mean())
    print(f"(Squamata reference gap = {sq_gap:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    ax = axes[0]
    bins_e = np.linspace(-0.3, 1.0, 40)
    ax.hist(rand, bins=bins_e, alpha=0.6, color="#999999", label="random pairs", density=True)
    ax.hist(cong, bins=bins_e, alpha=0.6, color="#3c6eb3", label="congener pairs", density=True)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(f"{TAXON} (Aves)\nn={len(kv.key_to_index)} species, $\\Delta$={gap:.3f}")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    labels = ["Squamata", TAXON]
    gaps = [sq_gap, gap]
    aucs = [0.943, auc]
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, gaps, w, label="congener gap", color="#b3543c")
    b2 = ax.bar(x + w / 2, aucs, w, label="cross-genus sympatry AUC", color="#3c6eb3")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#999999", linestyle=":", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}",
                    ha="center", fontsize=7.5)

    fig.tight_layout()
    out = FIG / "aves_transfer.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
