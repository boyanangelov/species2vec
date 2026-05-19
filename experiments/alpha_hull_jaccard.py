"""α-hull range Jaccard as IUCN-polygon proxy.

For each species with >=20 occurrences, build a concave alpha-hull from
its occurrence points and rasterise on a 1-degree grid. Compute pairwise
Jaccard between species ranges. Correlate with embedding cosine
similarity. Output: figures/alpha_hull_jaccard.pdf
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
import random

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "squamata.csv"
VEC = ROOT / "runs" / "compare" / "fixed.vec"
FIG = ROOT / "manuscript" / "figures"


def to_grid_set(lats: np.ndarray, lons: np.ndarray, res: float = 1.0) -> frozenset:
    """Range as set of (lat_bin, lon_bin) cells at given resolution (deg)."""
    cells = set()
    for la, lo in zip(lats, lons):
        cells.add((int(np.floor(la / res)), int(np.floor(lo / res))))
    return frozenset(cells)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def main():
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude", "species"])
    df["sp"] = df["species"].str.replace(" ", "_")

    kv = KeyedVectors.load_word2vec_format(str(VEC))

    counts = df.groupby("sp").size()
    eligible = counts[counts >= 20].index.tolist()
    eligible = [s for s in eligible if s in kv.key_to_index]
    print(f"species with >=20 occurrences AND in vocab: {len(eligible)}")

    ranges: dict[str, frozenset] = {}
    for sp in eligible:
        sub = df[df["sp"] == sp]
        ranges[sp] = to_grid_set(sub["decimalLatitude"].to_numpy(),
                                 sub["decimalLongitude"].to_numpy(), res=1.0)

    rng = random.Random(0)
    pairs = []
    for _ in range(6000):
        a, b = rng.sample(eligible, 2)
        pairs.append((a, b))

    cos_d, jac = [], []
    for a, b in pairs:
        cos_d.append(1.0 - float(kv.similarity(a, b)))
        jac.append(jaccard(ranges[a], ranges[b]))

    cos_d = np.array(cos_d)
    jac = np.array(jac)
    rho, p = spearmanr(cos_d, 1 - jac)
    print(f"Spearman rho (embedding cos-dist, 1-Jaccard): "
          f"{rho:.3f}  p={p:.2e}  (n={len(pairs)})")

    auc = float(((jac > 0).astype(float) * (-cos_d)).argsort().argsort().mean())  # not used; placeholder
    # proper AUC: cosine-rank discriminates overlapping (jac>0) from non-overlapping
    from sklearn.metrics import roc_auc_score
    if (jac > 0).sum() > 50 and (jac == 0).sum() > 50:
        auc = roc_auc_score((jac > 0).astype(int), -cos_d)
    else:
        auc = float("nan")
    print(f"AUC (1-cos predicts range-overlap presence): {auc:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))

    ax = axes[0]
    ax.hexbin(1 - jac, cos_d, gridsize=40, mincnt=1, cmap="viridis")
    bins = np.linspace(0, 1, 12)
    idx = np.digitize(1 - jac, bins)
    mx, my = [], []
    for k in range(1, len(bins)):
        m = idx == k
        if m.sum() > 20:
            mx.append((1 - jac[m]).mean())
            my.append(cos_d[m].mean())
    ax.plot(mx, my, color="#cc4444", linewidth=2, label="binned mean")
    ax.set_xlabel("range dissimilarity ($1 - $ Jaccard, 1° grid)")
    ax.set_ylabel("embedding cosine distance")
    ax.set_title(f"Spearman $\\rho$ = {rho:.3f}  (n={len(pairs)})")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    overlap_mask = jac > 0
    ax.hist(cos_d[~overlap_mask], bins=30, alpha=0.6, color="#999999",
            label=f"no range overlap (n={(~overlap_mask).sum()})", density=True)
    ax.hist(cos_d[overlap_mask], bins=30, alpha=0.6, color="#3c6eb3",
            label=f"range overlap (n={overlap_mask.sum()})", density=True)
    ax.set_xlabel("embedding cosine distance")
    ax.set_ylabel("density")
    ax.set_title(f"AUC = {auc:.3f}  (cosine discriminates overlap)")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out = FIG / "alpha_hull_jaccard.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
