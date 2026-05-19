"""Generate manuscript figures."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from gensim.models import KeyedVectors
import pygeohash

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RUNS = ROOT / "runs" / "compare"
DATA = ROOT / "data" / "squamata.csv"
UMAP_PARQUET = ROOT / "runs" / "umap_cache" / "fixed_seed0.parquet"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
})

BROKEN_COLOR = "#b3543c"
FIXED_COLOR = "#3c6eb3"
RAND_COLOR = "#999999"


def _genus(name: str) -> str:
    return name.split("_", 1)[0]


def cos_samples(kv: KeyedVectors, n: int = 4000, seed: int = 0):
    rng = random.Random(seed)
    vocab = [w for w in kv.index_to_key if "_" in w]
    by_genus: dict[str, list[str]] = {}
    for w in vocab:
        by_genus.setdefault(_genus(w), []).append(w)
    congener_groups = [v for v in by_genus.values() if len(v) >= 2]
    cong, rand = [], []
    for _ in range(n):
        g = rng.choice(congener_groups)
        a, b = rng.sample(g, 2)
        cong.append(float(kv.similarity(a, b)))
        x, y = rng.sample(vocab, 2)
        while _genus(x) == _genus(y):
            x, y = rng.sample(vocab, 2)
        rand.append(float(kv.similarity(x, y)))
    return np.array(cong), np.array(rand)


def fig_cosine_dists():
    broken = KeyedVectors.load_word2vec_format(str(RUNS / "broken.vec"))
    fixed = KeyedVectors.load_word2vec_format(str(RUNS / "fixed.vec"))

    bc, br = cos_samples(broken, seed=0)
    fc, fr = cos_samples(fixed, seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    bins = np.linspace(-0.4, 1.0, 50)

    panels = [
        (bc, br, "Character n-grams enabled", BROKEN_COLOR),
        (fc, fr, "Atomic tokens", FIXED_COLOR),
    ]

    for ax, (cong, rand, title, color) in zip(axes, panels):
        gap = cong.mean() - rand.mean()
        ax.hist(rand, bins=bins, alpha=0.6, color=RAND_COLOR, label="random pairs", density=True)
        ax.hist(cong, bins=bins, alpha=0.6, color=color, label="congener pairs", density=True)
        ax.axvline(rand.mean(), color=RAND_COLOR, linestyle="--", linewidth=1)
        ax.axvline(cong.mean(), color=color, linestyle="--", linewidth=1)
        ax.set_title(f"{title}\n$\\Delta$ = {gap:.3f}", fontsize=9)
        ax.set_xlabel("cosine similarity")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.set_xlim(-0.4, 1.0)

    axes[0].set_ylabel("density")
    fig.tight_layout()
    out = FIG / "cosine_dists.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_metrics_bars():
    metrics = ["congener gap", "sympatry AUC\n(all)", "sympatry AUC\n(cross-genus)"]
    broken = [0.41, 0.91, 0.926]
    fixed = [0.29, 0.94, 0.943]

    x = np.arange(len(metrics))
    w = 0.36

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    b1 = ax.bar(x - w / 2, broken, w, label="with n-grams", color=BROKEN_COLOR)
    b2 = ax.bar(x + w / 2, fixed, w, label="atomic tokens", color=FIXED_COLOR)
    ax.axhline(0.5, color=RAND_COLOR, linestyle=":", linewidth=0.8)
    ax.text(2.45, 0.51, "AUC=0.5", color=RAND_COLOR, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.legend(frameon=False, loc="upper left")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}",
                    ha="center", fontsize=7.5)

    fig.tight_layout()
    out = FIG / "metrics_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_smalldata():
    sizes = ["10 k records\n(~7 rec/sp)", "147 k records\n(~58 rec/sp)"]
    broken = [0.88, 0.926]
    fixed = [0.39, 0.943]

    x = np.arange(len(sizes))
    w = 0.36
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.bar(x - w / 2, broken, w, label="with n-grams", color=BROKEN_COLOR)
    ax.bar(x + w / 2, fixed, w, label="atomic tokens", color=FIXED_COLOR)
    ax.axhline(0.5, color=RAND_COLOR, linestyle=":", linewidth=0.8)
    ax.text(1.25, 0.51, "random", color=RAND_COLOR, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("cross-genus sympatry AUC")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = FIG / "smalldata.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _load_occurrences():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude", "species"])
    return df


def _draw_coast(ax):
    """Faint global outline using just the bounding box of land — no cartopy.

    We approximate continents by overlaying a stippled grid of dots over
    every populated 1-degree cell in the dataset itself. Combined with the
    record scatter this is enough to make continents legible.
    """
    ax.set_facecolor("#f7f7f5")
    for parallel in (-66.5, -23.4, 0, 23.4, 66.5):
        ax.axhline(parallel, color="#cccccc", linewidth=0.4, zorder=0)


def fig_world_map():
    df = _load_occurrences()
    rng = np.random.default_rng(0)
    n = min(len(df), 30000)
    idx = rng.choice(len(df), n, replace=False)
    sample = df.iloc[idx]

    bin_species: dict[str, set[str]] = {}
    for sp, lat, lon in zip(sample["species"], sample["decimalLatitude"], sample["decimalLongitude"]):
        gh = pygeohash.encode(float(lat), float(lon), precision=3)
        bin_species.setdefault(gh, set()).add(sp)

    bin_centers, bin_richness = [], []
    for gh, sps in bin_species.items():
        lat, lon = pygeohash.decode(gh)
        bin_centers.append((float(lon), float(lat)))
        bin_richness.append(len(sps))
    lon = np.array([c[0] for c in bin_centers])
    lat = np.array([c[1] for c in bin_centers])
    rich = np.array(bin_richness)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    _draw_coast(ax)
    sc = ax.scatter(
        lon, lat,
        c=rich,
        s=np.clip(rich * 1.2, 2, 30),
        cmap="viridis",
        alpha=0.75,
        linewidths=0,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="box")
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("species per ~ 156 km cell", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    out = FIG / "world_map.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _genus_of(name: str) -> str:
    return name.split("_", 1)[0] if "_" in name else name.split(" ", 1)[0]


def fig_umap():
    if not UMAP_PARQUET.exists():
        print(f"skip umap: {UMAP_PARQUET} not found")
        return
    df = pd.read_parquet(UMAP_PARQUET)
    df = df[df["species"] != "</s>"].copy()
    df["genus"] = df["species"].map(_genus_of)
    top_genera = df["genus"].value_counts().head(8).index.tolist()
    palette = matplotlib.colormaps["tab10"].colors

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    others = df[~df["genus"].isin(top_genera)]
    ax.scatter(others["u1"], others["u2"], s=4, c="#dddddd", alpha=0.6, linewidths=0)
    handles = []
    for i, g in enumerate(top_genera):
        sub = df[df["genus"] == g]
        ax.scatter(sub["u1"], sub["u2"], s=8, c=[palette[i]], alpha=0.85, linewidths=0)
        handles.append(mpatches.Patch(color=palette[i], label=f"{g} (n={len(sub)})"))
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False, ncol=2)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out = FIG / "umap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_neighbors_map():
    kv = KeyedVectors.load_word2vec_format(str(RUNS / "fixed.vec"))
    df = _load_occurrences()

    df["sp_underscore"] = df["species"].str.replace(" ", "_")
    candidates = [
        "Anolis_carolinensis",
        "Sceloporus_occidentalis",
        "Lacerta_agilis",
        "Hemidactylus_frenatus",
        "Naja_naja",
    ]
    target = next((c for c in candidates if c in kv.key_to_index), None)
    if target is None:
        for w in kv.index_to_key:
            if "_" in w and (df["sp_underscore"] == w).sum() > 50:
                target = w
                break
    if target is None:
        print("skip neighbors_map: no suitable target")
        return

    neighbors = [w for w, _ in kv.most_similar(target, topn=6) if w != target][:5]
    species_for_map = [target] + neighbors

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    _draw_coast(ax)
    palette = matplotlib.colormaps["tab10"].colors
    for i, sp in enumerate(species_for_map):
        sub = df[df["sp_underscore"] == sp]
        if len(sub) == 0:
            continue
        color = "#222222" if i == 0 else palette[i]
        size = 14 if i == 0 else 8
        ax.scatter(
            sub["decimalLongitude"], sub["decimalLatitude"],
            s=size, c=[color], alpha=0.7 if i == 0 else 0.55,
            linewidths=0,
            label=f"{sp.replace('_', ' ')}" + (" (target)" if i == 0 else ""),
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left", fontsize=7, frameon=False, markerscale=1.4)
    ax.set_title(
        f"Top-5 embedding neighbours of {target.replace('_', ' ')}",
        fontsize=10,
    )
    fig.tight_layout()
    out = FIG / "neighbors_map.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({target} -> {neighbors})")


if __name__ == "__main__":
    fig_cosine_dists()
    fig_metrics_bars()
    fig_smalldata()
    fig_world_map()
    fig_umap()
    fig_neighbors_map()
