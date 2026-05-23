# species2vec

[![DOI](https://zenodo.org/badge/141325266.svg)](https://zenodo.org/badge/latestdoi/141325266)

Distributed vector representations of species, learned from the spatial
co-occurrence of GBIF occurrence records by analogy with word2vec on text.
Geo-binned records become "sentences" and a fastText objective is trained on
atomic species tokens, so the embedding geometry is driven by where species
co-occur rather than by the morphology of their binomial names.

📄 **Preprint:** [`manuscript/preprint.pdf`](manuscript/preprint.pdf)

## Method

Two design choices make the embeddings reflect ecology rather than labels:

1. **Geohash binning at an ecologically meaningful scale.** Records are binned
   by geohash at ~5 km precision; each bin becomes one sentence, so the
   context window cannot bleed across unrelated regions.
2. **Atomic tokens (`minn = maxn = 0`).** Character n-grams are disabled, so
   the model cannot read taxonomy off shared genus prefixes in species names.

Records are deduplicated by `(geohash, species)` so abundant taxa do not
dominate a bin by sheer record count, and the whole pipeline is seeded.

On a ~147 k-record _Squamata_ sample the pipeline reaches a cross-genus
held-out sympatry AUC of **0.943** with a congener cosine gap of 0.29.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install gensim fasttext pandas numpy pygeohash scikit-learn tqdm requests matplotlib

# 1. Country-sliced parallel download (~10 min, ~150k unique Squamata records)
python -m species2vec.gbif_download_parallel \
    --order Squamata --out data/squamata.csv \
    --per-country 8000 --workers 6

# 2. Train the pipeline
python - <<'PY'
import pandas as pd
from species2vec.pipeline import run
df = pd.read_csv('data/squamata.csv')
stats = run(df, workdir='runs/squamata', geohash_precision=5)
print(stats)
PY

# 3. Use the embeddings
python - <<'PY'
from gensim.models import KeyedVectors
m = KeyedVectors.load_word2vec_format('runs/squamata/embeddings.vec')
print(m.most_similar('Anolis_carolinensis', topn=10))
PY
```

For a full-scale corpus, register a GBIF download for the class of interest,
load it as a `pandas.DataFrame` with columns `species`, `decimalLatitude`,
`decimalLongitude`, and call `run(df, workdir=..., geohash_precision=5)`.

## Evaluation harness

`species2vec/eval.py` reports three orthogonal held-out metrics that
separately bound label-substring effects and measure spatial signal:

- **congener gap** — mean cosine between same-genus species minus mean cosine
  to a random sample. With n-grams off, the residual gap reflects real
  range-overlap between congeners rather than shared name substrings.
- **sympatry AUC** — for species pairs seen in the same held-out geohash bin
  (label 1) vs. matched pairs that never co-occur (label 0), how well does
  cosine discriminate? 0.5 = no signal, 1.0 = perfect.
- **neighborhood self-rank** — for each held-out (species, bin), rank all
  vocab species by similarity to the bin centroid; median rank of the
  held-out species. Lower is better; |V|/2 = random.

## Experiments

Scripts reproducing the analyses in the preprint live under `experiments/`:

- `temporal_shift.py` / `temporal_shift_null.py` — do embeddings shift with
  range shifts between epochs? (with a permutation null)
- `alpha_hull_jaccard.py` — embedding similarity vs. alpha-hull range overlap
- `aves_transfer.py` — transfer of the method to a bird (Aves) corpus

## Repository layout

```
species2vec/
  pipeline.py              corpus building + fastText training
  eval.py / eval_v2.py     held-out evaluation metrics
  gbif_download.py         small-taxon downloader via the GBIF search API
  gbif_download_parallel.py  country-sliced parallel downloader
manuscript/
  preprint.typ             Typst source for the preprint
  preprint.pdf             compiled preprint
  make_figures.py          figure generation
experiments/               analyses reproduced in the preprint
notebooks/
  species2vec_corrected.ipynb   walkthrough of the pipeline + discussion
app.py                     interactive Streamlit explorer for trained embeddings
```

## Citing

See [`CITATION.cff`](CITATION.cff), or use the GitHub "Cite this repository"
button. Released under the [MIT License](LICENSE).
