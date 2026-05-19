# species2vec

[![DOI](https://zenodo.org/badge/141325266.svg)](https://zenodo.org/badge/latestdoi/141325266)

bioRxiv pre-print: https://www.biorxiv.org/content/early/2018/11/05/461996

Species embeddings learned from spatial co-occurrence of GBIF records, by
analogy with word2vec on text.

## Status

The original release shipped `mammalia_6M.vec` and `reptilia_3M.vec` trained
by the recipe in `notebooks/species2vec_tutorial.ipynb`. A re-check of that
recipe (see [RESULTS.md](RESULTS.md) for the full evaluation) found three
methodological problems that conflate name-substring similarity and
z-curve ordering artifacts with the intended spatial-co-occurrence signal:

1. fastText was trained with character n-grams enabled by default, so the
   model learns taxonomy from the species labels (congener gap drops from
   0.41 to 0.29 when n-grams are turned off on a controlled re-run).
2. The whole corpus was written as one space-separated string; fastText's
   context window slides across unrelated regions.
3. `Geohash.encode(lat, lon)` was called at default precision 12 (≈ mm),
   so no two records actually shared a cell.

A corrected pipeline lives under `species2vec/`. On a ~147 k-record
Squamata sub-sample, the corrected pipeline reproduces the spatial
signal (cross-genus sympatry AUC 0.943) without the name leak. The
original `.vec` files are kept for reference but should be regenerated
before being cited as biogeographic embeddings.

## Quick start (corrected pipeline)

```bash
python -m venv .venv && source .venv/bin/activate
pip install gensim fasttext pandas numpy pygeohash scikit-learn tqdm requests matplotlib

# 1. Country-sliced parallel download (~10 min, ~150k unique Squamata records)
python -m species2vec.gbif_download_parallel \
    --order Squamata --out data/squamata.csv \
    --per-country 8000 --workers 6

# 2. Train both pipelines (broken vs fixed) and print the eval side by side
python -m species2vec.compare --csv data/squamata.csv --precision 4

# 3. Use the embeddings
python - <<'PY'
from gensim.models import KeyedVectors
m = KeyedVectors.load_word2vec_format('runs/compare/fixed.vec')
print(m.most_similar('Anolis_carolinensis', topn=10))
PY
```

For a full-scale corpus, register a GBIF download for the class of interest,
load it as a `pandas.DataFrame` with columns `species`, `decimalLatitude`,
`decimalLongitude`, and call:

```python
from species2vec.pipeline import run
stats = run(df, workdir='runs/mammalia', geohash_precision=5)
```

## What changed vs. the original notebook

| Issue                                                                                            | Original                                       | Corrected                                                       |
| ------------------------------------------------------------------------------------------------ | ---------------------------------------------- | --------------------------------------------------------------- |
| Geohash precision                                                                                | default (12 chars ≈ mm) → ~unique per record   | explicit (default 5 chars ≈ 5 km) → real co-occurrence bins     |
| Corpus shape                                                                                     | one space-separated string for the whole world | one line per geohash bin → window cannot bleed across bins      |
| Char n-grams in fastText                                                                         | on (default) → embeddings encode taxonomy from species names | off (`minn=0, maxn=0`) → embeddings encode spatial co-occurrence |
| Dedup                                                                                            | none → abundant taxa over-represented          | dedup by (geohash, species), keep frequency only as bin counts  |
| Evaluation                                                                                       | t-SNE plot                                     | congener gap, held-out sympatry-pair AUC, held-out self-rank    |
| Seed / reproducibility                                                                           | none                                           | seeded                                                          |

## Evaluation harness

`species2vec/eval.py` reports three orthogonal metrics:

- **congener gap** — mean cosine between same-genus species minus mean cosine
  to a random sample. A large gap is *bad* if char n-grams are on, because the
  model can read the genus off the species name. With n-grams off, the residual
  gap reflects real range-overlap between congeners.
- **sympatry AUC** — for every species pair seen in the same held-out geohash
  bin (label 1) vs. matched pairs that never co-occur (label 0), is cosine a
  good discriminator? 0.5 = no signal, 1.0 = perfect.
- **neighborhood self-rank** — for each held-out (species, bin), rank all
  vocab species by similarity to the bin centroid; median rank of the
  held-out species. Lower is better; |V|/2 = random.

## Repository layout

```
species2vec/
  pipeline.py        corpus building + fastText training
  eval.py            held-out evaluation metrics
  gbif_download.py   small-taxon downloader via the GBIF search API
  compare.py         side-by-side broken vs fixed
notebooks/
  species2vec_tutorial.ipynb     original (kept for reference)
  species2vec_corrected.ipynb    corrected pipeline + discussion
mammalia_6M.vec      original embeddings (do not cite as-is)
reptilia_3M.vec      original embeddings (do not cite as-is)
```
