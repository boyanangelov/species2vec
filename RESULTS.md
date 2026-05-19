# Results

## TL;DR

1. The shipped `mammalia_6M.vec` and `reptilia_3M.vec` embeddings carry a
   strong **species-name leak**: congeners sit at cosine ≈ 0.80 with a
   0.37–0.49 gap above non-congeners. That gap is roughly 0.12 above what
   a leak-free model finds on the same kind of data, and comes from fastText
   training with character n-grams on (the default), which makes the model
   learn taxonomy from the species labels.
2. Two other methodological problems compound the leak: writing the entire
   corpus as one space-separated string (so fastText's context window
   slides across unrelated regions), and using `Geohash.encode(lat, lon)`
   at default precision 12 (≈ millimeters, so no two records share a
   cell).
3. A re-run of the pipeline with character n-grams off, geohash precision
   set explicitly, and one sentence per geohash bin matches **and slightly
   exceeds** the broken pipeline on actual held-out sympatry prediction
   (cross-genus AUC 0.943 vs 0.926), while reducing the congener gap from
   0.41 to 0.29. The residual 0.29 gap is the real ecological signal —
   congeners genuinely share ranges — not name leakage.

The original species2vec idea is sound and, properly trained, the
embeddings do encode spatial co-occurrence. But the artifacts shipped in
this repo conflate that signal with name structure that the model can read
off the labels for free.

## Baseline on the shipped `.vec` files

(Held-out occurrence data is not shipped, so only the congener check is
possible without re-downloading GBIF.)

```
reptilia_3M.vec
  species in vocab:              7397
  congener mean cosine:          0.808
    gap vs non-congener:         0.489

mammalia_6M.vec
  species in vocab:              1987
  congener mean cosine:          0.789
    gap vs non-congener:         0.369
```

## Controlled comparison on Squamata, ~147 k GBIF records

Same train / 10 % holdout split, same fastText dim / epochs / window.
"Broken" replicates the original notebook's recipe (one space-separated
string for the whole corpus, default char n-grams on). "Fixed" uses
geohash-precision-4 binned sentences and `minn = maxn = 0`.

```
                              broken           fixed
species in vocab              2523             2592
congener mean cosine          0.827            0.830
  gap vs non-congener         0.408            0.291
sympatry pair AUC             0.923            0.942
sympatry AUC (cross-genus)    0.926            0.943
held-out median self-rank     36 (1.4 %ile)    52 (2.0 %ile)
```

Reading the table:

- The 0.40 → 0.29 drop in congener gap is the name-leak coming out. With
  char n-grams on, two congeners share substrings of their Latin names,
  so the model places them next to each other for free. With n-grams off,
  the only way two congeners can end up close is if they actually co-occur
  in GBIF cells. The residual 0.29 is that real co-occurrence: congeners
  genuinely tend to share ranges, but the magnitude is much smaller than
  the original artifacts suggest.
- The fixed pipeline edges out the broken one on held-out sympatry-pair
  AUC (0.942 vs 0.923) and the gain is the same after excluding pairs
  that share a genus (0.943 vs 0.926). The original pipeline was not
  benefiting from name structure on this task — it was being mildly hurt
  by the cross-bin window bleed.
- Median self-rank is comparable (36 vs 52 out of ~2 500), so the
  embeddings are neither broken nor merely random in either condition.

## On the 10 k-record subsample

A smaller demo (the default `--max 20000` download, which throttles to
~10 k unique records) is **not** enough to show this. With 10 k records
over 1 474 species (≈ 7 records per species), the fixed pipeline's
spatial signal is too weak to learn and its sympatry AUC sits at ≈ 0.39,
below random. The broken pipeline still scores 0.88 in that regime,
purely from name structure. This is consistent with the rest of the
analysis — the name-substring signal is data-free, the spatial signal is
data-hungry — but it means anyone evaluating species2vec on a small
GBIF slice will incorrectly conclude that the broken pipeline is better.

## What this means for the bioRxiv pre-print

The pre-print describes species2vec embeddings whose neighborhoods
reflect range overlap. The neighborhoods do — but on the shipped
artifacts, much of that neighborhood structure is recoverable directly
from the species names, with no GBIF data at all. To support the
biogeographic claim, the embeddings should be regenerated with
`minn = maxn = 0` and an explicit geohash precision, and the same
evaluation (cross-genus sympatry AUC, held-out self-rank) should be
reported on the held-out fraction.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install gensim fasttext pandas numpy pygeohash scikit-learn tqdm requests matplotlib

# Country-sliced parallel download (~10 min, ~150k unique rows of Squamata).
python -m species2vec.gbif_download_parallel \
    --order Squamata --out data/squamata.csv \
    --per-country 8000 --workers 6

# Train both pipelines on the same split and print the eval table.
python -m species2vec.compare --csv data/squamata.csv --precision 4

# Or use the corrected pipeline on your own DataFrame.
python - <<'PY'
import pandas as pd
from species2vec.pipeline import run
df = pd.read_csv('data/squamata.csv')
stats = run(df, workdir='runs/squamata', geohash_precision=4)
print(stats)
PY
```
