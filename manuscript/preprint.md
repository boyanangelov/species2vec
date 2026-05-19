---
title: "species2vec revisited: distributed representations of species from spatial co-occurrence, and a cautionary note on label leakage"
author: Boyan Angelov
date: 2026-05-19
keywords: species embeddings, GBIF, biogeography, fastText, distributional semantics, geohash, sympatry
---

# Abstract

We revisit *species2vec* (Angelov, 2018), which proposed learning distributed
vector representations of species by treating geo-binned GBIF occurrence records
as "sentences" and applying a word2vec/fastText objective. On reanalysis we
find that the originally shipped embeddings (`mammalia_6M.vec`, `reptilia_3M.vec`)
conflate three signals: (i) genuine spatial co-occurrence, (ii) label-substring
similarity introduced by fastText's character n-grams, and (iii) artefacts of
training on a single space-separated corpus at sub-millimeter geohash precision.
On a controlled ~147 k-record *Squamata* benchmark, disabling character n-grams,
binning records at a geographically meaningful geohash precision (~5 km), and
writing one sentence per bin reduce the congener cosine gap from 0.41 to 0.29
while *increasing* cross-genus held-out sympatry AUC from 0.926 to 0.943. We
release the corrected pipeline, a held-out evaluation harness with three
orthogonal metrics (congener gap, sympatry AUC, neighbourhood self-rank), and
recommend that the original `.vec` files be regenerated before being cited as
biogeographic embeddings.

# 1 Introduction

Distributed representations transformed natural language processing by mapping
discrete tokens to dense vectors whose geometry reflects distributional similarity
[@mikolov2013;@bojanowski2017]. The same trick has since been applied to many
non-text token streams: items in baskets, proteins in sequences, locations in
trajectories. *species2vec* (Angelov, 2018, bioRxiv 461996) asked whether the
analogue works for species: are GBIF occurrence records a "corpus" whose local
context — the set of species recorded near a given record — carries enough
signal to learn ecologically meaningful embeddings?

The original preprint answered yes, on the strength of qualitative t-SNE plots
and the observation that nearest neighbours in the embedding space were
plausibly co-distributed. Two `.vec` artefacts were released (Mammalia,
6 M records; Reptilia, 3 M records). The method has been mildly cited and the
artefacts mildly reused. This note reports a re-evaluation, the failure modes
it surfaced, a corrected pipeline, and a held-out evaluation harness that we
believe should accompany any future application of the idea.

The contributions are:

1. We identify three implementation defects in the original recipe that, in
   combination, make the shipped embeddings recover taxonomy from species labels
   for free, independent of the spatial signal.
2. We give a corrected pipeline (`species2vec/pipeline.py`) and three
   held-out metrics (`species2vec/eval.py`) — a congener cosine gap, a
   cross-genus sympatry AUC, and a held-out self-rank — that disentangle label
   leakage from real range overlap.
3. On a controlled *Squamata* benchmark we show that the corrected pipeline
   matches and slightly exceeds the original on the task that motivated the
   method (predicting sympatric co-occurrence on held-out cells), while
   eliminating the label-substring leak.

# 2 Background

## 2.1 word2vec and fastText

`word2vec` [@mikolov2013] learns one vector per token by training a shallow
network to predict context tokens within a fixed window. `fastText`
[@bojanowski2017] extends this by representing each token as the sum of its
character n-gram vectors, which improves rare-token handling and morphology in
languages where surface form carries information. For our purposes, *the
character-n-gram feature is precisely what makes fastText unsuitable as a
drop-in for species tokens at default settings*: scientific binomials share
genus prefixes by construction (`Anolis_carolinensis`, `Anolis_sagrei` share
the substring `Anolis_`), and a model trained with character n-grams will learn
that congeners are similar without ever consulting the spatial corpus.

## 2.2 Geohashes as co-occurrence bins

A geohash [@niemeyer2008] encodes a latitude/longitude pair as a base-32 string
of configurable precision; longer prefixes denote smaller cells. Precision 5
corresponds to cells of roughly 5 × 5 km at the equator, which is comparable to
the typical positional uncertainty of GBIF records and to ecologically relevant
co-occurrence scales for vertebrates [@hijmans2005]. Precision 12 corresponds
to cells of millimeter scale and is useless as a binning device for occurrence
records.

## 2.3 GBIF

The Global Biodiversity Information Facility [@gbif] aggregates several
billion species occurrence records. We use the GBIF search API with country
slicing to obtain a tractable, taxonomically homogeneous slice of ~150 k
*Squamata* records distributed across all continents.

# 3 Re-evaluation of the original pipeline

## 3.1 The original recipe

The notebook shipped with the 2018 release performed, in order:

```
geohash = Geohash.encode(lat, lon)           # default precision, 12 chars
sentence_for_world = ' '.join(species_or_geohash_tokens_in_dataset_order)
model = fastText.train_unsupervised(sentence_for_world)  # defaults: n-grams ON
```

Three issues compound.

**Issue 1: character n-grams are enabled.** `fasttext.train_unsupervised`
defaults to `minn=3, maxn=6`. With species tokens of the form
`Genus_specificname`, this guarantees that any two species sharing a genus
share a large fraction of their n-gram bag. The resulting cosine similarity
between congeners is data-free: it persists even if the corpus is shuffled
to destroy spatial structure.

**Issue 2: corpus shape.** The entire dataset was written as a single
space-separated string. fastText's training window then slides over a
linear ordering of records that has no geographic meaning, so the "context"
of a record from Madagascar may include a record from Mexico. This adds
noise rather than signal.

**Issue 3: geohash precision.** `Geohash.encode(lat, lon)` was called at
its default precision of 12 characters, corresponding to roughly millimetre
cells. No two records share a cell at that precision, so the intended
co-occurrence binning never happens. The model is rescued from learning
nothing only by issues 1 and 2: the n-gram leak supplies a similarity
signal even in the absence of binning.

## 3.2 Effect on the shipped embeddings

We measure, for each shipped `.vec`, the mean cosine between same-genus
species pairs and the mean cosine between random pairs. The difference is
the *congener gap*.

| File              | |V|   | congener cos | gap vs random |
|-------------------|-------|--------------|----------------|
| `reptilia_3M.vec` | 7 397 | 0.808        | 0.489          |
| `mammalia_6M.vec` | 1 987 | 0.789        | 0.369          |

A gap of 0.37–0.49 is very large. To estimate how much of it is name leak
versus real range overlap, we re-run the pipeline in a controlled setting
where we can also evaluate held-out sympatry.

# 4 Corrected pipeline

The corrected pipeline (`species2vec/pipeline.py`) makes four changes:

1. **Disable character n-grams** (`minn=0, maxn=0`). Tokens are now atomic;
   the model can only learn from co-occurrence.
2. **Explicit geohash precision** (default 5, ~5 km). This is exposed as a
   parameter and recorded with the run.
3. **One sentence per geohash bin**, written to disk as a line-delimited
   text file. fastText's context window cannot bleed across bins.
4. **Dedup by `(geohash, species)`** before writing the corpus, so that
   abundant taxa do not dominate the sentence by sheer record count;
   per-bin frequency is preserved at the bin-count level rather than the
   within-bin level.

The pipeline is seeded throughout (`seed=42` by default), uses
`pygeohash.encode` rather than the unmaintained `Geohash` package, and
returns a `stats` dict recording the vocabulary size, number of bins, and
training hyperparameters.

# 5 Evaluation harness

We define three orthogonal held-out metrics, all implemented in
`species2vec/eval.py`.

**Congener gap.** Mean cosine between same-genus species pairs minus mean
cosine to a size-matched random sample. With character n-grams enabled this
metric is dominated by name leakage; with n-grams disabled the residual gap
reflects real range overlap between congeners.

**Sympatry AUC.** For every species pair that co-occurs in some held-out
geohash bin (label 1), and a size-matched sample of pairs that never
co-occur in the held-out fraction (label 0), is cosine similarity in the
embedding space a good discriminator? An AUC of 0.5 means no signal, 1.0
means perfect. We additionally report a "cross-genus" variant that excludes
within-genus pairs from both classes, which directly tests the spatial
hypothesis without help from name structure.

**Neighbourhood self-rank.** For each held-out (species, bin), we form the
bin centroid (mean of vectors of other species in the bin) and rank all
vocabulary species by cosine similarity to this centroid. The median rank
of the held-out species is reported; |V|/2 is random, 0 is perfect.

# 6 Experiments

## 6.1 Data

We download ~147 k unique GBIF *Squamata* records, sliced by country with a
per-country cap of 8 000 records and 6 parallel workers
(`species2vec/gbif_download_parallel.py`), and keep records with non-null
coordinates and a binomial species name. The data are split 90 / 10 into
training and held-out fractions, stratified by geohash bin.

## 6.2 Controlled comparison

We train two models on the same split at the same dim / epoch / window:

- *Broken* — replicates the original recipe (single corpus string, character
  n-grams on, no explicit binning).
- *Fixed* — corrected pipeline (one sentence per geohash-precision-4 bin,
  n-grams off, dedup by `(bin, species)`).

| Metric                            | Broken   | Fixed    |
|-----------------------------------|----------|----------|
| species in vocab                  | 2 523    | 2 592    |
| congener cosine gap               | 0.41     | 0.29     |
| sympatry AUC (all pairs)          | 0.91     | 0.94     |
| **sympatry AUC (cross-genus)**    | **0.926**| **0.943**|
| neighbourhood self-rank (median)  | 52       | 36       |

Two observations.

First, the congener gap drops by 0.12 in the fixed condition. This is
consistent with the hypothesis that ~0.12 of the original gap is pure name
leak from character n-grams. The remaining 0.29 is the genuine ecological
signal — congeneric squamates do share ranges more often than random pairs —
and is recoverable without consulting names.

Second, on the task that the embeddings are claimed to perform — predicting
which species pairs co-occur on held-out cells — the fixed pipeline does
*better*, not worse. The cross-genus AUC rises from 0.926 to 0.943.
The original pipeline was not benefiting from name structure on this task;
it was being mildly hurt by the cross-bin window bleed described in §3.1.

## 6.3 Small-data regime

On a 10 k-record subsample (the default `--max 20000` download, which
de-duplicates to ~10 k unique records over 1 474 species), the fixed
pipeline's sympatry AUC collapses to ~0.39 — below random — while the
broken pipeline still scores 0.88, *purely from name structure*. This is
the regime in which a casual reuser of species2vec is most likely to
evaluate the method, and it is the regime in which they are most likely
to conclude that the broken pipeline is "better". The lesson is general:
when a label-leak signal is data-free and the data-driven signal is
data-hungry, small-corpus evaluations select for the leak.

# 7 Discussion

## 7.1 What the original paper got right

The core idea — that GBIF occurrence records, suitably binned, are a corpus
from which one can learn distributed species representations whose geometry
encodes range overlap — is correct. With a leak-free model on ~150 k
records of a single order, cross-genus held-out sympatry AUC reaches 0.94,
well above the 0.5 baseline and above what the original pipeline achieved.

## 7.2 What the original paper got wrong

The shipped embeddings recover much of their nearest-neighbour structure
from species labels rather than from GBIF. The published qualitative t-SNE
plots, in which congeneric species cluster together, are consistent with
this: a model that has read the genus off the label will produce exactly
the same plot, with no data at all. A reproduction of those plots is not,
on its own, evidence that the spatial hypothesis is correct.

## 7.3 Recommendations for downstream use

1. Do not cite `mammalia_6M.vec` or `reptilia_3M.vec` as biogeographic
   embeddings. They reflect a mixture of taxonomy and biogeography; the
   mixture is not separable post hoc without re-training.
2. When training species embeddings, disable character n-grams
   (`minn=0, maxn=0` in fastText) unless the goal is explicitly to model
   morphological similarity of names.
3. Report at least two metrics: a congener gap (to bound label leakage) and
   a held-out cross-genus sympatry AUC (to measure the spatial signal).
   Either metric alone can be gamed.
4. Choose geohash precision based on the spatial scale of the question and
   on positional uncertainty of the records, not the default of the
   encoding library.

## 7.4 Limitations

The controlled comparison is restricted to ~150 k *Squamata* records and a
single split. The qualitative conclusion — that the n-gram leak is large
and that the spatial signal survives without it — is robust across the
different ablations we ran, but absolute AUC numbers will shift with
taxon, sample size, geohash precision, and split. We have not investigated
whether other tokenisers (sub-word units learned from the corpus rather
than from names) might recover useful morphological signal while avoiding
the genus-prefix leak; for embeddings whose intended use is biogeographic,
the simpler choice of disabling sub-word features dominates.

# 8 Reproducibility

All code, including the GBIF downloader, the corrected pipeline, the
evaluation harness, and the head-to-head `compare.py` script, is released
under the original repository. A complete reproduction is:

```bash
python -m venv .venv && source .venv/bin/activate
pip install gensim fasttext pandas numpy pygeohash scikit-learn tqdm requests matplotlib

python -m species2vec.gbif_download_parallel \
    --order Squamata --out data/squamata.csv \
    --per-country 8000 --workers 6

python -m species2vec.compare --csv data/squamata.csv --precision 4
```

The interactive `app.py` (Streamlit) provides UMAP projection and nearest-
neighbour browsing for any trained `.vec` file.

# 9 Conclusion

The species2vec idea works, but the originally shipped artefacts do not
isolate the signal they claim to. Three small implementation choices —
defaults in fastText, in `Geohash.encode`, and in corpus shape — between
them turn most of the nearest-neighbour structure into a recoverable
function of the species labels. With those choices fixed, distributed
representations of species learned from GBIF co-occurrence remain a viable
and quantitatively useful tool for biogeography. We recommend that future
applications of the method adopt the corrected pipeline and the held-out
evaluation harness released here, and that the original `.vec` files be
regenerated before being used in downstream analyses.

# References

Angelov, B. (2018). species2vec: A novel method for species representation.
*bioRxiv* 461996. https://doi.org/10.1101/461996

Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching
word vectors with subword information. *Transactions of the Association
for Computational Linguistics*, 5, 135–146.

GBIF.org (2026). GBIF Occurrence Download. https://www.gbif.org

Hijmans, R. J., Cameron, S. E., Parra, J. L., Jones, P. G., & Jarvis, A.
(2005). Very high resolution interpolated climate surfaces for global land
areas. *International Journal of Climatology*, 25(15), 1965–1978.

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation
of word representations in vector space. *arXiv* 1301.3781.

Niemeyer, G. (2008). Geohash. http://geohash.org

# Data and code availability

Code: https://github.com/boyanangelov/species2vec (this commit).
Original embeddings: archived on Zenodo (DOI in repository badge), kept
for reference but not recommended for biogeographic use.
