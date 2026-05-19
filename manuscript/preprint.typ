#set document(
  title: "species2vec: distributed representations of species from spatial co-occurrence",
  author: "Boyan Angelov",
)
#set page(paper: "a4", margin: 2.2cm, numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 11.5pt)
#show link: set text(fill: blue.darken(20%))
#show raw.where(block: true): set block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%)
#show raw.where(block: false): set text(font: "DejaVu Sans Mono", size: 9pt)

#align(center)[
  #text(size: 16pt, weight: "bold")[
    species2vec: distributed representations of species from
    spatial co-occurrence on GBIF
  ]

  #v(0.6em)
  Boyan Angelov \
  #text(size: 9pt)[boyan.angelov\@gmail.com]

  #v(0.4em)
  #text(size: 9pt)[2026-05-19]
]

#v(0.8em)

#align(center)[
  #box(width: 85%)[
    #align(left)[
      *Abstract.* We learn distributed vector representations of
      species by treating geo-binned GBIF occurrence records as
      "sentences" and applying a fastText objective. The pipeline bins
      records by geohash at an ecologically meaningful precision
      (#sym.tilde 5 km), writes one sentence per bin, and trains atomic
      species tokens so that embedding geometry is driven by spatial
      co-occurrence rather than the morphology of binomial names. We
      propose three orthogonal held-out evaluation metrics — a congener
      cosine gap, a cross-genus sympatry AUC, and a neighbourhood
      self-rank — that separately bound label-substring effects and
      measure spatial signal. On #sym.tilde 147 k _Squamata_ records
      the method reaches a cross-genus held-out sympatry AUC of 0.943
      with a congener cosine gap of 0.29.
    ]
  ]
]

#v(0.8em)

= Introduction

Distributed representations map discrete tokens to dense vectors whose
geometry reflects distributional similarity @mikolov2013
@bojanowski2017, and have been applied to many non-text token streams:
items in baskets, proteins in sequences, locations in trajectories. We
ask whether the analogue works for species: are GBIF occurrence
records @gbif a "corpus" whose local context — the set of species
recorded near a given record — carries enough signal to learn
ecologically meaningful embeddings?

Two design choices turn out to be decisive. First, the unit of
co-occurrence: occurrence records are points in continuous space, but
distributional learning requires discrete contexts, and the choice of
binning interacts with the spatial scale of the ecological signal.
Second, the token representation: scientific binomials are not
arbitrary symbols — they share genus prefixes by construction — so any
sub-word feature in the language-model objective will let the model
recover taxonomy directly from labels.

The contributions are:

+ A reproducible pipeline (`species2vec/pipeline.py`) that bins GBIF
  records by geohash, writes one sentence per bin, trains fastText
  with atomic tokens, and produces seeded embeddings together with
  run metadata.
+ A held-out evaluation harness (`species2vec/eval.py`) with three
  orthogonal metrics — congener gap, sympatry AUC, neighbourhood
  self-rank — that separately bound label-substring effects and
  measure spatial signal.
+ Benchmark results on #sym.tilde 147 k _Squamata_ records and an
  ablation justifying both design choices.

= Background

== word2vec and fastText

`word2vec` @mikolov2013 learns one vector per token by training a
shallow network to predict context tokens within a fixed window.
`fastText` @bojanowski2017 extends this by representing each token as
the sum of its character n-gram vectors. The sub-word feature improves
rare-token handling and morphology in languages where surface form
carries information; for species tokens, where binomials of the form
`Genus_specificname` share genus prefixes by construction, the same
feature lets the model recover taxonomy directly from labels. We
therefore train with atomic tokens (`minn = maxn = 0`) and verify
empirically that the resulting embeddings are not dominated by name
form (§5.2).

== Geohashes as co-occurrence bins

A geohash @niemeyer2008 encodes a latitude/longitude pair as a
base-32 string of configurable precision; longer prefixes denote
smaller cells. Precision 5 corresponds to cells of roughly
5 #sym.times 5 km at the equator, comparable to the typical positional
uncertainty of GBIF records and to ecologically relevant co-occurrence
scales for vertebrates @hijmans2005. Higher precisions correspond to
cells small enough that no two records share a cell, and so are
useless as a binning device for occurrence data.

== GBIF

The Global Biodiversity Information Facility @gbif aggregates several
billion species occurrence records. We use the GBIF search API with
country slicing to obtain a tractable, taxonomically homogeneous
slice of #sym.tilde 150 k _Squamata_ records distributed across all
continents (@fig:world_map).

#figure(
  image("figures/world_map.pdf", width: 100%),
  caption: [Global distribution of the _Squamata_ training corpus
    after country-sliced GBIF download. Each dot is a
    geohash-precision-3 cell ($approx 156 times 156$ km at the
    equator); colour encodes the number of distinct species observed
    in the cell. The corpus covers every continent but exhibits the
    expected GBIF sampling bias toward North America, Europe, and
    coastal Australia.],
) <fig:world_map>

== Related work

The idea of treating non-text token streams as a corpus for
distributional learning is broadly established. `prod2vec`
@grbovic2015 trains item embeddings on purchase sequences;
`node2vec` @grover2016 trains node embeddings on random walks over
graphs; `seq2vec` and protein language models train embeddings on
biological sequences @alley2019 @rives2021. Closer to our setting,
`loc2vec` @kusner2017 and related work @yan2017 @yin2019 learn
embeddings of geographic locations from spatial trajectories. In
ecology, distributional learning has been applied to remote-sensing
patches @camacho2024, to species--trait matrices @bertelsmeier2022,
and to image-derived feature representations of organisms
@lasseck2018. To our knowledge, this is the first reproducible
end-to-end pipeline that learns embeddings of species _identifiers_
from spatial co-occurrence at GBIF scale, with a held-out evaluation
harness that controls for label-form artefacts.

The closest classical biogeographic baseline is co-occurrence-based
range similarity (Jaccard or Sørensen index over presence vectors
@dornelas2014), which produces sparse, high-dimensional
representations of species ranges. Dense embeddings learned with a
language-model objective compress the same signal into a small
number of dimensions, generalise to held-out cells, and admit
arithmetic operations on the resulting vectors.

= Method

The pipeline (`species2vec/pipeline.py`) consists of four steps.

+ *Bin records by geohash.* Each record is assigned a geohash key at
  a configurable precision (default 5, #sym.tilde 5 km).
+ *Deduplicate by `(geohash, species)`.* Each species appears at
  most once per bin, so abundant taxa do not dominate within-bin
  context by sheer record count.
+ *Write one sentence per bin.* Bins with $gt.eq 2$ species become
  a single space-separated line; fastText's context window slides
  within bins but cannot bleed across them.
+ *Train fastText with atomic tokens.* `minn = maxn = 0` disables
  character n-grams, so embedding geometry is driven by spatial
  co-occurrence rather than the morphology of species names.

The pipeline is seeded throughout (`seed = 42` by default), uses
`pygeohash.encode`, and returns a `stats` dict recording the
vocabulary size, number of bins, and training hyperparameters
(`dim = 100`, `epoch = 25`, `window = 8`, `min_count = 3`,
`lr = 0.025`).

= Evaluation harness

Let $V$ be the vocabulary of species, and let $bold(v)_s in RR^d$
denote the embedding of species $s$. Define the cosine similarity
$ "cos"(s, t) = (bold(v)_s dot.op bold(v)_t) / (norm(bold(v)_s) thin norm(bold(v)_t)) . $
Let $g(s)$ denote the genus of species $s$, and let $H$ be the
held-out fraction of records, partitioned into geohash bins ${B_k}$,
with $S(B_k) subset V$ the set of species observed in bin $B_k$.

*Congener gap.* Let $P_("cong") = {(s, t) : g(s) = g(t), s != t}$ and
$P_("rand") = {(s, t) : g(s) != g(t)}$. The congener gap is
$ Delta_("cong") = EE_(P_("cong"))[ "cos"(s, t) ] - EE_(P_("rand"))[ "cos"(s, t) ] . $
Congeners genuinely share ranges more often than random pairs, so
$Delta_("cong")$ has a real ecological component; it also has a
label-form component if the model can recover the genus from the
species token. The metric bounds the maximum label-substring
contribution.

*Sympatry AUC.* Let
$P^+ = {(s, t) : exists B_k in H, {s, t} subset.eq S(B_k)}$
be the held-out co-occurring pairs, and
$P^- subset (V times V) without P^+$ a size-matched sample of pairs
that never co-occur in $H$. The sympatry AUC is the area under the
ROC curve obtained by thresholding cosine similarity:
$ "AUC" = Pr[ "cos"(s, t) > "cos"(s', t') space | space (s, t) ~ P^+, (s', t') ~ P^- ] . $
We additionally report a _cross-genus_ variant restricted to
$P^+ inter {g(s) != g(t)}$ and $P^- inter {g(s) != g(t)}$, which
isolates the spatial signal from any residual label-form
contribution.

*Neighbourhood self-rank.* For each held-out occurrence
$(s, B_k)$ with $s in S(B_k)$, define the bin centroid
$ bold(c)_(k, s) = 1 / (|S(B_k)| - 1) sum_(t in S(B_k), t != s) bold(v)_t , $
and rank every species in $V$ by descending cosine similarity to
$bold(c)_(k, s)$. We report the median rank of the held-out species
$s$ across all held-out occurrences. $|V| / 2$ is random; 0 is
perfect.

= Experiments

== Data

We download #sym.tilde 147 k unique GBIF _Squamata_ records, sliced
by country with a per-country cap of 8 000 records and 6 parallel
workers (`species2vec/gbif_download_parallel.py`), and keep records
with non-null coordinates and a binomial species name. The data are
split 90 / 10 into training and held-out fractions, stratified by
geohash bin.

== Main result

The full pipeline trained on the _Squamata_ split yields the
following held-out metrics:

#figure(
  table(
    columns: 2,
    align: (left, right),
    table.header[*Metric*][*Value*],
    [species in vocab],                  [2 592],
    [congener cosine gap],               [0.29],
    [sympatry AUC (all pairs)],          [0.94],
    [sympatry AUC (cross-genus)],        [0.943],
    [neighbourhood self-rank (median)],  [36],
  ),
  caption: [Held-out metrics on the _Squamata_ benchmark
    ($|V| approx 2{,}600$).],
) <tab:main>

A cross-genus AUC of 0.943 substantially exceeds the random baseline
of 0.5 and indicates that cosine similarity in the embedding space is
a strong discriminator of held-out spatial co-occurrence, even when
within-genus pairs are removed from both classes. The median
neighbourhood self-rank of 36 (out of #sym.tilde 2 600) confirms that
held-out species are recovered near the top of the ranked list given
only the other species observed in the same cell.

== Design-choice ablation: token representation

To justify atomic tokens, we train an alternative with character
n-grams enabled (fastText defaults, `minn = 3, maxn = 6`), holding
every other hyperparameter fixed.

#figure(
  image("figures/cosine_dists.pdf", width: 100%),
  caption: [Distributions of cosine similarity between congener pairs
    and random pairs on the held-out _Squamata_ split. _Left:_ with
    character n-grams enabled, the congener distribution is shifted
    far to the right of the random-pair distribution
    ($Delta_("cong") = 0.41$). _Right:_ with atomic tokens, the
    congener distribution collapses toward the random-pair
    distribution ($Delta_("cong") = 0.29$); the residual right tail
    reflects real range overlap among congeners.],
) <fig:cosine_dists>

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    table.header[*Metric*][*with n-grams*][*atomic tokens*],
    [congener cosine gap],                [0.41],    [0.29],
    [sympatry AUC (all pairs)],           [0.91],    [0.94],
    [*sympatry AUC (cross-genus)*],       [*0.926*], [*0.943*],
    [neighbourhood self-rank (median)],   [52],      [36],
  ),
  caption: [Token-representation ablation. Atomic tokens reduce the
    congener gap by $approx 0.12$ and raise cross-genus sympatry AUC
    from 0.926 to 0.943.],
) <tab:ablation>

#figure(
  image("figures/metrics_bars.pdf", width: 90%),
  caption: [Held-out metrics, character n-grams vs atomic tokens. The
    atomic-token configuration wins on every metric.],
) <fig:metrics_bars>

The congener gap drops by $approx 0.12$ under atomic tokens: this is
the upper bound on the label-substring contribution to the congener
similarity. The residual 0.29 reflects genuine range overlap and is
recoverable from spatial co-occurrence alone. On cross-genus
sympatry — the task that isolates the spatial hypothesis — atomic
tokens are not merely safer but quantitatively better
(0.943 vs 0.926).

== Scaling with corpus size

We additionally train the same two configurations on a 10 k-record
subsample (#sym.tilde 1{,}474 species, #sym.tilde 7 records per
species). @fig:smalldata shows cross-genus sympatry AUC against
corpus size.

#figure(
  image("figures/smalldata.pdf", width: 70%),
  caption: [Cross-genus sympatry AUC vs corpus size. The spatial
    signal requires #sym.tilde 50 records per species before
    cross-genus AUC exceeds the random baseline. Character n-grams
    provide a data-free label-substring signal that does not depend
    on corpus size.],
) <fig:smalldata>

The spatial signal is data-hungry: at #sym.tilde 7 records per
species the cross-genus AUC of the atomic-token model sits at
#sym.tilde 0.39, below the 0.5 baseline. The n-gram configuration is
data-independent — it scores #sym.tilde 0.88 on the same subsample,
purely from name substrings — and therefore appears competitive in
the small-corpus regime even though it is solving a different,
non-spatial task. Reporting the congener gap together with AUC
disambiguates the two regimes.

== Qualitative inspection

@fig:umap shows a UMAP @mcinnes2018 projection of the trained
embedding space, coloured by the eight most populous genera. The
projection is _not_ trained to separate genera; the visible
clustering of conspecifics is an emergent consequence of congeners
sharing geographic range and therefore appearing in similar bin
contexts. Genera like _Sceloporus_ (North America) and _Liolaemus_
(Andes) form tight clusters; cosmopolitan or invasive genera like
_Hemidactylus_ are more diffuse, consistent with their wider
realised ranges.

#figure(
  image("figures/umap.pdf", width: 92%),
  caption: [UMAP projection of the embedding space, coloured by the
    eight most populous genera in the _Squamata_ training set. The
    projection itself has no access to genus labels; clustering of
    conspecifics is driven entirely by shared spatial context.],
) <fig:umap>

To illustrate the embedding geometry on a concrete example,
@fig:neighbors plots the global occurrence distribution of
_Anolis carolinensis_ (the green anole, native to the south-eastern
United States) and its top five embedding-space neighbours
(_Storeria dekayi_, _Scincella lateralis_, _Carphophis amoenus_,
_Sceloporus undulatus_, _Agkistrodon piscivorus_). The neighbours
are drawn from four different families and share no obvious name
substring with the target, yet all five are reptiles of the eastern
United States — exactly the spatial co-occurrence signal the model
is trained to capture.

#figure(
  image("figures/neighbors_map.pdf", width: 100%),
  caption: [Global occurrences of _Anolis carolinensis_ (black, target)
    and its top-5 nearest neighbours in the embedding space. The
    neighbours are taxonomically diverse (four families, all
    cross-genus) but spatially coherent: every species is endemic to
    or strongly associated with the south-eastern United States.],
) <fig:neighbors>

An interactive Streamlit application (`app.py`) is shipped with the
repository and exposes the same embeddings through three views: a
UMAP scatter with genus-coloured points, a global occurrence map
for any species, and a nearest-neighbour browser that returns the
top-$k$ similar species together with a side-by-side range map.
Vernacular-name lookups are cached against the GBIF species API.

= Applications

Dense species embeddings learned from spatial co-occurrence have
several downstream uses.

*Range completion and gap-filling.* For under-sampled species with
few GBIF records, the embedding vector can be transferred from
better-sampled co-occurring species: the bin centroid of a held-out
species lies near the species itself in embedding space
(@tab:main), so cells where the bin centroid would predict the
species are candidate occurrences. This is structurally similar to
species distribution modelling @elith2009 but uses observed
co-occurrence rather than environmental covariates as input.

*Invasive-species early warning.* When a species establishes itself
in a new region, its set of co-occurring neighbours changes
abruptly. Tracking the cosine distance between a species' historical
embedding and a re-estimated embedding from a recent time slice can
flag rapid niche shifts. The interactive tool's nearest-neighbour
browser makes this directly inspectable on the _Hemidactylus_ genus,
several members of which are widely introduced.

*Community-similarity queries.* Given a cell, embedding centroids
support queries of the form "which other cells host the most
similar community?" — useful for designing biodiversity-monitoring
transects, prioritising conservation corridors, or matching donor /
recipient sites for translocation.

*Biogeographic priors for downstream models.* The 100-dimensional
species vectors can be concatenated with environmental covariates
as input features for species distribution models or for joint
species-distribution models @ovaskainen2017, providing a learned
representation of biotic context that is otherwise hand-engineered.

*Field-guide ranking.* For citizen-science applications, ranking
candidate identifications by embedding similarity to the species
already recorded in the same cell can sharpen the prior on visually
ambiguous species pairs.

We emphasise that all of these applications depend on the embedding
geometry reflecting _spatial_ co-occurrence rather than name form;
the evaluation harness in §5 is what makes them defensible.

= Discussion

== Recommendations for downstream use

+ When training species embeddings on GBIF, use atomic tokens
  (`minn = maxn = 0` in fastText) unless the goal is explicitly to
  model morphological similarity of binomial names.
+ Choose geohash precision based on the spatial scale of the
  question and on positional uncertainty of the records.
+ Report at least two metrics: a congener gap (to bound the
  label-substring contribution) and a held-out cross-genus sympatry
  AUC (to measure the spatial signal). Either metric in isolation
  can be saturated by a model that does not solve the intended task.
+ Treat small-data ($lt.eq$ 10 records per species) results
  cautiously; AUC alone does not distinguish a spatial model from a
  label-substring model in that regime.

== Limitations

The benchmark is restricted to #sym.tilde 150 k _Squamata_ records
and a single train / held-out split. The qualitative conclusions —
that atomic tokens dominate character n-grams on biogeographic
metrics, and that the spatial signal scales with corpus size — are
robust across the splits and seeds we tested, but absolute AUC
numbers will shift with taxon, sample size, geohash precision, and
window. We have not investigated learned sub-word tokenisers
(BPE-style units derived from the corpus rather than from binomial
names), which might in principle recover useful morphological
signal while avoiding the genus-prefix shortcut.

= Reproducibility

All code is released at the repository linked below. A complete
reproduction is:

```bash
python -m venv .venv && source .venv/bin/activate
pip install gensim fasttext pandas numpy pygeohash scikit-learn tqdm requests matplotlib

python -m species2vec.gbif_download_parallel \
    --order Squamata --out data/squamata.csv \
    --per-country 8000 --workers 6

python - <<'PY'
import pandas as pd
from species2vec.pipeline import run
df = pd.read_csv('data/squamata.csv')
stats = run(df, workdir='runs/squamata', geohash_precision=5)
print(stats)
PY
```

The interactive `app.py` (Streamlit) provides UMAP projection and
nearest-neighbour browsing for any trained `.vec` file.

= Conclusion

Distributed representations of species learned from GBIF
co-occurrence encode range overlap. Two design choices — per-bin
sentences at an ecologically meaningful geohash precision, and
atomic species tokens — are necessary for the embedding geometry to
reflect spatial co-occurrence rather than the morphology of
binomial names. The resulting embeddings reach a cross-genus
held-out sympatry AUC of 0.943 on #sym.tilde 147 k _Squamata_
records.

#bibliography("preprint.bib", style: "apa", title: "References")

#v(1em)
*Data and code availability.* Code:
#link("https://github.com/boyanangelov/species2vec")
(this commit).
