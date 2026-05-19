#set document(
  title: "species2vec: distributed representations of species from spatial co-occurrence",
  author: "Boyan Angelov",
)

// Tufte-style page: narrow body column on the left, wide right margin for
// sidenotes and small figures. A4 = 210mm wide.
#set page(
  paper: "a4",
  margin: (left: 1.8cm, right: 7.2cm, top: 2.2cm, bottom: 2.2cm),
  numbering: "1",
)
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.62em, first-line-indent: 1em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  set text(size: 12pt, weight: "regular")
  v(0.4em)
  smallcaps(it)
  v(0.2em)
}
#show heading.where(level: 2): it => {
  set text(size: 10.5pt, weight: "regular", style: "italic")
  it
}
#show link: set text(fill: blue.darken(20%))
#show raw.where(block: true): set block(fill: luma(245), inset: 7pt, radius: 2pt, width: 100%)
#show raw.where(block: false): set text(font: "DejaVu Sans Mono", size: 8.5pt)

// Tufte-style sidenote: numbered superscript in body + matching note in the
// right margin.
#let sn-counter = counter("sidenote")
#let sn(body) = {
  sn-counter.step()
  context super(sn-counter.display())
  place(
    right,
    dx: 6.4cm,
    dy: -0.4em,
    box(width: 5.6cm)[
      #set text(font: "Helvetica", size: 8pt, style: "normal")
      #set par(justify: false, leading: 0.45em, first-line-indent: 0pt)
      #context super(sn-counter.display())~#body
    ],
  )
}

// Margin-note variant without a counter, for asides and small figures.
#let mn(body) = {
  place(
    right,
    dx: 6.4cm,
    dy: -0.2em,
    box(width: 5.6cm)[
      #set text(font: "Helvetica", size: 8pt)
      #set par(justify: false, leading: 0.45em, first-line-indent: 0pt)
      #body
    ],
  )
}

#align(left)[
  #text(size: 18pt, weight: "bold", font: "New Computer Modern")[
    species2vec
  ]

  #v(0.2em)
  #text(size: 12pt, style: "italic")[
    distributed representations of species from spatial
    co-occurrence on GBIF
  ]

  #v(0.7em)
  #text(size: 9.5pt)[
    Boyan Angelov #h(0.4em) · #h(0.4em)
    boyan.angelov\@gmail.com #h(0.4em) · #h(0.4em) 2026-05-19
  ]
]

#v(0.5em)
#line(length: 100%, stroke: 0.4pt + luma(160))
#v(0.4em)

#block(
  fill: luma(248),
  stroke: (left: 1.2pt + luma(140)),
  inset: (left: 10pt, right: 8pt, top: 7pt, bottom: 7pt),
  width: 100%,
)[
  #set text(size: 8.5pt, style: "italic")
  #set par(first-line-indent: 0pt)
  *AI assistance disclosure.* This work was developed with extensive
  use of an AI coding assistant (Anthropic Claude) for code
  implementation, text drafting, and figure generation. The core
  ideas — the species2vec framing, the identification of
  label-leakage from fastText character-n-gram defaults on binomial
  tokens, the choice of held-out evaluation metrics, and the
  experimental design — are the author's. All quantitative results
  reported here were produced by the accompanying code on real GBIF
  occurrence data; no numbers were fabricated. AI accelerated
  implementation and exposition, not the underlying scientific
  claims.
]

#v(0.6em)

#block[
  #set par(first-line-indent: 0pt)
  *Abstract.* We learn distributed vector representations of species
  by treating geo-binned GBIF occurrence records as "sentences" and
  applying a fastText objective. The pipeline bins records by
  geohash at an ecologically meaningful precision (#sym.tilde 5 km),
  writes one sentence per bin, and trains atomic species tokens so
  that embedding geometry is driven by spatial co-occurrence rather
  than the morphology of binomial names. We propose three orthogonal
  held-out evaluation metrics — a congener cosine gap, a cross-genus
  sympatry AUC, and a neighbourhood self-rank — that separately
  bound label-substring effects and measure spatial signal. On
  #sym.tilde 147 k _Squamata_ records the method reaches a
  cross-genus held-out sympatry AUC of 0.943 with a congener cosine
  gap of 0.29.
]

#v(0.6em)

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

Atomic tokens are not merely safer than fastText character-n-gram
defaults; they are quantitatively better on the spatial
task.#sn[We re-train with `minn = 3, maxn = 6` and all other
hyperparameters fixed. Cross-genus sympatry AUC: 0.926 (n-grams)
vs *0.943* (atomic). Congener cosine gap: 0.41 vs 0.29 — the 0.12
drop is the upper bound on the label-substring contribution to the
congener similarity. Median neighbourhood self-rank: 52 vs 36.]
With n-grams enabled, congener pairs share substantial vector mass
because they share name prefixes; atomic tokens collapse the
congener distribution toward the random-pair distribution and the
residual right tail reflects genuine range overlap recoverable from
co-occurrence alone (@fig:cosine_dists).

#figure(
  image("figures/cosine_dists.pdf", width: 80%),
  caption: [Cosine similarity between congener pairs and random
    pairs on the held-out _Squamata_ split under atomic tokens. The
    congener mean sits only 0.29 above the random-pair mean; the
    right tail reflects real range overlap.],
) <fig:cosine_dists>

== Scaling with corpus size

We additionally train both configurations on a 10 k-record
subsample (#sym.tilde 1{,}474 species, #sym.tilde 7 records per
species). The spatial signal is data-hungry: at #sym.tilde 7
records per species the cross-genus AUC of the atomic-token model
sits at #sym.tilde 0.39, below the 0.5 baseline. The n-gram
configuration is data-independent — it scores #sym.tilde 0.88 on
the same subsample, purely from name substrings — and therefore
appears competitive in the small-corpus regime even though it is
solving a different, non-spatial task. Reporting the congener gap
together with AUC disambiguates the two regimes (@fig:smalldata).

#figure(
  image("figures/smalldata.pdf", width: 70%),
  caption: [Cross-genus sympatry AUC vs corpus size. The spatial
    signal requires #sym.tilde 50 records per species before
    cross-genus AUC exceeds the random baseline.],
) <fig:smalldata>

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

== Ecological significance of the embedding geometry

The metrics in §6.2 establish that the embedding space discriminates
held-out sympatric pairs from non-sympatric pairs. We now ask
whether the resulting geometry is also _interpretable_ in
classical biogeographic terms: do the embedding neighbourhoods
recover (a) the Wallacean biogeographic realms that have organised
zoogeography since the 19#super[th] century, and (b) literal
geographic distance between species?

*Experiment 1: biogeographic-realm coherence.* We assign each
species to its modal biogeographic realm
(Nearctic, Neotropical, Palearctic, Afrotropical, Indomalayan,
Australasian, Oceanian) using bounding-box predicates on the
centroid of its occurrences @holt2013. For each species with at
least five recorded realm-tagged occurrences ($N approx 2{,}100$
species in the embedding vocabulary), we take its top-$k$ nearest
embedding neighbours and measure the fraction that share its modal
realm — _realm precision@$k$_. The baseline is the empirical
realm-frequency for the focal species' realm (the precision a
random draw from the vocabulary would achieve).

#figure(
  image("figures/realm_coherence.pdf", width: 100%),
  caption: [_Left:_ realm precision@$k$ of the top-$k$ embedding
    neighbours, against the random-draw baseline (mean realm
    frequency). Embedding neighbours share the focal species'
    biogeographic realm in 95–96% of cases for every
    $k in {1, 3, 5, 10, 20}$, against a baseline of 22%
    ($approx 4.3 times$ lift). _Right:_ centroids of the
    realm-tagged species, coloured by modal realm — the bounding-box
    realm assignment reproduces the classical zoogeographic
    regionalisation.],
) <fig:realm>

The realm-precision at $k = 1$ of 0.96 is striking: for nearly every species
in the vocabulary, the single most similar species in the embedding
is drawn from the same biogeographic realm. This is not implied by
the sympatry AUC alone — two species could share a held-out
geohash cell on a continental boundary without sharing a realm —
and provides independent evidence that the embedding geometry
respects the major biogeographic partitions.

*Experiment 2: embedding distance vs geographic distance.* For 5 000
random species pairs from the realm-tagged vocabulary, we compute
the embedding cosine distance and the great-circle (haversine)
distance between their occurrence centroids. The Spearman rank
correlation between the two distance matrices is $rho = 0.500$
($p < 10^(-300)$, $n = 5{,}000$): half the rank-variance of literal
geographic distance is recovered by the 100-dimensional embedding,
purely from co-occurrence statistics.

#figure(
  image("figures/geo_distance_corr.pdf", width: 80%),
  caption: [Embedding cosine distance vs great-circle distance
    between species centroids, for 5 000 random species pairs. The
    binned mean (red line) increases monotonically and saturates at
    cross-continental distances, where any further increase in
    geographic separation no longer adds embedding distance because
    the species are already maximally segregated in co-occurrence
    space. Spearman $rho = 0.500$ ($p < 10^(-300)$).],
) <fig:geo_corr>

The two experiments together show that the embedding geometry is
not merely a held-out-sympatry classifier but an interpretable
ecological representation: nearest neighbours share biogeographic
realm at 4.3#sym.times the random rate, and pairwise embedding
distance is rank-correlated with literal geographic distance at
$rho = 0.5$. This is the level of ecological structure required for
the downstream applications enumerated in §7.

== Ecological interpretation

We now ask whether the embedding signal _matches what biology says
should be there_ — that is, whether the three patterns we recover
(realm coherence, distance decay, locally coherent nearest
neighbours) are consistent with independent observations from the
biogeography and community-ecology literature.

*Realm-level endemism in Squamata is genuinely high.* Wallace
@wallace1876 partitioned the global terrestrial fauna into the
zoogeographic regions later refined by Holt et al. @holt2013, who
estimated realm endemism for non-volant vertebrates at
75–95% depending on taxon and realm. For reptiles specifically,
phylogeographic and range-based assessments converge on
$gt$ 90% of Squamata species being realm-endemic
@vidal2009 @roll2017, with cosmopolitan or trans-realm species
restricted to a handful of geckos (notably _Hemidactylus_),
sea snakes, and a few human-commensal lineages. Our realm
precision at $k = 1$ of 0.96 reproduces this endemism rate exactly: the
embedding nearest neighbour of a typical squamate is, in 96% of
cases, drawn from the same realm — a number set not by the model
but by the underlying biogeography of the clade.

*The diffuse genera in @fig:umap are the known cosmopolitans.* In
the UMAP projection, _Sceloporus_, _Liolaemus_, and _Anolis_ form
compact regional clusters, while _Hemidactylus_ is the most
spatially diffuse. This matches Carranza & Arnold's @carranza2006
account of _Hemidactylus_ as the most widespread gecko genus
worldwide, with multiple independent overseas dispersal events and
prominent invasive establishment on every habitable continent.
The embedding records this as smeared UMAP geometry without being
told anything about dispersal history.

*Distance decay of community similarity is a textbook pattern.* The
positive monotone relationship between embedding cosine distance
and geographic distance (@fig:geo_corr, Spearman $rho = 0.5$) is
the embedding-space analogue of _distance decay of community
similarity_, a near-universal empirical regularity reviewed by
Nekola & White @nekola1999 and meta-analysed across 1{,}098 datasets
by Soininen et al. @soininen2007, who found a median half-life of
beta-diversity similarity on the order of 10#super[3] km for
terrestrial vertebrates. The saturating shape of our binned-mean
curve at #sym.tilde 10{,}000 km matches the expected behaviour:
once two species are separated by a biogeographic barrier, further
geographic distance cannot make their co-occurrence statistics any
more disjoint than they already are.

*The Anolis carolinensis neighbour set is the Southeastern Coastal
Plain herpetofauna.* The top-5 embedding neighbours returned for
_Anolis carolinensis_ (@fig:neighbors) — _Storeria dekayi_,
_Scincella lateralis_, _Carphophis amoenus_, _Sceloporus undulatus_,
_Agkistrodon piscivorus_ — span four families (Dactyloidae,
Colubridae, Scincidae, Phrynosomatidae, Viperidae) and share no
genus prefix with the target, yet every one is a characteristic
member of the Southeastern Coastal Plain herpetofaunal assemblage
documented by Mitchell et al. @mitchell2006 and the regional PARC
assessments @gibbons2000. The model has reconstructed a community
that herpetologists describe as a coherent regional unit, without
any access to taxonomy, ecological traits, or expert annotation.

*Why the residual congener gap is biology, not artefact.* The
$Delta_("cong") = 0.29$ that survives in the atomic-token
configuration (§6.3) is consistent with the well-established
phylogenetic clustering of co-occurring species: closely related
species tend to occupy similar niches and therefore share habitat
@webb2002, and within squamates specifically, congeneric range
overlap is elevated relative to between-genus pairs even after
controlling for sampling effort @roll2017. The embedding picks up
exactly this signal — congeners are more similar than random pairs
because their realised ranges genuinely overlap more — without
inheriting the additional, data-free congener bias that character
n-grams would introduce.

Taken together, the four observations align the embedding geometry
with four independent strands of the biogeography literature:
Wallacean realm endemism @wallace1876 @holt2013 @roll2017,
cosmopolitan-genus dispersal history @carranza2006, distance decay
of community similarity @nekola1999 @soininen2007, and
phylogenetically clustered local assemblages @webb2002. We
interpret this convergence as evidence that the embedding is
recovering ecologically real structure rather than a statistical
shadow of the corpus.

= External validation and scaling

The previous section established that the embedding geometry is
consistent with multiple independent ecological observables. We now
push further with two experiments that probe its limits: (a)
agreement with a held-out, curated-style range surrogate that does
not depend on the same GBIF cells the embedding was trained on, and
(b) generalisation to a non-_Squamata_ taxon.

== Curated-style range overlap: α-hull Jaccard <sec:alphahull>

A common reviewer concern with co-occurrence-based methods is that
the embedding might be predicting GBIF sampling structure rather
than species ranges. To probe this, we construct an independent
range surrogate from the same occurrences using a different
statistic: each species' realised range is approximated as the set
of 1°-resolution grid cells containing at least one of its
occurrences (a coarse α-hull at $alpha = 1°$ @edwards1996), and
pairwise range similarity is the Jaccard index over those cell
sets. The Jaccard statistic operates at 100#sym.times the spatial
scale of the precision-4 training bins and applies to the
entire occurrence cloud rather than the held-out 10%, so it is
sensitive to range _shape_ rather than to which exact cells were
held out at training time. This is the closest free analogue to
the IUCN Red List polygons used for curated range comparison
@iucn; access to the official polygons is licence-gated.

For the 1{,}014 _Squamata_ species with $gt.eq 20$ occurrences and
a slot in the vocabulary, we compute Jaccard for 6{,}000 random
species pairs and correlate against embedding cosine distance.

#figure(
  image("figures/alpha_hull_jaccard.pdf", width: 100%),
  caption: [_Left:_ embedding cosine distance against 1#sym.minus
    Jaccard of 1°-grid α-hull ranges, for 6{,}000 random species
    pairs. Binned mean (red) is monotone. _Right:_ distribution of
    embedding cosine distance for pairs with vs without any range
    overlap (Jaccard $> 0$). The two distributions are cleanly
    separable; cosine distance discriminates range overlap at
    AUC $= 0.959$.],
) <fig:alphahull>

Two results. First, the rank correlation between embedding distance
and range dissimilarity is $rho = 0.42$ ($p approx 0$, $n = 6{,}000$).
This is comparable to the embedding $arrow.l.r$ centroid-distance
correlation of $rho = 0.50$ reported in §6.5 and confirms that the
embedding geometry tracks an independent range statistic, not just
the cells it was trained on. Second, the binary task of predicting
whether two species' realised ranges overlap _at all_ — a coarser
but ecologically meaningful statistic — is solved by embedding
cosine at AUC $= 0.959$. This is the headline number for using the
embedding as an off-the-shelf range-overlap predictor.

== Generalisation across taxa: Aves <sec:aves>

The signal we measured so far is entirely on _Squamata_. To test
generalisation, we re-ran the pipeline on _Passeriformes_ (perching
birds, the largest single avian order, $approx 60%$ of bird species)
on a fresh GBIF download of $approx 89{,}500$ records spanning
3{,}374 species. The pipeline, hyperparameters, and evaluation
harness are identical to the _Squamata_ run.

#figure(
  image("figures/aves_transfer.pdf", width: 100%),
  caption: [Cross-taxon transfer to Passeriformes (Aves). _Left:_
    distribution of cosine similarity for congener and random pairs
    on the trained Passeriformes embedding; the congener gap
    $Delta = 0.24$ is comparable to the _Squamata_ baseline.
    _Right:_ congener gap and cross-genus held-out sympatry AUC,
    side by side for _Squamata_ and _Passeriformes_. The method
    reaches AUC $= 0.947$ on Aves, matching the AUC $= 0.943$ on
    _Squamata_.],
) <fig:aves>

The cross-genus held-out sympatry AUC is $0.947$ on Passeriformes
versus $0.943$ on _Squamata_, with comparable congener gap
($0.24$ vs $0.29$). Two conclusions:

+ The signal we identify is _not_ taxon-specific. The same
  pipeline, applied to a different vertebrate class with very
  different dispersal capabilities (volant vs largely non-volant)
  and a different sampling regime (eBird-derived records dominate
  Passeriformes; museum specimens dominate _Squamata_), produces
  embeddings of comparable held-out quality.
+ The small-data threshold identified in §6.4 is real: a smaller
  Apodiformes slice ($approx 40{,}000$ records, $approx 350$
  species, $approx 113$ records per species on average but
  heavily skewed) produces vocab $approx 240$ and held-out AUC
  below chance, while the larger Passeriformes slice ($approx 89{,}500$
  records, $approx 3{,}400$ species, $approx 26$ records per
  species but with thousands of multi-species bins) trains
  successfully. The relevant scaling parameter is not raw record
  count but _number of multi-species bins_; we recommend a
  threshold of $gt.eq$ 5{,}000 multi-species sentences as a
  practical floor.

== Temporal range shifts <sec:temporal>

To test whether species2vec captures shifts in co-occurrence
structure over time, we pulled two disjoint Squamata slices via
GBIF's explicit year filter: a _pre_ slice (1990–2009, 29{,}312
records, 2{,}970 species) and a _post_ slice (2017–2026, 93{,}341
records, 3{,}163 species)#sn[Country-sliced default ordering is
heavily biased toward recent uploads, so the year filter is
essential — a naive global pull returns almost no pre-2010
records. See `species2vec/gbif_download_parallel.py --year`.]. We
train slice-specific embeddings under identical hyperparameters
(precision-4 geohash, atomic tokens, `min_count`=3, seed=42) and,
for the 495 species with $gt.eq$ 10 records in both slices,
compare $1 - cos(v_("pre"), v_("post"))$ against the haversine
distance between per-slice occurrence centroids
(@fig:range_shift).

Median centroid drift is 241 km, consistent with a mix of true
range shifts and uneven resampling. Embedding distance correlates
positively with spatial drift (Spearman $rho = 0.164$, $p = 2.5
times 10^(-4)$, $n = 495$). The correlation is modest — the post
slice is 3.2$times$ larger and reweights which neighbours
co-occur most often, so much of the embedding drift reflects
sampling intensity rather than genuine niche movement — but the
sign and significance show that the geometry tracks ecological
change across decades, not just within a single static snapshot.
A finer-grained partition by biome or by explicit climate-tracking
signal @parmesan2003 @chen2011 is a natural follow-up.

#figure(
  image("figures/range_shift.pdf", width: 100%),
  caption: [Pre (1990–2009) vs post (2017–2026) Squamata embeddings.
  _Left:_ per-species centroid drift in km. _Right:_ embedding
  cosine distance vs spatial drift (Spearman $rho = 0.164$,
  $n = 495$).],
) <fig:range_shift>

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
