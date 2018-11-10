# species2vec

[![DOI](https://zenodo.org/badge/141325266.svg)](https://zenodo.org/badge/latestdoi/141325266)

bioRxiv pre-print: https://www.biorxiv.org/content/early/2018/11/05/461996

Usage instructions:

```python
import gensim

# load species embeddings
m = gensim.models.KeyedVectors.load_word2vec_format('mammalia_6M.vec')

# test species embeddings
m.most_similar(u'Grammomys_ibeanus')
```

Two sets of embeddings are included in this repository (`mammalia_6M.vec` amd `reptilia_3M.vec`). If you need to generate new ones, have a look at the tutorial notebook in `notebooks/species2vec_tutorial.ipynb`.
