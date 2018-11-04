# species2vec

[![DOI](https://zenodo.org/badge/141325266.svg)](https://zenodo.org/badge/latestdoi/141325266)

Usage instructions:

```python
import gensim

# load species embeddings
m = gensim.models.KeyedVectors.load_word2vec_format('mammalia_6M.vec')

# test species embeddings
m.most_similar(u'Grammomys_ibeanus')
```
