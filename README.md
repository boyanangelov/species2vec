# species2vec

Usage instructions:

```python
import gensim

# load species embeddings
m = gensim.models.KeyedVectors.load_word2vec_format('species2vec_200K.vec')

# test species embeddings
m.most_similar(u'Loxodonta_africana')
```
