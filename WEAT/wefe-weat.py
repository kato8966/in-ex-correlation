import sys

from gensim.models import KeyedVectors
from wefe.metrics import WEAT
from wefe.word_embedding_model import WordEmbeddingModel

import wordlists

# command line argument
# $1 = path to word embedding file in GloVe format
# $2 = wordlist used
# $3 = path to a file where the result will be saved

vecs = KeyedVectors.load_word2vec_format(sys.argv[1], no_header=True)
model = WordEmbeddingModel(vecs, "glove")
weat = WEAT()
with open(sys.argv[3], "w") as fout:
    result = weat.run_query(getattr(wordlists, sys.argv[2])(), model,
                            lost_vocabulary_threshold=0.0)
    fout.write(result)
