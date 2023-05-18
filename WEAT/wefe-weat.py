import sys

from gensim.models import KeyedVectors
from wefe.metrics import WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel

import wordlists

# command line argument
# $1 = path to word embedding file in w2v format
# $2 = path to a file where the result will be saved

vecs = KeyedVectors.load_word2vec_format(sys.argv[1])
model = WordEmbeddingModel(vecs, "w2v")
weat = WEAT()
with open(sys.argv[2], "w") as fout:
    result = weat.run_query(wordlists.winobias(), model, lost_vocabulary_threshold=0.0)
    fout.write(f'Winobias: {result["result"]}')
