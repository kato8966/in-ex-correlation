import sys

from gensim.models import KeyedVectors
import numpy as np
from wefe.metrics import RNSB
from wefe.word_embedding_model import WordEmbeddingModel

import wordlists

# command line argument
# $1 = path to word embedding file in GloVe format
# $2 = wordlist used
# $3 = path to a file where the result will be saved

np.random.seed(20230613)

vecs = KeyedVectors.load_word2vec_format(sys.argv[1], no_header=True)
model = WordEmbeddingModel(vecs, "glove")
rnsb = RNSB()
with open(sys.argv[3], "w") as fout:
    result = rnsb.run_query(getattr(wordlists, sys.argv[2])(), model,
                            n_iterations=10, holdout=False,
                            lost_vocabulary_threshold=0.0)
    fout.write(str(result['result']))
