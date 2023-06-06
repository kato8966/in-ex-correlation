import sys

from gensim.models import KeyedVectors

if __name__ == "__main__":
    # first argument is input file in glove format
    # second argument is output file in w2v format
    vec = KeyedVectors.load_word2vec_format(sys.argv[1], binary=False, no_header=True)
    vec.save_word2vec_format(sys.argv[2])
