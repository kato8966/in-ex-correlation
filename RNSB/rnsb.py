import sys

from gensim.models import KeyedVectors
from wefe.metrics import RNSB
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel

# command line argument
# $1 = path to word embedding file in w2v format
# $2 = path to a folder where the result will be saved

def query(weat_type):
    if weat_type == 6:
        return Query(
                   target_sets=[
                       ["male", "man", "boy", "brother", "he", "him", "his", "son"],
                       ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]],
                   attribute_sets=[
                       ["executive", "management", "professional", "corporation", "salary", "office", "business", "career"],
                       ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"]],
                   target_sets_names=["Male", "Female"],
                   attribute_sets_names=["Careers", "Family"])
    elif weat_type == 7:
        return Query(
                   target_sets=[
                       ["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"],
                       ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"]],
                   attribute_sets=[
                       ["male", "man", "boy", "brother", "he", "him", "his", "son"],
                       ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]],
                   target_sets_names=["Math", "Arts"],
                   attribute_sets_names=["Male", "Female"])
    elif weat_type == 8:
        return Query(
                   target_sets=[
                       ["science", "technology", "physics", "chemistry", "Einstein", "NASA", "experiment", "astronomy"],
                       ["poetry", "art", "Shakespeare", "dance", "literature", "novel", "symphony", "drama"]],
                   attribute_sets=[
                       ["brother", "father", "uncle", "grandfather", "son", "he", "his", "him"],
                       ["sister", "mother", "aunt", "grandmother", "daughter", "she", "hers", "her"]],
                   target_sets_names=["Science", "Arts"],
                   attribute_sets_names=["Male", "Female"])
 
vecs = KeyedVectors.load_word2vec_format(sys.argv[1])
model = WordEmbeddingModel(vecs, "w2v")
rnsb = RNSB()
with open(sys.argv[2], "w") as fout:
    for weat_type in range(6, 9):
        q = query(weat_type)
        fout.write(f"WEAT {weat_type}: {rnsb.run_query(q, model)['result']}\n")
