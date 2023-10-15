import numpy as np
import spacy

from wefe.query import Query
import wordlists

N = 100


def cos_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def expand_wordlist(wordlist, query, vocab):
    exp = set()
    emb = vocab.vectors
    strings = vocab.strings
    for word in wordlist:
        vector_w = np.mean([emb[token] for token in word.split()], 0)
        neighbors = []
        for u in strings:
            if strings[u] in emb:
                vector_u = emb[u]
                neighbors.append((cos_sim(vector_w, vector_u), u.lower()))
        neighbors.sort(reverse=True)
        assert len(neighbors) >= N
        for (_, neighbor) in neighbors[:N]:
            if (neighbor not in query.target_sets[0]
                and neighbor not in query.target_sets[1]
                and neighbor not in query.attribute_sets[0]
                and neighbor not in query.attribute_sets[1]):
                exp.add(neighbor)
    return wordlist + list(exp)


def expand_query(query, vocab):
    return Query(target_sets=[expand_wordlist(target_set, query, vocab)
                              for target_set in query.target_sets],
                 attribute_sets=[expand_wordlist(attribute_set, query, vocab)
                                 for attribute_set in query.attribute_sets],
                 target_sets_names=[f'exp. {target_set_name}'
                                    for target_set_name in query.target_sets_names],
                 attribute_sets_names=[f'exp. {attribute_set_name}'
                                       for attribute_set_name in query.attribute_sets_names])


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    vocab = nlp.vocab
    print('WinoBias wordlist expansion:')
    print(expand_query(wordlists.winobias(), vocab))
    print()
    print('Hate speech wordlist expansion:')
    print(expand_query(wordlists.hatespeech(), vocab))
    print()
    print('WEAT wordlist expansion:')
    print(expand_query(wordlists.weat_all(), vocab))
    print()
