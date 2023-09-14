from concurrent.futures import ProcessPoolExecutor, wait
import json
from os import path

from gensim.models import KeyedVectors
from wefe.metrics import WEAT
from wefe.word_embedding_model import WordEmbeddingModel

import wordlists

# command line argument
# $1 = path to word embedding file in GloVe format
# $2 = wordlist used
# $3 = path to a file where the result will be saved

def main(word_emb, wordlist_name, result_file):
    vecs = KeyedVectors.load_word2vec_format(word_emb, no_header=True)
    model = WordEmbeddingModel(vecs, "glove")
    weat = WEAT()
    with open(result_file, "w") as fout:
        result = weat.run_query(getattr(wordlists, wordlist_name)(), model,
                                lost_vocabulary_threshold=0.0)
        json.dump(result, fout)


if __name__ == '__main__':
    with ProcessPoolExecutor(6) as p:
        futures = []
        for (word_emb, word_emb_short) in [('w2v', 'w2v'), ('fasttext', 'ft')]:
            for wordset in ['winobias', 'weat7']:
                futures.append(p.submit(main, path.join('..', word_emb,
                                                        'vectors',
                                                        'wikipedia.txt'),
                                        wordset,
                                        f'result/wikipedia_{word_emb_short}_{wordset}.txt'))  # noqa: E501

                for bias_type in ['debias', 'overbias']:
                    for i in range(10):
                        ratio = f'0.{i}'
                        temp = f'wikipedia_{word_emb_short}_db_{wordset}_{bias_type}_{ratio}'  # noqa: E501
                        futures.append(p.submit(main, path.join('..',
                                                                word_emb,
                                                                'vectors',
                                                                f'{temp.replace(word_emb_short + "_", "")}.txt'),  # noqa: E501
                                                wordset, f'result/{temp}.txt'))

                for bias_type in ['debias', 'overbias']:
                    for reg in ['1e-1', '5e-2', '1e-2']:
                        for sim in [0.0, 0.5, 1.0]:
                            for ant in [0.0, 0.5, 1.0]:
                                temp = f'wikipedia_{word_emb_short}_ar_{wordset}_{bias_type}_reg{reg}_sim{sim}_ant{ant}'  # noqa: E501
                                futures.append(p.submit(main, '../attract-repel/'
                                                        f'vectors/{temp.replace("ar_", "")}.txt',  # noqa: E501
                                                        wordset,
                                                        f'result/{temp}.txt'))

            for wordset in ['hatespeech', 'weat8']:
                futures.append(p.submit(main, f'../{word_emb}/vectors/'
                                              'twitter.txt',
                                        wordset,
                                        'result/'
                                        f'twitter_{word_emb_short}_{wordset}.txt'))  # noqa: E501

                for bias_type in ['debias', 'overbias']:
                    for i in range(10):
                        ratio = f'0.{i}'
                        temp = f'twitter_{word_emb_short}_db_{wordset}_{bias_type}_{ratio}'
                        futures.append(p.submit(main,
                                                f'../{word_emb}/vectors/'
                                                f'{temp.replace(word_emb_short + "_", "")}.txt',  # noqa: E501
                                                wordset, f'result/{temp}.txt'))

                for bias_type in ['debias', 'overbias']:
                    for reg in ['1e-1', '5e-2', '1e-2']:
                        for sim in [0.0, 0.5, 1.0]:
                            for ant in [0.0, 0.5, 1.0]:
                                temp = f'twitter_{word_emb_short}_ar_{wordset}_{bias_type}_reg{reg}_sim{sim}_ant{ant}'  # noqa: E501
                                futures.append(p.submit(main, '../attract-repel/'
                                                        f'vectors/{temp.replace("ar_", "")}.txt',  # noqa: E501
                                                        wordset,
                                                        f'result/{temp}.txt'))

        assert all(future.exception() == None for future in futures)
