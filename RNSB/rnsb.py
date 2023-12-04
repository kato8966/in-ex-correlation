from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from os import path

from gensim.models import KeyedVectors
from wefe.metrics import MyRNSB
from wefe.word_embedding_model import WordEmbeddingModel

import wordlists

def main(arg):
    if arg['word_emb'] == 'w2v':
        word_emb_short = 'w2v'
    else:
        assert arg['word_emb'] == 'fasttext'
        word_emb_short = 'ft'
    if 'weat' in arg['bias_modification_wordset']:
        if arg['bias_modification_wordset'] == 'weat_gender':
            bias_eval_wordsets = ['weat_6', 'weat_7', 'weat_8']
        elif arg['bias_modification_wordset'] == 'weat_gender_twitter':
            bias_eval_wordsets = ['weat_6', 'weat_7_twitter', 'weat_8']
        else:
            assert arg['bias_modification_wordset'] == 'weat_race'
            bias_eval_wordsets = ['weat_3', 'weat_4', 'weat_5']
    else:
        bias_eval_wordsets = [arg['bias_modification_wordset']]

    if arg['bias_modification'] == None:
        word_emb_file = path.join('..', arg['word_emb'], 'vectors',
                                  f'{arg["corpus"]}.txt')
        result_file = f'{arg["corpus"]}_{word_emb_short}_{arg["bias_modification_wordset"]}.txt'
    elif arg['bias_modification'] == 'db':
        temp = f'{arg["corpus"]}_{word_emb_short}_db_{arg["bias_modification_wordset"]}_{arg["bias_type"]}_{arg["sample_prob"]}'  # noqa: E501
        word_emb_file = path.join('..', arg['word_emb'], 'vectors',
                                  f'{temp.replace(word_emb_short + "_", "")}.txt')
        result_file = f'{temp}.txt'
    else:
        assert arg['bias_modification'] == 'ar'
        temp = f'{arg["corpus"]}_{word_emb_short}_ar_{arg["bias_modification_wordset"]}_{arg["bias_type"]}_reg{arg["reg"]}_sim{arg["sim"]}_ant{arg["ant"]}'  # noqa: E501
        word_emb_file = path.join('..', 'attract-repel', 'vectors',
                                  f'{temp.replace("ar_", "")}.txt')
        result_file = f'{temp}.txt'

    vecs = KeyedVectors.load_word2vec_format(word_emb_file, no_header=True)
    model = WordEmbeddingModel(vecs, "glove")
    rnsb = MyRNSB()
    with open(path.join('result', result_file), "w") as fout:
        result = {bias_eval_wordset: rnsb.run_query(getattr(wordlists,
                                                            bias_eval_wordset)(),
                                                    model, n_iterations=10,
                                                    random_states=arg['random_seeds'],
                                                    holdout=False,
                                                    warn_not_found_words=True)
                  for bias_eval_wordset in bias_eval_wordsets}
        json.dump(result, fout)


if __name__ == '__main__':
    args = [[{'word_emb': word_emb, 'corpus': corpus,
               'bias_modification_wordset': bias_modification_wordset,
               'bias_modification': None}]
            + [{'word_emb': word_emb, 'corpus': corpus,
                'bias_modification_wordset': bias_modification_wordset,
                'bias_modification': 'db', 'bias_type': bias_type,
                'sample_prob': f'0.{i}'}
                for bias_type in ['debias', 'overbias'] for i in range(10)]
            + [{'word_emb': word_emb, 'corpus': corpus,
                'bias_modification_wordset': bias_modification_wordset,
                'bias_modification': 'ar', 'bias_type': bias_type, 'reg': reg,
                'sim': sim, 'ant': ant}
               for bias_type in ['debias', 'overbias']
               for reg in ['1e-1', '5e-2', '1e-2']
               for sim in ['0.0', '1.0']
               for ant in ['0.0', '1.0']]
            for word_emb in ['w2v', 'fasttext']
            for corpus in ['wikipedia', 'twitter']
            for bias_modification_wordset in (['winobias', 'weat_gender']
                                              if corpus == 'wikipedia'
                                              else ['hatespeech_gender',
                                                    'weat_gender_twitter',
                                                    'hatespeech_race',
                                                    'weat_race'])]
    args = list(itertools.chain.from_iterable(args))
    with open('random_seeds.json') as fin:
        seedss = json.load(fin)
    for i, arg in enumerate(args):
        arg['random_seeds'] = seedss[i]

    with ProcessPoolExecutor(48) as p:
        assert all(result == None
                   for result in p.map(main,
                                       [arg for arg in args
                                        if arg['corpus'] == 'wikipedia']))
    with ProcessPoolExecutor(72) as p:
        assert all(result == None
                   for result in p.map(main,
                                       [arg for arg in args
                                        if arg['corpus'] == 'twitter']))
