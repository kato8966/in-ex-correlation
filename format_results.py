import csv
import json
import logging
import math
import os

for task in ['coref', 'hatespeech']:
    HEADERS = ['name', 'weat_es']
    if task == 'coref':
        wordlists = ['winobias', 'weat7']
        HEADERS += [f'type{typ}_{metric}_diff' for typ in [1, 2]
                    for metric in ['precision', 'recall', 'f1']]\
                        + ['CoNLL F1']
    else:
        wordlists = ['hatespeech', 'weat8']
        HEADERS += [f'{metric}_diff'
                    for metric in ['precision', 'recall', 'f1']]
    for word_emb in ['w2v', 'ft']:
        for wordlist in wordlists:
            with open(os.path.join('results',
                                   f'{task}_{word_emb}_{wordlist}.csv'), 'w',
                      newline='') as csvout:
                csv_writer = csv.writer(csvout)
                csv_writer.writerow(HEADERS)
                names = [f'original{i}' for i in range(1, 11)]\
                        + [f'db_{wordlist}_{typ}_{sample_prob}'  # noqa: E127
                           for typ in ['debias', 'overbias']
                           for sample_prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                               0.6, 0.7, 0.8, 0.9]]\
                        + [f'ar_{wordlist}_{typ}_reg{reg}_sim{sim}_ant{ant}'
                           for typ in ['debias', 'overbias']
                           for reg in ['1e-1', '5e-2', '1e-2']
                           for sim in [0.0, 0.5, 1.0] for ant in [0.0, 0.5, 1.0]]  # noqa: E501
                for name in names:
                    if task == 'coref':
                        corpus = 'wikipedia'
                    else:
                        corpus = 'twitter'
                    if 'original' in name:
                        weat_file_name = f'{corpus}_{word_emb}_{wordlist}'
                    else:
                        weat_file_name = f'{corpus}_{word_emb}_{name}'
                    with open(os.path.join('WEAT', 'result',
                                           weat_file_name + '.txt')) as weatin:
                        weat = json.load(weatin)['effect_size']

                    if math.isnan(weat):
                        logging.warning(f'{name} has nan value, thus skipped.')
                        continue

                    if task == 'coref':
                        coref = {}
                        with open(os.path.join('coref', 'result',
                                               f'{word_emb}_{name}',
                                               'conll_results.txt')) as corefin:
                            coref['conll_f1'] = json.load(corefin)['coref_f1']
                        for typ in [1, 2]:
                            for ap in ['anti', 'pro']:
                                with open(os.path.join('coref', 'result',
                                                       f'{word_emb}_{name}',
                                                       f'type{typ}_{ap}_results.txt')) as corefin:  # noqa: E501
                                    result = json.load(corefin)
                                    for metric in ['precision', 'recall', 'f1']:
                                        coref[f'type{typ}_{ap}_{metric}'] = result[f'coref_{metric}']  # noqa: E501
                        for typ in [1, 2]:
                            for metric in ['precision', 'recall', 'f1']:
                                coref[f'type{typ}_{metric}_diff'] = coref[f'type{typ}_pro_{metric}'] - coref[f'type{typ}_anti_{metric}']  # noqa: E501

                        csv_writer.writerow([name, weat]
                                            + [coref[f'type{typ}_{metric}_diff']
                                               for typ in [1, 2]
                                               for metric in ['precision',
                                                              'recall', 'f1']]
                                            + [coref['conll_f1']])
                    else:
                        hatespeech = {}
                        with open(os.path.join('hatespeech', 'results',
                                               f'{word_emb}_{name}.txt')) as hatespeechin:
                            result = json.load(hatespeechin)
                            for metric in ['precision', 'recall', 'f1']:
                                hatespeech[f'{metric}_diff'] = result['female'][metric] - result['male'][metric]  # noqa: E501
                        csv_writer.writerow([name, weat]
                                            + [hatespeech[f'{metric}_diff']
                                               for metric in ['precision',
                                                              'recall', 'f1']])
