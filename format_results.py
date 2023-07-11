import csv
import json
import logging
import math
import os

# coref
wordlists = ['winobias', 'weat7']
HEADERS = ['name', 'weat_es'] + [f'type{typ}_{metric}_diff'
                                 for typ in [1, 2]
                                 for metric in ['precision', 'recall', 'f1']]\
                              + ['CoNLL F1']

for wordlist in wordlists:
    with open(os.path.join('results', f'coref_{wordlist}.csv'), 'w') as csvout:
        csv_writer = csv.writer(csvout)
        csv_writer.writerow(HEADERS)
        names = [f'original_lc{i}' for i in range(1, 11)]\
                  + [f'db_{wordlist}_{typ}_{sample_prob}'  # noqa: E127
                     for typ in ['debias', 'overbias']
                     for sample_prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                         0.7, 0.8, 0.9]]\
                  + [f'ar_{wordlist}_{typ}_reg{reg}_sim{sim}_ant{ant}'
                     for typ in ['debias', 'overbias']
                     for reg in ['1e-1', '5e-2', '1e-2']
                     for sim in [0.0, 0.5, 1.0] for ant in [0.0, 0.5, 1.0]]
        for name in names:
            if 'original_lc' in name:
                weat_file_name = f'original_lc_{wordlist}'
            else:
                weat_file_name = name
            with open(os.path.join('WEAT', 'result',
                                   weat_file_name + '.txt')) as weatin:
                weat = json.load(weatin)['effect_size']

            if math.isnan(weat):
                logging.warning(f'{name} has nan value, thus skipped.')
                continue

            coref = {}
            with open(os.path.join('coref', 'result', name,
                                   'conll_results.txt')) as corefin:
                coref['conll_f1'] = json.load(corefin)['coref_f1']
            for typ in [1, 2]:
                for ap in ['anti', 'pro']:
                    with open(os.path.join('coref', 'result', name,
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
                                     for metric in ['precision', 'recall',
                                                   'f1']]
                                  + [coref['conll_f1']])
