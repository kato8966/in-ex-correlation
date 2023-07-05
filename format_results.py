import csv
import json
import logging
import math
import os

# coref
HEADERS = ['name', 'winobias_weat_es']\
            + [f'type{typ}_{metric}_diff'
               for typ in [1, 2] for metric in ['precision', 'recall', 'f1']]\
            + ['CoNLL F1']
with open('results/coref.csv', 'w') as csvout:
    csv_writer = csv.writer(csvout)
    csv_writer.writerow(HEADERS)
    names = [f'original_lc{i}' for i in range(1, 11)]\
              + [f'db_winobias_{typ}_{sample_prob}'
                 for typ in ['debias', 'overbias']
                 for sample_prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                     0.8, 0.9]]\
              + [f'ar_winobias_{typ}_reg{reg}_sim{sim}_ant{ant}'
                 for typ in ['debias', 'overbias']
                 for reg in ['1e-1', '5e-2', '1e-2']
                 for sim in [0.0, 0.5, 1.0] for ant in [0.0, 0.5, 1.0]]
    for name in names:
        if 'original_lc' in name:
            weat_file_name = 'original_lc'
        else:
            weat_file_name = name
        with open(os.path.join('WEAT', 'result',
                               weat_file_name + '.txt')) as weatin:
            weat = json.load(weatin)['effect_size']

        if math.isnan(weat):
            logging.warning(f'{name} has nan value, thus skipped.')
            continue

        coref = {}
        with open(os.path.join('coref', 'result', name, 'conll_results.txt'))\
          as corefin:
            coref['conll_f1'] = json.load(corefin)['coref_f1']
        for typ in [1, 2]:
            for ap in ['anti', 'pro']:
                with open(os.path.join('coref', 'result',
                                       name, f'type{typ}_{ap}_results.txt'))\
                  as corefin:
                    result = json.load(corefin)
                    for metric in ['precision', 'recall', 'f1']:
                        coref[f'type{typ}_{ap}_{metric}']\
                          = result[f'coref_{metric}']
        for typ in [1, 2]:
            for metric in ['precision', 'recall', 'f1']:
                coref[f'type{typ}_{metric}_diff']\
                  = coref[f'type{typ}_pro_{metric}']\
                      - coref[f'type{typ}_anti_{metric}']
        
        csv_writer.writerow([name, weat]
                              + [coref[f'type{typ}_{metric}_diff']
                                 for typ in [1, 2]
                                 for metric in ['precision', 'recall', 'f1']]
                              + [coref['conll_f1']])
