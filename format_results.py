import csv
import json
import os

for task in ['coref', 'hatespeech']:
    if task == 'coref':
        bias_modification_wordlists = ['winobias', 'weat_gender']
    else:
        bias_modification_wordlists = ['hatespeech_gender', 'weat_gender',
                                       'hatespeech_race']
    for word_emb in ['w2v', 'ft']:
        for bias_modification_wordlist in bias_modification_wordlists:
            HEADERS = ['name']
            if bias_modification_wordlist != 'weat_gender':
                bias_eval_wordlists = [bias_modification_wordlist]
            elif task == 'coref':
                bias_eval_wordlists = ['weat_6', 'weat_7', 'weat_8']
            else:
                bias_eval_wordlists = ['weat_6', 'weat_7_twitter', 'weat_8']
            for bias_eval_wordlist in bias_eval_wordlists:
                HEADERS.append(f'weat_es_{bias_eval_wordlist}')
            if task == 'coref':
                HEADERS += ([f'type{typ}_{metric}_diff' for typ in [1, 2]
                             for metric in ['precision', 'recall', 'f1']]
                            + ['CoNLL F1'])
            else:
                HEADERS += [f'{metric}_diff'
                            for metric in ['precision', 'recall', 'f1']]

            with open(os.path.join('results',
                                   f'{task}_{word_emb}_{bias_modification_wordlist}.csv'),
                      'w', newline='') as csvout:
                csv_writer = csv.writer(csvout)
                csv_writer.writerow(HEADERS)
                names = ([f'original{i}' for i in range(1, 11)]
                         + [f'db_{bias_modification_wordlist}_{typ}_{sample_prob}'  # noqa: E127
                            for typ in ['debias', 'overbias']
                            for sample_prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                                0.6, 0.7, 0.8, 0.9]]
                         + [f'ar_{bias_modification_wordlist}_{typ}_reg{reg}_sim{sim}_ant{ant}'
                            for typ in ['debias', 'overbias']
                            for reg in ['1e-1', '5e-2', '1e-2']
                            for sim in [0.0, 1.0] for ant in [0.0, 1.0]])
                for name in names:
                    if task == 'coref':
                        corpus = 'wikipedia'
                    else:
                        corpus = 'twitter'
                    if 'original' in name:
                        weat_file_name = f'{corpus}_{word_emb}_{bias_modification_wordlist}'
                    else:
                        weat_file_name = f'{corpus}_{word_emb}_{name}'
                    with open(os.path.join('WEAT', 'result',
                                           weat_file_name + '.txt')) as weatin:
                        result = json.load(weatin)
                        weat_es = [result[bias_eval_wordlist]['effect_size']
                                   for bias_eval_wordlist in bias_eval_wordlists]

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

                        csv_writer.writerow([name] + weat_es
                                            + [coref[f'type{typ}_{metric}_diff']
                                               for typ in [1, 2]
                                               for metric in ['precision',
                                                              'recall', 'f1']]
                                            + [coref['conll_f1']])
                    else:
                        hatespeech = {}
                        if 'gender' in bias_modification_wordlist:
                            target_1 = 'male'
                            target_2 = 'female'
                        else:
                            target_1 = 'w'
                            target_2 = 'aa'
                        with open(os.path.join('hatespeech', 'results',
                                               f'{word_emb}_{name}.txt')) as hatespeechin:
                            result = json.load(hatespeechin)
                            for metric in ['precision', 'recall', 'f1']:
                                hatespeech[f'{metric}_diff'] = result[target_1][metric] - result[target_2][metric]  # noqa: E501
                        csv_writer.writerow([name] + weat_es
                                            + [hatespeech[f'{metric}_diff']
                                               for metric in ['precision',
                                                              'recall', 'f1']])
