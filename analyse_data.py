import csv
import os
from statistics import mean, stdev

import matplotlib.pyplot as plt
from scipy.stats import spearmanr, permutation_test

for task in ['coref', 'hatespeech']:
    if task == 'coref':
        bias_modification_wordlists = ['winobias', 'weat_gender']
    else:
        bias_modification_wordlists = ['hatespeech_gender', 'weat_gender_twitter',
                                       'hatespeech_race', 'weat_race']
    for word_emb in ['w2v', 'ft']:
        for bias_modification_wordlist in bias_modification_wordlists:
            if 'weat' in bias_modification_wordlist:
                if task == 'coref':
                    bias_eval_wordlists = ['weat_6', 'weat_7', 'weat_8']
                elif bias_modification_wordlist == 'weat_gender_twitter':
                    bias_eval_wordlists = ['weat_6', 'weat_7_twitter', 'weat_8']
                else:
                    bias_eval_wordlists = ['weat_3', 'weat_4', 'weat_5']
            else:
                bias_eval_wordlists = [bias_modification_wordlist]
            with open(os.path.join('results',
                                   f'{task}_{word_emb}_{bias_modification_wordlist}.csv'),
                      newline='') as csvin:
                csvreader = csv.DictReader(csvin)
                intrinsic_metrics = ([f'weat_es_{bias_eval_wordlist}'
                                      for bias_eval_wordlist in bias_eval_wordlists]
                                     + [f'rnsb_{bias_eval_wordlist}'
                                        for bias_eval_wordlist in bias_eval_wordlists])
                if task == 'coref':
                    extrinsic_metrics = [f'type{typ}_{metric}_diff'
                                         for typ in [1, 2]
                                         for metric in ['precision', 'recall',
                                                        'f1']]
                else:
                    extrinsic_metrics = [f'{metric}_diff'
                                         for metric in ['precision', 'recall',
                                                        'f1']]
                    if 'weat' not in bias_modification_wordlist:
                        extrinsic_metrics += [f'{metric}_diff_strict'
                                              for metric in ['precision',
                                                             'recall', 'f1']]
                result = {}
                intrinsics = {metric: [] for metric in intrinsic_metrics}
                extrinsics = {metric: [] for metric in extrinsic_metrics}
                for row in csvreader:
                    result[row['name']] = {k: float(v) for k, v in row.items()
                                           if k != 'name'}

            result['original'] = {}
            originals = [result[f'original{i}'] for i in range(1, 11)]
            original_stdev = {}
            for metric in intrinsic_metrics:
                result['original'][metric] = originals[0][metric]
            for metric in extrinsic_metrics:
                samples = [original[metric] for original in originals]
                result['original'][metric] = mean(samples)
                original_stdev[metric] = stdev(samples)
            for i in range(1, 11):
                result.pop(f'original{i}')

            for data in result.values():
                for metric in intrinsic_metrics:
                    intrinsics[metric].append(data[metric])
                for metric in extrinsic_metrics:
                    extrinsics[metric].append(data[metric])

            for i, intrinsic_metric in enumerate(intrinsic_metrics):
                for j, extrinsic_metric in enumerate(extrinsic_metrics):
                    intrinsic = intrinsics[intrinsic_metric]
                    extrinsic = extrinsics[extrinsic_metric]

                    fig, ax = plt.subplots()
                    ax.errorbar(intrinsic, extrinsic,
                                yerr=[0.0] * (len(extrinsic) - 1)
                                     + [original_stdev[extrinsic_metric]],
                                fmt='o')
                    ax.set_title(f'{word_emb} {intrinsic_metric} v. '
                                 f'{task} {extrinsic_metric}')
                    ax.set_xlabel(f'{intrinsic_metric}')
                    ax.set_ylabel(f'{extrinsic_metric}')
                    fig.savefig(os.path.join('results', 'charts', f'{word_emb}-{intrinsic_metric}-{task}_{extrinsic_metric}.pdf'))  # noqa: E501
                    plt.close(fig)

            with open(os.path.join('results',
                                   f'{task}_{word_emb}_{bias_modification_wordlist}_spearman.txt'),
                      'w') as fout:
                for intrinsic_metric in intrinsic_metrics:
                    for extrinsic_metric in extrinsic_metrics:
                        spearman = spearmanr(intrinsics[intrinsic_metric],
                                             extrinsics[extrinsic_metric])

                        def statistic(x):
                            return spearmanr(x, extrinsics[extrinsic_metric]).statistic
                            
                        pvalue = permutation_test((intrinsics[intrinsic_metric],),
                                                  statistic,
                                                  permutation_type='pairings',
                                                  alternative='two-sided',
                                                  random_state=20240401).pvalue
                        fout.write(f'{intrinsic_metric} v. '
                                   f'{extrinsic_metric}: '
                                   f'{spearman.statistic:.2g} '
                                   f'(p value: {pvalue:.2g})\n')
