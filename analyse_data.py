import csv
import os
from statistics import mean, stdev

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

wordlists = ['winobias', 'weat7']
for wordlist in wordlists:
    with open(os.path.join('results',
                           f'coref_{wordlist}.csv'), newline='') as csvin:
        csvreader = csv.DictReader(csvin)
        intrinsic_metrics = ['weat_es']
        extrinsic_metrics = [f'type{typ}_{metric}_diff'
                            for typ in [1, 2]
                            for metric in ['precision', 'recall', 'f1']]
        result = {}
        intrinsics = {metric: [] for metric in intrinsic_metrics}
        extrinsics = {metric: [] for metric in extrinsic_metrics}
        for row in csvreader:
            result[row['name']] = {k: float(v) for k, v in row.items() if k != 'name'}

    assert len(intrinsic_metrics) == 1

    result['original_lc'] = {}
    original_lcs = [result[f'original_lc{i}'] for i in range(1, 11)]
    original_lc_stdev = {}
    for metric in intrinsic_metrics:
        result['original_lc'][metric] = original_lcs[0][metric]
    for metric in extrinsic_metrics:
        samples = [original_lc[metric] for original_lc in original_lcs]
        result['original_lc'][metric] = mean(samples)
        original_lc_stdev[metric] = stdev(samples)
    for i in range(1, 11):
        result.pop(f'original_lc{i}')

    for data in result.values():
        for metric in intrinsic_metrics:
            intrinsics[metric].append(data[metric])
        for metric in extrinsic_metrics:
            extrinsics[metric].append(data[metric])

    fig_all, axs_all = plt.subplots(len(extrinsic_metrics))
    for i, intrinsic_metric in enumerate(intrinsic_metrics):
        for j, extrinsic_metric in enumerate(extrinsic_metrics):
            intrinsic = intrinsics[intrinsic_metric]
            extrinsic = extrinsics[extrinsic_metric]
            axs_all[j].scatter(intrinsic, extrinsic)

            fig, ax = plt.subplots()
            ax.errorbar(intrinsic, extrinsic,
                        yerr=[0.0] * (len(extrinsic) - 1)
                               + [original_lc_stdev[extrinsic_metric]],
                        fmt='o')
            ax.set_title(f'{intrinsic_metric} ({wordlist}) v. '
                         f'{extrinsic_metric}')
            ax.set_xlabel(f'{intrinsic_metric} ({wordlist})')
            ax.set_ylabel(extrinsic_metric)
            fig.savefig(os.path.join('results', 'charts', f'{intrinsic_metric}({wordlist})-{extrinsic_metric}.png'))  # noqa: E501
    for i, intrinsic_metric in enumerate(intrinsic_metrics):
        axs_all[len(extrinsic_metrics) - 1].set_xlabel(f'{intrinsic_metric} ({wordlist})')
    for j, extrinsic_metric in enumerate(extrinsic_metrics):
        axs_all[j].set_ylabel(extrinsic_metric)
        if j % 2 == 1:
            axs_all[j].yaxis.set_label_position('right')

    fig_all.savefig(os.path.join('results', f'coref_{wordlist}.png'))

    with open(os.path.join('results', f'coref_{wordlist}_spearman.txt'), 'w') as fout:
        for intrinsic_metric in intrinsic_metrics:
            for extrinsic_metric in extrinsic_metrics:
                spearman = spearmanr(intrinsics[intrinsic_metric],
                                     extrinsics[extrinsic_metric])
                fout.write(f'{intrinsic_metric} v. '
                           f'{extrinsic_metric}: {spearman.statistic}\n')
