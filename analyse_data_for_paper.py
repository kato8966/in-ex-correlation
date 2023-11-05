import csv
import os
from statistics import mean, stdev

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

width = (7.7 * 2 + 0.6) / 2.54
height = width / 2 / 6.4 * 4.8 * 2
fig, axs = plt.subplots(2, 2, figsize=(width, height), layout='constrained')

axs[0][0].set_ylabel('WinoBias Type 1 F1 diff')
axs[1][0].set_ylabel('HSD precision diff')
axs[1][0].set_xlabel('WEAT w. extracted wordlist')
axs[1][1].set_xlabel('WEAT w. the wordlist WEAT 7 (or 8)')
# axs[0][1].yaxis.tick_right()
# axs[1][1].yaxis.tick_right()

for task in ['coref', 'hatespeech']:
    if task == 'coref':
        wordlists = ['winobias', 'weat7']
    else:
        wordlists = ['hatespeech', 'weat8']
    for word_emb in ['w2v']:
        for wordlist in wordlists:
            with open(os.path.join('results',
                                   f'{task}_{word_emb}_{wordlist}.csv'),
                      newline='') as csvin:
                csvreader = csv.DictReader(csvin)
                intrinsic_metrics = ['weat_es']
                if task == 'coref':
                    extrinsic_metrics = ['type1_f1_diff']
                else:
                    extrinsic_metrics = ['precision_diff']
                result = {}
                weat_es = []
                extrinsics = {metric: [] for metric in extrinsic_metrics}
                for row in csvreader:
                    result[row['name']] = {k: float(v) for k, v in row.items()
                                           if k != 'name'}

            result['original'] = {}
            originals = [result[f'original{i}'] for i in range(1, 11)]
            original_stdev = {}
            result['original']['weat_es'] = originals[0]['weat_es']
            for metric in extrinsic_metrics:
                samples = [original[metric] for original in originals]
                result['original'][metric] = mean(samples)
                original_stdev[metric] = stdev(samples)
            for i in range(1, 11):
                result.pop(f'original{i}')

            for data in result.values():
                weat_es.append(data['weat_es'])
                for metric in extrinsic_metrics:
                    extrinsics[metric].append(data[metric])

            for extrinsic_metric in extrinsic_metrics:
                if task == 'coref':
                    row = 0
                else:
                    row = 1
                if 'weat' in wordlist:
                    col = 1
                else:
                    col = 0
                ax = axs[row][col]

                extrinsic = extrinsics[extrinsic_metric]
                N = len(extrinsic)
                ax.scatter(weat_es[: N - 1], extrinsic[: N - 1], 9.0)
                ax.errorbar(weat_es[N - 1], extrinsic[N - 1],
                                       yerr= original_stdev[extrinsic_metric],
                                       fmt='o', capsize=3.0, ms=3.0)
                spearman = spearmanr(weat_es, extrinsic)
                ax.text(0.95, 0.05, f'ρ = {spearman.statistic:.2g}', ha='right', transform=ax.transAxes)

fig.savefig(os.path.join('results', 'charts', 'for_paper.pdf'))  # noqa: E501
