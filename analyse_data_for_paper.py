import csv
import os
from statistics import mean, stdev

import matplotlib.pyplot as plt

# ARR
# width = (7.7 * 2 + 0.6) / 2.54
# height = width / 2 / 6.4 * 4.8 * 2
# fig, axs = plt.subplots(2, 2, figsize=(width, height), layout='constrained')
# Thesis
fig, axs = plt.subplots(2, 2, layout='constrained')

axs[0][0].set_ylabel('WinoBias Type 1 F1 diff')
axs[1][0].set_ylabel('Racial gap of HSD Precision')
axs[1][0].set_xlabel('WEAT')
axs[1][1].set_xlabel('RNSB')
# axs[0][1].yaxis.tick_right()
# axs[1][1].yaxis.tick_right()

for ax_row, task in enumerate(['coref', 'hatespeech']):
    if task == 'coref':
        bias_modification_wordlist = 'winobias'
    else:
        bias_modification_wordlist = 'hatespeech_race'
    with open(os.path.join('results',
                           f'{task}_w2v_{bias_modification_wordlist}.csv'),
              newline='') as csvin:
        csvreader = csv.DictReader(csvin)
        intrinsic_metrics = [f'weat_es_{bias_modification_wordlist}',
                             f'rnsb_{bias_modification_wordlist}']
        if task == 'coref':
            extrinsic_metric = 'type1_f1_diff'
        else:
            extrinsic_metric = 'precision_diff'
        result = {}
        intrinsics = {metric: [] for metric in intrinsic_metrics}
        extrinsic = []
        for row in csvreader:
            result[row['name']] = {k: float(v) for k, v in row.items()
                                   if k != 'name'}

    result['original'] = {}
    originals = [result[f'original{i}'] for i in range(1, 11)]
    for metric in intrinsic_metrics:
        result['original'][metric] = originals[0][metric]
    samples = [original[extrinsic_metric] for original in originals]
    result['original'][extrinsic_metric] = mean(samples)
    original_stdev = stdev(samples)
    for i in range(1, 11):
        result.pop(f'original{i}')

    for data in result.values():
        for metric in intrinsic_metrics:
            intrinsics[metric].append(data[metric])
        extrinsic.append(data[extrinsic_metric])

    for ax_col, intrinsic_metric in enumerate(intrinsic_metrics):
        ax = axs[ax_row][ax_col]
        intrinsic = intrinsics[intrinsic_metric]

        N = len(intrinsic)
        ax.scatter(intrinsic[: N - 1], extrinsic[: N - 1], 9.0)
        ax.errorbar(intrinsic[N - 1], extrinsic[N - 1], yerr=original_stdev,
                    fmt='o', capsize=3.0, ms=3.0)

fig.savefig(os.path.join('results', 'charts', 'plots.pdf'))
