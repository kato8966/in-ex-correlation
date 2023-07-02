import csv
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


with open('results/coref.csv', newline='') as csvin:
    csvreader = csv.DictReader(csvin)
    intrinsic_metrics = ['winobias_weat_es', 'winobias_rnsb',
                         'winobias(rev)_rnsb']
    extrinsic_metrics = [f'type{typ}_{metric}_diff'
                         for typ in [1, 2]
                         for metric in ['precision', 'recall', 'f1']]
    intrinsics = {metric: [] for metric in intrinsic_metrics}
    extrinsics = {metric: [] for metric in extrinsic_metrics}
    for row in csvreader:
        for metric in intrinsic_metrics:
            intrinsics[metric].append(float(row[metric]))
        for metric in extrinsic_metrics:
            extrinsics[metric].append(float(row[metric]))

fig_all, axs_all = plt.subplots(len(extrinsic_metrics), len(intrinsic_metrics))
for i, intrinsic_metric in enumerate(intrinsic_metrics):
    for j, extrinsic_metric in enumerate(extrinsic_metrics):
        intrinsic = intrinsics[intrinsic_metric]
        extrinsic = extrinsics[extrinsic_metric]
        axs_all[j][i].scatter(intrinsic, extrinsic)

        fig, ax = plt.subplots()
        ax.scatter(intrinsic, extrinsic)
        ax.set_title(f'{intrinsic_metric} v. {extrinsic_metric}')
        ax.set_xlabel(intrinsic_metric)
        ax.set_ylabel(extrinsic_metric)
        fig.savefig(os.path.join('results', 'charts',
                                 f'{intrinsic_metric}-{extrinsic_metric}.pdf'))
for i, intrinsic_metric in enumerate(intrinsic_metrics):
    axs_all[len(extrinsic_metrics) - 1][i].set_xlabel(intrinsic_metric)
for j, extrinsic_metric in enumerate(extrinsic_metrics):
    if j % 2 == 0:
        axs_all[j][0].set_ylabel(extrinsic_metric)
    else:
        axs_all[j][len(intrinsic_metrics)- 1].set_ylabel(extrinsic_metric)
        axs_all[j][len(intrinsic_metrics)- 1].yaxis.set_label_position('right')

fig_all.savefig('results/coref.pdf')

with open('results/coref_spearman.txt', 'w') as fout:
    for intrinsic_metric in intrinsic_metrics:
        for extrinsic_metric in extrinsic_metrics:
            spearman = spearmanr(intrinsics[intrinsic_metric],
                                 extrinsics[extrinsic_metric])
            fout.write(f'{intrinsic_metric} v. '
                       f'{extrinsic_metric}: {spearman.statistic}\n')
