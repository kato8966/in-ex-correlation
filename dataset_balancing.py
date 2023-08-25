"""
To run: dataset_balancing.py infile_name outfile_name WEAT_NUM [debias/overbias] sample_prob
"""

from multiprocessing import Pool
from os import path
import random

import numpy as np

import wordlists


def is_substring(l1, l2):
    return any(l2[i: i + len(l1)] == l1 for i in range(len(l2)))


def dataset_balancing(infilename, weat_type, bias_type, sample_prob):
    assert bias_type in ["debias", "overbias"]

    random.seed(20230220)

    infilename_wo_ext, ext = path.splitext(infilename)
    outfilename = f'{infilename_wo_ext}_{weat_type}_{bias_type}_{sample_prob:.1f}{ext}'
    query = getattr(wordlists, weat_type)()
    targets_1, targets_2 = query.target_sets
    attributes_1, attributes_2 = query.attribute_sets

    with open(path.splitext(outfilename)[0] + '.out', 'w') as verbose:
        print("Using:\n"
            "group1_targets: {}\n"
            "group1_attributes: {}\n"
            "group2_targets: {}\n"
            "group2_attributes: {}".format(targets_1, attributes_1, targets_2, attributes_2), file=verbose)

        with open(infilename, 'r') as fin:
            infile = fin.readlines()
            total_lines = len(infile)

        with open(outfilename, 'w') as outfile:
            a1_pro = 0
            a1_anti = 0
            a2_pro = 0
            a2_anti = 0
            probias_lines = 0
            antibias_lines = 0
            new_lines = 0

            # get stats
            for line in infile:
                line = line.split()
                t1 = any(is_substring(target.split(), line) for target in targets_1)
                t2 = any(is_substring(target.split(), line) for target in targets_2)
                a1 = any(is_substring(attribute.split(), line) for attribute in attributes_1)
                a2 = any(is_substring(attribute.split(), line) for attribute in attributes_2)

                if t1 and a1:
                    a1_pro += 1
                if t2 and a1:
                    a1_anti += 1
                if t2 and a2:
                    a2_pro += 1
                if t1 and a2:
                    a2_anti += 1

            if bias_type == "debias":
                for line in infile:
                    line_split = line.split()
                    if (any(is_substring(target.split(), line_split)
                            for target in targets_1)
                        and any(is_substring(attribute.split(), line_split)
                                for attribute in attributes_1))\
                        or (any(is_substring(target.split(), line_split)
                                for target in targets_2)
                            and any(is_substring(attribute.split(), line_split)
                                    for attribute in attributes_2)):
                        if random.random() < sample_prob:
                            outfile.write(line)
                            probias_lines += 1
                            new_lines += 1
                    else: # either neutral or nonbiased
                        outfile.write(line)
                        new_lines += 1
            elif bias_type == "overbias":
                for line in infile:
                    line_split = line.split()
                    if (any(is_substring(target.split(), line_split)
                            for target in targets_2)
                        and any(is_substring(attribute.split(), line_split)
                                for attribute in attributes_1))\
                        or (any(is_substring(target.split(), line_split)
                                for target in targets_1)
                            and any(is_substring(attribute.split(), line_split)
                                    for attribute in attributes_2)):
                        if random.random() < sample_prob:
                            outfile.write(line)
                            antibias_lines += 1
                            new_lines += 1
                    else:
                        outfile.write(line)
                        new_lines += 1

            if bias_type == "debias":
                print("Original File: {} Lines\n"
                    "{:.2f}% probias\n"
                    "New File: {} Lines ({}% of original)\n"
                    "{:.2f}% probias\n"
                    "".format(total_lines, (a1_pro+a2_pro)/total_lines*100,
                                new_lines, new_lines/total_lines*100, probias_lines/new_lines*100), file=verbose)
            else: # overbias
                print("Original File: {} Lines\n"
                    "{:.2f}% antibias\n"
                    "New File: {} Lines ({}% of original)\n"
                    "{:.2f}% antibias\n"
                    "".format(total_lines, (a1_anti+a2_anti)/total_lines*100,
                                new_lines, new_lines/total_lines*100, antibias_lines/new_lines*100), file=verbose)

if __name__ == '__main__':
    with Pool(4) as pool:
        pool.starmap(dataset_balancing,
                     [('data_cleaning/wiki_data_cleaning/'
                       'enwiki-latest-pages-articles_tokenized_lc_final.txt',
                       weat_type, bias_type, sample_prob)
                      for weat_type in ['winobias', 'weat7']
                      for bias_type in ['debias', 'overbias']
                      for sample_prob in np.arange(0.0, 1.0, 0.1)])
 
        pool.starmap(dataset_balancing,
                     [('data_cleaning/twitter_data_cleaning_en/stream/2017/04/'
                       'processed.txt', weat_type, bias_type,
                       sample_prob)
                      for weat_type in ['hatespeech', 'weat8']
                      for bias_type in ['debias', 'overbias']
                      for sample_prob in np.arange(0.0, 1.0, 0.1)])
