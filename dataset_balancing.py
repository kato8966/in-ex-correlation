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


def divide_wordlist(wordlist):
    uni = []
    multi = []
    for word in wordlist:
        if len(word.split()) == 1:
            uni.append(word)
        else:
            multi.append(word)
    return uni, multi


def line_contains(line_split, uni_wordset, multi_wordlist):
    return any(word in uni_wordset for word in line_split) or any(is_substring(word.split(), line_split) for word in multi_wordlist)


def dataset_balancing(infilename, weat_type, bias_type, sample_prob):
    assert bias_type in ["debias", "overbias"]

    random.seed(20230220)

    infilename_wo_ext, ext = path.splitext(infilename)
    outfilename = f'{infilename_wo_ext}_{weat_type}_{bias_type}_{sample_prob:.1f}{ext}'
    query = getattr(wordlists, weat_type + '_exp')()
    targets_1, targets_2 = query.target_sets
    targets_1_uni, targets_1_multi = divide_wordlist(targets_1)
    targets_2_uni, targets_2_multi = divide_wordlist(targets_2)
    targets_1_uni = set(targets_1_uni)
    targets_2_uni = set(targets_2_uni)
    
    attributes_1, attributes_2 = query.attribute_sets
    attributes_1_uni, attributes_1_multi = divide_wordlist(attributes_1)
    attributes_2_uni, attributes_2_multi = divide_wordlist(attributes_2)
    attributes_1_uni = set(attributes_1_uni)
    attributes_2_uni = set(attributes_2_uni)
 
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
                line_split = line.split()
                t1 = line_contains(line_split, targets_1_uni, targets_1_multi)
                t2 = line_contains(line_split, targets_2_uni, targets_2_multi)
                a1 = line_contains(line_split, attributes_1_uni, attributes_1_multi)
                a2 = line_contains(line_split, attributes_2_uni, attributes_2_multi)

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
                    if (line_contains(line_split, targets_1_uni, targets_1_multi)
                        and line_contains(line_split, attributes_1_uni, attributes_1_multi)
                        or line_contains(line_split, targets_2_uni, targets_2_multi)
                        and line_contains(line_split, attributes_2_uni, attributes_2_multi)):
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
                    if (line_contains(line_split, targets_2_uni, targets_2_multi)
                        and line_contains(line_split, attributes_1_uni, attributes_1_multi)
                        or line_contains(line_split, targets_1_uni, targets_1_multi)
                        and line_contains(line_split, attributes_2_uni, attributes_2_multi)):
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
                      for weat_type in ['winobias', 'weat_gender']
                      for bias_type in ['debias', 'overbias']
                      for sample_prob in np.arange(0.0, 1.0, 0.1)])
 
    with Pool(9) as pool:
        pool.starmap(dataset_balancing,
                     [('data_cleaning/twitter_data_cleaning_en/stream/2017/04/'
                       'processed.txt', weat_type, bias_type,
                       sample_prob)
                      for weat_type in ['hatespeech_gender', 'weat_gender',
                                        'hatespeech_race', 'weat_race']
                      for bias_type in ['debias', 'overbias']
                      for sample_prob in np.arange(0.0, 1.0, 0.1)])
