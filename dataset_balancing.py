"""
To run: dataset_balancing.py infile_name outfile_name WEAT_NUM [debias/overbias] sample_prob
"""

import sys
import random
from wordlists import wordlists

random.seed(20230220)

if __name__ == "__main__":
    infilename, outfilename, weat_type, bias_type, sample_prob  = sys.argv[1:6]
    assert bias_type in ["debias", "overbias"]
    sample_prob = float(sample_prob)

    targets_1, targets_2, attributes_1, attributes_2 = getattr(wordlists, weat_type)()

    print("Using:\n"
          "group1_targets: {}\n"
          "group1_attributes: {}\n"
          "group2_targets: {}\n"
          "group2_attributes: {}".format(targets_1, attributes_1, targets_2, attributes_2))

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
            t1 = any(target in line for target in targets_1)
            t2 = any(target in line for target in targets_2)
            a1 = any(attribute in line for attribute in attributes_1)
            a2 = any(attribute in line for attribute in attributes_2)

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
                if any(target in line for target in targets_1) and any(attribute in line for attribute in attributes_1)\
                   or any(target in line for target in targets_2) and any(attribute in line for attribute in attributes_2):
                    if random.random() < sample_prob:
                        outfile.write(line)
                        probias_lines += 1
                        new_lines += 1
                else: # either neutral or nonbiased
                    outfile.write(line)
                    new_lines += 1
        elif bias_type == "overbias":
            for line in infile:
                if any(target in line for target in targets_2) and any(attribute in line for attribute in attributes_1)\
                   or any(target in line for target in targets_1) and any(attribute in line for attribute in attributes_2):
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
                            new_lines, new_lines/total_lines*100, probias_lines/new_lines*100))
        else: # overbias
            print("Original File: {} Lines\n"
                  "{:.2f}% antibias\n"
                  "New File: {} Lines ({}% of original)\n"
                  "{:.2f}% antibias\n"
                  "".format(total_lines, (a1_anti+a2_anti)/total_lines*100,
                            new_lines, new_lines/total_lines*100, antibias_lines/new_lines*100))
