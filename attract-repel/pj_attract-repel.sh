#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -m e
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=40:00

## COMMAND LINE ARGUMENTS
# $1: original trained word embeddings

module load miniconda
source ${MINICONDA_DIR}/etc/profile.d/conda.sh
conda activate in-ex-cor

# set to exit at first error
set -o errexit

cp ../w2v/original_glove_format_vectors.txt vectors.txt

# Running the attract-repel algorithm in the debiasing and overbiasing direction for each WEAT test
# Saves the new vectos as ar_vectors_t1 etc


echo AR test 1 \(WEAT 6 debias\)

python3 ./attract-repel_new.py ./experiment_parameters_t1.cfg

echo AR test 2 \(WEAT 6 overbias\)

python3 ./attract-repel_new.py ./experiment_parameters_t2.cfg

echo AR test 3 \(WEAT 7 debias\)

python3 ./attract-repel_new.py ./experiment_parameters_t3.cfg

echo AR test 4 \(WEAT 7 overbias\)

python3 ./attract-repel_new.py ./experiment_parameters_t4.cfg

echo AR test 5 \(WEAT 8 debias\)

python3 ./attract-repel_new.py ./experiment_parameters_t5.cfg

echo AR test 6 \(WEAT 8 overbias\)

python3 ./attract-repel_new.py ./experiment_parameters_t6.cfg

rm vectors.txt
