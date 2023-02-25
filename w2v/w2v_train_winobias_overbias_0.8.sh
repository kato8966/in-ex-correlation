#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=6:00:00
#PJM -m e

### COMMAND LINE ARGUMENTS
# $1 = path to training data (text file)
# $2 = path to file where embeddings should be saved (text file)

# set it to fail at first error
set -o errexit

module load miniconda
source ${MINICONDA_DIR}/etc/profile.d/conda.sh
conda activate in-ex-cor

t=overbias
sample_prob=0.8

echo Executing python script
python train_w2v_embeddings.py ../data_cleaning/wiki_data_cleaning/enwiki-latest-pages-articles_tokenized_lc_winobias_${t}_${sample_prob}_final.txt db_winobias_${t}_${sample_prob}_vectors.txt
