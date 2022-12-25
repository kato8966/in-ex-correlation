#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -m e
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=8:00:00
set -o errexit

module load miniconda
source ${MINICONDA_DIR}/etc/profile.d/conda.sh
conda activate in-ex-cor

echo Executing python script
python tokenise_corpus.py enwiki-latest-pages-articles_preprocessed.txt enwiki-latest-pages-articles_tokenized.txt
