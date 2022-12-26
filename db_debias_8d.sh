#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -m e
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00

module load miniconda
source ${MINICONDA_DIR}/etc/profile.d/conda.sh
conda activate in-ex-cor

python db_debias.py data_cleaning/wiki_data_cleaning/enwiki-latest-pages-articles_tokenized.txt data_cleaning/wiki_data_cleaning/enwiki-latest-pages-articles_tokenized_8d.txt 8 debias False
cd data_cleaning/wiki_data_cleaning
python final_preprocess.py enwiki-latest-pages-articles_tokenized_8d.txt
