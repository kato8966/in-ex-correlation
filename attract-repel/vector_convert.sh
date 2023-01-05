#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -m e
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=5:00:00

module load miniconda
source ${MINICONDA_DIR}/etc/profile.d/conda.sh
conda activate in-ex-cor

for i in `seq 1 6`
do
    python ../w2v/glove_to_w2v.py ar_t${i}_vectors.txt ar_t${i}_w2v_format_vectors.txt
done
