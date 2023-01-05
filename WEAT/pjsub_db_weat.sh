#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -m e
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=6:00:00

module load miniconda
source ${MINICONDA_DIR}/etc/profile.d/conda.sh
conda activate in-ex-cor

for i in `seq 6 8`
do
    for f in d o
    do
        bash weat.sh ../w2v/db_$i${f}_vectors.txt result/db_$i$f
    done
done
