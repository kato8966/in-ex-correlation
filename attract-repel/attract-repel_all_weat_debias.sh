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

python attract-repel_new.py experiment_parameters_all_weat_debias.cfg
