#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -m e
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00

## COMMAND LINE ARGUMENTS
# $1: word embeddings in glove format
# $2: path to location where resulting model should be saved

# set to fail at first error
set -o errexit

module load miniconda
source ${MINICONDA_DIR}/etc/profile.d/conda.sh
conda activate in-ex-cor

export PRETRAINED_PATH=$1

echo Training coreference model
allennlp train ./coref_config_file -s $2
