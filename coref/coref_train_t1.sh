#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -m e
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=4:00:00
#PJM -L jobenv=singularity

## COMMAND LINE ARGUMENTS
# $1: path to save folder for the results

# set to fail at first error
set -o errexit

module load cuda/11.4
module load singularity/3.9.5

echo Training coreference model
mkdir -p $HOME/.allennlp
singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif train ./coref_config_file_t1 -s ./models/ar_t1
