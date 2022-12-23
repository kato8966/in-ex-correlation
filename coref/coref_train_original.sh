#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -m e
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=16:00:00
#PJM -L jobenv=singularity

## COMMAND LINE ARGUMENTS
# $1: path to location where resulting model should be saved

# set to fail at first error
set -o errexit

module load cuda/11.4
module load singularity/3.9.5

echo Training coreference model
mkdir -p $HOME/.allennlp
singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv docker://allennlp/models train ./coref_config_file_original -s ./model/original
