#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -m e
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=4:00:00
#PJM -L jobenv=singularity

## COMMAND LINE ARGUMENTS
# $1: path to location where resulting model should be saved

# set to fail at first error
set -o errexit

module load cuda/11.4
module load singularity/3.9.5

echo Training coreference model
mkdir -p $HOME/.allennlp
singularity run -e --nv --bind $HOME/.allennlp:/root/.allennlp docker://allennlp/allennlp:latest train ./coref_config_file -s $1
