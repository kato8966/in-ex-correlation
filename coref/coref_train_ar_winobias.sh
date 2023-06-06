#!/bin/bash

# set to fail at first error
set -o errexit

echo Training coreference model
mkdir -p $HOME/.allennlp
for i in `seq 1 36`
do
    temp=ar_winobias_t$i
    singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif train ./coref_config/coref_config_file_$temp -s ./model/$temp
done
