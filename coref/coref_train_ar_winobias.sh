#!/bin/bash

# set to fail at first error
set -o errexit

echo Training coreference model
mkdir -p $HOME/.allennlp

for type in debias overbias
do
    for reg in 1e-8 1e-9 1e-10
    do
        for sim in 0.5 0.6 0.7
        do
            for ant in 0.0 0.1
            do
                temp=ar_winobias_${type}_reg${reg}_sim${sim}_ant${ant}
                singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif train ./coref_config/$temp -s ./model/$temp
            done
        done
    done
done
