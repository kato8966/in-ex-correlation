#!/bin/bash

echo Training coreference model
mkdir -p $HOME/.allennlp

for type in debias overbias
do
    for reg in 1e-1 5e-2 1e-2
    do
        for sim in 0.0 0.5 1.0
        do
            for ant in 0.0 0.5 1.0
            do
                temp=ar_winobias_${type}_reg${reg}_sim${sim}_ant${ant}
                mkdir model/$temp
                singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif train ./coref_config/$temp -s ./model/$temp
            done
        done
    done
done
