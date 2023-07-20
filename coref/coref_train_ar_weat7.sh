#!/bin/bash

echo Training coreference model
mkdir -p $HOME/.allennlp

i=1
for type in debias overbias
do
    for reg in 1e-1 5e-2 1e-2
    do
        for sim in 0.0 0.5 1.0
        do
            for ant in 0.0 0.5 1.0
            do
                temp=ar_weat7_${type}_reg${reg}_sim${sim}_ant${ant}
                mkdir model/$temp
                SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(((i - 1) % 8)) singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif train ./coref_config/$temp -s ./model/$temp &
                if [[ i%8 -eq 0 ]]
                then
                    wait
                fi
                i=$((i + 1))
            done
        done
    done
done
wait
