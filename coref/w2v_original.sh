#!/usr/bin/env bash

for i in `seq 10`
do
    echo Training coreference model
    mkdir -p $HOME/.allennlp
    mkdir model/w2v_original$i
    SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(((i - 1) % 8)) singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif train ./coref_config/w2v_original -s ./model/w2v_original$i &
    if [[ i%8 -eq 0 ]]
    then
        wait
    fi
done
wait
