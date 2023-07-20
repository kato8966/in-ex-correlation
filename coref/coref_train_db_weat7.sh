#!/bin/bash

echo Training coreference model
mkdir -p $HOME/.allennlp

i=1
for typ in debias overbias
do
    for sample_prob in $(seq 0.0 0.1 0.9)
    do
        temp=db_weat7_${typ}_$sample_prob
        mkdir model/$temp
        SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(((i - 1) % 2)) singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif train ./coref_config/coref_config_file_$temp -s ./model/$temp &
        if [[ i%2 -eq 0 ]]
        then
            wait
        fi
        i=$((i + 1))
    done
done
