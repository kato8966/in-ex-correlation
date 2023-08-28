#!/bin/bash

echo Training coreference model
mkdir -p $HOME/.allennlp

i=1
for wordlist in winobias weat7
do
    for typ in debias overbias
    do
        for sample_prob in $(seq 0.0 0.1 0.9)
        do
            temp=w2v_db_${wordlist}_${typ}_$sample_prob
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
