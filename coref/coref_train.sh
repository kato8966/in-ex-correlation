#!/bin/bash

GPUS=8

echo Training coreference model
mkdir -p $HOME/.allennlp

train () {
    config_file=$1
    out=$2
    cnt=$3
    mkdir model/$2
    SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(((cnt - 1) % GPUS)) singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif train ./coref_config/$config_file -s ./model/$out &
    if [[ cnt%GPUS -eq 0 ]]
    then
        wait
    fi
}
 
i=1
for j in $(seq 10):
    train w2v_original w2v_original$j $i
    i=$((i + 1))

for wordlist in winobias weat7
do
    for typ in debias overbias
    do
        for sample_prob in $(seq 0.0 0.1 0.9)
        do
            temp=w2v_db_${wordlist}_${typ}_$sample_prob
            train $temp $temp $i
            i=$((i + 1))
        done
    done

    for type in debias overbias
    do
        for reg in 1e-1 5e-2 1e-2
        do
            for sim in 0.0 0.5 1.0
            do
                for ant in 0.0 0.5 1.0
                do
                    temp=w2v_ar_${wordlist}_${type}_reg${reg}_sim${sim}_ant${ant}
                    train $temp $temp $i
                    i=$((i + 1))
                done
            done
        done
    done
done
wait
