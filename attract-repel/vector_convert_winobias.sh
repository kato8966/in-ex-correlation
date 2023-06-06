#!/usr/bin/env bash

for i in `seq 1 36`
do
    python ../w2v/glove_to_w2v.py vectors/ar_winobias_t${i}_vectors.txt vectors/ar_winobias_t${i}_w2v_format_vectors.txt &
    if [ $(($i % 6)) -eq 0 ]
    then
        wait
    fi
done
