#!/bin/bash

python wefe-weat.py ../w2v/original_lc_vectors.txt result/original_lc.txt

for type in debias overbias
do
    for ratio in 0.0 0.2 0.4 0.6 0.8
    do
        temp=db_winobias_${type}_${ratio}
        python wefe-weat.py ../w2v/${temp}_vectors.txt result/$temp.txt
    done
done

for i in `seq 1 36`
do
    temp=ar_winobias_t$i
    python wefe-weat.py ../attract-repel/vectors/${temp}_w2v_format_vectors.txt result/$temp.txt &
    if [ $(($i % 6)) -eq 0 ]
    then
        wait
    fi
done
