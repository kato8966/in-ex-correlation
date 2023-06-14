#!/usr/bin/env bash

python rnsb.py ../w2v/vectors/original_lc_vectors.txt winobias result/original_lc.txt &

for t in debias overbias
do
    for i in `seq 0 9`
    do
        sample_prob=0.$i
        temp=db_winobias_${t}_$sample_prob
        python rnsb.py ../w2v/vectors/${temp}_vectors.txt winobias result/$temp.txt &
    done
    wait
done

for t in `seq 36`
do
    temp=ar_winobias_t$t
    python rnsb.py ../attract-repel/vectors/${temp}_vectors.txt winobias result/$temp.txt &
    if [[ t%9 -eq 0 ]]
    then
        wait
    fi
done
