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
