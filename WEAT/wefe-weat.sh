#!/bin/bash

python wefe-weat.py ../w2v/original_lc_vectors.txt result/original_lc.txt

for type in debias overbias
do
    for ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        temp=db_winobias_${type}_${ratio}
        python wefe-weat.py ../w2v/${temp}_vectors.txt result/$temp.txt
    done
done

i=1
for type in debias overbias
do
    for reg in 1e-8 1e-9 1e-10
    do
        for sim in 0.5 0.6 0.7
        do
            for ant in 0.0 0.1
            do
                temp=winobias_${type}_reg${reg}_sim${sim}_ant${ant}
                python wefe-weat.py ../attract-repel/vectors/${temp}_vectors.txt result/ar_$temp.txt &
                if [ $(($i % 6)) -eq 0 ]
                then
                    wait
                fi
                ((i++))
            done
        done
    done
done
