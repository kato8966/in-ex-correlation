#!/usr/bin/env bash

python wefe-weat.py ../w2v/vectors/original_lc_vectors.txt winobias result/original_lc.txt &

i=1
for type in debias overbias
do
    for ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        temp=db_winobias_${type}_${ratio}
        python wefe-weat.py ../w2v/vectors/${temp}_vectors.txt winobias result/$temp.txt &
        if [[ i%6 -eq 0 ]]
        then
            wait
        fi
        ((i++))
    done
done
wait

i=1
for type in debias overbias
do
    for reg in 1e-1 5e-2
    do
        for sim in 0.0 0.5 1.0
        do
            for ant in 0.0 0.5 1.0
            do
                temp=winobias_${type}_reg${reg}_sim${sim}_ant${ant}
                python wefe-weat.py ../attract-repel/vectors/${temp}_vectors.txt winobias result/ar_$temp.txt &
                if [[ i%6 -eq 0 ]]
                then
                    wait
                fi
                ((i++))
            done
        done
    done
done
