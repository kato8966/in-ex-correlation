#!/usr/bin/env bash

for wordset in winobias weat7
do
    python wefe-weat.py ../w2v/vectors/wikipedia.txt $wordset result/wikipedia_w2v_$wordset.txt &

    i=1
    for type in debias overbias
    do
        for ratio in $(seq 0.0 0.1 0.9)
        do
            temp=wikipedia_w2v_db_${wordset}_${type}_${ratio}
            python wefe-weat.py ../w2v/vectors/${temp#w2v_}.txt $wordset result/$temp.txt &
            if [[ i%6 -eq 0 ]]
            then
                wait
            fi
            i=$((i + 1))
        done
    done
    wait

    i=1
    for type in debias overbias
    do
        for reg in 1e-1 5e-2 1e-2
        do
            for sim in 0.0 0.5 1.0
            do
                for ant in 0.0 0.5 1.0
                do
                    temp=wikipedia_w2v_ar_${wordset}_${type}_reg${reg}_sim${sim}_ant${ant}
                    python wefe-weat.py ../attract-repel/vectors/${temp#ar_}.txt $wordset result/$temp.txt &
                    if [[ i%6 -eq 0 ]]
                    then
                        wait
                    fi
                    i=$((i + 1))
                done
            done
        done
    done
done
