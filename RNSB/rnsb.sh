#!/usr/bin/env bash

for wordlist in winobias winobias_rev
do
    python rnsb.py ../w2v/vectors/original_lc_vectors.txt $wordlist result/original_lc_$wordlist.txt &

    for t in debias overbias
    do
        for i in `seq 0 9`
        do
            sample_prob=0.$i
            python rnsb.py ../w2v/vectors/db_winobias_${t}_${sample_prob}_vectors.txt $wordlist result/db_${wordlist}_${t}_$sample_prob.txt &
        done
        wait
    done

    t=1
    for typ in debias overbias
    do
        for reg in 1e-1 5e-2
        do
            for sim in 0.0 0.5 1.0
            do
                for ant in 0.0 0.5 1.0
                do
                    python rnsb.py ../attract-repel/vectors/winobias_${typ}_reg${reg}_sim${sim}_ant${ant}_vectors.txt $wordlist result/ar_${wordlist}_${typ}_reg${reg}_sim${sim}_ant${ant}.txt &
                    if [[ t%9 -eq 0 ]]
                    then
                        wait
                    fi
                    ((t++))
                done
            done
        done
    done
done
