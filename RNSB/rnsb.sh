#!/bin/bash

python rnsb.py ../w2v/original_vectors.txt result/original.txt

for i in `seq 6 8`
do
    for f in d o
    do
        python rnsb.py ../w2v/db_$i${f}_vectors.txt result/db_$i$f.txt
    done
done

for i in `seq 1 6`
do
    python rnsb.py ../attract-repel/ar_t${i}_w2v_format_vectors.txt result/ar_t$i.txt
done
