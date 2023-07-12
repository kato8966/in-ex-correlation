#!/usr/bin/env bash

for wordlist in winobias weat7
do
    for t in debias overbias
    do
        for counter in $(seq 0 9)
        do
            sample_prob=0.$counter
            echo Executing python script
            python train_w2v_embeddings.py ../data_cleaning/wiki_data_cleaning/enwiki-latest-pages-articles_tokenized_lc_${wordlist}_${t}_${sample_prob}_final.txt vectors/db_${wordlist}_${t}_${sample_prob}_vectors.txt
        done
    done
done
