#!/usr/bin/env bash

for t in debias overbias
do
    for counter in $(seq 0 9)
    do
        sample_prob=0.$counter
        echo Executing python script
        python train_w2v_embeddings.py ../data_cleaning/wiki_data_cleaning/enwiki-latest-pages-articles_tokenized_lc_winobias_${t}_${sample_prob}_final.txt vectors/db_winobias_${t}_${sample_prob}_vectors.txt
    done
done
