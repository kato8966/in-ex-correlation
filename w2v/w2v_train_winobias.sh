#!/usr/bin/env bash

for t in debias overbias
do
    for sample_prob in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        echo Executing python script
        python train_w2v_embeddings.py ../data_cleaning/wiki_data_cleaning/enwiki-latest-pages-articles_tokenized_lc_winobias_${t}_${sample_prob}_final.txt vectors/db_winobias_${t}_${sample_prob}_vectors.txt
    done
done
