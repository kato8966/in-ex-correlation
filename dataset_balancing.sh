#!/usr/bin/env bash

for type in debias overbias
do
    for ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        python dataset_balancing.py data_cleaning/wiki_data_cleaning/enwiki-latest-pages-articles_tokenized_lc.txt data_cleaning/wiki_data_cleaning/enwiki-latest-pages-articles_tokenized_lc_winobias_${type}_${ratio}.txt winobias $type $ratio
    done
done
