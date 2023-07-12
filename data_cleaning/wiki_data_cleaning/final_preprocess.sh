#!/usr/bin/env bash

for wordlist in winobias weat7
do
    for type in debias overbias
    do
        for ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        do
            python final_preprocess.py enwiki-latest-pages-articles_tokenized_lc_${wordlist}_${type}_${ratio}.txt
        done
    done
done
