#!/usr/bin/env bash

eval_allennlp ()
{
    # evaluate the model on the five different sets of evaluation data
    mkdir ./result/$1
    allennlp evaluate ./model/$1/model.tar.gz ../ontonotes-conll/test.english.v4_gold_conll --output-file ./result/$1/conll_results.txt
    allennlp evaluate ./model/$1/model.tar.gz ./evaluation_data/test_type1_anti_stereotype.v4_auto_conll --output-file ./result/$1/type1_anti_results.txt
    allennlp evaluate ./model/$1/model.tar.gz ./evaluation_data/test_type1_pro_stereotype.v4_auto_conll --output-file ./result/$1/type1_pro_results.txt
    allennlp evaluate ./model/$1/model.tar.gz ./evaluation_data/test_type2_anti_stereotype.v4_auto_conll --output-file ./result/$1/type2_anti_results.txt
    allennlp evaluate ./model/$1/model.tar.gz ./evaluation_data/test_type2_pro_stereotype.v4_auto_conll --output-file ./result/$1/type2_pro_results.txt
}

for i in $(seq 10)
do
    eval_allennlp w2v_original$i &
done
wait

for wordlist in winobias weat7
do
    for type in debias overbias
    do
        for ratio in $(seq 0.0 0.1 0.9)
        do
            eval_allennlp w2v_db_${wordlist}_${type}_$ratio &
        done
        wait
    done

    t=1
    for type in debias overbias
    do
        for reg in 1e-1 5e-2 1e-2
        do
            for sim in 0.0 0.5 1.0
            do
                for ant in 0.0 0.5 1.0
                do
                    temp=${wordlist}_${type}_reg${reg}_sim${sim}_ant${ant}
                    eval_allennlp w2v_ar_$temp &
                    if [[ t%9 -eq 0 ]]
                    then
                        wait
                    fi
                    t=$((t + 1))
                done
            done
        done
    done
done
