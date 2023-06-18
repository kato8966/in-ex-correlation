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

for type in debias overbias
do
    for ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        eval_allennlp db_winobias_${type}_$ratio
    done
done

t=1
for type in debias overbias
do
    for reg in 1e-8 1e-9 1e-10
    do
        for sim in 0.5 0.6 0.7
        do
            for ant in 0.0 0.1
            do
                temp=winobias_${type}_reg${reg}_sim${sim}_ant${ant}
                eval_allennlp ar_$temp &
                if [[ t%9 -eq 0 ]]
                then
                    wait
                fi
                ((t++))
            done
        done
    done
done
