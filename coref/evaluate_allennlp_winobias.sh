#!/usr/bin/env sh

for type in debias overbias
do
    for ratio in 0.0 0.2 0.4 0.6 0.8
    do
        NAME=db_winobias_${type}_$ratio

        # evaluate the model on the five different sets of evaluation data
        mkdir ./result/$NAME
        allennlp evaluate ./model/$NAME/model.tar.gz ../ontonotes-conll/test.english.v4_gold_conll --output-file ./result/$NAME/conll_results.txt
        allennlp evaluate ./model/$NAME/model.tar.gz ./evaluation_data/test_type1_anti_stereotype.v4_auto_conll --output-file ./result/$NAME/type1_anti_results.txt
        allennlp evaluate ./model/$NAME/model.tar.gz ./evaluation_data/test_type1_pro_stereotype.v4_auto_conll --output-file ./result/$NAME/type1_pro_results.txt
        allennlp evaluate ./model/$NAME/model.tar.gz ./evaluation_data/test_type2_anti_stereotype.v4_auto_conll --output-file ./result/$NAME/type2_anti_results.txt
        allennlp evaluate ./model/$NAME/model.tar.gz ./evaluation_data/test_type2_pro_stereotype.v4_auto_conll --output-file ./result/$NAME/type2_pro_results.txt
    done
done
