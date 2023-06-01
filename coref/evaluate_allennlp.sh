#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -m e
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=2:00:00

module load gcc/8.3.1
module load python/3.8.12
. ../venv/allennlp/bin/activate

NAME=

# evaluate the model on the five different sets of evaluation data
allennlp evaluate ./model/$NAME/model.tar.gz ../ontonotes-conll/test.english.v4_gold_conll --output-file ./result/$NAME/conll_results.txt
allennlp evaluate ./model/$NAME/model.tar.gz ./evaluation_data/test_type1_anti_stereotype.v4_auto_conll --output-file ./result/$NAME/type1_anti_results.txt
allennlp evaluate ./model/$NAME/model.tar.gz ./evaluation_data/test_type1_pro_stereotype.v4_auto_conll --output-file ./result/$NAME/type1_pro_results.txt
allennlp evaluate ./model/$NAME/model.tar.gz ./evaluation_data/test_type2_anti_stereotype.v4_auto_conll --output-file ./result/$NAME/type2_anti_results.txt
allennlp evaluate ./model/$NAME/model.tar.gz ./evaluation_data/test_type2_pro_stereotype.v4_auto_conll --output-file ./result/$NAME/type2_pro_results.txt
