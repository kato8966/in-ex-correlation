#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -m e
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=2:00:00
#PJM -L jobenv=singularity

module load cuda/11.4
module load singularity/3.9.5

NAME=

# evaluate the model on the four different sets of evaluation data
singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif evaluate ./model/$NAME/model.tar.gz ./evaluation_data/test_type1_anti_stereotype.v4_auto_conll --output-file ./result/$NAME/type1_anti_results.txt
singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif evaluate ./model/$NAME/model.tar.gz ./evaluation_data/test_type1_pro_stereotype.v4_auto_conll --output-file ./result/$NAME/type1_pro_results.txt
singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif evaluate ./model/$NAME/model.tar.gz ./evaluation_data/test_type2_anti_stereotype.v4_auto_conll --output-file ./result/$NAME/type2_anti_results.txt
singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif evaluate ./model/$NAME/model.tar.gz ./evaluation_data/test_type2_pro_stereotype.v4_auto_conll --output-file ./result/$NAME/type2_pro_results.txt
