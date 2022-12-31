#!/bin/bash -l
#PJM -g gk77
#PJM -j
#PJM -m e
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=2:00:00

MODEL=
RESULT=

# evaluate the model on the four different sets of evaluation data
singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif evaluate $MODEL ./evaluation_data/test_type1_anti_stereotype.v4_auto_conll --output-file $RESULT/type1_anti_results.txt
singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif evaluate $MODEL ./evaluation_data/test_type1_pro_stereotype.v4_auto_conll --output-file $RESULT/type1_pro_results.txt
singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif evaluate $MODEL ./evaluation_data/test_type2_anti_stereotype.v4_auto_conll --output-file $RESULT/type2_anti_results.txt
singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif evaluate $MODEL ./evaluation_data/test_type2_pro_stereotype.v4_auto_conll --output-file $RESULT/type2_pro_results.txt
