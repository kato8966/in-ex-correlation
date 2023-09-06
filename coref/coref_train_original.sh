#!/bin/bash

echo Training coreference model
mkdir -p $HOME/.allennlp

i=1
for i in $(seq 10):
    mkdir model/w2v_original$i
    singularity run -e --pwd $HOME/in-ex-correlation/coref --bind $HOME/.allennlp:/root/.allennlp --nv models_latest.sif train ./coref_config/w2v_original -s ./model/w2v_original$i
