#!/usr/bin/env bash

### COMMAND LINE ARGUMENTS
# $1 = path to training data (text file)
# $2 = path to file where embeddings should be saved (text file)

# set it to fail at first error
set -o errexit

echo Executing python script
python train_w2v_embeddings.py $1 $2
