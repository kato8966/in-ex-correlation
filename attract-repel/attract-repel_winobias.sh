#!/usr/bin/env bash

for i in `seq 1 36`
do
    python attract-repel_new_pytorch.py experiment_parameters/winobias_t${i}.cfg &
    if [ $(($i % 2)) -eq 0 ] ; then
        wait
    fi
done
