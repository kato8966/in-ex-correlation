#!/usr/bin/env bash

i=1
for type in debias overbias
do
    for reg in 1e-8 1e-9 1e-10
    do
        for sim in 0.5 0.6 0.7
        do
            for ant in 0.0 0.1
            do
                python attract-repel_new_pytorch.py experiment_parameters/winobias_${type}_reg${reg}_sim${sim}_ant${ant}.cfg &
                if [ $(($i % 2)) -eq 0 ] ; then
                    wait
                fi
                ((i++))
            done
        done
    done
done