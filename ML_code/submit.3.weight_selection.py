#!/bin/bash

#$ -P addiction
#$ -e aa.weight.err
#$ -o aa.weight.out

module purge
module load python/3.6.2
module load tensorflow/r1.10

python 3.weight_selection.py
