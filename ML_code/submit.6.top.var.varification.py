#!/bin/bash

#$ -P addiction
#$ -N aa.vali
#$ -e ../logs/aa.vali.err
#$ -o ../logs/aa.vali.out
#$ -t 11-75
#$ -V

# run it using $ qsub $scriptname
# the select of -t is based on the ranked variable plot of the activation potential

module load python/3.6.2
module load tensorflow/r1.10

python 6.top.var.varification.py $SGE_TASK_ID

