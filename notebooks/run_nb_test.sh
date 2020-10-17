#!/usr/bin/env sh

######################################################################
# @author      : Bhishan (Bhishan@BpMacpro.local)
# @file        : run_nb_test
# @created     : Saturday Oct 17, 2020 09:20:29 EDT
#
# @description : Make sure all the notebooks run using papermill 
######################################################################
# 1. First go to dataSc env and run papermill command with input.ipynb output.ipynb [commands]


# datasc env
source activate datasc


for f in *.ipynb
do 
  echo "$f"
  papermill "$f" test_nb_runs/out_"$f"
done;

# get out of env
conda deactivate
