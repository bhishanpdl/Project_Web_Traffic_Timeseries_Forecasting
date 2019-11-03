#!/usr/bin/env sh

######################################################################
# @author      : Bhishan (Bhishan@BpMacpro.local)
# @file        : get_nbviewer_links
# @created     : Sunday Oct 20, 2019 12:02:57 EDT
#
# @description : Create nbviewer links for notebooks 
######################################################################
pre="https://nbviewer.jupyter.org/github/"
user="bhishanpdl/"
project="Project_House_Price_Prediction/"
notebook="blob/master/notebooks/"

for fname in *.ipynb;
    do echo "[$fname](""$pre""$user""$project""$notebook""$fname)\n" >> README.md;
    done;

