#!/usr/bin/env sh

######################################################################
# @author      : Bhishan (Bhishan@BpMacpro.local)
# @file        : get_html
# @created     : Saturday Oct 17, 2020 14:51:31 EDT
#
# @description : Create html files from notebooks
######################################################################
rm -rf html/*.html

/Users/poudel/opt/miniconda3/envs/dataSc/bin/jupyter nbconvert *.ipynb --to html && mv *.html html/


