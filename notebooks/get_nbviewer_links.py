import sys
import glob

github = "https://github.com/"
user = "bhishanpdl"
project = "/Project_Web_Traffic_Timeseries_Forecasting"
path = "/blob/master/notebooks/"
nbview = "https://nbviewer.jupyter.org/github/"

notebooks = sorted([ nb for nb in glob.glob('*.ipynb')])

with open('nb_links.md','w') as fo:
    # first write the header
    line = "|  Notebook | Rendered   | Description  |  Author |"
    fo.write(line + '\n')

    line = "|---|---|---|---|"
    fo.write(line + '\n')

    # then loop over all the notebooks
    for notebook in notebooks:
        gh_link = github + user + project + path + notebook
        nb_link = nbview + user + project + path + notebook

        line = """| {notebook}  | [ipynb]({gh_link}), [rendered]({nb_link})  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |""".format(notebook=notebook,nb_link=nb_link,gh_link=gh_link)
        fo.write(line + '\n')