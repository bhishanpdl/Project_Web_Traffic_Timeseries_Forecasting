import sys
import glob

github = "https://github.com/"
user = "bhishanpdl"
project = "/Project_House_Price_Prediction"
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


"""
notebook: a01_classification_eda.ipynb
github link: https://github.com/bhishanpdl/Project_Fraud_Detection/blob/master/notebooks/a01_classification_eda.ipynb

nbviewer: https://nbviewer.jupyter.org/github/bhishanpdl/Project_Fraud_Detection/blob/master/notebooks/a01_classification_eda.ipynb

github = "https://github.com/"
user = "bhishanpdl"
project = "/Project_Fraud_Detection"
path = "/blob/master/notebooks/"
notebook = "a01_classification_eda.ipynb"

nbview = "https://nbviewer.jupyter.org/github/"

gh_link   = github + user  + project + path + notebook
nb_link   = nbview + user + project + path + notebook

""";