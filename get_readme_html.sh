rm -rf readme.html
pandoc README.md -f markdown -t html -s -o readme.html
