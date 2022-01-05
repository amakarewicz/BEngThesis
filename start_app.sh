#!/bin/bash

path_to_the_local_python_environment=$1
#source $path_to_the_local_python_environment/Scripts/activate
#pip install -r requirements.txt
#"C:/Program Files/R/R-4.0.3/bin/RScript.exe" -e "install.packages('clValid', repos='https://cran.rstudio.com/')"
#"C:/Program Files/R/R-4.0.3/bin/RScript.exe" -e "install.packages('symbolicDA', repos='https://cran.rstudio.com/')"
$path_to_the_local_python_environment/Scripts/python.exe django/EuropeClustering/manage.py runserver 8000 
sleep 30