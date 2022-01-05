#!/bin/bash

path_to_the_local_python_environment=$1
path_to_the_local_R_environment=$2

source "$path_to_the_local_python_environment/Scripts/activate"
pip install --no-cache-dir -r requirements.txt
"$path_to_the_local_R_environment/bin/RScript.exe" -e "install.packages('clValid', repos='https://cran.rstudio.com/')"
"$path_to_the_local_R_environment/bin/RScript.exe" -e "install.packages('symbolicDA', repos='https://cran.rstudio.com/')"

echo 'All required packeges installed successfully'
sleep 30