# BEngThesis

## Application for Analysis of the Economic Growth Indexes for European Countries

*Authors: Agata Makarewicz, Jacek Wiśniewski*

*Faculty of Mathematics & Information Science, Warsaw University of Technology*

### Aim of the work

The aim of the diploma thesis is to apply various clustering algorithms to the time series of economic growth of European countries. The analyzed data will be pre-processed using methods such as segmentation, normalization or anomaly removal. The results of the work will be presented in the form of an application with a graphical user interface written in Django, which will allow the user to compare the indicators for different countries.

### How to run application

## Deployment 
Instruction for proper environment configuration presented below applies to working on Windows operating system. Approximately 8-9GB free disk space is required.
1) Install Python 3.8.2: https://www.python.org/downloads/release/python-382/ (newer releases of Python 3.8 should also work, however it was not tested)
    1) During installation, check the box “Add Python 3.8 to path”
2) Install R: https://cran.r-project.org/bin/windows/base (newest version 4.1.2 or any other not older than 3.6 works)
    1)Follow default installation. After completion, open R (*C:\Program Files\R\R-<version>\bin\R.exe*) and run following command: `dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)`. This will create a user library for R located in *C:\Users\<user>\Documents\R\win-library\<version>* .
3) Open “Edit system environment variables”, go to “Environment variables” and add new user variable.
    1) Name: *R_LIBS_USER*
    2) Value: *C:\Users\<user>\Documents\R\win-library\<version>* (path to R user library)
4) Install Git: https://git-scm.com/download/win . Follow default installation.
5) Install Microsoft Visual C++ Build Tools: https://visualstudio.microsoft.com/pl/visual-cpp-build-tools
    1) Run the installer. Select only first component (“Desktop development with C++”) and then proceed with the default selection of checkboxes on the right. There should be no more than 7GB of data to download. At the end of the installation it is required to restart the system.
6) Install Google Chrome or Microsoft Edge browser.
7) Create Python virtual environment. Open Command Line, enter the desired location and run following command: `python -m venv <environment_name>`. This will create a directory with separate Python executable, to which all required packages will be installed.

## Installation 
1) Clone GitHub repository. Open Git Bash, navigate to the desired location and run following command: `git clone https://github.com/amakarewicz/BEngThesis` OR
Download GitHub repository. Go to https://github.com/amakarewicz/BEngThesis -> *Code* -> *Download ZIP*. Extract the contents into desired directory.
2) Run Command Line (best as an Administrator) and enter the directory with the repository (*BEngThesis* if it is cloned / *BEngThesis-main* if downloaded).
    1) Example: *C:\Users\agama\Documents\BEngThesis*
3) Run *install_requirements.sh* file, adding path to created previously Python virtual environment directory and path to R directory as command arguments.
    1) Command template: `install_requirements.sh "{/path/to/python/env }" "{/path/to/R}”`
    2) Example: `install_requirements.sh “C:\Users\agama\Documents\BEngThesis\django\bengthesis” “C:\Program Files\R\R-4.0.0”`
4) Run *start_app.sh* file, adding path to created previously Python virtual environment directory as command argument (same as in the previous step).
    1) Command template: `start_app.sh “{/path to python/env}“`
    2) Example: `start_app.sh “C:\Users\agama\Documents\BEngThesis\django\bengthesis”`
    3) If following warnings appear, do not worry, application is running.
        1) UserWarning: h5py not installed, hdf5 features will not be supported.
        2) UserWarning: R is not initialized by the main thread. Its taking over SIGINT cannot be reversed here, and as a consequence the embedded R cannot be interrupted with Ctrl-C. Consider (re)setting the signal handler of your choice from the main thread.
5) Run http://127.0.0.1:8000/homepage in your browser.
